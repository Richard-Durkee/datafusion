// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Arrow-IPC backed [`PartitionSpiller`] for `(group_idx, T::Native)` keys.
//!
//! Bridges [`SpillableHashSet`] to DataFusion's `SpillManager` for use inside
//! `PrimitiveDistinctCountGroupsAccumulator`. Each grace-hash partition gets
//! its own set of spill files; keys are serialized as a two-column Arrow
//! RecordBatch `[group_idx: UInt32, value: T]` in IPC streaming format.

use std::collections::HashMap;
use std::hash::Hash;
use std::mem::size_of;
use std::sync::Arc;

use arrow::array::{ArrayRef, AsArray, PrimitiveArray, UInt32Array};
use arrow::datatypes::{ArrowPrimitiveType, DataType, Field, Schema, SchemaRef};
use arrow::ipc::reader::StreamReader;
use arrow::record_batch::RecordBatch;

use crate::spill::spill_manager::SpillManager;
use datafusion_execution::spill_file::SpillFile;
use datafusion_execution::runtime_env::RuntimeEnv;
use datafusion_functions_aggregate_common::spillable_hash_set::{
    PartitionSpiller, SpillError,
};

/// Arrow-IPC spiller for `(usize, T::Native)` keys used by the primitive
/// distinct count accumulator.
///
/// Each `append()` call creates one IPC spill file for the partition. At
/// `read_all()` time, all files for a partition are read back sequentially
/// using Arrow's synchronous `StreamReader` (via `SpillFile::path()`).
pub struct ArrowPrimitiveSpiller<T: ArrowPrimitiveType> {
    spill_manager: SpillManager,
    /// Per-partition: accumulated spill files (one per append call).
    partition_files: HashMap<usize, Vec<Arc<dyn SpillFile>>>,
    /// Schema for the two-column batch: [group_idx: UInt32, value: T]
    schema: SchemaRef,
    /// Running count of key pairs spilled, for memory accounting.
    spilled_key_count: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: ArrowPrimitiveType> ArrowPrimitiveSpiller<T>
where
    T::Native: Eq + Hash,
{
    /// Create a new spiller. The provided `SpillManager`'s schema is ignored;
    /// we use our own two-column schema internally.
    pub fn new(base_spill_manager: &SpillManager) -> Self {
        let schema = Arc::new(Schema::new(vec![
            Field::new("group_idx", DataType::UInt32, false),
            Field::new("value", T::DATA_TYPE, false),
        ]));
        // Clone the base and rebuild with our schema. SpillManager is Clone
        // (wraps Arc<RuntimeEnv> internally), so this is cheap.
        let mut spill_manager = base_spill_manager.clone();
        // We need a SpillManager with our 2-column schema. Since there's no
        // `with_schema` method, construct fresh from the same env + metrics.
        let env: Arc<RuntimeEnv> = Arc::new(base_spill_manager.env().clone());
        spill_manager = SpillManager::new(
            env,
            base_spill_manager.metrics.clone(),
            Arc::clone(&schema),
        );
        Self {
            spill_manager,
            partition_files: HashMap::new(),
            schema,
            spilled_key_count: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Approximate bytes of keys currently on disk.
    pub fn spilled_bytes(&self) -> usize {
        self.spilled_key_count * (size_of::<u32>() + size_of::<T::Native>())
    }
}

impl<T: ArrowPrimitiveType> PartitionSpiller<(usize, T::Native)>
    for ArrowPrimitiveSpiller<T>
where
    T::Native: Eq + Hash + Clone,
{
    fn append(
        &mut self,
        partition: usize,
        keys: &[(usize, T::Native)],
    ) -> Result<(), SpillError> {
        if keys.is_empty() {
            return Ok(());
        }

        // Build Arrow arrays from the key pairs.
        let group_indices: Vec<u32> = keys.iter().map(|(g, _)| *g as u32).collect();
        let values: Vec<T::Native> = keys.iter().map(|(_, v)| v.clone()).collect();
        let num_keys = keys.len();

        let group_array: ArrayRef = Arc::new(UInt32Array::from(group_indices));
        let value_array: ArrayRef =
            Arc::new(PrimitiveArray::<T>::from_iter_values(values));

        let batch = RecordBatch::try_new(
            Arc::clone(&self.schema),
            vec![group_array, value_array],
        )
        .map_err(|e| SpillError(format!("failed to create batch: {e}")))?;

        let spill_file = self
            .spill_manager
            .spill_record_batch_and_finish(
                &[batch],
                &format!("DistinctCountSpill_p{partition}"),
            )
            .map_err(|e| SpillError(format!("spill write failed: {e}")))?;

        if let Some(file) = spill_file {
            self.partition_files.entry(partition).or_default().push(file);
            self.spilled_key_count += num_keys;
        }

        Ok(())
    }

    fn read_all(
        &mut self,
        partition: usize,
    ) -> Result<Vec<(usize, T::Native)>, SpillError> {
        let Some(files) = self.partition_files.remove(&partition) else {
            return Ok(Vec::new());
        };

        let mut result = Vec::new();

        for file in &files {
            let path = file
                .path()
                .ok_or_else(|| SpillError("spill file has no local path".into()))?;

            let f = std::fs::File::open(path)
                .map_err(|e| SpillError(format!("open spill file: {e}")))?;

            let reader = StreamReader::try_new(f, None)
                .map_err(|e| SpillError(format!("IPC stream reader: {e}")))?;

            for batch_result in reader {
                let batch = batch_result
                    .map_err(|e| SpillError(format!("read batch: {e}")))?;
                let groups =
                    batch.column(0).as_primitive::<arrow::datatypes::UInt32Type>();
                let values = batch.column(1).as_primitive::<T>();
                for i in 0..batch.num_rows() {
                    result.push((groups.value(i) as usize, values.value(i)));
                }
            }
        }

        // Keys have been read back; decrement the spilled count.
        self.spilled_key_count = self.spilled_key_count.saturating_sub(result.len());

        Ok(result)
    }
}
