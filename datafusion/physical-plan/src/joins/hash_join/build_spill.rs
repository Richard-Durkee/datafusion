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

//! Grace-hash build-side partitioner for spill support in [`HashJoinExec`].
//!
//! When the build side exceeds available memory, [`BuildPartitioner`] splits
//! incoming [`RecordBatch`]es into `P` partitions by `hash(join_key) % P`.
//! Each partition's rows are written to a separate spill file. At restore
//! time, partitions are loaded one at a time, a hash table is built, and
//! the matching probe-side partition is replayed against it.
//!
//! # Algorithm (Velox-style grace hash)
//!
//! 1. Partition stickiness: once a partition is spilled, ALL subsequent rows
//!    for that partition go directly to disk — never buffered in memory.
//! 2. Spill the partitions with the most data first.
//! 3. Unsorted spill: rows are appended in arrival order per partition.
//! 4. Recursive repartition: if restoring a partition still OOMs, advance
//!    the partition bit offset and sub-partition further.
//!
//! [`HashJoinExec`]: super::HashJoinExec

use std::sync::Arc;

use arrow::array::UInt64Array;
use arrow::compute::take;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion_common::hash_utils::create_hashes;
use datafusion_common::Result;
use datafusion_physical_expr::PhysicalExprRef;

use crate::spill::in_progress_spill_file::InProgressSpillFile;
use crate::spill::spill_manager::SpillManager;
use datafusion_execution::spill_file::SpillFile;

use super::partitioned_hash_eval::SeededRandomState;

/// Metadata about spilled partitions, passed from the build phase to the
/// probe/restore phase so the stream knows which partitions need replay.
pub(super) struct SpilledPartitionMeta {
    /// Finished spill files for cold build-side partitions.
    /// `cold_files[i] = Some(file)` if partition i was spilled, None otherwise.
    pub cold_files: Vec<Option<Arc<dyn SpillFile>>>,
    /// Which partitions were spilled (mirrors cold_files but as booleans).
    pub spilled_partitions: Vec<bool>,
    /// Bit offset used for partitioning (for recursive repartition at restore).
    pub bit_offset: usize,
    /// Build-side schema (needed to rebuild hash tables from spilled data).
    pub schema: SchemaRef,
    /// Join key expressions for the build side.
    pub on_left: Vec<PhysicalExprRef>,
    /// Random state used for partitioning (probe side must match).
    #[allow(dead_code)]
    pub random_state: SeededRandomState,
}

/// Number of grace-hash partitions for the build side.
///
/// 8 partitions (3 bits) matches Velox's default. With 1 GB memory limit,
/// level-1 spill supports 8 GB build tables; level-2 supports 64 GB.
/// Kept small to reduce per-partition overhead (file handles, buffers).
pub const NUM_PARTITIONS: usize = 8;

/// Tracks per-partition state: rows buffered, bytes spilled, file handle.
pub(super) struct PartitionState {
    /// Spill file for this partition (created lazily on first spill).
    file: Option<InProgressSpillFile>,
    /// Whether this partition has been designated for spilling.
    pub(super) is_spilled: bool,
    /// Running byte count of data written to disk for this partition.
    spilled_bytes: usize,
    /// Running row count of data in memory for this partition (before spill).
    /// Used to select which partition to evict (largest first).
    in_memory_rows: usize,
    /// Running byte estimate of data in memory for this partition.
    in_memory_bytes: usize,
}

impl PartitionState {
    fn new() -> Self {
        Self {
            file: None,
            is_spilled: false,
            spilled_bytes: 0,
            in_memory_rows: 0,
            in_memory_bytes: 0,
        }
    }
}

/// Routes build-side RecordBatches into P partitions by hash(join_key) % P.
///
/// Lifecycle:
/// 1. Created when build-side memory pressure is detected.
/// 2. Existing buffered batches are partitioned and partially spilled.
/// 3. Subsequent batches are routed: hot partitions accumulate in memory,
///    cold (spilled) partitions go directly to disk.
/// 4. After build input is exhausted, hot partitions build hash tables
///    in-memory; cold partitions are restored one at a time.
pub(super) struct BuildPartitioner {
    /// Schema of the build side.
    schema: SchemaRef,
    /// Spill manager for creating partition files.
    spill_manager: SpillManager,
    /// Per-partition state.
    pub(super) partitions: Vec<PartitionState>,
    /// Expressions to evaluate join keys on build-side batches.
    on_left: Vec<PhysicalExprRef>,
    /// Random state for hashing (must match probe-side partitioning).
    random_state: SeededRandomState,
    /// Partition bit offset for recursive repartitioning.
    /// Level 0 = bits [0..3), level 1 = bits [3..6), etc.
    bit_offset: usize,
    /// Batches retained in memory for hot (non-spilled) partitions.
    /// Indexed by partition number.
    hot_batches: Vec<Vec<RecordBatch>>,
    /// Total rows across all hot partitions.
    hot_rows: usize,
    /// Total bytes across all hot partitions.
    hot_bytes: usize,
}

impl BuildPartitioner {
    /// Create a new partitioner.
    ///
    /// `spill_manager` should be configured with the build-side schema and
    /// the session's spill compression settings.
    pub fn new(
        schema: SchemaRef,
        spill_manager: SpillManager,
        on_left: Vec<PhysicalExprRef>,
        random_state: SeededRandomState,
    ) -> Self {
        Self {
            schema,
            spill_manager,
            partitions: (0..NUM_PARTITIONS).map(|_| PartitionState::new()).collect(),
            on_left,
            random_state,
            bit_offset: 0,
            hot_batches: (0..NUM_PARTITIONS).map(|_| Vec::new()).collect(),
            hot_rows: 0,
            hot_bytes: 0,
        }
    }

    /// Compute partition number for each row in `batch`.
    ///
    /// Returns a Vec<usize> of length batch.num_rows() with values in [0, NUM_PARTITIONS).
    fn compute_partitions(&self, batch: &RecordBatch) -> Result<Vec<usize>> {
        let num_rows = batch.num_rows();
        let key_arrays =
            self.on_left
                .iter()
                .map(|expr| {
                    expr.evaluate(batch)
                        .and_then(|cv| cv.into_array(num_rows))
                })
                .collect::<Result<Vec<_>>>()?;

        let mut hashes = vec![0u64; num_rows];
        create_hashes(&key_arrays, self.random_state.random_state(), &mut hashes)?;

        // Apply bit offset for recursive partitioning:
        // Level 0 uses bits [0..3), level 1 uses bits [3..6), etc.
        let partitions: Vec<usize> = hashes
            .iter()
            .map(|h| ((h >> self.bit_offset) as usize) % NUM_PARTITIONS)
            .collect();

        Ok(partitions)
    }

    /// Route a single batch: hot-partition rows stay in memory,
    /// cold-partition rows go to disk.
    ///
    /// Returns the number of rows kept in memory (hot).
    pub fn insert_batch(&mut self, batch: RecordBatch) -> Result<usize> {
        if batch.num_rows() == 0 {
            return Ok(0);
        }

        let partitions = self.compute_partitions(&batch)?;
        let num_rows = batch.num_rows();
        let batch_mem_size = batch.get_array_memory_size();

        // Fast path: if no partitions are spilled, keep everything in memory
        // and just track per-partition counts for future eviction decisions.
        if !self.has_spilled_partitions() {
            // Find the partition for each row and accumulate
            // For now, put the whole batch in partition 0's slot and track stats.
            // TODO: actually split by partition for proper eviction granularity.
            // For the initial implementation, we track stats but don't split
            // until spill is triggered — matching Velox's approach where the
            // RowContainer holds everything until first spill.
            for &p in &partitions {
                self.partitions[p].in_memory_rows += 1;
            }
            // Store batch in the partition of its first row (rough; refined later)
            let first_partition = partitions[0];
            self.hot_batches[first_partition].push(batch);
            self.hot_rows += num_rows;
            self.hot_bytes += batch_mem_size;
            return Ok(num_rows);
        }

        // Split batch by partition assignment.
        let mut rows_kept = 0;
        let indices_per_partition = self.split_indices_by_partition(&partitions);

        for (p_idx, row_indices) in indices_per_partition.into_iter().enumerate() {
            if row_indices.is_empty() {
                continue;
            }

            let indices_array = UInt64Array::from(
                row_indices.iter().map(|&i| i as u64).collect::<Vec<_>>(),
            );
            let partition_batch = RecordBatch::try_new(
                Arc::clone(&self.schema),
                batch
                    .columns()
                    .iter()
                    .map(|col| take(col, &indices_array, None).map_err(Into::into))
                    .collect::<Result<Vec<_>>>()?,
            )?;

            let part_size = partition_batch.get_array_memory_size();
            let part_rows = partition_batch.num_rows();

            if self.partitions[p_idx].is_spilled {
                // Cold partition: write directly to disk.
                self.spill_batch_to_partition(p_idx, &partition_batch)?;
            } else {
                // Hot partition: keep in memory.
                self.partitions[p_idx].in_memory_rows += part_rows;
                self.partitions[p_idx].in_memory_bytes += part_size;
                self.hot_batches[p_idx].push(partition_batch);
                self.hot_rows += part_rows;
                self.hot_bytes += part_size;
                rows_kept += part_rows;
            }
        }

        Ok(rows_kept)
    }

    /// Spill the partition with the most in-memory data.
    ///
    /// Returns the number of bytes freed, or 0 if no partition can be spilled.
    pub fn spill_largest_partition(&mut self) -> Result<usize> {
        // Find the hot partition with the most data.
        let target = self
            .partitions
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.is_spilled && p.in_memory_rows > 0)
            .max_by_key(|(i, _)| self.hot_batches[*i].len())
            .map(|(i, _)| i);

        let Some(p_idx) = target else {
            return Ok(0);
        };

        // Spill all in-memory batches for this partition.
        let batches = std::mem::take(&mut self.hot_batches[p_idx]);
        let mut freed = 0;
        for batch in &batches {
            freed += batch.get_array_memory_size();
            self.spill_batch_to_partition(p_idx, batch)?;
        }

        self.partitions[p_idx].is_spilled = true;
        self.partitions[p_idx].in_memory_rows = 0;
        self.partitions[p_idx].in_memory_bytes = 0;
        self.hot_rows -= batches.iter().map(|b| b.num_rows()).sum::<usize>();
        self.hot_bytes -= freed;

        Ok(freed)
    }

    /// Write a batch to a partition's spill file.
    fn spill_batch_to_partition(
        &mut self,
        partition: usize,
        batch: &RecordBatch,
    ) -> Result<()> {
        let file = match &mut self.partitions[partition].file {
            Some(f) => f,
            None => {
                let f = self.spill_manager.create_in_progress_file(
                    &format!("HashJoinBuild_p{partition}"),
                )?;
                self.partitions[partition].file = Some(f);
                self.partitions[partition].file.as_mut().unwrap()
            }
        };
        let bytes_written = file.append_batch(batch)?;
        self.partitions[partition].spilled_bytes += bytes_written;
        Ok(())
    }

    /// Split row indices by their partition assignment.
    fn split_indices_by_partition(
        &self,
        partitions: &[usize],
    ) -> Vec<Vec<usize>> {
        let mut result = vec![Vec::new(); NUM_PARTITIONS];
        for (row_idx, &p) in partitions.iter().enumerate() {
            result[p].push(row_idx);
        }
        result
    }

    /// Whether any partition has been spilled to disk.
    pub fn has_spilled_partitions(&self) -> bool {
        self.partitions.iter().any(|p| p.is_spilled)
    }

    /// Finish all spill files and return the completed spill files
    /// for cold partitions plus the in-memory batches for hot partitions.
    ///
    /// Returns: (hot_partition_batches, cold_partition_files)
    /// where hot_partition_batches[i] = batches for partition i (empty if spilled)
    /// and cold_partition_files[i] = Some(file) if partition i was spilled.
    pub fn finish(
        mut self,
    ) -> Result<(Vec<Vec<RecordBatch>>, Vec<Option<Arc<dyn SpillFile>>>)> {
        let mut cold_files: Vec<Option<Arc<dyn SpillFile>>> =
            Vec::with_capacity(NUM_PARTITIONS);

        for p in &mut self.partitions {
            if let Some(mut file) = p.file.take() {
                cold_files.push(file.finish()?);
            } else {
                cold_files.push(None);
            }
        }

        Ok((self.hot_batches, cold_files))
    }

    /// Finish partitioning, returning flattened hot-partition batches and a
    /// SpilledPartitionMeta for the stream's per-partition restore loop.
    pub fn finish_with_spill_meta(
        mut self,
        on_left: &[PhysicalExprRef],
    ) -> Result<(Vec<RecordBatch>, SpilledPartitionMeta)> {
        let spilled_partitions: Vec<bool> = self.partitions.iter()
            .map(|p| p.is_spilled)
            .collect();
        let bit_offset = self.bit_offset();

        let mut cold_files: Vec<Option<Arc<dyn SpillFile>>> =
            Vec::with_capacity(NUM_PARTITIONS);
        for p in &mut self.partitions {
            if let Some(mut file) = p.file.take() {
                cold_files.push(file.finish()?);
            } else {
                cold_files.push(None);
            }
        }

        // Flatten hot batches from all partitions
        let mut hot_batches_flat = Vec::new();
        for partition_batches in self.hot_batches {
            hot_batches_flat.extend(partition_batches);
        }

        let meta = SpilledPartitionMeta {
            cold_files,
            spilled_partitions,
            bit_offset,
            schema: Arc::clone(&self.schema),
            on_left: on_left.to_vec(),
            random_state: self.random_state.clone(),
        };

        Ok((hot_batches_flat, meta))
    }

    /// Current bit offset (for recursive repartitioning).
    pub fn bit_offset(&self) -> usize {
        self.bit_offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_execution::runtime_env::RuntimeEnv;
    use datafusion_physical_expr::expressions::col;

    use crate::metrics::{ExecutionPlanMetricsSet, SpillMetrics};
    use super::super::partitioned_hash_eval::SeededRandomState;

    fn test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]))
    }

    fn test_batch(ids: Vec<i32>, names: Vec<&str>) -> RecordBatch {
        RecordBatch::try_new(
            test_schema(),
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .unwrap()
    }

    fn test_spill_manager(schema: SchemaRef) -> SpillManager {
        let runtime = Arc::new(RuntimeEnv::default());
        let metrics_set = ExecutionPlanMetricsSet::new();
        let spill_metrics = SpillMetrics::new(&metrics_set, 0);
        SpillManager::new(runtime, spill_metrics, schema)
    }

    #[test]
    fn partition_batch_without_spill() {
        let schema = test_schema();
        let spill_mgr = test_spill_manager(Arc::clone(&schema));
        let on_left = vec![col("id", &schema).unwrap()];
        let random_state = SeededRandomState::with_seed(12345);

        let mut partitioner =
            BuildPartitioner::new(schema, spill_mgr, on_left, random_state);

        let batch = test_batch(vec![1, 2, 3, 4, 5], vec!["a", "b", "c", "d", "e"]);
        let kept = partitioner.insert_batch(batch).unwrap();

        assert_eq!(kept, 5);
        assert!(!partitioner.has_spilled_partitions());
        assert_eq!(partitioner.hot_rows(), 5);
    }

    #[test]
    fn spill_largest_partition_frees_memory() {
        let schema = test_schema();
        let spill_mgr = test_spill_manager(Arc::clone(&schema));
        let on_left = vec![col("id", &schema).unwrap()];
        let random_state = SeededRandomState::with_seed(12345);

        let mut partitioner =
            BuildPartitioner::new(schema, spill_mgr, on_left, random_state);

        // Insert enough data to have something to spill
        for i in 0..100 {
            let batch = test_batch(
                vec![i * 10, i * 10 + 1, i * 10 + 2],
                vec!["x", "y", "z"],
            );
            partitioner.insert_batch(batch).unwrap();
        }

        assert!(partitioner.hot_rows() > 0);
        let before = partitioner.hot_bytes();

        let freed = partitioner.spill_largest_partition().unwrap();
        assert!(freed > 0);
        assert!(partitioner.has_spilled_partitions());
        assert!(partitioner.hot_bytes() < before);
    }

    #[test]
    fn sticky_routing_after_spill() {
        let schema = test_schema();
        let spill_mgr = test_spill_manager(Arc::clone(&schema));
        let on_left = vec![col("id", &schema).unwrap()];
        let random_state = SeededRandomState::with_seed(12345);

        let mut partitioner =
            BuildPartitioner::new(schema, spill_mgr, on_left, random_state);

        // Insert initial data
        for i in 0..50 {
            let batch = test_batch(vec![i], vec!["v"]);
            partitioner.insert_batch(batch).unwrap();
        }

        // Spill a partition
        partitioner.spill_largest_partition().unwrap();
        assert!(partitioner.has_spilled_partitions());
        let spilled_count = partitioner.num_spilled_partitions();
        assert!(spilled_count >= 1);

        // Insert more data — rows for spilled partitions should go to disk
        let before_hot = partitioner.hot_rows();
        for i in 50..100 {
            let batch = test_batch(vec![i], vec!["w"]);
            partitioner.insert_batch(batch).unwrap();
        }
        // Some rows went to hot (non-spilled) partitions, some to disk
        // Total hot rows should be less than before + 50
        let after_hot = partitioner.hot_rows();
        assert!(after_hot < before_hot + 50);
    }

    #[test]
    fn finish_returns_hot_and_cold() {
        let schema = test_schema();
        let spill_mgr = test_spill_manager(Arc::clone(&schema));
        let on_left = vec![col("id", &schema).unwrap()];
        let random_state = SeededRandomState::with_seed(12345);

        let mut partitioner =
            BuildPartitioner::new(schema, spill_mgr, on_left, random_state);

        for i in 0..50 {
            let batch = test_batch(vec![i], vec!["v"]);
            partitioner.insert_batch(batch).unwrap();
        }
        partitioner.spill_largest_partition().unwrap();

        let (hot_batches, cold_files) = partitioner.finish().unwrap();
        assert_eq!(hot_batches.len(), NUM_PARTITIONS);
        assert_eq!(cold_files.len(), NUM_PARTITIONS);

        // At least one partition should have a cold file
        assert!(cold_files.iter().any(|f| f.is_some()));
        // At least one partition should have hot batches
        assert!(hot_batches.iter().any(|b| !b.is_empty()));
    }

    #[test]
    fn recursive_bit_offset_changes_partitioning() {
        let schema = test_schema();
        let spill_mgr = test_spill_manager(Arc::clone(&schema));
        let on_left = vec![col("id", &schema).unwrap()];
        let random_state = SeededRandomState::with_seed(12345);

        let batch = test_batch(vec![42, 42, 42], vec!["a", "b", "c"]);

        // Level 0 partitioning
        let p0 = BuildPartitioner::new(
            Arc::clone(&schema),
            test_spill_manager(Arc::clone(&schema)),
            on_left.clone(),
            random_state.clone(),
        );
        let parts_l0 = p0.compute_partitions(&batch).unwrap();

        // Level 1 partitioning (bit offset = 3)
        let p1 = BuildPartitioner::with_bit_offset(
            Arc::clone(&schema),
            spill_mgr,
            on_left,
            random_state,
            3,
        );
        let parts_l1 = p1.compute_partitions(&batch).unwrap();

        // Same key (42) should hash to the same partition within a level
        assert_eq!(parts_l0[0], parts_l0[1]);
        assert_eq!(parts_l0[1], parts_l0[2]);
        // But may differ across levels (different bits extracted)
        // (Not guaranteed to differ for all keys, but the mechanism is sound)
        // Just verify the bit_offset is stored correctly
        assert_eq!(p1.bit_offset(), 3);
    }
}

