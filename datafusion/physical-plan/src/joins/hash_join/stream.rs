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

//! Stream implementation for Hash Join
//!
//! This module implements [`HashJoinStream`], the streaming engine for
//! [`super::HashJoinExec`]. See comments in [`HashJoinStream`] for more details.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::task::Poll;

use crate::coalesce::{LimitedBatchCoalescer, PushBatchStatus};
use crate::joins::Map;
use crate::joins::MapOffset;
use crate::joins::PartitionMode;
use crate::joins::hash_join::exec::JoinLeftData;
use crate::joins::hash_join::shared_bounds::{
    PartitionBounds, PartitionBuildData, SharedBuildAccumulator,
};
use crate::joins::utils::{
    OnceFut, equal_rows_arr, get_final_indices_from_shared_bitmap, matchable_join_keys,
};
use crate::stream::EmptyRecordBatchStream;
use crate::{
    RecordBatchStream, SendableRecordBatchStream, handle_state,
    hash_utils::create_hashes,
    joins::utils::{
        BuildProbeJoinMetrics, ColumnIndex, JoinFilter, JoinHashMapType,
        StatefulStreamResult, adjust_indices_by_join_type, apply_join_filter_to_indices,
        build_batch_empty_build_side, build_batch_from_indices,
        need_produce_result_in_final,
    },
};

use arrow::array::{Array, ArrayRef, UInt32Array, UInt64Array};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion_common::{
    JoinSide, JoinType, NullEquality, Result, internal_datafusion_err, internal_err,
};
use datafusion_physical_expr::PhysicalExprRef;

use datafusion_common::hash_utils::RandomState;
use datafusion_execution::runtime_env::RuntimeEnv;
use datafusion_physical_expr_common::utils::evaluate_expressions_to_arrays;
use futures::{Stream, StreamExt, ready};

/// Represents build-side of hash join.
pub(super) enum BuildSide {
    /// Indicates that build-side not collected yet
    Initial(BuildSideInitialState),
    /// Indicates that build-side data has been collected
    Ready(BuildSideReadyState),
}

/// Container for BuildSide::Initial related data
pub(super) struct BuildSideInitialState {
    /// Future for building hash table from build-side input
    pub(super) left_fut: OnceFut<JoinLeftData>,
}

/// Container for BuildSide::Ready related data
pub(super) struct BuildSideReadyState {
    /// Collected build-side data
    left_data: Arc<JoinLeftData>,
}

impl BuildSide {
    /// Tries to extract BuildSideInitialState from BuildSide enum.
    /// Returns an error if state is not Initial.
    fn try_as_initial_mut(&mut self) -> Result<&mut BuildSideInitialState> {
        match self {
            BuildSide::Initial(state) => Ok(state),
            _ => internal_err!("Expected build side in initial state"),
        }
    }

    /// Tries to extract BuildSideReadyState from BuildSide enum.
    /// Returns an error if state is not Ready.
    fn try_as_ready(&self) -> Result<&BuildSideReadyState> {
        match self {
            BuildSide::Ready(state) => Ok(state),
            _ => internal_err!("Expected build side in ready state"),
        }
    }

    /// Tries to extract BuildSideReadyState from BuildSide enum.
    /// Returns an error if state is not Ready.
    fn try_as_ready_mut(&mut self) -> Result<&mut BuildSideReadyState> {
        match self {
            BuildSide::Ready(state) => Ok(state),
            _ => internal_err!("Expected build side in ready state"),
        }
    }
}

/// Represents state of HashJoinStream
///
/// Expected state transitions performed by HashJoinStream are:
///
/// ```text
///
///       WaitBuildSide
///             │
///             ▼
///  ┌─► FetchProbeBatch ───► ExhaustedProbeSide ───► Completed
///  │          │
///  │          ▼
///  └─ ProcessProbeBatch
/// ```
#[derive(Debug, Clone)]
pub(super) enum HashJoinStreamState {
    /// Initial state for HashJoinStream indicating that build-side data not collected yet
    WaitBuildSide,
    /// Waiting for bounds to be reported by all partitions
    WaitPartitionBoundsReport,
    /// Indicates that build-side has been collected, and stream is ready for fetching probe-side
    FetchProbeBatch,
    /// Indicates that non-empty batch has been fetched from probe-side, and is ready to be processed
    ProcessProbeBatch(ProcessProbeBatchState),
    /// Indicates that probe-side has been fully processed
    ExhaustedProbeSide,
    /// Restoring spilled partitions one at a time after the hot-partition
    /// probe is complete. Each cold partition is loaded, hash table rebuilt,
    /// and probed independently.
    RestoreSpilledPartition,
    /// Indicates that HashJoinStream execution is completed
    Completed,
}

impl HashJoinStreamState {
    /// Tries to extract ProcessProbeBatchState from HashJoinStreamState enum.
    /// Returns an error if state is not ProcessProbeBatchState.
    fn try_as_process_probe_batch_mut(&mut self) -> Result<&mut ProcessProbeBatchState> {
        match self {
            HashJoinStreamState::ProcessProbeBatch(state) => Ok(state),
            _ => internal_err!("Expected hash join stream in ProcessProbeBatch state"),
        }
    }
}

/// Container for HashJoinStreamState::ProcessProbeBatch related data
#[derive(Debug, Clone)]
pub(super) struct ProcessProbeBatchState {
    /// Current probe-side batch
    batch: RecordBatch,
    /// Probe-side on expressions values
    values: Vec<ArrayRef>,
    /// Combined validity of the probe-side key columns, set when NULL keys
    /// exist and cannot match (`NullEquality::NullEqualsNothing`); NULL rows
    /// are skipped during JoinHashMap lookups
    valid_keys: Option<NullBuffer>,
    /// Starting offset for JoinHashMap lookups
    offset: MapOffset,
    /// Max joined probe-side index from current batch
    joined_probe_idx: Option<usize>,
}

impl ProcessProbeBatchState {
    fn advance(&mut self, offset: MapOffset, joined_probe_idx: Option<usize>) {
        self.offset = offset;
        if joined_probe_idx.is_some() {
            self.joined_probe_idx = joined_probe_idx;
        }
    }
}

/// Lifecycle of this partition's build-data report to the shared coordinator.
///
/// `Scheduled` means the reporting `OnceFut` has been constructed but is lazy:
/// the coordinator has not necessarily observed the report. Only `Delivered`
/// guarantees the coordinator saw it, so `Drop` must still cancel a `Scheduled`
/// partition — otherwise sibling partitions can wait forever for a report that
/// never runs.
#[derive(Debug, PartialEq, Eq)]
enum BuildReportState {
    NotReported,
    Scheduled,
    Delivered,
    Canceled,
    Finalized,
}

/// Owns the stream-side lifecycle for one partition's build-data report.
struct BuildReportHandle {
    partition: usize,
    mode: PartitionMode,
    build_accumulator: Option<Arc<SharedBuildAccumulator>>,
    waiter: Option<OnceFut<()>>,
    state: BuildReportState,
}

impl BuildReportHandle {
    fn new(
        partition: usize,
        mode: PartitionMode,
        build_accumulator: Option<Arc<SharedBuildAccumulator>>,
    ) -> Self {
        Self {
            partition,
            mode,
            build_accumulator,
            waiter: None,
            state: BuildReportState::NotReported,
        }
    }

    fn has_accumulator(&self) -> bool {
        self.build_accumulator.is_some()
    }

    fn schedule(&mut self, build_data: PartitionBuildData) {
        let Some(build_accumulator) = &self.build_accumulator else {
            // Defensive no-op terminal state; current callers avoid scheduling
            // unless an accumulator is present.
            self.finalize();
            return;
        };

        debug_assert!(matches!(self.state, BuildReportState::NotReported));
        let acc = Arc::clone(build_accumulator);
        self.waiter = Some(OnceFut::new(async move {
            acc.report_build_data(build_data).await
        }));
        self.state = BuildReportState::Scheduled;
    }

    fn poll_delivery(&mut self, cx: &mut std::task::Context<'_>) -> Poll<Result<()>> {
        if let Some(ref mut fut) = self.waiter {
            ready!(fut.get_shared(cx))?;
            if !matches!(self.state, BuildReportState::Delivered) {
                debug_assert!(matches!(self.state, BuildReportState::Scheduled));
                self.state = BuildReportState::Delivered;
            }
        }
        Poll::Ready(Ok(()))
    }

    fn cancel_pending(&mut self) {
        if matches!(
            self.state,
            BuildReportState::Delivered
                | BuildReportState::Canceled
                | BuildReportState::Finalized
        ) {
            return;
        }

        if self.mode == PartitionMode::Partitioned
            && let Some(build_accumulator) = &self.build_accumulator
        {
            build_accumulator.report_canceled_partition(self.partition);
            self.state = BuildReportState::Canceled;
        } else {
            self.finalize();
        }
    }

    fn finalize(&mut self) {
        self.state = BuildReportState::Finalized;
    }

    #[cfg(test)]
    fn state(&self) -> &BuildReportState {
        &self.state
    }
}

impl Drop for BuildReportHandle {
    fn drop(&mut self) {
        self.cancel_pending();
    }
}

/// [`Stream`] for [`super::HashJoinExec`] that does the actual join.
///
/// This stream:
///
/// - Collecting the build side (left input) into a hash map
/// - Iterating over the probe side (right input) in streaming fashion
/// - Looking up matches against the hash table and applying join filters
/// - Producing joined [`RecordBatch`]es incrementally
/// - Emitting unmatched rows for outer/semi/anti joins in the final stage
pub(super) struct HashJoinStream {
    /// Partition identifier for debugging and determinism
    partition: usize,
    /// Input schema
    schema: Arc<Schema>,
    /// equijoin columns from the right (probe side)
    on_right: Vec<PhysicalExprRef>,
    /// optional join filter
    filter: Option<JoinFilter>,
    /// type of the join (left, right, semi, etc)
    join_type: JoinType,
    /// right (probe) input
    right: SendableRecordBatchStream,
    /// Random state used for hashing initialization
    random_state: RandomState,
    /// Metrics
    join_metrics: BuildProbeJoinMetrics,
    /// Information of index and left / right placement of columns
    column_indices: Vec<ColumnIndex>,
    /// Defines the null equality for the join.
    null_equality: NullEquality,
    /// State of the stream
    state: HashJoinStreamState,
    /// Build side
    build_side: BuildSide,
    /// Maximum output batch size
    batch_size: usize,
    /// Scratch space for computing hashes
    hashes_buffer: Vec<u64>,
    /// Scratch space for probe indices during hash lookup
    probe_indices_buffer: Vec<u32>,
    /// Scratch space for build indices during hash lookup
    build_indices_buffer: Vec<u64>,
    /// Specifies whether the right side has an ordering to potentially preserve
    right_side_ordered: bool,
    /// Owns this partition's build-data report lifecycle.
    build_report: BuildReportHandle,
    /// Partitioning mode to use
    mode: PartitionMode,
    /// Output buffer for coalescing small batches into larger ones with optional fetch limit.
    /// Uses `LimitedBatchCoalescer` to efficiently combine batches and absorb limit with 'fetch'
    output_buffer: LimitedBatchCoalescer,
    /// Whether this is a null-aware anti join
    null_aware: bool,
    /// Runtime environment for spill file I/O during per-partition restore
    runtime_env: Arc<RuntimeEnv>,
    /// Probe-side spill writers per partition (populated during initial probe
    /// when spill_meta exists, finalized during finalize_probe_spill).
    /// probe_spill_writers[i] = Some(writer) if partition i has cold probe rows.
    probe_spill_writers: Option<Vec<Option<crate::spill::in_progress_spill_file::InProgressSpillFile>>>,
    /// Finalized probe spill files (set by finalize_probe_spill, consumed by restore).
    probe_spill_files: Option<Vec<Option<Arc<dyn datafusion_execution::spill_file::SpillFile>>>>,
    /// Index of the next cold partition to restore (iterates through spill_meta.cold_files)
    next_cold_partition: usize,
}

impl RecordBatchStream for HashJoinStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

/// Executes lookups by hash against JoinHashMap and resolves potential
/// hash collisions.
/// Returns build/probe indices satisfying the equality condition, along with
/// (optional) starting point for next iteration.
///
/// # Example
///
/// For `LEFT.b1 = RIGHT.b2`:
/// LEFT (build) Table:
/// ```text
///  a1  b1  c1
///  1   1   10
///  3   3   30
///  5   5   50
///  7   7   70
///  9   8   90
///  11  8   110
///  13   10  130
/// ```
///
/// RIGHT (probe) Table:
/// ```text
///  a2   b2  c2
///  2    2   20
///  4    4   40
///  6    6   60
///  8    8   80
/// 10   10  100
/// 12   10  120
/// ```
///
/// The result is
/// ```text
/// "+----+----+-----+----+----+-----+",
/// "| a1 | b1 | c1  | a2 | b2 | c2  |",
/// "+----+----+-----+----+----+-----+",
/// "| 9  | 8  | 90  | 8  | 8  | 80  |",
/// "| 11 | 8  | 110 | 8  | 8  | 80  |",
/// "| 13 | 10 | 130 | 10 | 10 | 100 |",
/// "| 13 | 10 | 130 | 12 | 10 | 120 |",
/// "+----+----+-----+----+----+-----+"
/// ```
///
/// And the result of build and probe indices are:
/// ```text
/// Build indices: 4, 5, 6, 6
/// Probe indices: 3, 3, 4, 5
/// ```
#[expect(clippy::too_many_arguments)]
pub(super) fn lookup_join_hashmap(
    build_hashmap: &dyn JoinHashMapType,
    build_side_values: &[ArrayRef],
    probe_side_values: &[ArrayRef],
    null_equality: NullEquality,
    hashes_buffer: &[u64],
    valid_keys: Option<&NullBuffer>,
    limit: usize,
    offset: MapOffset,
    probe_indices_buffer: &mut Vec<u32>,
    build_indices_buffer: &mut Vec<u64>,
) -> Result<(UInt64Array, UInt32Array, Option<MapOffset>)> {
    let next_offset = build_hashmap.get_matched_indices_with_limit_offset(
        hashes_buffer,
        valid_keys,
        limit,
        offset,
        probe_indices_buffer,
        build_indices_buffer,
    );

    let build_indices_unfiltered: UInt64Array =
        std::mem::take(build_indices_buffer).into();
    let probe_indices_unfiltered: UInt32Array =
        std::mem::take(probe_indices_buffer).into();

    // TODO: optimize equal_rows_arr to avoid allocation of intermediate arrays
    // https://github.com/apache/datafusion/issues/12131
    let (build_indices, probe_indices) = equal_rows_arr(
        &build_indices_unfiltered,
        &probe_indices_unfiltered,
        build_side_values,
        probe_side_values,
        null_equality,
    )?;

    // Reclaim buffers
    *build_indices_buffer = build_indices_unfiltered.into_parts().1.into();
    *probe_indices_buffer = probe_indices_unfiltered.into_parts().1.into();

    Ok((build_indices, probe_indices, next_offset))
}

/// Counts the number of distinct elements in the input array.
///
/// The input array must be sorted (e.g., `[0, 1, 1, 2, 2, ...]`) and contain no null values.
#[inline]
fn count_distinct_sorted_indices(indices: &UInt32Array) -> usize {
    if indices.is_empty() {
        return 0;
    }

    debug_assert!(indices.null_count() == 0);

    let values_buf = indices.values();
    let values = values_buf.as_ref();
    let mut iter = values.iter();
    let Some(&first) = iter.next() else {
        return 0;
    };

    let mut count = 1usize;
    let mut last = first;
    for &value in iter {
        if value != last {
            last = value;
            count += 1;
        }
    }
    count
}

impl HashJoinStream {
    #[expect(clippy::too_many_arguments)]
    pub(super) fn new(
        partition: usize,
        schema: Arc<Schema>,
        on_right: Vec<PhysicalExprRef>,
        filter: Option<JoinFilter>,
        join_type: JoinType,
        right: SendableRecordBatchStream,
        random_state: RandomState,
        join_metrics: BuildProbeJoinMetrics,
        column_indices: Vec<ColumnIndex>,
        null_equality: NullEquality,
        state: HashJoinStreamState,
        build_side: BuildSide,
        batch_size: usize,
        hashes_buffer: Vec<u64>,
        right_side_ordered: bool,
        build_accumulator: Option<Arc<SharedBuildAccumulator>>,
        mode: PartitionMode,
        null_aware: bool,
        fetch: Option<usize>,
        runtime_env: Arc<RuntimeEnv>,
    ) -> Self {
        // Create output buffer with coalescing and optional fetch limit.
        let output_buffer =
            LimitedBatchCoalescer::new(Arc::clone(&schema), batch_size, fetch);

        Self {
            partition,
            schema,
            on_right,
            filter,
            join_type,
            right,
            random_state,
            join_metrics,
            column_indices,
            null_equality,
            state,
            build_side,
            batch_size,
            hashes_buffer,
            probe_indices_buffer: Vec::with_capacity(batch_size),
            build_indices_buffer: Vec::with_capacity(batch_size),
            right_side_ordered,
            build_report: BuildReportHandle::new(partition, mode, build_accumulator),
            mode,
            output_buffer,
            null_aware,
            runtime_env,
            probe_spill_writers: None,
            probe_spill_files: None,
            next_cold_partition: 0,
        }
    }

    /// Returns the next state after the build side has been fully collected
    /// and any required build-side coordination has completed.
    fn state_after_build_ready(
        join_type: JoinType,
        left_data: &JoinLeftData,
    ) -> HashJoinStreamState {
        let build_empty = !left_data.has_build_rows();
        // The map can be empty even when the build side has rows: under
        // `NullEqualsNothing`, build rows with a NULL join key are omitted. For
        // join types whose every output row requires a build match, that still
        // guarantees an empty result, so we can skip scanning the probe side.
        let map_empty = !left_data.has_matchable_build_rows();

        if (build_empty && join_type.empty_build_side_produces_empty_result())
            || (map_empty && join_type.empty_map_produces_empty_result())
        {
            HashJoinStreamState::Completed
        } else {
            HashJoinStreamState::FetchProbeBatch
        }
    }

    /// Transitions state after build-side data has been collected, automatically
    /// reporting build data to the accumulator when one is present.
    ///
    /// If a `build_accumulator` is configured, this method constructs the
    /// appropriate [`PartitionBuildData`], schedules the reporting future, and
    /// returns [`HashJoinStreamState::WaitPartitionBoundsReport`]. Otherwise it
    /// delegates to [`Self::state_after_build_ready`].
    fn transition_after_build_collected(
        &mut self,
        left_data: &Arc<JoinLeftData>,
    ) -> HashJoinStreamState {
        if !self.build_report.has_accumulator() {
            return Self::state_after_build_ready(self.join_type, left_data.as_ref());
        }

        let pushdown = left_data.membership().clone();
        let bounds = left_data
            .bounds
            .clone()
            .unwrap_or_else(|| PartitionBounds::new(vec![]));
        // Arrow tracks null counts per array, so this costs no data scan.
        let keys_have_null = left_data
            .values()
            .iter()
            .any(|array| array.null_count() > 0);

        let build_data = match self.mode {
            PartitionMode::Partitioned => PartitionBuildData::Partitioned {
                partition_id: self.partition,
                pushdown,
                bounds,
                keys_have_null,
            },
            PartitionMode::CollectLeft => PartitionBuildData::CollectLeft {
                pushdown,
                bounds,
                keys_have_null,
            },
            PartitionMode::Auto => unreachable!(
                "PartitionMode::Auto should not be present at execution time. This is a bug in DataFusion, please report it!"
            ),
        };

        self.build_report.schedule(build_data);
        HashJoinStreamState::WaitPartitionBoundsReport
    }

    /// Separate implementation function that unpins the [`HashJoinStream`] so
    /// that partial borrows work correctly
    fn poll_next_impl(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Result<RecordBatch>>> {
        loop {
            // First, check if we have any completed batches ready to emit
            if let Some(batch) = self.output_buffer.next_completed_batch() {
                return self
                    .join_metrics
                    .baseline
                    .record_poll(Poll::Ready(Some(Ok(batch))));
            }

            // Check if the coalescer has finished (limit reached and flushed)
            if self.output_buffer.is_finished() {
                return Poll::Ready(None);
            }

            return match self.state {
                HashJoinStreamState::WaitBuildSide => {
                    handle_state!(ready!(self.collect_build_side(cx)))
                }
                HashJoinStreamState::WaitPartitionBoundsReport => {
                    handle_state!(ready!(self.wait_for_partition_bounds_report(cx)))
                }
                HashJoinStreamState::FetchProbeBatch => {
                    handle_state!(ready!(self.fetch_probe_batch(cx)))
                }
                HashJoinStreamState::ProcessProbeBatch(_) => {
                    handle_state!(self.process_probe_batch())
                }
                HashJoinStreamState::ExhaustedProbeSide => {
                    handle_state!(self.process_unmatched_build_batch())
                }
                HashJoinStreamState::RestoreSpilledPartition => {
                    handle_state!(ready!(self.restore_spilled_partition(cx)))
                }
                HashJoinStreamState::Completed if !self.output_buffer.is_empty() => {
                    // Flush any remaining buffered data
                    self.output_buffer.finish()?;
                    // Continue loop to emit the flushed batch
                    continue;
                }
                HashJoinStreamState::Completed => Poll::Ready(None),
            };
        }
    }

    /// Optional step to wait until build-side information (hash maps or bounds) has been reported by all partitions.
    /// This state is only entered if a build accumulator is present.
    ///
    /// ## Why wait?
    ///
    /// The dynamic filter is only built once all partitions have reported their information (hash maps or bounds).
    /// If we do not wait here, the probe-side scan may start before the filter is ready.
    /// This can lead to the probe-side scan missing the opportunity to apply the filter
    /// and skip reading unnecessary data.
    fn wait_for_partition_bounds_report(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Result<StatefulStreamResult<Option<RecordBatch>>>> {
        ready!(self.build_report.poll_delivery(cx))?;
        let build_side = self.build_side.try_as_ready()?;
        self.state =
            Self::state_after_build_ready(self.join_type, build_side.left_data.as_ref());
        Poll::Ready(Ok(StatefulStreamResult::Continue))
    }

    /// Collects build-side data by polling `OnceFut` future from initialized build-side
    ///
    /// Updates build-side to `Ready`, and state to `FetchProbeSide`
    fn collect_build_side(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Result<StatefulStreamResult<Option<RecordBatch>>>> {
        let build_timer = self.join_metrics.build_time.timer();
        // build hash table from left (build) side, if not yet done
        let left_data = ready!(
            self.build_side
                .try_as_initial_mut()?
                .left_fut
                .get_shared(cx)
        )?;
        build_timer.done();

        // Note: For null-aware anti join, we need to check the probe side (right) for NULLs,
        // not the build side (left). The probe-side NULL check happens during process_probe_batch.
        // The probe_side_has_null flag will be set there if any probe batch contains NULL.

        self.state = self.transition_after_build_collected(&left_data);

        self.build_side = BuildSide::Ready(BuildSideReadyState { left_data });
        Poll::Ready(Ok(StatefulStreamResult::Continue))
    }

    /// Fetches next batch from probe-side
    ///
    /// If non-empty batch has been fetched, updates state to `ProcessProbeBatchState`,
    /// otherwise updates state to `ExhaustedProbeSide`
    fn fetch_probe_batch(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Result<StatefulStreamResult<Option<RecordBatch>>>> {
        match ready!(self.right.poll_next_unpin(cx)) {
            None => {
                // Finalize probe-side spill files before releasing the stream
                self.finalize_probe_spill()?;

                // Release the probe-side input pipeline's resources. The schema
                // is preserved so callers that still query `self.right.schema()`
                // (e.g. for unmatched-build emission) keep working.
                let right_schema = self.right.schema();
                self.right = Box::pin(EmptyRecordBatchStream::new(right_schema));
                self.state = HashJoinStreamState::ExhaustedProbeSide;
            }
            Some(Ok(batch)) => {
                // If spill_meta exists, partition probe rows: hot rows proceed
                // to ProcessProbeBatch, cold rows get spilled for later restore.
                let batch = self.maybe_spill_cold_probe_rows(batch)?;
                if batch.num_rows() == 0 {
                    // All rows in this batch went to cold partitions
                    self.state = HashJoinStreamState::FetchProbeBatch;
                    return Poll::Ready(Ok(StatefulStreamResult::Continue));
                }

                // Precalculate hash values for fetched batch
                let keys_values = evaluate_expressions_to_arrays(&self.on_right, &batch)?;

                let valid_keys = if let Map::HashMap(_) =
                    self.build_side.try_as_ready()?.left_data.map()
                {
                    self.hashes_buffer.clear();
                    self.hashes_buffer.resize(batch.num_rows(), 0);
                    create_hashes(
                        &keys_values,
                        &self.random_state,
                        &mut self.hashes_buffer,
                    )?;
                    matchable_join_keys(&keys_values, self.null_equality)
                } else {
                    None
                };

                self.join_metrics.input_batches.add(1);
                self.join_metrics.input_rows.add(batch.num_rows());

                self.state =
                    HashJoinStreamState::ProcessProbeBatch(ProcessProbeBatchState {
                        batch,
                        values: keys_values,
                        valid_keys,
                        offset: (0, None),
                        joined_probe_idx: None,
                    });
            }
            Some(Err(err)) => return Poll::Ready(Err(err)),
        };

        Poll::Ready(Ok(StatefulStreamResult::Continue))
    }

    /// Joins current probe batch with build-side data and produces batch with matched output
    ///
    /// Updates state to `FetchProbeBatch`
    fn process_probe_batch(
        &mut self,
    ) -> Result<StatefulStreamResult<Option<RecordBatch>>> {
        let state = self.state.try_as_process_probe_batch_mut()?;
        let build_side = self.build_side.try_as_ready_mut()?;

        self.join_metrics
            .probe_hit_rate
            .add_total(state.batch.num_rows());

        let timer = self.join_metrics.join_time.timer();

        // Null-aware anti join semantics:
        // For LeftAnti: output LEFT (build) rows where LEFT.key NOT IN RIGHT.key
        // 1. If RIGHT (probe) contains NULL in any batch, no LEFT rows should be output
        // 2. LEFT rows with NULL keys should not be output (handled in final stage)
        if self.null_aware {
            // Mark that we've seen a probe batch with actual rows (probe side is non-empty)
            // Only set this if batch has rows - empty batches don't count
            // Use shared atomic state so all partitions can see this global information
            if state.batch.num_rows() > 0 {
                build_side
                    .left_data
                    .probe_side_non_empty
                    .store(true, Ordering::Relaxed);
            }

            // Check if probe side (RIGHT) contains NULL
            // Since null_aware validation ensures single column join, we only check the first column
            let probe_key_column = &state.values[0];
            if probe_key_column.null_count() > 0 {
                // Found NULL in probe side - set shared flag to prevent any output
                build_side
                    .left_data
                    .probe_side_has_null
                    .store(true, Ordering::Relaxed);
            }

            // If probe side has NULL (detected in this or any other partition), return empty result
            if build_side
                .left_data
                .probe_side_has_null
                .load(Ordering::Relaxed)
            {
                timer.done();
                self.state = HashJoinStreamState::FetchProbeBatch;
                return Ok(StatefulStreamResult::Continue);
            }
        }

        let is_empty = !build_side.left_data.has_matchable_build_rows();

        if is_empty {
            let result = build_batch_empty_build_side(
                &self.schema,
                build_side.left_data.batch(),
                &state.batch,
                &self.column_indices,
                self.join_type,
            )?;
            timer.done();
            self.output_buffer.push_batch(result)?;
            self.state = HashJoinStreamState::FetchProbeBatch;

            return Ok(StatefulStreamResult::Continue);
        }

        // get the matched by join keys indices
        let (left_indices, right_indices, next_offset) = match build_side.left_data.map()
        {
            Map::HashMap(map) => lookup_join_hashmap(
                map.as_ref(),
                build_side.left_data.values(),
                &state.values,
                self.null_equality,
                &self.hashes_buffer,
                state.valid_keys.as_ref(),
                self.batch_size,
                state.offset,
                &mut self.probe_indices_buffer,
                &mut self.build_indices_buffer,
            )?,
            Map::ArrayMap(array_map) => {
                let next_offset = array_map.get_matched_indices_with_limit_offset(
                    &state.values,
                    self.batch_size,
                    state.offset,
                    &mut self.probe_indices_buffer,
                    &mut self.build_indices_buffer,
                )?;
                (
                    UInt64Array::from(self.build_indices_buffer.clone()),
                    UInt32Array::from(self.probe_indices_buffer.clone()),
                    next_offset,
                )
            }
        };

        let distinct_right_indices_count = count_distinct_sorted_indices(&right_indices);

        self.join_metrics
            .probe_hit_rate
            .add_part(distinct_right_indices_count);

        self.join_metrics.avg_fanout.add_part(left_indices.len());

        self.join_metrics
            .avg_fanout
            .add_total(distinct_right_indices_count);

        // apply join filter if exists
        let (left_indices, right_indices) = if let Some(filter) = &self.filter {
            apply_join_filter_to_indices(
                build_side.left_data.batch(),
                &state.batch,
                left_indices,
                right_indices,
                filter,
                JoinSide::Left,
                None,
                self.join_type,
            )?
        } else {
            (left_indices, right_indices)
        };

        // mark joined left-side indices as visited, if required by join type
        if need_produce_result_in_final(self.join_type) {
            let mut bitmap = build_side.left_data.visited_indices_bitmap().lock();
            left_indices.iter().flatten().for_each(|x| {
                bitmap.set_bit(x as usize, true);
            });
        }

        // The goals of index alignment for different join types are:
        //
        // 1) Right & FullJoin -- to append all missing probe-side indices between
        //    previous (excluding) and current joined indices.
        // 2) SemiJoin -- deduplicate probe indices in range between previous
        //    (excluding) and current joined indices.
        // 3) AntiJoin -- return only missing indices in range between
        //    previous and current joined indices.
        //    Inclusion/exclusion of the indices themselves don't matter
        //
        // As a summary -- alignment range can be produced based only on
        // joined (matched with filters applied) probe side indices, excluding starting one
        // (left from previous iteration).

        // if any rows have been joined -- get last joined probe-side (right) row
        // it's important that index counts as "joined" after hash collisions checks
        // and join filters applied.
        let last_joined_right_idx = match right_indices.len() {
            0 => None,
            n => Some(right_indices.value(n - 1) as usize),
        };

        // Calculate range and perform alignment.
        // In case probe batch has been processed -- align all remaining rows.
        let index_alignment_range_start = state.joined_probe_idx.map_or(0, |v| v + 1);
        let index_alignment_range_end = if next_offset.is_none() {
            state.batch.num_rows()
        } else {
            last_joined_right_idx.map_or(0, |v| v + 1)
        };

        let (left_indices, right_indices) = adjust_indices_by_join_type(
            left_indices,
            right_indices,
            index_alignment_range_start..index_alignment_range_end,
            self.join_type,
            self.right_side_ordered,
        )?;

        // Build output batch and push to coalescer
        let (build_batch, probe_batch, join_side) =
            if self.join_type == JoinType::RightMark {
                (&state.batch, build_side.left_data.batch(), JoinSide::Right)
            } else {
                (build_side.left_data.batch(), &state.batch, JoinSide::Left)
            };

        let batch = build_batch_from_indices(
            &self.schema,
            build_batch,
            probe_batch,
            &left_indices,
            &right_indices,
            &self.column_indices,
            join_side,
            self.join_type,
        )?;

        let push_status = self.output_buffer.push_batch(batch)?;

        timer.done();

        // If limit reached, finish and move to Completed state
        if push_status == PushBatchStatus::LimitReached {
            self.output_buffer.finish()?;
            self.state = HashJoinStreamState::Completed;
            return Ok(StatefulStreamResult::Continue);
        }

        if next_offset.is_none() {
            self.state = HashJoinStreamState::FetchProbeBatch;
        } else {
            state.advance(
                next_offset
                    .ok_or_else(|| internal_datafusion_err!("unexpected None offset"))?,
                last_joined_right_idx,
            )
        };

        Ok(StatefulStreamResult::Continue)
    }

    /// Processes unmatched build-side rows for certain join types and produces output batch
    ///
    /// Updates state to `Completed`
    fn process_unmatched_build_batch(
        &mut self,
    ) -> Result<StatefulStreamResult<Option<RecordBatch>>> {
        let timer = self.join_metrics.join_time.timer();

        if !need_produce_result_in_final(self.join_type) {
            self.state = self.next_state_after_probe();
            return Ok(StatefulStreamResult::Continue);
        }

        let build_side = self.build_side.try_as_ready()?;

        // For null-aware anti join, if probe side had NULL, no rows should be output
        // Check shared atomic state to get global knowledge across all partitions
        if self.null_aware
            && build_side
                .left_data
                .probe_side_has_null
                .load(Ordering::Relaxed)
        {
            timer.done();
            self.state = self.next_state_after_probe();
            return Ok(StatefulStreamResult::Continue);
        }
        if !build_side.left_data.report_probe_completed() {
            self.state = self.next_state_after_probe();
            return Ok(StatefulStreamResult::Continue);
        }

        // use the global left bitmap to produce the left indices and right indices
        let (mut left_side, mut right_side) = get_final_indices_from_shared_bitmap(
            build_side.left_data.visited_indices_bitmap(),
            self.join_type,
            true,
        );

        // For null-aware anti join, filter out LEFT rows with NULL in join keys
        // BUT only if the probe side (RIGHT) was non-empty. If probe side is empty,
        // NULL NOT IN (empty) = TRUE, so NULL rows should be returned.
        // Use shared atomic state to get global knowledge across all partitions
        if self.null_aware
            && self.join_type == JoinType::LeftAnti
            && build_side
                .left_data
                .probe_side_non_empty
                .load(Ordering::Relaxed)
        {
            // Since null_aware validation ensures single column join, we only check the first column
            let build_key_column = &build_side.left_data.values()[0];

            // Filter out indices where the key is NULL
            let filtered_indices: Vec<u64> = left_side
                .iter()
                .filter_map(|idx| {
                    let idx_usize = idx.unwrap() as usize;
                    if build_key_column.is_null(idx_usize) {
                        None // Skip rows with NULL keys
                    } else {
                        Some(idx.unwrap())
                    }
                })
                .collect();

            left_side = UInt64Array::from(filtered_indices);

            // Update right_side to match the new length
            let mut builder = arrow::array::UInt32Builder::with_capacity(left_side.len());
            builder.append_nulls(left_side.len());
            right_side = builder.finish();
        }

        self.join_metrics.input_batches.add(1);
        self.join_metrics.input_rows.add(left_side.len());

        timer.done();

        self.state = self.next_state_after_probe();

        // Push final unmatched indices to output buffer
        if !left_side.is_empty() {
            let empty_right_batch = RecordBatch::new_empty(self.right.schema());
            let batch = build_batch_from_indices(
                &self.schema,
                build_side.left_data.batch(),
                &empty_right_batch,
                &left_side,
                &right_side,
                &self.column_indices,
                JoinSide::Left,
                self.join_type,
            )?;
            let push_status = self.output_buffer.push_batch(batch)?;

            // If limit reached, finish the coalescer
            if push_status == PushBatchStatus::LimitReached {
                self.output_buffer.finish()?;
            }
        }

        Ok(StatefulStreamResult::Continue)
    }

    /// Determine next state after hot-partition probe is done.
    /// If spill_meta exists, transition to RestoreSpilledPartition.
    /// Otherwise, go to Completed.
    fn next_state_after_probe(&self) -> HashJoinStreamState {
        // If we're in the middle of per-partition restore (probe spill state populated),
        // always loop back to RestoreSpilledPartition until all partitions are done.
        if self.probe_spill_files.is_some() || self.probe_spill_writers.is_some() {
            return HashJoinStreamState::RestoreSpilledPartition;
        }

        // First probe pass: if spill_meta exists, begin restore loop.
        let has_spill = self
            .build_side
            .try_as_ready()
            .ok()
            .and_then(|bs| bs.left_data.spill_meta.as_ref())
            .is_some();
        if has_spill {
            HashJoinStreamState::RestoreSpilledPartition
        } else {
            HashJoinStreamState::Completed
        }
    }

    /// If spill_meta exists, split probe batch into hot rows (returned) and
    /// cold rows (spilled to per-partition files for later restore).
    /// If no spill_meta, returns the batch unchanged.
    fn maybe_spill_cold_probe_rows(
        &mut self,
        batch: RecordBatch,
    ) -> Result<RecordBatch> {
        use super::build_spill::NUM_PARTITIONS;

        let spill_meta = match self
            .build_side
            .try_as_ready()
            .ok()
            .and_then(|bs| bs.left_data.spill_meta.as_ref())
        {
            Some(meta) => meta,
            None => return Ok(batch), // No spill, pass through
        };

        let spilled_partitions = &spill_meta.spilled_partitions;

        // Lazily initialize probe spill writers
        if self.probe_spill_writers.is_none() {
            self.probe_spill_writers = Some((0..NUM_PARTITIONS).map(|_| None).collect());
        }

        // Compute partition assignment for each probe row using same hash as build
        let keys_values = evaluate_expressions_to_arrays(&self.on_right, &batch)?;
        let mut hashes = vec![0u64; batch.num_rows()];
        create_hashes(&keys_values, &self.random_state, &mut hashes)?;

        let bit_offset = spill_meta.bit_offset;
        let num_partitions = NUM_PARTITIONS;

        // Separate hot vs cold indices
        let mut hot_indices: Vec<u32> = Vec::new();
        let mut cold_by_partition: Vec<Vec<u32>> = vec![Vec::new(); num_partitions];

        for (row_idx, hash) in hashes.iter().enumerate() {
            let partition = ((*hash >> bit_offset) as usize) % num_partitions;
            if spilled_partitions[partition] {
                cold_by_partition[partition].push(row_idx as u32);
            } else {
                hot_indices.push(row_idx as u32);
            }
        }

        // Spill cold rows to per-partition writers (O(1) append per batch)
        for (part_idx, indices) in cold_by_partition.iter().enumerate() {
            if indices.is_empty() {
                continue;
            }

            // Select rows for this partition
            let indices_arr = UInt32Array::from(indices.clone());
            let columns: Vec<ArrayRef> = batch
                .columns()
                .iter()
                .map(|col| arrow::compute::take(col, &indices_arr, None).unwrap())
                .collect();
            let part_batch = RecordBatch::try_new(batch.schema(), columns)?;

            // Get or create writer for this partition
            let writers = self.probe_spill_writers.as_mut().unwrap();
            if writers[part_idx].is_none() {
                let spill_mgr = crate::spill::spill_manager::SpillManager::new(
                    Arc::clone(&self.runtime_env),
                    crate::metrics::SpillMetrics::new(
                        &crate::metrics::ExecutionPlanMetricsSet::new(),
                        0,
                    ),
                    batch.schema(),
                );
                writers[part_idx] = Some(
                    spill_mgr.create_in_progress_file("probe_spill")?
                );
            }
            writers[part_idx].as_mut().unwrap().append_batch(&part_batch)?;
        }

        // Return only hot rows
        if hot_indices.len() == batch.num_rows() {
            return Ok(batch); // All hot, no filtering needed
        }
        if hot_indices.is_empty() {
            return Ok(RecordBatch::new_empty(batch.schema()));
        }

        let indices_arr = UInt32Array::from(hot_indices);
        let columns: Vec<ArrayRef> = batch
            .columns()
            .iter()
            .map(|col| arrow::compute::take(col, &indices_arr, None).unwrap())
            .collect();
        Ok(RecordBatch::try_new(batch.schema(), columns)?)
    }

    /// Finalize probe spill writers — finish all open IPC files and store
    /// the completed file references for per-partition restore.
    fn finalize_probe_spill(&mut self) -> Result<()> {
        if let Some(writers) = self.probe_spill_writers.take() {
            let mut files: Vec<Option<Arc<dyn datafusion_execution::spill_file::SpillFile>>> =
                Vec::with_capacity(writers.len());
            for writer_opt in writers {
                if let Some(mut writer) = writer_opt {
                    files.push(writer.finish()?);
                } else {
                    files.push(None);
                }
            }
            self.probe_spill_files = Some(files);
        }
        Ok(())
    }

    /// Process spilled partitions one at a time.
    /// For each cold partition: read build-side IPC → build hash table →
    /// read probe-side IPC → set as new right stream → transition to FetchProbeBatch.
    fn restore_spilled_partition(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> Poll<Result<StatefulStreamResult<Option<RecordBatch>>>> {
        // Get spill_meta from the original build side
        let build_side = self.build_side.try_as_ready()?;
        let spill_meta = match &build_side.left_data.spill_meta {
            Some(meta) => meta,
            None => {
                self.state = HashJoinStreamState::Completed;
                return Poll::Ready(Ok(StatefulStreamResult::Continue));
            }
        };

        // Find the next cold partition that has a spill file
        let num_partitions = spill_meta.cold_files.len();
        while self.next_cold_partition < num_partitions {
            let idx = self.next_cold_partition;
            self.next_cold_partition += 1;

            let build_file = match &spill_meta.cold_files[idx] {
                Some(f) => Arc::clone(f),
                None => continue, // Not spilled (hot partition or empty)
            };

            // Read all build-side batches from spill file
            let build_schema = Arc::clone(&spill_meta.schema);

            // Collect build batches synchronously via IPC StreamReader
            let build_batches: Vec<RecordBatch> = {
                use arrow::ipc::reader::StreamReader;
                let path = build_file.path().expect("local spill file");
                let file = std::fs::File::open(path)?;
                let reader = StreamReader::try_new(file, None)?;
                reader.into_iter().collect::<std::result::Result<Vec<_>, _>>()?
            };

            let num_rows: usize = build_batches.iter().map(|b| b.num_rows()).sum();
            if num_rows == 0 {
                continue; // Empty partition, skip
            }

            // Build hash table for this partition
            let mut reservation = datafusion_execution::memory_pool::MemoryConsumer::new(
                format!("HashJoinRestore[{}]", idx),
            )
            .register(&self.runtime_env.memory_pool);

            let on_left = spill_meta.on_left.clone();
            let config = datafusion_common::config::ConfigOptions::new();

            let new_left_data = super::exec::build_hash_table_from_batches(
                build_batches,
                num_rows,
                &build_schema,
                &on_left,
                &self.random_state,
                &mut reservation,
                &self.join_metrics,
                need_produce_result_in_final(self.join_type),
                1, // single probe thread for restored partition
                &config,
                self.null_equality,
                None, // no bounds for restored partitions
                None, // no array_map counter
            )?;

            // Read probe-side spill file for this partition (if exists)
            let probe_stream: SendableRecordBatchStream =
                if let Some(Some(probe_file)) = self
                    .probe_spill_files
                    .as_ref()
                    .and_then(|files| files.get(idx))
                {
                    let probe_schema = self.right.schema();
                    let probe_spill_mgr = crate::spill::spill_manager::SpillManager::new(
                        Arc::clone(&self.runtime_env),
                        crate::metrics::SpillMetrics::new(
                            &crate::metrics::ExecutionPlanMetricsSet::new(),
                            0,
                        ),
                        probe_schema,
                    );
                    probe_spill_mgr.read_spill_as_stream_unbuffered(
                        Arc::clone(probe_file),
                        None,
                    )?
                } else {
                    // No probe rows for this partition — empty stream
                    let probe_schema = self.right.schema();
                    Box::pin(EmptyRecordBatchStream::new(probe_schema))
                };

            // Replace build side and probe stream for this partition
            self.build_side = BuildSide::Ready(BuildSideReadyState {
                left_data: Arc::new(new_left_data),
            });
            self.right = probe_stream;
            self.state = HashJoinStreamState::FetchProbeBatch;

            return Poll::Ready(Ok(StatefulStreamResult::Continue));
        }

        // All cold partitions processed — clear probe_spill_files so
        // next_state_after_probe returns Completed if re-entered.
        self.probe_spill_files = None;
        self.state = HashJoinStreamState::Completed;
        Poll::Ready(Ok(StatefulStreamResult::Continue))
    }
}

impl Stream for HashJoinStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        self.poll_next_impl(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joins::hash_join::shared_bounds::{
        PushdownStrategy, completed_partitions_for_test,
        make_partitioned_accumulator_for_test,
    };

    fn empty_build_data(partition_id: usize) -> PartitionBuildData {
        PartitionBuildData::Partitioned {
            partition_id,
            pushdown: PushdownStrategy::Empty,
            bounds: PartitionBounds::new(vec![]),
            keys_have_null: false,
        }
    }

    fn partitioned_handle(acc: &Arc<SharedBuildAccumulator>) -> BuildReportHandle {
        BuildReportHandle::new(0, PartitionMode::Partitioned, Some(Arc::clone(acc)))
    }

    #[test]
    fn build_report_handle_cancels_scheduled_partition_on_drop() {
        let acc = Arc::new(make_partitioned_accumulator_for_test(2));

        {
            let mut handle = partitioned_handle(&acc);
            handle.schedule(empty_build_data(0));
            assert_eq!(handle.state(), &BuildReportState::Scheduled);
        }

        assert_eq!(completed_partitions_for_test(&acc), 1);
    }

    #[test]
    fn build_report_handle_does_not_cancel_delivered_partition_on_drop() {
        let acc = Arc::new(make_partitioned_accumulator_for_test(1));

        {
            let mut handle = partitioned_handle(&acc);
            handle.schedule(empty_build_data(0));
            let mut cx = std::task::Context::from_waker(futures::task::noop_waker_ref());
            assert!(matches!(handle.poll_delivery(&mut cx), Poll::Ready(Ok(()))));
            assert_eq!(handle.state(), &BuildReportState::Delivered);
        }

        assert_eq!(completed_partitions_for_test(&acc), 1);
    }

    #[test]
    fn build_report_handle_cancel_pending_is_idempotent() {
        let acc = Arc::new(make_partitioned_accumulator_for_test(2));
        let mut handle = partitioned_handle(&acc);
        handle.schedule(empty_build_data(0));

        handle.cancel_pending();
        handle.cancel_pending();

        assert_eq!(handle.state(), &BuildReportState::Canceled);
        assert_eq!(completed_partitions_for_test(&acc), 1);
    }

    #[test]
    fn build_report_handle_no_accumulator_finalizes() {
        let mut handle = BuildReportHandle::new(0, PartitionMode::Partitioned, None);

        handle.schedule(empty_build_data(0));
        handle.cancel_pending();

        assert_eq!(handle.state(), &BuildReportState::Finalized);
    }
}
