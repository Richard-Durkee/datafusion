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

//! [`SpillableAccumulator`]: optional trait for accumulators that can shed
//! memory by spilling internal state to disk.
//!
//! # Design
//!
//! DataFusion's existing spill path for grouped aggregation works at the
//! *stream* level: `GroupedHashAggregateStream::spill()` calls
//! `accumulator.state()` to serialize the whole state, writes it to IPC, and
//! reloads it via `merge_batch()`. This works for bounded-state accumulators
//! (e.g. `SUM`, `AVG`), but for unbounded accumulators like `COUNT(DISTINCT)`
//! that hold a per-group hash set, the merge side OOMs just as badly as the
//! update side.
//!
//! `SpillableAccumulator` is a secondary trait that accumulators can
//! *optionally* implement to provide **internal** memory shedding — evicting
//! cold partitions from their own data structures rather than dumping the
//! entire state to the stream-level spill path.
//!
//! The stream discovers which accumulators support internal spill via
//! `Any::downcast_mut` (since `GroupsAccumulator: Any`), and calls
//! `spill_internal()` before resorting to the full state-serialization path.
//!
//! # Lifecycle
//!
//! 1. Stream creates the accumulator, then calls `set_spill_manager()` to
//!    inject the I/O backend.
//! 2. On memory pressure, stream calls `spill_internal()`. The accumulator
//!    evicts the largest cold partition and reports how many bytes were freed.
//! 3. At evaluate time, the accumulator's `drain_distinct()` reads back cold
//!    partitions one at a time, bounded to a single partition's memory.

use crate::spill::spill_manager::SpillManager;
use datafusion_common::Result;

/// Trait for accumulators that support internal spill to disk.
///
/// Implemented alongside [`GroupsAccumulator`]; discovered at runtime via
/// `Any::downcast_mut`.
///
/// [`GroupsAccumulator`]: datafusion_expr_common::groups_accumulator::GroupsAccumulator
pub trait SpillableAccumulator: Send {
    /// Provide the I/O backend for spill operations.
    ///
    /// Must be called once before `spill_internal()`. The stream calls this
    /// immediately after constructing the accumulator.
    fn set_spill_manager(&mut self, spill_manager: SpillManager);

    /// Whether this accumulator has been given a `SpillManager` and is
    /// capable of spilling.
    fn can_spill(&self) -> bool;

    /// Shed memory by evicting one or more internal partitions to disk.
    ///
    /// Returns the approximate number of bytes freed. Returns `Ok(0)` when
    /// there is nothing left to evict (all partitions are already cold or
    /// empty).
    fn spill_internal(&mut self) -> Result<usize>;

    /// Whether the accumulator currently has any spilled state on disk.
    fn has_spilled(&self) -> bool;
}
