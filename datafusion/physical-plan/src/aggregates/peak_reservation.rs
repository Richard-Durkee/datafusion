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

//! [`PeakTrackedReservation`]: a memory reservation that reports its peak size.

use datafusion_common::Result;
use datafusion_execution::memory_pool::MemoryReservation;

use crate::metrics::{ExecutionPlanMetricsSet, Gauge, MetricBuilder};

/// A [`MemoryReservation`] that records the largest size it reaches in the
/// `peak_mem_used` metric of its aggregate operator.
///
/// The peak is updated after every successful grow or resize. This type
/// deliberately does not deref to [`MemoryReservation`], so every size change
/// goes through it and none can be missed.
#[derive(Debug)]
pub(super) struct PeakTrackedReservation {
    reservation: MemoryReservation,
    peak_mem_used: Gauge,
}

impl PeakTrackedReservation {
    /// Wraps `reservation` and registers the `peak_mem_used` metric for
    /// `partition` in `metrics`.
    pub(super) fn new(
        reservation: MemoryReservation,
        metrics: &ExecutionPlanMetricsSet,
        partition: usize,
    ) -> Self {
        let peak_mem_used =
            MetricBuilder::new(metrics).peak_memory_usage("peak_mem_used", partition);
        let tracked = Self {
            reservation,
            peak_mem_used,
        };
        tracked.record_peak();
        tracked
    }

    /// Current size of the reservation in bytes.
    #[cfg(test)]
    fn size(&self) -> usize {
        self.reservation.size()
    }

    /// See [`MemoryReservation::try_grow`].
    pub(super) fn try_grow(&self, capacity: usize) -> Result<()> {
        self.reservation.try_grow(capacity)?;
        self.record_peak();
        Ok(())
    }

    /// See [`MemoryReservation::try_resize`].
    pub(super) fn try_resize(&self, capacity: usize) -> Result<()> {
        self.reservation.try_resize(capacity)?;
        self.record_peak();
        Ok(())
    }

    /// See [`MemoryReservation::try_shrink`]. Shrinking never raises the peak.
    pub(super) fn try_shrink(&self, capacity: usize) -> Result<usize> {
        self.reservation.try_shrink(capacity)
    }

    /// See [`MemoryReservation::free`].
    pub(super) fn free(&self) -> usize {
        self.reservation.free()
    }

    /// Returns an empty sibling reservation for the same consumer. Its size is
    /// not included in this reservation's peak.
    pub(super) fn new_empty(&self) -> MemoryReservation {
        self.reservation.new_empty()
    }

    /// Returns an empty sibling reservation for the same consumer that reports
    /// into the same `peak_mem_used` metric, e.g. for the stream that replays
    /// spilled data after this one has released its memory.
    pub(super) fn new_empty_tracked(&self) -> Self {
        Self {
            reservation: self.reservation.new_empty(),
            peak_mem_used: self.peak_mem_used.clone(),
        }
    }

    fn record_peak(&self) {
        self.peak_mem_used.set_max(self.reservation.size());
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion_execution::memory_pool::{
        GreedyMemoryPool, MemoryConsumer, MemoryPool,
    };

    use super::*;

    fn tracked(limit: usize) -> (PeakTrackedReservation, ExecutionPlanMetricsSet) {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(limit));
        let metrics = ExecutionPlanMetricsSet::new();
        let reservation = MemoryConsumer::new("test").register(&pool);
        (
            PeakTrackedReservation::new(reservation, &metrics, 0),
            metrics,
        )
    }

    fn peak(metrics: &ExecutionPlanMetricsSet) -> usize {
        metrics
            .clone_inner()
            .sum_by_name("peak_mem_used")
            .expect("peak_mem_used is registered")
            .as_usize()
    }

    #[test]
    fn records_the_largest_size_reached() -> Result<()> {
        let (reservation, metrics) = tracked(1000);
        assert_eq!(peak(&metrics), 0);

        reservation.try_grow(100)?;
        reservation.try_resize(400)?;
        assert_eq!(peak(&metrics), 400);

        // Shrinking and freeing keep the peak.
        reservation.try_resize(50)?;
        reservation.try_shrink(50)?;
        reservation.free();
        assert_eq!(reservation.size(), 0);
        assert_eq!(peak(&metrics), 400);

        reservation.try_grow(300)?;
        assert_eq!(peak(&metrics), 400);
        Ok(())
    }

    #[test]
    fn failed_grow_does_not_raise_the_peak() -> Result<()> {
        let (reservation, metrics) = tracked(100);
        reservation.try_grow(60)?;
        assert!(reservation.try_grow(60).is_err());
        assert!(reservation.try_resize(200).is_err());
        assert_eq!(peak(&metrics), 60);
        Ok(())
    }
}
