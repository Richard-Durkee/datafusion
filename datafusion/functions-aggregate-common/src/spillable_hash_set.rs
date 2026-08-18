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

//! [`SpillableHashSet`]: a grace-hash partitioned distinct-key set that can
//! spill cold partitions to durable storage under memory pressure.
//!
//! # Why this exists
//!
//! `COUNT(DISTINCT x)` accumulators track every distinct value in an
//! in-memory hash set. For high-cardinality inputs that set grows without
//! bound. The existing relief valve — serializing the whole set through
//! `state()` and rebuilding it on the merge side via `merge_batch()` — just
//! relocates the peak allocation to the merge side, so it still OOMs.
//!
//! `SpillableHashSet` replaces the bare set with a *grace-hash partitioned*
//! one. The key space is split into `P` partitions by `hash(key) % P`. While
//! there is memory, every partition is **hot** (an in-memory dedup set) and
//! behaves exactly like a plain `HashSet`. Under memory pressure the caller
//! evicts the largest hot partition to disk, making it **cold**. Subsequent
//! inserts routed to a cold partition are appended to disk instead of being
//! deduplicated eagerly — their distinctness decision is *deferred* to drain
//! time.
//!
//! # Correctness invariant
//!
//! A given key always hashes to the same partition. Therefore a cold
//! partition's on-disk contents (plus any not-yet-flushed pending keys) hold
//! **every** occurrence of every key routed there. At drain time each cold
//! partition is read back independently, deduplicated in isolation, and
//! emitted. Peak extra memory during drain is bounded to a single partition,
//! never the whole key space.
//!
//! # Layering
//!
//! The core logic here has no DataFusion (or Arrow) dependency — durable
//! storage is delegated to the [`PartitionSpiller`] trait so the partitioning
//! and drain logic can be unit-tested with an in-memory backend. Production
//! wires an Arrow-IPC + `SpillManager` backed implementation from the
//! physical-plan layer.

use std::collections::HashSet;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash};

/// Default number of grace-hash partitions.
///
/// Matches the partition count used by the equivalent Trino `MarkDistinct`
/// spill implementation. Large enough that evicting one partition frees a
/// meaningful chunk of memory, small enough that per-partition bookkeeping is
/// cheap.
pub const DEFAULT_NUM_PARTITIONS: usize = 32;

/// Error returned by a [`PartitionSpiller`] backend.
///
/// Intentionally dependency-free (a plain message) so the core type can be
/// exercised without pulling in `datafusion_common::DataFusionError`. The
/// physical-plan backend converts `DataFusionError` into this and back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpillError(pub String);

impl std::fmt::Display for SpillError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "spill error: {}", self.0)
    }
}

impl std::error::Error for SpillError {}

/// Durable storage backend for cold (spilled) partitions.
///
/// The [`SpillableHashSet`] owns no I/O; it hands a partition's keys to this
/// trait for persistence and reads them back once, per partition, at drain
/// time. Implementations only need append-and-read-all semantics — random
/// access is never required.
pub trait PartitionSpiller<K> {
    /// Append `keys` to the durable representation of `partition`.
    ///
    /// Called both when a hot partition is first evicted (the whole resident
    /// set) and when a cold partition's pending buffer is flushed.
    fn append(&mut self, partition: usize, keys: &[K]) -> Result<(), SpillError>;

    /// Read back **every** key previously appended to `partition`.
    ///
    /// May contain duplicates (both across appends and within one append);
    /// the caller deduplicates. Invoked at most once per partition at drain.
    fn read_all(&mut self, partition: usize) -> Result<Vec<K>, SpillError>;
}

/// Outcome of [`SpillableHashSet::insert`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Insert {
    /// Key was not previously present in its (hot) partition. Callers keeping
    /// an eager running count should increment on this variant only.
    New,
    /// Key was already present in its (hot) partition.
    Duplicate,
    /// Key was routed to a cold partition; whether it is distinct cannot be
    /// known until drain. Eager counts must **not** be incremented here — the
    /// count for cold partitions is recomputed at drain instead.
    Deferred,
}

/// A distinct-key set with grace-hash partitioning and per-partition spill.
///
/// Generic over the key type `K`, the storage backend `S`, and the routing
/// hasher `H` (defaulting to [`RandomState`]). The routing hasher is a single
/// fixed instance for the lifetime of the set so key→partition assignment is
/// stable, which is what makes the correctness invariant hold.
pub struct SpillableHashSet<K, S, H = RandomState>
where
    K: Eq + Hash + Clone,
    S: PartitionSpiller<K>,
    H: BuildHasher,
{
    num_partitions: usize,
    /// Per-partition resident dedup sets. `Some` = hot (in memory);
    /// `None` = cold (evicted to `spiller`).
    resident: Vec<Option<HashSet<K>>>,
    /// Per-partition keys inserted into a cold partition but not yet flushed
    /// to `spiller`. Empty for hot partitions.
    pending: Vec<Vec<K>>,
    /// True once any partition has been evicted to disk.
    spilled: bool,
    spiller: S,
    /// Fixed hasher used solely for key→partition routing.
    router: H,
    /// Pending keys per cold partition are flushed to the spiller once this
    /// many accumulate, bounding the in-memory footprint of cold partitions.
    flush_threshold: usize,
}

impl<K, S> SpillableHashSet<K, S, RandomState>
where
    K: Eq + Hash + Clone,
    S: PartitionSpiller<K>,
{
    /// Create a set with the given partition count and storage backend, using
    /// a randomly-seeded routing hasher.
    pub fn new(num_partitions: usize, spiller: S) -> Self {
        Self::with_router(num_partitions, spiller, RandomState::new())
    }
}

impl<K, S, H> SpillableHashSet<K, S, H>
where
    K: Eq + Hash + Clone,
    S: PartitionSpiller<K>,
    H: BuildHasher,
{
    /// Create a set with an explicit routing hasher. Useful for deterministic
    /// tests (inject a fixed-seed hasher so partition placement is known).
    pub fn with_router(num_partitions: usize, spiller: S, router: H) -> Self {
        assert!(num_partitions > 0, "num_partitions must be positive");
        Self {
            num_partitions,
            resident: (0..num_partitions).map(|_| Some(HashSet::new())).collect(),
            pending: (0..num_partitions).map(|_| Vec::new()).collect(),
            spilled: false,
            spiller,
            router,
            // 8192 keys ~= a batch's worth; keeps cold-partition buffers small
            // without flushing on every single insert.
            flush_threshold: 8192,
        }
    }

    /// Override the pending-buffer flush threshold (keys per cold partition).
    pub fn with_flush_threshold(mut self, flush_threshold: usize) -> Self {
        assert!(flush_threshold > 0, "flush_threshold must be positive");
        self.flush_threshold = flush_threshold;
        self
    }

    #[inline]
    fn partition_of(&self, key: &K) -> usize {
        // `% num_partitions` on a well-mixed 64-bit hash. Not a power-of-two
        // mask on purpose: DEFAULT_NUM_PARTITIONS is a power of two anyway, but
        // keeping the modulo makes non-power-of-two partition counts valid too.
        (self.router.hash_one(key) as usize) % self.num_partitions
    }

    /// Insert a key, returning whether it was newly seen, a duplicate, or
    /// deferred to a cold partition. See [`Insert`].
    pub fn insert(&mut self, key: K) -> Result<Insert, SpillError> {
        let p = self.partition_of(&key);
        // Hot partition: dedup in memory, exactly like a plain HashSet.
        if let Some(set) = self.resident[p].as_mut() {
            return Ok(if set.insert(key) {
                Insert::New
            } else {
                Insert::Duplicate
            });
        }
        // Cold partition: buffer and lazily flush; distinctness deferred.
        self.pending[p].push(key);
        if self.pending[p].len() >= self.flush_threshold {
            self.flush_partition(p)?;
        }
        Ok(Insert::Deferred)
    }

    /// Flush a cold partition's pending buffer to the backend.
    fn flush_partition(&mut self, p: usize) -> Result<(), SpillError> {
        if !self.pending[p].is_empty() {
            let keys = std::mem::take(&mut self.pending[p]);
            self.spiller.append(p, &keys)?;
        }
        Ok(())
    }

    /// Evict the largest hot partition to disk, returning the keys that were
    /// evicted (so a caller maintaining an eager per-key running count can
    /// reconcile it). Returns `Ok(None)` when no hot partition holds any keys.
    ///
    /// This is the memory-pressure hook: call it repeatedly until enough
    /// memory has been released.
    pub fn spill_largest_partition(&mut self) -> Result<Option<Vec<K>>, SpillError> {
        let target = self
            .resident
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.as_ref().map(|s| (i, s.len())))
            .filter(|&(_, len)| len > 0)
            .max_by_key(|&(_, len)| len)
            .map(|(i, _)| i);

        let Some(p) = target else {
            return Ok(None);
        };

        // Take the resident set, mark the partition cold, persist its keys.
        let set = self.resident[p].take().expect("target partition is hot");
        let keys: Vec<K> = set.into_iter().collect();
        self.spiller.append(p, &keys)?;
        self.spilled = true;
        Ok(Some(keys))
    }

    /// Visit every distinct key exactly once.
    ///
    /// Hot partitions are drained from their in-memory sets. Each cold
    /// partition is read back independently, deduplicated in a temporary set,
    /// emitted, then dropped — so peak extra memory is bounded to one
    /// partition rather than the entire key space.
    ///
    /// Consumes the set's contents (it is empty afterward).
    pub fn drain_distinct<F>(&mut self, mut f: F) -> Result<(), SpillError>
    where
        F: FnMut(K),
    {
        for p in 0..self.num_partitions {
            match self.resident[p].take() {
                Some(set) => {
                    for key in set {
                        f(key);
                    }
                }
                None => {
                    // Cold: fold pending into disk, then read+dedup in isolation.
                    self.flush_partition(p)?;
                    let all = self.spiller.read_all(p)?;
                    let distinct: HashSet<K> = all.into_iter().collect();
                    for key in distinct {
                        f(key);
                    }
                }
            }
        }
        self.spilled = false;
        Ok(())
    }

    /// Number of keys currently held in memory across all hot partitions.
    /// Cold partitions contribute their unflushed pending buffers only.
    pub fn resident_len(&self) -> usize {
        let hot: usize = self.resident.iter().flatten().map(|s| s.len()).sum();
        let pend: usize = self.pending.iter().map(|v| v.len()).sum();
        hot + pend
    }

    /// Whether any partition has been evicted to disk.
    pub fn is_spilled(&self) -> bool {
        self.spilled
    }

    /// Number of partitions.
    pub fn num_partitions(&self) -> usize {
        self.num_partitions
    }

    /// Number of hot (in-memory) partitions remaining.
    pub fn num_hot_partitions(&self) -> usize {
        self.resident.iter().filter(|s| s.is_some()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::hash::Hasher;

    /// In-memory [`PartitionSpiller`] for tests — no disk, no DataFusion.
    #[derive(Default)]
    struct InMemorySpiller<K> {
        data: HashMap<usize, Vec<K>>,
        append_calls: usize,
    }

    impl<K: Clone> PartitionSpiller<K> for InMemorySpiller<K> {
        fn append(&mut self, partition: usize, keys: &[K]) -> Result<(), SpillError> {
            self.data.entry(partition).or_default().extend_from_slice(keys);
            self.append_calls += 1;
            Ok(())
        }

        fn read_all(&mut self, partition: usize) -> Result<Vec<K>, SpillError> {
            Ok(self.data.get(&partition).cloned().unwrap_or_default())
        }
    }

    /// Deterministic routing hasher: `hash_one(u64) == u64`, so a `u64` key `k`
    /// always routes to partition `k % num_partitions`. Lets tests reason about
    /// exact partition placement.
    #[derive(Clone, Default)]
    struct IdentityBuildHasher;

    struct IdentityHasher(u64);

    impl BuildHasher for IdentityBuildHasher {
        type Hasher = IdentityHasher;
        fn build_hasher(&self) -> IdentityHasher {
            IdentityHasher(0)
        }
    }

    impl Hasher for IdentityHasher {
        fn finish(&self) -> u64 {
            self.0
        }
        fn write(&mut self, bytes: &[u8]) {
            for (i, b) in bytes.iter().enumerate() {
                self.0 ^= (*b as u64) << ((i % 8) * 8);
            }
        }
        fn write_u64(&mut self, i: u64) {
            self.0 = i;
        }
    }

    fn collect_sorted<S, H>(set: &mut SpillableHashSet<u64, S, H>) -> Vec<u64>
    where
        S: PartitionSpiller<u64>,
        H: BuildHasher,
    {
        let mut out = Vec::new();
        set.drain_distinct(|k| out.push(k)).unwrap();
        out.sort_unstable();
        out
    }

    #[test]
    fn dedup_without_spilling_matches_plain_hashset() {
        let mut set = SpillableHashSet::new(DEFAULT_NUM_PARTITIONS, InMemorySpiller::default());
        for k in [1u64, 2, 2, 3, 3, 3, 7, 7, 42] {
            set.insert(k).unwrap();
        }
        assert!(!set.is_spilled());
        assert_eq!(collect_sorted(&mut set), vec![1, 2, 3, 7, 42]);
    }

    #[test]
    fn insert_reports_new_vs_duplicate_when_hot() {
        let mut set = SpillableHashSet::new(DEFAULT_NUM_PARTITIONS, InMemorySpiller::default());
        assert_eq!(set.insert(5u64).unwrap(), Insert::New);
        assert_eq!(set.insert(5u64).unwrap(), Insert::Duplicate);
        assert_eq!(set.insert(6u64).unwrap(), Insert::New);
    }

    #[test]
    fn spilling_evicts_keys_and_reduces_resident_memory() {
        let mut set =
            SpillableHashSet::with_router(4, InMemorySpiller::default(), IdentityBuildHasher);
        // partitions by k % 4: 0->{0,4}, 1->{1,5}, 2->{2,6}, 3->{3,7}
        for k in 0u64..8 {
            set.insert(k).unwrap();
        }
        assert_eq!(set.resident_len(), 8);
        assert_eq!(set.num_hot_partitions(), 4);

        let evicted = set.spill_largest_partition().unwrap().unwrap();
        assert!(set.is_spilled());
        // one partition of 2 keys evicted
        assert_eq!(evicted.len(), 2);
        assert_eq!(set.resident_len(), 6);
        assert_eq!(set.num_hot_partitions(), 3);
    }

    #[test]
    fn inserts_to_cold_partition_are_deferred() {
        let mut set =
            SpillableHashSet::with_router(4, InMemorySpiller::default(), IdentityBuildHasher);
        for k in 0u64..8 {
            set.insert(k).unwrap();
        }
        // Evict every partition so we know which one is cold.
        while set.spill_largest_partition().unwrap().is_some() {}
        assert_eq!(set.num_hot_partitions(), 0);

        // Any insert now lands in a cold partition -> deferred.
        assert_eq!(set.insert(100u64).unwrap(), Insert::Deferred);
        // Re-inserting an already-spilled key is also deferred (can't dedup).
        assert_eq!(set.insert(0u64).unwrap(), Insert::Deferred);
    }

    #[test]
    fn drain_is_correct_across_hot_and_cold_partitions() {
        let mut set =
            SpillableHashSet::with_router(4, InMemorySpiller::default(), IdentityBuildHasher);
        // Fill all partitions.
        for k in 0u64..8 {
            set.insert(k).unwrap();
        }
        // Evict partition 0 ({0,4}) — it becomes cold.
        set.spill_largest_partition().unwrap();
        // New inserts: some to the cold partition 0 (8, and duplicate 0/4),
        // some to still-hot partitions.
        for k in [8u64, 0, 4, 12, 9, 9, 3] {
            set.insert(k).unwrap();
        }
        // Expected distinct universe.
        let expected: Vec<u64> = {
            let mut v: Vec<u64> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12];
            v.sort_unstable();
            v
        };
        assert_eq!(collect_sorted(&mut set), expected);
    }

    #[test]
    fn cold_partition_duplicates_are_deduped_at_drain() {
        let mut set =
            SpillableHashSet::with_router(2, InMemorySpiller::default(), IdentityBuildHasher);
        set.insert(0u64).unwrap();
        set.insert(1u64).unwrap();
        // Force both partitions cold.
        while set.spill_largest_partition().unwrap().is_some() {}
        // Hammer the same few keys many times into cold partitions.
        for _ in 0..1000 {
            set.insert(0u64).unwrap();
            set.insert(1u64).unwrap();
            set.insert(2u64).unwrap();
        }
        assert_eq!(collect_sorted(&mut set), vec![0, 1, 2]);
    }

    #[test]
    fn pending_buffer_flushes_at_threshold() {
        let spiller = InMemorySpiller::<u64>::default();
        let mut set = SpillableHashSet::with_router(2, spiller, IdentityBuildHasher)
            .with_flush_threshold(4);
        // Make partition 0 cold.
        set.insert(0u64).unwrap();
        set.insert(2u64).unwrap();
        set.spill_largest_partition().unwrap();
        // Insert 4 keys routed to partition 0 (all even) -> triggers a flush.
        for k in [10u64, 20, 30, 40] {
            assert_eq!(set.insert(k).unwrap(), Insert::Deferred);
        }
        // Drain must still see every distinct value exactly once.
        let mut out = Vec::new();
        set.drain_distinct(|k| out.push(k)).unwrap();
        out.sort_unstable();
        // partition 0 (evens): {0,2,10,20,30,40}; partition 1 (odds): {} in this run
        assert_eq!(out, vec![0, 2, 10, 20, 30, 40]);
    }

    #[test]
    fn tuple_keys_group_and_value() {
        // Mirrors the accumulator key: (group_idx, value). Distinctness is per
        // (group, value) pair.
        let mut set = SpillableHashSet::new(8, InMemorySpiller::default());
        let rows = [(0usize, 10u64), (0, 10), (0, 11), (1, 10), (1, 10), (2, 99)];
        for r in rows {
            set.insert(r).unwrap();
        }
        // Force spilling to exercise the cold path with tuple keys.
        set.spill_largest_partition().unwrap();
        set.insert((0usize, 12u64)).unwrap();
        set.insert((0usize, 10u64)).unwrap();

        let mut counts: HashMap<usize, usize> = HashMap::new();
        set.drain_distinct(|(g, _v)| *counts.entry(g).or_default() += 1)
            .unwrap();
        assert_eq!(counts.get(&0), Some(&3)); // {10, 11, 12}
        assert_eq!(counts.get(&1), Some(&1)); // {10}
        assert_eq!(counts.get(&2), Some(&1)); // {99}
    }
}
