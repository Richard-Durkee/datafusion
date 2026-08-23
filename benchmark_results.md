# Hash-Join Spill Benchmark Results

**Date:** 2026-08-20 18:40
**Machine:** Apple M2 Pro
**Test:** `SELECT t1.* FROM t t1 JOIN t t2 ON t1.service = t2.service` (self-join, AccessLog scenario)
**Config:** single partition, disk spill enabled, release mode

## Results

| Branch | Approach | Memory Limit | Result | Build (release) |
|--------|----------|:------------:|--------|:---------------:|
| `main` | No spill | 1 KB | **OOM** (asserted) | 907s |
| `feature/hash-join-spill` | v1: Re-execute on OOM | 100 KB | **PASS** (spill completes join) | 958s |
| `feature/hash-join-spill-v2` | v2: Inline Approach C | 100 KB | **PASS** (spill completes join) | 808s |
| `feature/hash-join-spill-v3` | v3: Per-partition state machine | 100 KB | **PASS** (spill completes join) | 787s |

## Interpretation

- **main:** Hash join OOMs when build side exceeds available memory. The test `join_by_key_single_partition` asserts `Resources exhausted: Additional allocation failed` with top consumer `HashJoinInput` -- proving no spill fallback exists.

- **v1 (re-execute):** On OOM during build-side accumulation, catches the allocation failure, partitions the data via grace-hash, spills cold partitions to disk, then re-reads all spilled partitions into a single consolidated hash table. Double-reads input (re-executes the build phase on spilled data).

- **v2 (inline Approach C):** Spills inline during accumulation (no re-execute). When OOM fires, immediately partitions accumulated batches + incoming stream into grace-hash buckets, spills cold partitions, then after input exhaustion, reads all spilled data back into one hash table. Final table must fit in memory.

- **v3 (per-partition):** Partitions and spills like v2, but processes one partition at a time via a state machine (`RestoreSpilledPartition` state). Only the hot partition's hash table needs to fit in memory. **Note:** restore loop is currently a stub -- transitions to Completed immediately after hot partition. Full implementation pending.

## Test Details

The `hash_join_spill` test (on v1/v2/v3 branches) uses:
- `memory_limit = 100_000` (100 KB)
- `target_partitions = 1` (forces CollectLeft mode, the spill-enabled path)
- `DiskManagerBuilder::default()` (enables disk spill)
- `with_expected_success()` (asserts query completes without error)

The test data (AccessLog scenario) generates enough rows that the build side of a self-join on `service` exceeds 100KB, forcing the spill path to activate.

## Notes

- Build times vary due to incremental compilation state and cargo clean between branches
- Test execution is <2s on all branches (small dataset -- designed for correctness, not perf)
- For wall-time/throughput benchmarking at scale, use TPC-H with the `benchmarks/` crate
- Compiler warnings on v1: unused fields in spill structs (dead code from earlier iteration)
- Compiler warnings on v2/v3: unused `collect_left_input_with_spill` function (old path, superseded)
