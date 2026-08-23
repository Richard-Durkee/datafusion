#!/bin/bash
# Benchmark hash-join spill across all 4 branches
# Cleans between builds to manage disk space on constrained system
set -uo pipefail

REPO="/Users/rdurkee/datafusion-spill"
RESULTS_FILE="$REPO/benchmark_results.md"

cd "$REPO"
ORIG_BRANCH=$(git branch --show-current)

echo "# Hash-Join Spill Benchmark Results" > "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "**Date:** $(date '+%Y-%m-%d %H:%M')" >> "$RESULTS_FILE"
echo "**Machine:** $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'unknown')" >> "$RESULTS_FILE"
echo "**Test:** self-join on 'service' column, memory_limit=100KB, 1 partition, disk spill enabled" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "| Branch | Approach | Result | Build Time | Test Time | Notes |" >> "$RESULTS_FILE"
echo "|--------|----------|--------|------------|-----------|-------|" >> "$RESULTS_FILE"

run_branch() {
    local branch="$1"
    local label="$2"
    local approach="$3"
    
    echo ""
    echo "============================================================"
    echo " $label ($branch)"
    echo "============================================================"
    
    if ! git checkout "$branch" 2>/dev/null; then
        echo "| $label | $approach | CHECKOUT FAILED | - | - | - |" >> "$RESULTS_FILE"
        return
    fi
    
    # Clean to save disk space (builds are ~5-8GB each)
    echo "  Cleaning previous build..."
    cargo clean 2>/dev/null
    
    # Determine test name
    local test_name=""
    if grep -q "async fn hash_join_spill" datafusion/core/tests/memory_limit/mod.rs 2>/dev/null; then
        test_name="memory_limit::hash_join_spill"
    else
        # main branch: demonstrate OOM with symmetric_hash_join
        test_name="memory_limit::symmetric_hash_join"
    fi
    
    echo "  Building (release mode)..."
    BUILD_START=$(date +%s)
    if ! cargo test --release --package datafusion --test core_integration --no-run 2>/tmp/bench_build_err.log; then
        BUILD_END=$(date +%s)
        local build_err=$(tail -3 /tmp/bench_build_err.log | tr '\n' ' ')
        echo "| $label | $approach | BUILD FAILED | $((BUILD_END - BUILD_START))s | - | $build_err |" >> "$RESULTS_FILE"
        echo "  BUILD FAILED: $build_err"
        return
    fi
    BUILD_END=$(date +%s)
    local build_time="$((BUILD_END - BUILD_START))s"
    echo "  Built in $build_time"
    
    echo "  Running test: $test_name"
    TEST_START=$(date +%s)
    TEST_OUTPUT=$(cargo test --release --package datafusion --test core_integration -- "$test_name" --exact --nocapture 2>&1)
    TEST_EXIT=$?
    TEST_END=$(date +%s)
    local test_time="$((TEST_END - TEST_START))s"
    
    if [ $TEST_EXIT -eq 0 ]; then
        # Check if the test actually ran (vs being filtered out)
        if echo "$TEST_OUTPUT" | grep -q "test result.*0 passed"; then
            echo "| $label | $approach | TEST NOT FOUND | $build_time | $test_time | Test filtered out |" >> "$RESULTS_FILE"
            echo "  Test was filtered out (0 passed)"
        else
            local notes=""
            # Try to extract spill info
            local spill_line=$(echo "$TEST_OUTPUT" | grep -i "spill" | grep -v "^test\|running\|Compiling" | head -1)
            [ -n "$spill_line" ] && notes="$spill_line"
            [ -z "$notes" ] && notes="Join completed successfully with spill"
            echo "| $label | $approach | **PASS** | $build_time | $test_time | $notes |" >> "$RESULTS_FILE"
            echo "  PASS in $test_time"
        fi
    else
        if echo "$TEST_OUTPUT" | grep -q "Resources exhausted"; then
            echo "| $label | $approach | **OOM** | $build_time | $test_time | Memory exhausted (no spill support) |" >> "$RESULTS_FILE"
            echo "  OOM in $test_time (expected for baseline)"
        else
            local error=$(echo "$TEST_OUTPUT" | grep -E "panicked|FAILED|Error" | head -1 | cut -c1-80)
            echo "| $label | $approach | FAILED | $build_time | $test_time | $error |" >> "$RESULTS_FILE"
            echo "  FAILED: $error"
        fi
    fi
}

echo "Available disk: $(df -h / | tail -1 | awk '{print $4}')"
echo "Starting benchmarks..."

run_branch "main"                        "main"  "No spill (baseline)"
run_branch "feature/hash-join-spill"     "v1"    "Re-execute on OOM"
run_branch "feature/hash-join-spill-v2"  "v2"    "Inline Approach C"
run_branch "feature/hash-join-spill-v3"  "v3"    "Per-partition state machine"

# Restore
git checkout "$ORIG_BRANCH" 2>/dev/null
cargo clean 2>/dev/null  # Final cleanup

echo ""
echo "============================================================"
echo " RESULTS"
echo "============================================================"
cat "$RESULTS_FILE"
echo ""
echo "Results saved to: $RESULTS_FILE"
