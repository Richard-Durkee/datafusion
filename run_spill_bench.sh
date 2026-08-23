#!/bin/bash
# Run spill_bench example across all 4 branches in release mode
# Copies the benchmark file to each branch before building
set -uo pipefail

REPO="/Users/rdurkee/datafusion-spill"
BENCH_FILE="datafusion-examples/examples/spill_bench.rs"
RESULTS_DIR="$REPO/bench_results"
TIMESTAMP=$(date '+%Y%m%d_%H%M')

cd "$REPO"
ORIG_BRANCH=$(git branch --show-current)

# Save the benchmark file content (it only exists on v3)
BENCH_CONTENT=$(cat "$BENCH_FILE")

mkdir -p "$RESULTS_DIR"

BRANCHES=(
    "main"
    "feature/hash-join-spill"
    "feature/hash-join-spill-v2"
    "feature/hash-join-spill-v3"
)

LABELS=(
    "main_baseline"
    "v1_reexecute"
    "v2_approach_c"
    "v3_per_partition"
)

echo "=== Hash-Join Spill Benchmark (Release Mode) ==="
echo "Time: $(date)"
echo "Machine: $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
echo "Disk free: $(df -h / | tail -1 | awk '{print $4}')"
echo ""

for i in "${!BRANCHES[@]}"; do
    branch="${BRANCHES[$i]}"
    label="${LABELS[$i]}"
    outfile="$RESULTS_DIR/${label}_${TIMESTAMP}.txt"
    
    echo "============================================================"
    echo " [$((i+1))/4] $label ($branch)"
    echo "============================================================"
    
    # Checkout branch
    if ! git checkout "$branch" 2>/dev/null; then
        echo "CHECKOUT FAILED" | tee "$outfile"
        continue
    fi
    
    # Clean previous build to save disk
    echo "  Cleaning..."
    cargo clean 2>/dev/null
    
    # Place benchmark file (may not exist on this branch)
    echo "$BENCH_CONTENT" > "$BENCH_FILE"
    
    # Build in release mode
    echo "  Building (release)..."
    BUILD_START=$(date +%s)
    if ! cargo build --release --example spill_bench 2>/tmp/bench_build_${label}.log; then
        BUILD_END=$(date +%s)
        echo "  BUILD FAILED after $((BUILD_END - BUILD_START))s"
        echo "  Last error: $(tail -5 /tmp/bench_build_${label}.log)"
        echo "BUILD_FAILED" > "$outfile"
        echo "build_time: $((BUILD_END - BUILD_START))s" >> "$outfile"
        echo "error: $(tail -3 /tmp/bench_build_${label}.log)" >> "$outfile"
        # Remove our injected file before switching branches
        git checkout -- "$BENCH_FILE" 2>/dev/null || rm -f "$BENCH_FILE"
        continue
    fi
    BUILD_END=$(date +%s)
    echo "  Built in $((BUILD_END - BUILD_START))s"
    
    # Run benchmark (release binary is in target/release/examples/)
    echo "  Running benchmark..."
    /usr/bin/time -l ./target/release/examples/spill_bench > "$outfile" 2>/tmp/bench_time_${label}.txt
    RUN_EXIT=$?
    
    # Append /usr/bin/time output (has peak RSS)
    echo "" >> "$outfile"
    echo "--- /usr/bin/time ---" >> "$outfile"
    # Extract peak RSS from macOS time output
    PEAK_RSS=$(grep "maximum resident set size" /tmp/bench_time_${label}.txt | awk '{print $1}')
    REAL_TIME=$(grep "real" /tmp/bench_time_${label}.txt | awk '{print $1}')
    echo "time_peak_rss_bytes: ${PEAK_RSS:-unknown}" >> "$outfile"
    echo "time_real: ${REAL_TIME:-unknown}" >> "$outfile"
    echo "build_time_secs: $((BUILD_END - BUILD_START))" >> "$outfile"
    echo "exit_code: $RUN_EXIT" >> "$outfile"
    
    echo "  Done. Results in $outfile"
    echo ""
    cat "$outfile" | grep -E "status:|wall_time|throughput|peak_rss|spill|time_peak_rss"
    echo ""
    
    # Clean up injected file
    git checkout -- "$BENCH_FILE" 2>/dev/null || rm -f "$BENCH_FILE"
done

# Restore original branch and file
git checkout "$ORIG_BRANCH" 2>/dev/null
echo "$BENCH_CONTENT" > "$BENCH_FILE"

echo ""
echo "============================================================"
echo " ALL RESULTS"
echo "============================================================"
echo ""
for i in "${!LABELS[@]}"; do
    label="${LABELS[$i]}"
    outfile="$RESULTS_DIR/${label}_${TIMESTAMP}.txt"
    echo "--- $label ---"
    if [ -f "$outfile" ]; then
        cat "$outfile"
    else
        echo "  (no results)"
    fi
    echo ""
done

echo "Results directory: $RESULTS_DIR"
