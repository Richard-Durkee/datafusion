// Hash-Join Spill Benchmark
// Generates 500K rows in small batches, runs self-join with 20MB memory limit,
// measures wall time, peak RSS, bytes spilled, throughput.
//
// Usage: cargo run --release --example spill_bench

use arrow::array::{Int64Builder, RecordBatch, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::disk_manager::DiskManagerBuilder;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::execute_stream;
use datafusion::prelude::*;
use futures::StreamExt;
use std::sync::Arc;
use std::time::Instant;

const DEFAULT_NUM_ROWS: usize = 500_000;
const BATCH_SIZE: usize = 8192;
// Memory limit calibration for hybrid spill approach:
// - Must be LESS than total data so spill triggers during batch accumulation
// - Must be MORE than (hot_partition_data + hash_table_for_ALL_rows) since
//   v3 hybrid restores all cold partitions before building the final hash table
// For 500K rows (~43MB data, ~17MB hash table), limit = 30MB works:
//   triggers spill at ~30MB accumulated, then restore+hash_table ~60MB... actually
//   this exceeds pool. The hybrid limitation means we need limit > data+hash_table.
//
// REALITY: hybrid approach only helps with accumulation PEAK, not final size.
// For real benchmarking, use limit between (single_batch_size) and (total_data).
// The query will spill during accumulation but may OOM at hash-table phase if
// total_data + hash_table > pool.
//
// Use env vars BENCH_ROWS and BENCH_MEM_MB to override for experimentation.
const DEFAULT_MEMORY_LIMIT: usize = 30 * 1024 * 1024; // 30 MB
const MEMORY_FRACTION: f64 = 0.95;

fn get_config() -> (usize, usize) {
    let num_rows = std::env::var("BENCH_ROWS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_NUM_ROWS);
    let memory_limit = std::env::var("BENCH_MEM_MB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|mb| mb * 1024 * 1024)
        .unwrap_or(DEFAULT_MEMORY_LIMIT);
    (num_rows, memory_limit)
}

/// Generate a table with skewed join keys to simulate realistic workload
/// Returns multiple small batches to allow incremental memory accumulation
fn generate_data(num_rows: usize, batch_size: usize) -> Vec<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("join_key", DataType::Utf8, false),
        Field::new("payload", DataType::Utf8, false),
    ]));

    let mut batches = Vec::new();
    let mut row_offset = 0;

    while row_offset < num_rows {
        let this_batch_size = (num_rows - row_offset).min(batch_size);

        let mut id_builder = Int64Builder::with_capacity(this_batch_size);
        let mut key_builder =
            StringBuilder::with_capacity(this_batch_size, this_batch_size * 12);
        let mut payload_builder =
            StringBuilder::with_capacity(this_batch_size, this_batch_size * 32);

        for j in 0..this_batch_size {
            let i = row_offset + j;
            id_builder.append_value(i as i64);

            // High cardinality keys: num_rows/50 distinct keys, 50 rows each.
            // Join output = num_rows * 50 rows (manageable).
            // Build side still needs full accumulation before probe, forcing spill.
            let key_idx = i % (num_rows / 50).max(1);
            key_builder.append_value(format!("service_{key_idx:05}"));

            // Payload to add memory pressure (~32 bytes each)
            payload_builder.append_value(format!("payload_data_{i:010}_{i:010}"));
        }

        batches.push(
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(id_builder.finish()),
                    Arc::new(key_builder.finish()),
                    Arc::new(payload_builder.finish()),
                ],
            )
            .unwrap(),
        );

        row_offset += this_batch_size;
    }

    batches
}

/// Recursively extract spill metrics from a plan tree
fn extract_spill_metrics(plan: &dyn ExecutionPlan) -> (usize, usize, usize) {
    let (mut count, mut bytes, mut rows) = (0, 0, 0);

    if let Some(metrics) = plan.metrics() {
        count += metrics.spill_count().unwrap_or(0);
        bytes += metrics.spilled_bytes().unwrap_or(0);
        rows += metrics.spilled_rows().unwrap_or(0);
    }

    for child in plan.children() {
        let (c, b, r) = extract_spill_metrics(child.as_ref());
        count += c;
        bytes += b;
        rows += r;
    }

    (count, bytes, rows)
}

/// Extract total output rows from plan
fn extract_output_rows(plan: &dyn ExecutionPlan) -> usize {
    // The root plan's output_rows is what we want
    if let Some(metrics) = plan.metrics() {
        if let Some(rows) = metrics.output_rows() {
            return rows;
        }
    }
    0
}

/// Get current RSS in bytes via /usr/bin/ps (portable, no libc dependency)
fn get_current_rss_bytes() -> usize {
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok();
    output
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<usize>().ok())
        .map(|kb| kb * 1024) // ps reports in KB
        .unwrap_or(0)
}

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    let (num_rows, memory_limit) = get_config();

    println!("=== Hash-Join Spill Benchmark ===");
    println!("Rows: {num_rows}");
    println!("Memory limit: {} MB", memory_limit / 1024 / 1024);
    println!();

    // Generate data
    let gen_start = Instant::now();
    let batches = generate_data(num_rows, BATCH_SIZE);
    let gen_time = gen_start.elapsed();
    let total_mem: usize = batches.iter().map(|b| b.get_array_memory_size()).sum();
    let total_rows_gen: usize = batches.iter().map(|b| b.num_rows()).sum();
    println!(
        "Data generated: {} rows in {} batches, ~{} MB in memory ({:.2}s)",
        total_rows_gen,
        batches.len(),
        total_mem / 1024 / 1024,
        gen_time.as_secs_f64()
    );

    // Create memory-constrained session with disk spill enabled
    let runtime = RuntimeEnvBuilder::new()
        .with_disk_manager_builder(DiskManagerBuilder::default())
        .with_memory_limit(memory_limit, MEMORY_FRACTION)
        .build_arc()?;

    let config = SessionConfig::new().with_target_partitions(1);
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_runtime_env(runtime)
        .with_default_features()
        .build();

    let ctx = SessionContext::new_with_state(state);

    // Register table with multiple batches (allows incremental accumulation)
    let schema = batches[0].schema();
    let table = datafusion::datasource::MemTable::try_new(
        schema,
        vec![batches],
    )?;
    ctx.register_table("t", Arc::new(table))?;

    // Warm up: get the physical plan first and explain it
    let df = ctx.sql("SELECT t1.id, t1.join_key, t1.payload FROM t t1 JOIN t t2 ON t1.join_key = t2.join_key").await?;
    let plan = df.create_physical_plan().await?;

    println!();
    println!("Query: self-join on join_key (500K x 500K)");
    println!("Plan: {:?}", plan.name());

    // Print explain
    let df_explain = ctx.sql("EXPLAIN SELECT t1.id, t1.join_key, t1.payload FROM t t1 JOIN t t2 ON t1.join_key = t2.join_key").await?;
    df_explain.show().await?;
    println!();

    // Record RSS before query
    let _rss_before = get_current_rss_bytes();

    // Execute and measure
    let exec_start = Instant::now();
    let df = ctx.sql("SELECT t1.id, t1.join_key, t1.payload FROM t t1 JOIN t t2 ON t1.join_key = t2.join_key").await?;
    let plan = df.create_physical_plan().await?;

    // Execute all partitions
    let task_ctx = ctx.task_ctx();
    let stream = execute_stream(Arc::clone(&plan), task_ctx)?;

    let mut total_rows: usize = 0;
    let mut batches_collected = 0;
    let mut stream = stream;
    while let Some(batch_result) = stream.next().await {
        match batch_result {
            Ok(batch) => {
                total_rows += batch.num_rows();
                batches_collected += 1;
            }
            Err(e) => {
                let exec_time = exec_start.elapsed();
                println!("QUERY FAILED after {:.3}s: {e}", exec_time.as_secs_f64());
                println!();
                println!("--- RESULT ---");
                println!("status: FAILED");
                println!("error: {e}");
                println!("wall_time_secs: {:.3}", exec_time.as_secs_f64());
                return Ok(());
            }
        }
    }

    let exec_time = exec_start.elapsed();
    let rss_after = get_current_rss_bytes();

    // Extract metrics
    let (spill_count, spilled_bytes, spilled_rows) = extract_spill_metrics(plan.as_ref());
    let output_rows = extract_output_rows(plan.as_ref());
    let actual_output = if output_rows > 0 { output_rows } else { total_rows };

    let throughput = actual_output as f64 / exec_time.as_secs_f64();
    let peak_rss_mb = rss_after as f64 / 1024.0 / 1024.0;

    println!("--- RESULT ---");
    println!("status: OK");
    println!("wall_time_secs: {:.3}", exec_time.as_secs_f64());
    println!("output_rows: {actual_output}");
    println!("batches: {batches_collected}");
    println!("throughput_rows_per_sec: {:.0}", throughput);
    println!("peak_rss_mb: {:.1}", peak_rss_mb);
    println!("spill_count: {spill_count}");
    println!("spilled_bytes: {spilled_bytes}");
    println!("spilled_rows: {spilled_rows}");
    println!(
        "spilled_mb: {:.2}",
        spilled_bytes as f64 / 1024.0 / 1024.0
    );

    Ok(())
}
