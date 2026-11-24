// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Benchmarks interning a single primitive group-by column.

use std::hint::black_box;
use std::sync::Arc;

use arrow::array::{ArrayRef, Decimal128Array, Int32Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use datafusion_physical_plan::aggregates::group_values::new_group_values;
use datafusion_physical_plan::aggregates::order::GroupClustering;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const BATCH_SIZE: usize = 8192;
const NUM_ROWS: usize = 2 * 1024 * 1024;

/// Keys drawn uniformly from `0..cardinality`, in batches of `BATCH_SIZE`.
fn batches(data_type: &DataType, cardinality: u64) -> Vec<ArrayRef> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..NUM_ROWS / BATCH_SIZE)
        .map(|_| {
            let keys = (0..BATCH_SIZE).map(|_| rng.random_range(0..cardinality));
            match data_type {
                DataType::Int32 => {
                    Arc::new(Int32Array::from_iter_values(keys.map(|k| k as i32)))
                        as ArrayRef
                }
                DataType::Int64 => {
                    Arc::new(Int64Array::from_iter_values(keys.map(|k| k as i64)))
                }
                DataType::Decimal128(p, s) => Arc::new(
                    Decimal128Array::from_iter_values(keys.map(|k| k as i128))
                        .with_precision_and_scale(*p, *s)
                        .unwrap(),
                ),
                other => unimplemented!("{other}"),
            }
        })
        .collect()
}

fn intern(c: &mut Criterion) {
    let mut group = c.benchmark_group("primitive_group_values");
    group.sample_size(10);
    for data_type in [
        DataType::Int64,
        DataType::Int32,
        DataType::Decimal128(18, 2),
    ] {
        let schema =
            Arc::new(Schema::new(vec![Field::new("k", data_type.clone(), false)]));
        for cardinality in [1_000, 100_000, 2_000_000] {
            let batches = batches(&data_type, cardinality);
            group.bench_function(
                BenchmarkId::new(format!("{data_type}"), cardinality),
                |b| {
                    b.iter(|| {
                        let mut group_values =
                            new_group_values(Arc::clone(&schema), &GroupClustering::None)
                                .unwrap();
                        let mut groups = Vec::with_capacity(BATCH_SIZE);
                        for batch in &batches {
                            group_values
                                .intern(std::slice::from_ref(batch), &mut groups)
                                .unwrap();
                        }
                        black_box(group_values.len())
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, intern);
criterion_main!(benches);
