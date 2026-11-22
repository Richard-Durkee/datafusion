// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::array::ArrayRef;
use arrow::datatypes::{Field, Int64Type};
use arrow::util::bench_util::{
    create_primitive_array_with_seed,
    create_string_array_with_len_range_and_prefix_and_seed,
};
use criterion::{Criterion, criterion_group};
use datafusion_common::config::ConfigOptions;
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs};
use datafusion_functions::core::{greatest, least};
use std::hint::black_box;
use std::sync::Arc;

// DataFusion's default execution batch size.
const ROWS: usize = 8192;

fn int64(null_density: f32, seed: u64) -> ArrayRef {
    Arc::new(create_primitive_array_with_seed::<Int64Type>(
        ROWS,
        null_density,
        seed,
    ))
}

fn utf8(null_density: f32, seed: u64) -> ArrayRef {
    Arc::new(
        create_string_array_with_len_range_and_prefix_and_seed::<i32>(
            ROWS,
            null_density,
            8,
            24,
            "",
            seed,
        ),
    )
}

fn criterion_benchmark(c: &mut Criterion) {
    let cases = [
        ("int64", int64(0.0, 1), int64(0.0, 2)),
        ("int64 nulls", int64(0.1, 1), int64(0.1, 2)),
        ("int64 nulls on one side", int64(0.1, 1), int64(0.0, 2)),
        ("utf8", utf8(0.0, 1), utf8(0.0, 2)),
        ("utf8 nulls", utf8(0.1, 1), utf8(0.1, 2)),
    ];
    let config_options = Arc::new(ConfigOptions::default());

    for (name, lhs, rhs) in cases {
        let return_field = Arc::new(Field::new("f", lhs.data_type().clone(), true));
        let args = vec![ColumnarValue::Array(lhs), ColumnarValue::Array(rhs)];
        let arg_fields = args
            .iter()
            .enumerate()
            .map(|(idx, arg)| {
                Field::new(format!("arg_{idx}"), arg.data_type(), true).into()
            })
            .collect::<Vec<_>>();

        for func in [greatest(), least()] {
            c.bench_function(&format!("{} {name}", func.name()), |b| {
                b.iter(|| {
                    black_box(
                        func.invoke_with_args(ScalarFunctionArgs {
                            args: args.clone(),
                            arg_fields: arg_fields.clone(),
                            number_rows: ROWS,
                            return_field: Arc::clone(&return_field),
                            config_options: Arc::clone(&config_options),
                        })
                        .unwrap(),
                    )
                })
            });
        }
    }
}

criterion_group!(benches, criterion_benchmark);
