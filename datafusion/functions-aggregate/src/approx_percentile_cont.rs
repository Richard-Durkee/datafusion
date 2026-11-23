// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::fmt::Debug;
use std::mem::size_of_val;
use std::sync::Arc;

use arrow::array::{Array, Float16Array};
use arrow::array::{AsArray, BooleanArray, ListArray, PrimitiveArray, UInt64Array};
use arrow::buffer::{BooleanBuffer, OffsetBuffer, ScalarBuffer};
use arrow::compute::{filter, is_not_null};
use arrow::datatypes::FieldRef;
use arrow::datatypes::{Float16Type, Float32Type, Float64Type};
use arrow::{
    array::{ArrayRef, Float32Array, Float64Array},
    datatypes::{DataType, Field},
};
use datafusion_common::cast::{as_float64_array, as_list_array, as_uint64_array};
use datafusion_common::types::{NativeType, logical_float64};
use datafusion_common::{
    Result, ScalarValue, downcast_value, internal_err, not_impl_err, plan_datafusion_err,
    plan_err,
};
use datafusion_expr::expr::{AggregateFunction, Sort};
use datafusion_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion_expr::utils::format_state_name;
use datafusion_expr::{
    Accumulator, AggregateUDFImpl, Coercion, Documentation, Expr, Signature,
    TypeSignature, TypeSignatureClass, Volatility,
};
use datafusion_expr::{DistinctHandling, EmitTo, GroupsAccumulator};
use datafusion_functions_aggregate_common::tdigest::{
    Centroid, DEFAULT_MAX_SIZE, TDigest,
};
use datafusion_macros::user_doc;
use datafusion_physical_expr_common::physical_expr::PhysicalExpr;

use crate::utils::{PercentileParam, get_scalar_value};

create_func!(ApproxPercentileCont, approx_percentile_cont_udaf);

// state field index for the percentile argument
pub const STATE_PERCENTILE_IDX: usize = 6;

/// Computes the approximate percentile continuous of a set of numbers
pub fn approx_percentile_cont(
    order_by: Sort,
    percentile: Expr,
    centroids: Option<Expr>,
) -> Expr {
    let expr = order_by.expr.clone();

    let args = if let Some(centroids) = centroids {
        vec![expr, percentile, centroids]
    } else {
        vec![expr, percentile]
    };

    Expr::AggregateFunction(AggregateFunction::new_udf(
        approx_percentile_cont_udaf(),
        args,
        false,
        None,
        vec![order_by],
        None,
    ))
}

#[user_doc(
    doc_section(label = "Approximate Functions"),
    description = "Returns the approximate percentile of input values using the t-digest algorithm.",
    syntax_example = "approx_percentile_cont(percentile [, centroids]) WITHIN GROUP (ORDER BY expression)",
    sql_example = r#"```sql
> SELECT approx_percentile_cont(0.75) WITHIN GROUP (ORDER BY column_name) FROM table_name;
+------------------------------------------------------------------+
| approx_percentile_cont(0.75) WITHIN GROUP (ORDER BY column_name) |
+------------------------------------------------------------------+
| 65.0                                                             |
+------------------------------------------------------------------+
> SELECT approx_percentile_cont(0.75, 100) WITHIN GROUP (ORDER BY column_name) FROM table_name;
+-----------------------------------------------------------------------+
| approx_percentile_cont(0.75, 100) WITHIN GROUP (ORDER BY column_name) |
+-----------------------------------------------------------------------+
| 65.0                                                                  |
+-----------------------------------------------------------------------+
```
An alternate syntax is also supported:
```sql
> SELECT approx_percentile_cont(column_name, 0.75) FROM table_name;
+-----------------------------------------------+
| approx_percentile_cont(column_name, 0.75)     |
+-----------------------------------------------+
| 65.0                                          |
+-----------------------------------------------+

> SELECT approx_percentile_cont(column_name, 0.75, 100) FROM table_name;
+----------------------------------------------------------+
| approx_percentile_cont(column_name, 0.75, 100)           |
+----------------------------------------------------------+
| 65.0                                                     |
+----------------------------------------------------------+
```
"#,
    standard_argument(name = "expression",),
    argument(
        name = "percentile",
        description = "Percentile to compute. Must be a float value between 0 and 1 (inclusive)."
    ),
    argument(
        name = "centroids",
        description = "Number of centroids to use in the t-digest algorithm. _Default is 100_. A higher number results in more accurate approximation but requires more memory."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ApproxPercentileCont {
    signature: Signature,
}

impl Default for ApproxPercentileCont {
    fn default() -> Self {
        Self::new()
    }
}

impl ApproxPercentileCont {
    /// Create a new [`ApproxPercentileCont`] aggregate function.
    pub fn new() -> Self {
        // Accept any numeric value paired with a float64 percentile
        let signature = Signature::one_of(
            vec![
                // 2 args - numeric, percentile (float)
                TypeSignature::Coercible(vec![
                    Coercion::new_implicit(
                        TypeSignatureClass::Float,
                        vec![TypeSignatureClass::Numeric],
                        NativeType::Float64,
                    ),
                    Coercion::new_implicit_native(
                        logical_float64(),
                        vec![TypeSignatureClass::Numeric],
                    ),
                ]),
                // 3 args - numeric, percentile (float), number of centroid for T-Digest (integer)
                TypeSignature::Coercible(vec![
                    Coercion::new_implicit(
                        TypeSignatureClass::Float,
                        vec![TypeSignatureClass::Numeric],
                        NativeType::Float64,
                    ),
                    Coercion::new_implicit_native(
                        logical_float64(),
                        vec![TypeSignatureClass::Numeric],
                    ),
                    Coercion::new_implicit(
                        TypeSignatureClass::Integer,
                        vec![TypeSignatureClass::Numeric],
                        NativeType::Int64,
                    ),
                ]),
            ],
            Volatility::Immutable,
        );
        Self { signature }
    }

    pub(crate) fn create_accumulator(
        &self,
        args: &AccumulatorArgs,
    ) -> Result<ApproxPercentileAccumulator> {
        let is_descending = args
            .order_bys
            .first()
            .map(|sort_expr| sort_expr.options.descending)
            .unwrap_or(false);

        let percentile = PercentileParam::try_new(
            &args.exprs[1],
            "APPROX_PERCENTILE_CONT",
            is_descending,
        )?;

        let tdigest_max_size = if args.exprs.len() == 3 {
            Some(validate_input_max_size_expr(&args.exprs[2])?)
        } else {
            None
        };

        let data_type = args.expr_fields[0].data_type();
        let accumulator: ApproxPercentileAccumulator = match data_type {
            DataType::Float16 | DataType::Float32 | DataType::Float64 => {
                if let Some(max_size) = tdigest_max_size {
                    ApproxPercentileAccumulator::new_with_max_size(
                        percentile,
                        data_type.clone(),
                        max_size,
                    )
                } else {
                    ApproxPercentileAccumulator::new(percentile, data_type.clone())
                }
            }
            other => {
                return not_impl_err!(
                    "Support for 'APPROX_PERCENTILE_CONT' for data type {other} is not implemented"
                );
            }
        };

        Ok(accumulator)
    }
}

fn validate_input_max_size_expr(expr: &Arc<dyn PhysicalExpr>) -> Result<usize> {
    let scalar_value = get_scalar_value(expr).map_err(|_e| {
        plan_datafusion_err!(
            "Tdigest max_size value for 'APPROX_PERCENTILE_CONT' must be a literal"
        )
    })?;

    let max_size = match scalar_value {
        ScalarValue::UInt8(Some(q)) => q as usize,
        ScalarValue::UInt16(Some(q)) => q as usize,
        ScalarValue::UInt32(Some(q)) => q as usize,
        ScalarValue::UInt64(Some(q)) => q as usize,
        ScalarValue::Int32(Some(q)) if q > 0 => q as usize,
        ScalarValue::Int64(Some(q)) if q > 0 => q as usize,
        ScalarValue::Int16(Some(q)) if q > 0 => q as usize,
        ScalarValue::Int8(Some(q)) if q > 0 => q as usize,
        sv => {
            return plan_err!(
                "Tdigest max_size value for 'APPROX_PERCENTILE_CONT' must be UInt > 0 literal (got data type {}).",
                sv.data_type()
            );
        }
    };

    Ok(max_size)
}

impl AggregateUDFImpl for ApproxPercentileCont {
    /// See [`TDigest::to_scalar_state()`] for a description of the serialized
    /// state.
    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        Ok(vec![
            Field::new(
                format_state_name(args.name, "max_size"),
                DataType::UInt64,
                false,
            ),
            Field::new(
                format_state_name(args.name, "sum"),
                DataType::Float64,
                false,
            ),
            Field::new(
                format_state_name(args.name, "count"),
                DataType::Float64,
                false,
            ),
            Field::new(
                format_state_name(args.name, "max"),
                DataType::Float64,
                false,
            ),
            Field::new(
                format_state_name(args.name, "min"),
                DataType::Float64,
                false,
            ),
            Field::new_list(
                format_state_name(args.name, "centroids"),
                Field::new_list_field(DataType::Float64, true),
                false,
            ),
            Field::new(
                format_state_name(args.name, "percentile"),
                DataType::Float64,
                true,
            ),
        ]
        .into_iter()
        .map(Arc::new)
        .collect())
    }

    fn name(&self) -> &str {
        "approx_percentile_cont"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    #[inline]
    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(self.create_accumulator(&acc_args)?))
    }

    fn groups_accumulator_supported(&self, args: AccumulatorArgs) -> bool {
        // A percentile that references a column must be constant within each
        // group only, which the row accumulator checks per group.
        !args.is_distinct && get_scalar_value(&args.exprs[1]).is_ok()
    }

    fn create_groups_accumulator(
        &self,
        args: AccumulatorArgs,
    ) -> Result<Box<dyn GroupsAccumulator>> {
        let accumulator = self.create_accumulator(&args)?;
        Ok(Box::new(ApproxPercentileGroupsAccumulator::new(
            accumulator.percentile,
            accumulator.return_type,
            accumulator.digest.max_size(),
        )))
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        // Defensive: the public signature already restricts callers to 2 or 3
        // arguments. This guards against aggregate planning accidentally
        // feeding state-field types (e.g. from `PartialReduce`) back into
        // `return_type`, which would otherwise silently choose the wrong type.
        if arg_types.len() > 3 {
            return plan_err!("approx_percentile_cont requires at most 3 arguments");
        }
        if !arg_types[0].is_numeric() {
            return plan_err!("approx_percentile_cont requires numeric input types");
        }
        if arg_types.len() == 3 && !arg_types[2].is_integer() {
            return plan_err!(
                "approx_percentile_cont requires integer centroids input types"
            );
        }
        Ok(arg_types[0].clone())
    }

    fn supports_within_group_clause(&self) -> bool {
        true
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn distinct_handling(&self) -> DistinctHandling {
        // Duplicate-sensitive, but the accumulator does not read
        // `is_distinct` and today silently returns the non-distinct answer.
        // The tag records the intent; enforcement is a follow-up change.
        DistinctHandling::Unsupported
    }
}

#[derive(Debug)]
pub struct ApproxPercentileAccumulator {
    digest: TDigest,
    percentile: PercentileParam,
    return_type: DataType,
}

impl ApproxPercentileAccumulator {
    pub fn new(percentile: PercentileParam, return_type: DataType) -> Self {
        Self {
            digest: TDigest::new(DEFAULT_MAX_SIZE),
            percentile,
            return_type,
        }
    }

    pub(crate) fn new_with_max_size(
        percentile: PercentileParam,
        return_type: DataType,
        max_size: usize,
    ) -> Self {
        Self {
            digest: TDigest::new(max_size),
            percentile,
            return_type,
        }
    }

    /// Callers must only invoke this when a percentile argument actually
    /// exists in the current batch.
    pub(crate) fn resolve_percentile(
        &mut self,
        percentile_array: &ArrayRef,
    ) -> Result<()> {
        self.percentile.resolve(percentile_array)
    }

    // pub(crate) for approx_percentile_cont_with_weight
    pub(crate) fn max_size(&self) -> usize {
        self.digest.max_size()
    }

    // pub(crate) for approx_percentile_cont_with_weight
    pub(crate) fn merge_digests(&mut self, digests: &[TDigest]) {
        let digests = digests.iter().chain(std::iter::once(&self.digest));
        self.digest = TDigest::merge_digests(digests)
    }

    // pub(crate) for approx_percentile_cont_with_weight
    pub(crate) fn convert_to_float(values: &ArrayRef) -> Result<Vec<f64>> {
        debug_assert!(
            values.null_count() == 0,
            "convert_to_float assumes nulls have already been filtered out"
        );
        match values.data_type() {
            DataType::Float64 => {
                let array = downcast_value!(values, Float64Array);
                Ok(array.values().iter().copied().collect::<Vec<_>>())
            }
            DataType::Float32 => {
                let array = downcast_value!(values, Float32Array);
                Ok(array.values().iter().map(|v| *v as f64).collect::<Vec<_>>())
            }
            DataType::Float16 => {
                let array = downcast_value!(values, Float16Array);
                Ok(array
                    .values()
                    .iter()
                    .map(|v| v.to_f64())
                    .collect::<Vec<_>>())
            }
            e => internal_err!(
                "APPROX_PERCENTILE_CONT is not expected to receive the type {e:?}"
            ),
        }
    }
}

impl Accumulator for ApproxPercentileAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let mut state: Vec<ScalarValue> =
            self.digest.to_scalar_state().into_iter().collect();
        state.push(ScalarValue::Float64(self.percentile.get().ok()));
        Ok(state)
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.len() > 1 {
            self.resolve_percentile(&values[1])?;
        }

        // Remove any nulls before computing the percentile
        let mut values = Arc::clone(&values[0]);
        if values.null_count() > 0 {
            values = filter(&values, &is_not_null(&values)?)?;
        }
        let sorted_values = &arrow::compute::sort(&values, None)?;
        let sorted_values = ApproxPercentileAccumulator::convert_to_float(sorted_values)?;
        self.digest = self.digest.merge_sorted_f64(&sorted_values);
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        if self.digest.count() == 0.0 {
            return ScalarValue::try_from(self.return_type.clone());
        }
        let q = self
            .digest
            .estimate_quantile(self.percentile.effective_percentile()?);

        // These acceptable return types MUST match the validation in
        // ApproxPercentile::create_accumulator.
        Ok(match &self.return_type {
            DataType::Float16 => ScalarValue::Float16(Some(half::f16::from_f64(q))),
            DataType::Float32 => ScalarValue::Float32(Some(q as f32)),
            DataType::Float64 => ScalarValue::Float64(Some(q)),
            v => unreachable!("unexpected return type {}", v),
        })
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if states.is_empty() {
            return Ok(());
        }

        if let Some(percentile_array) = states.get(STATE_PERCENTILE_IDX) {
            self.percentile.resolve(percentile_array)?;
        }

        let tdigest_states = &states[..STATE_PERCENTILE_IDX];
        let states = (0..tdigest_states[0].len())
            .map(|index| {
                tdigest_states
                    .iter()
                    .map(|array| ScalarValue::try_from_array(array, index))
                    .collect::<Result<Vec<_>>>()
                    .map(|state| TDigest::from_scalar_state(&state))
            })
            .collect::<Result<Vec<_>>>()?;

        self.merge_digests(&states);

        Ok(())
    }

    fn size(&self) -> usize {
        size_of_val(self) + self.digest.size() - size_of_val(&self.digest)
            + self.return_type.size()
            - size_of_val(&self.return_type)
    }
}

/// [`GroupsAccumulator`] for `approx_percentile_cont` with a literal
/// percentile and for `approx_median`: one [`TDigest`] per group.
///
/// Each batch updates and merges every group's digest the same way
/// [`ApproxPercentileAccumulator`] does under `GroupsAccumulatorAdapter`: the
/// group's values in the batch are sorted and merged in at once, and the
/// group's incoming digests are merged in row order followed by its own. So
/// the results are the same.
#[derive(Debug)]
pub(crate) struct ApproxPercentileGroupsAccumulator {
    digests: Vec<TDigest>,
    max_size: usize,
    percentile: PercentileParam,
    return_type: DataType,
    /// Number of centroids over all `digests`, for `size`.
    num_centroids: usize,
    /// Scratch for one batch: (group, value) or (group, row) pairs.
    pairs: Vec<(usize, f64)>,
    rows: Vec<(usize, usize)>,
    values: Vec<f64>,
}

impl ApproxPercentileGroupsAccumulator {
    pub(crate) fn new(
        percentile: PercentileParam,
        return_type: DataType,
        max_size: usize,
    ) -> Self {
        Self {
            digests: vec![],
            max_size,
            percentile,
            return_type,
            num_centroids: 0,
            pairs: vec![],
            rows: vec![],
            values: vec![],
        }
    }

    fn resize(&mut self, total_num_groups: usize) {
        let max_size = self.max_size;
        self.digests
            .resize_with(total_num_groups, || TDigest::new(max_size));
    }

    fn replace(&mut self, group: usize, digest: TDigest) {
        self.num_centroids -= self.digests[group].centroids().len();
        self.num_centroids += digest.centroids().len();
        self.digests[group] = digest;
    }

    fn take(&mut self, emit_to: EmitTo) -> Vec<TDigest> {
        let digests = emit_to.take_needed(&mut self.digests);
        self.num_centroids -= digests.iter().map(|d| d.centroids().len()).sum::<usize>();
        digests
    }

    /// The state columns, in the order of `state_fields`.
    fn to_state(&self, digests: &[TDigest]) -> Result<Vec<ArrayRef>> {
        let column = |f: fn(&TDigest) -> f64| -> ArrayRef {
            Arc::new(Float64Array::from_iter_values(digests.iter().map(f)))
        };
        let lengths = digests.iter().map(|d| d.centroids().len() * 2);
        let centroids: Float64Array = digests
            .iter()
            .flat_map(|d| d.centroids().iter().flat_map(|c| [c.mean(), c.weight()]))
            .collect();
        let centroids = ListArray::try_new(
            Arc::new(Field::new_list_field(DataType::Float64, true)),
            OffsetBuffer::from_lengths(lengths),
            Arc::new(centroids),
            None,
        )?;
        let percentile = self.percentile.get().ok();
        Ok(vec![
            Arc::new(UInt64Array::from_iter_values(
                digests.iter().map(|d| d.max_size() as u64),
            )),
            column(TDigest::sum),
            column(TDigest::count),
            column(TDigest::max),
            column(TDigest::min),
            Arc::new(centroids),
            Arc::new(Float64Array::from(vec![percentile; digests.len()])),
        ])
    }
}

/// The input values as `f64`. Entries for NULL rows are unspecified.
fn values_as_f64(values: &ArrayRef) -> Result<ScalarBuffer<f64>> {
    Ok(match values.data_type() {
        DataType::Float64 => values.as_primitive::<Float64Type>().values().clone(),
        DataType::Float32 => {
            let values = values.as_primitive::<Float32Type>().values();
            values.iter().map(|v| *v as f64).collect()
        }
        DataType::Float16 => {
            let values = values.as_primitive::<Float16Type>().values();
            values.iter().map(|v| v.to_f64()).collect()
        }
        e => {
            return internal_err!(
                "APPROX_PERCENTILE_CONT is not expected to receive the type {e:?}"
            );
        }
    })
}

/// Rows whose value is not NULL and that pass the filter, or `None` for all.
fn rows_to_update(
    values: &ArrayRef,
    opt_filter: Option<&BooleanArray>,
) -> Option<BooleanBuffer> {
    let filter = opt_filter.map(|filter| match filter.nulls() {
        Some(nulls) => filter.values() & nulls.inner(),
        None => filter.values().clone(),
    });
    match (values.logical_nulls(), filter) {
        (None, None) => None,
        (Some(nulls), None) => Some(nulls.into_inner()),
        (None, Some(filter)) => Some(filter),
        (Some(nulls), Some(filter)) => Some(nulls.inner() & &filter),
    }
}

impl GroupsAccumulator for ApproxPercentileGroupsAccumulator {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        if values.len() > 1 {
            self.percentile.resolve(&values[1])?;
        }
        self.resize(total_num_groups);
        let input = values_as_f64(&values[0])?;

        // Sort the batch's values by group, then by value, and merge each
        // group's sorted values into its digest.
        let mut pairs = std::mem::take(&mut self.pairs);
        pairs.clear();
        match rows_to_update(&values[0], opt_filter) {
            None => {
                pairs.extend(group_indices.iter().copied().zip(input.iter().copied()))
            }
            Some(rows) => {
                pairs.extend(rows.set_indices().map(|r| (group_indices[r], input[r])))
            }
        }
        pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
        let mut values = std::mem::take(&mut self.values);
        for run in pairs.chunk_by(|a, b| a.0 == b.0) {
            values.clear();
            values.extend(run.iter().map(|(_, v)| *v));
            let group = run[0].0;
            let digest = self.digests[group].merge_sorted_f64(&values);
            self.replace(group, digest);
        }
        self.pairs = pairs;
        self.values = values;
        Ok(())
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        total_num_groups: usize,
    ) -> Result<()> {
        if let Some(percentile) = values.get(STATE_PERCENTILE_IDX) {
            self.percentile.resolve(percentile)?;
        }
        self.resize(total_num_groups);
        let max_size = as_uint64_array(&values[0])?;
        let sum = as_float64_array(&values[1])?;
        let count = as_float64_array(&values[2])?;
        let max = as_float64_array(&values[3])?;
        let min = as_float64_array(&values[4])?;
        let centroids = as_list_array(&values[5])?;
        let centroid_values = as_float64_array(centroids.values())?.values();
        let offsets = centroids.value_offsets();
        let digest_at = |row: usize| {
            let (start, end) = (offsets[row] as usize, offsets[row + 1] as usize);
            TDigest::try_from_parts(
                max_size.value(row) as usize,
                sum.value(row),
                count.value(row),
                max.value(row),
                min.value(row),
                centroid_values[start..end]
                    .chunks(2)
                    .map(|c| Centroid::new(c[0], c[1]))
                    .collect(),
            )
        };

        // Group rows, keeping row order within each group.
        let mut rows = std::mem::take(&mut self.rows);
        rows.clear();
        rows.extend(group_indices.iter().enumerate().map(|(r, &g)| (g, r)));
        rows.sort_unstable();
        for run in rows.chunk_by(|a, b| a.0 == b.0) {
            let group = run[0].0;
            let incoming = run
                .iter()
                .map(|&(_, row)| digest_at(row))
                .collect::<Result<Vec<_>>>()?;
            let digest =
                TDigest::merge_digests(incoming.iter().chain([&self.digests[group]]));
            self.replace(group, digest);
        }
        self.rows = rows;
        Ok(())
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        let digests = self.take(emit_to);
        // Like the row accumulator, only groups with values need the percentile.
        let percentile = if digests.iter().any(|d| d.count() != 0.0) {
            self.percentile.effective_percentile()?
        } else {
            0.0
        };
        let quantiles: Vec<Option<f64>> = digests
            .iter()
            .map(|d| (d.count() != 0.0).then(|| d.estimate_quantile(percentile)))
            .collect();
        // These return types MUST match the validation in create_accumulator.
        Ok(match &self.return_type {
            DataType::Float16 => Arc::new(
                quantiles
                    .into_iter()
                    .map(|q| q.map(half::f16::from_f64))
                    .collect::<PrimitiveArray<Float16Type>>(),
            ),
            DataType::Float32 => Arc::new(
                quantiles
                    .into_iter()
                    .map(|q| q.map(|q| q as f32))
                    .collect::<Float32Array>(),
            ),
            DataType::Float64 => Arc::new(Float64Array::from(quantiles)),
            v => return internal_err!("unexpected return type {v}"),
        })
    }

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        let digests = self.take(emit_to);
        self.to_state(&digests)
    }

    fn convert_to_state(
        &self,
        values: &[ArrayRef],
        opt_filter: Option<&BooleanArray>,
    ) -> Result<Vec<ArrayRef>> {
        let input = values_as_f64(&values[0])?;
        let rows = rows_to_update(&values[0], opt_filter);
        let empty = TDigest::new(self.max_size);
        let digests: Vec<TDigest> = (0..values[0].len())
            .map(|row| {
                if rows.as_ref().is_none_or(|rows| rows.value(row)) {
                    empty.merge_sorted_f64(&[input[row]])
                } else {
                    empty.clone()
                }
            })
            .collect();
        self.to_state(&digests)
    }

    fn size(&self) -> usize {
        size_of_val(self)
            + self.digests.capacity() * size_of::<TDigest>()
            + self.num_centroids * size_of::<Centroid>()
            + self.pairs.capacity() * size_of::<(usize, f64)>()
            + self.rows.capacity() * size_of::<(usize, usize)>()
            + self.values.capacity() * size_of::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::DataType;

    use datafusion_functions_aggregate_common::tdigest::TDigest;

    use crate::approx_percentile_cont::ApproxPercentileAccumulator;
    use crate::utils::{PercentileParam, PercentileParamState};

    fn make_accumulator() -> ApproxPercentileAccumulator {
        ApproxPercentileAccumulator::new_with_max_size(
            PercentileParam {
                aggregate_fn_name: "APPROX_PERCENTILE_CONT".to_string(),
                state: PercentileParamState::Resolved(0.5),
                is_desc: false,
            },
            DataType::Float64,
            100,
        )
    }

    #[test]
    fn test_combine_approx_percentile_accumulator() {
        let mut digests: Vec<TDigest> = Vec::new();

        // one TDigest with 50_000 values from 1 to 1_000
        for _ in 1..=50 {
            let t = TDigest::new(100);
            let values: Vec<_> = (1..=1_000).map(f64::from).collect();
            let t = t.merge_unsorted_f64(values);
            digests.push(t)
        }

        let t1 = TDigest::merge_digests(&digests);
        let t2 = TDigest::merge_digests(&digests);

        let mut accumulator = make_accumulator();

        accumulator.merge_digests(&[t1]);
        assert_eq!(accumulator.digest.count(), 50_000.0);
        accumulator.merge_digests(&[t2]);
        assert_eq!(accumulator.digest.count(), 100_000.0);
    }

    mod groups {
        use std::sync::Arc;

        use arrow::array::{ArrayRef, BooleanArray, Float64Array};
        use arrow::datatypes::DataType;
        use datafusion_common::Result;
        use datafusion_expr::{EmitTo, GroupsAccumulator};
        use datafusion_functions_aggregate_common::aggregate::groups_accumulator::GroupsAccumulatorAdapter;

        use crate::approx_percentile_cont::{
            ApproxPercentileAccumulator, ApproxPercentileGroupsAccumulator,
        };
        use crate::utils::{PercentileParam, PercentileParamState};

        const MAX_SIZE: usize = 10;

        fn percentile() -> PercentileParam {
            PercentileParam {
                aggregate_fn_name: "APPROX_PERCENTILE_CONT".to_string(),
                state: PercentileParamState::Resolved(0.3),
                is_desc: false,
            }
        }

        /// The row accumulator under the adapter, which is what runs without
        /// the groups accumulator.
        fn adapter() -> GroupsAccumulatorAdapter {
            GroupsAccumulatorAdapter::new(|| {
                Ok(Box::new(ApproxPercentileAccumulator::new_with_max_size(
                    percentile(),
                    DataType::Float64,
                    MAX_SIZE,
                )))
            })
        }

        fn groups() -> ApproxPercentileGroupsAccumulator {
            ApproxPercentileGroupsAccumulator::new(
                percentile(),
                DataType::Float64,
                MAX_SIZE,
            )
        }

        /// Batches of (values, groups, filter) with NULLs, NaN, infinities,
        /// repeated groups and enough values per group to compress digests.
        fn batches() -> Vec<(ArrayRef, Vec<usize>, BooleanArray)> {
            (0..4)
                .map(|b| {
                    let n = 300;
                    let values: Vec<Option<f64>> = (0..n)
                        .map(|i| match (i * 7 + b) % 50 {
                            0 => None,
                            1 => Some(f64::NAN),
                            2 => Some(f64::INFINITY),
                            3 => Some(f64::NEG_INFINITY),
                            _ => Some(((i * 37 + b * 11) % 97) as f64 / 3.0),
                        })
                        .collect();
                    let groups = (0..n).map(|i| (i * 13 + b) % 11).collect();
                    let filter = (0..n)
                        .map(|i| (i % 23 != 0).then_some(i % 9 != 0))
                        .collect();
                    (
                        Arc::new(Float64Array::from(values)) as ArrayRef,
                        groups,
                        filter,
                    )
                })
                .collect()
        }

        fn run(
            acc: &mut dyn GroupsAccumulator,
            partial: &mut dyn GroupsAccumulator,
        ) -> Result<Vec<ArrayRef>> {
            let batches = batches();
            let num_groups = 12;
            for (values, groups, filter) in &batches[..2] {
                acc.update_batch(
                    &[Arc::clone(values)],
                    groups,
                    Some(filter),
                    num_groups,
                )?;
            }
            // Partial states, merged into `acc`: one from update_batch and one
            // from convert_to_state.
            let (values, groups, filter) = &batches[2];
            partial.update_batch(
                &[Arc::clone(values)],
                groups,
                Some(filter),
                num_groups,
            )?;
            let all_groups: Vec<usize> = (0..num_groups).collect();
            acc.merge_batch(&partial.state(EmitTo::All)?, &all_groups, num_groups)?;
            let (values, groups, filter) = &batches[3];
            let converted =
                partial.convert_to_state(&[Arc::clone(values)], Some(filter))?;
            acc.merge_batch(&converted, groups, num_groups)?;
            Ok(vec![
                acc.evaluate(EmitTo::First(5))?,
                acc.evaluate(EmitTo::All)?,
            ])
        }

        #[test]
        fn matches_row_accumulator() -> Result<()> {
            let expected = run(&mut adapter(), &mut adapter())?;
            let actual = run(&mut groups(), &mut groups())?;
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(&expected) {
                // Compare bits so NaN results compare equal.
                let bits = |a: &ArrayRef| -> Vec<Option<u64>> {
                    a.as_any()
                        .downcast_ref::<Float64Array>()
                        .unwrap()
                        .iter()
                        .map(|v| v.map(f64::to_bits))
                        .collect()
                };
                assert_eq!(bits(actual), bits(expected));
            }
            Ok(())
        }

        #[test]
        fn state_matches_row_accumulator() -> Result<()> {
            let (values, groups_idx, filter) = batches().remove(0);
            let mut expected = adapter();
            expected.update_batch(
                &[Arc::clone(&values)],
                &groups_idx,
                Some(&filter),
                12,
            )?;
            let mut actual = groups();
            actual.update_batch(
                &[Arc::clone(&values)],
                &groups_idx,
                Some(&filter),
                12,
            )?;
            let (expected, actual) =
                (expected.state(EmitTo::All)?, actual.state(EmitTo::All)?);
            for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(a.data_type(), e.data_type(), "column {i}");
                // NaN min/max of empty digests compare unequal, so compare
                // their debug output.
                assert_eq!(format!("{a:?}"), format!("{e:?}"), "column {i}");
            }
            Ok(())
        }
    }
}
