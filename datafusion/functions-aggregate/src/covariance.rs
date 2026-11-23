// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`CovarianceSample`]: covariance sample aggregations.

use arrow::array::{Array, ArrayRef, BooleanArray, Float64Array, UInt64Array};
use arrow::buffer::BooleanBuffer;
use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::cast::{as_float64_array, as_uint64_array};
use datafusion_common::{Result, ScalarValue, not_impl_err};
use datafusion_expr::{
    Accumulator, AggregateUDFImpl, Documentation, Signature, Volatility,
    function::{AccumulatorArgs, StateFieldsArgs},
    utils::format_state_name,
};
use datafusion_expr::{DistinctHandling, EmitTo, GroupsAccumulator};
use datafusion_functions_aggregate_common::stats::StatsType;
use datafusion_macros::user_doc;
use std::fmt::Debug;
use std::mem::size_of_val;
use std::sync::Arc;

make_udaf_expr_and_func!(
    CovarianceSample,
    covar_samp,
    y x,
    "Computes the sample covariance.",
    covar_samp_udaf
);

make_udaf_expr_and_func!(
    CovariancePopulation,
    covar_pop,
    y x,
    "Computes the population covariance.",
    covar_pop_udaf
);

#[user_doc(
    doc_section(label = "Statistical Functions"),
    description = "Returns the sample covariance of a set of number pairs.",
    syntax_example = "covar_samp(expression1, expression2)",
    sql_example = r#"```sql
> SELECT covar_samp(column1, column2) FROM table_name;
+-----------------------------------+
| covar_samp(column1, column2)      |
+-----------------------------------+
| 8.25                              |
+-----------------------------------+
```"#,
    standard_argument(name = "expression1", prefix = "First"),
    standard_argument(name = "expression2", prefix = "Second")
)]
#[derive(PartialEq, Eq, Hash, Debug)]
pub struct CovarianceSample {
    signature: Signature,
    aliases: Vec<String>,
}

impl Default for CovarianceSample {
    fn default() -> Self {
        Self::new()
    }
}

impl CovarianceSample {
    pub fn new() -> Self {
        Self {
            aliases: vec![String::from("covar")],
            signature: Signature::exact(
                vec![DataType::Float64, DataType::Float64],
                Volatility::Immutable,
            ),
        }
    }
}

impl AggregateUDFImpl for CovarianceSample {
    fn name(&self) -> &str {
        "covar_samp"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        let name = args.name;
        Ok(vec![
            Field::new(format_state_name(name, "count"), DataType::UInt64, true),
            Field::new(format_state_name(name, "mean1"), DataType::Float64, true),
            Field::new(format_state_name(name, "mean2"), DataType::Float64, true),
            Field::new(
                format_state_name(name, "algo_const"),
                DataType::Float64,
                true,
            ),
        ]
        .into_iter()
        .map(Arc::new)
        .collect())
    }

    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        if acc_args.is_distinct {
            return not_impl_err!("COVAR_SAMP(DISTINCT) aggregations are not available");
        }
        Ok(Box::new(CovarianceAccumulator::try_new(StatsType::Sample)?))
    }

    fn groups_accumulator_supported(&self, args: AccumulatorArgs) -> bool {
        !args.is_distinct
    }

    fn create_groups_accumulator(
        &self,
        args: AccumulatorArgs,
    ) -> Result<Box<dyn GroupsAccumulator>> {
        if args.is_distinct {
            return not_impl_err!("COVAR_SAMP(DISTINCT) aggregations are not available");
        }
        Ok(Box::new(CovarianceGroupsAccumulator::new(
            StatsType::Sample,
        )))
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn distinct_handling(&self) -> DistinctHandling {
        // The accumulator rejects `DISTINCT` with `not_impl_err!`. It only
        // works when the planner removes it first (`SingleDistinctToGroupBy`
        // for a single distinct field, e.g. `covar_samp(DISTINCT x, x)`).
        DistinctHandling::Unsupported
    }
}

#[user_doc(
    doc_section(label = "Statistical Functions"),
    description = "Returns the population covariance of a set of number pairs.",
    syntax_example = "covar_pop(expression1, expression2)",
    sql_example = r#"```sql
> SELECT covar_pop(column1, column2) FROM table_name;
+-----------------------------------+
| covar_pop(column1, column2)       |
+-----------------------------------+
| 7.63333333333                     |
+-----------------------------------+
```"#,
    standard_argument(name = "expression1", prefix = "First"),
    standard_argument(name = "expression2", prefix = "Second")
)]
#[derive(PartialEq, Eq, Hash, Debug)]
pub struct CovariancePopulation {
    signature: Signature,
}

impl Default for CovariancePopulation {
    fn default() -> Self {
        Self::new()
    }
}

impl CovariancePopulation {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![DataType::Float64, DataType::Float64],
                Volatility::Immutable,
            ),
        }
    }
}

impl AggregateUDFImpl for CovariancePopulation {
    fn name(&self) -> &str {
        "covar_pop"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        let name = args.name;
        Ok(vec![
            Field::new(format_state_name(name, "count"), DataType::UInt64, true),
            Field::new(format_state_name(name, "mean1"), DataType::Float64, true),
            Field::new(format_state_name(name, "mean2"), DataType::Float64, true),
            Field::new(
                format_state_name(name, "algo_const"),
                DataType::Float64,
                true,
            ),
        ]
        .into_iter()
        .map(Arc::new)
        .collect())
    }

    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        if acc_args.is_distinct {
            return not_impl_err!("COVAR_POP(DISTINCT) aggregations are not available");
        }
        Ok(Box::new(CovarianceAccumulator::try_new(
            StatsType::Population,
        )?))
    }

    fn groups_accumulator_supported(&self, args: AccumulatorArgs) -> bool {
        !args.is_distinct
    }

    fn create_groups_accumulator(
        &self,
        args: AccumulatorArgs,
    ) -> Result<Box<dyn GroupsAccumulator>> {
        if args.is_distinct {
            return not_impl_err!("COVAR_POP(DISTINCT) aggregations are not available");
        }
        Ok(Box::new(CovarianceGroupsAccumulator::new(
            StatsType::Population,
        )))
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn distinct_handling(&self) -> DistinctHandling {
        // The accumulator rejects `DISTINCT` with `not_impl_err!`. It only
        // works when the planner removes it first (`SingleDistinctToGroupBy`
        // for a single distinct field, e.g. `covar_pop(DISTINCT x, x)`).
        DistinctHandling::Unsupported
    }
}

/// An accumulator to compute covariance
/// The algorithm used is an online implementation and numerically stable. It is derived from the following paper
/// for calculating variance:
/// Welford, B. P. (1962). "Note on a method for calculating corrected sums of squares and products".
/// Technometrics. 4 (3): 419–420. doi:10.2307/1266577. JSTOR 1266577.
///
/// The algorithm has been analyzed here:
/// Ling, Robert F. (1974). "Comparison of Several Algorithms for Computing Sample Means and Variances".
/// Journal of the American Statistical Association. 69 (348): 859–866. doi:10.2307/2286154. JSTOR 2286154.
///
/// Though it is not covered in the original paper but is based on the same idea, as a result the algorithm is online,
/// parallelize and numerically stable.

#[derive(Debug)]
pub struct CovarianceAccumulator {
    state: CovarianceState,
    stats_type: StatsType,
}

/// The statistics of one group, shared by [`CovarianceAccumulator`] and
/// [`CovarianceGroupsAccumulator`].
#[derive(Debug, Default, Clone, Copy)]
struct CovarianceState {
    algo_const: f64,
    mean1: f64,
    mean2: f64,
    count: u64,
}

impl CovarianceState {
    /// Adds one row where both values are not NULL.
    #[inline]
    fn update(&mut self, value1: f64, value2: f64) {
        let new_count = self.count + 1;
        let delta1 = value1 - self.mean1;
        let new_mean1 = delta1 / new_count as f64 + self.mean1;
        let delta2 = value2 - self.mean2;
        let new_mean2 = delta2 / new_count as f64 + self.mean2;
        let new_c = delta1 * (value2 - new_mean2) + self.algo_const;

        self.count += 1;
        self.mean1 = new_mean1;
        self.mean2 = new_mean2;
        self.algo_const = new_c;
    }

    /// Merges a partial state with `c` rows into this one.
    #[inline]
    fn merge(&mut self, c: u64, mean1: f64, mean2: f64, algo_const: f64) {
        if c == 0_u64 {
            return;
        }
        let new_count = self.count + c;
        let new_mean1 = self.mean1 * self.count as f64 / new_count as f64
            + mean1 * c as f64 / new_count as f64;
        let new_mean2 = self.mean2 * self.count as f64 / new_count as f64
            + mean2 * c as f64 / new_count as f64;
        let delta1 = self.mean1 - mean1;
        let delta2 = self.mean2 - mean2;
        let new_c = self.algo_const
            + algo_const
            + delta1 * delta2 * self.count as f64 * c as f64 / new_count as f64;

        self.count = new_count;
        self.mean1 = new_mean1;
        self.mean2 = new_mean2;
        self.algo_const = new_c;
    }

    fn evaluate(&self, stats_type: &StatsType) -> Option<f64> {
        let count = match stats_type {
            StatsType::Population => self.count,
            StatsType::Sample => {
                if self.count > 0 {
                    self.count - 1
                } else {
                    self.count
                }
            }
        };

        (count != 0).then(|| self.algo_const / count as f64)
    }
}

impl CovarianceAccumulator {
    /// Creates a new `CovarianceAccumulator`
    pub fn try_new(s_type: StatsType) -> Result<Self> {
        Ok(Self {
            state: CovarianceState::default(),
            stats_type: s_type,
        })
    }

    pub fn get_count(&self) -> u64 {
        self.state.count
    }

    pub fn get_mean1(&self) -> f64 {
        self.state.mean1
    }

    pub fn get_mean2(&self) -> f64 {
        self.state.mean2
    }

    pub fn get_algo_const(&self) -> f64 {
        self.state.algo_const
    }
}

impl Accumulator for CovarianceAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let s = &self.state;
        Ok(vec![
            ScalarValue::from(s.count),
            ScalarValue::from(s.mean1),
            ScalarValue::from(s.mean2),
            ScalarValue::from(s.algo_const),
        ])
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let values1 = as_float64_array(&values[0])?;
        let values2 = as_float64_array(&values[1])?;

        for (value1, value2) in values1.iter().zip(values2) {
            let (Some(value1), Some(value2)) = (value1, value2) else {
                continue;
            };
            self.state.update(value1, value2);
        }

        Ok(())
    }

    fn retract_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let values1 = as_float64_array(&values[0])?;
        let values2 = as_float64_array(&values[1])?;
        let s = &mut self.state;

        for (value1, value2) in values1.iter().zip(values2) {
            let (Some(value1), Some(value2)) = (value1, value2) else {
                continue;
            };

            if s.count <= 1 {
                *s = CovarianceState::default();
                continue;
            }

            let new_count = s.count - 1;
            let delta1 = s.mean1 - value1;
            let new_mean1 = delta1 / new_count as f64 + s.mean1;
            let delta2 = s.mean2 - value2;
            let new_mean2 = delta2 / new_count as f64 + s.mean2;
            let new_c = s.algo_const - delta1 * (new_mean2 - value2);

            s.count -= 1;
            s.mean1 = new_mean1;
            s.mean2 = new_mean2;
            s.algo_const = new_c;
        }

        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let counts = as_uint64_array(&states[0])?;
        let means1 = as_float64_array(&states[1])?;
        let means2 = as_float64_array(&states[2])?;
        let cs = as_float64_array(&states[3])?;

        for i in 0..counts.len() {
            self.state.merge(
                counts.value(i),
                means1.value(i),
                means2.value(i),
                cs.value(i),
            );
        }
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        Ok(ScalarValue::Float64(self.state.evaluate(&self.stats_type)))
    }

    fn size(&self) -> usize {
        size_of_val(self)
    }

    fn supports_retract_batch(&self) -> bool {
        true
    }
}

/// [`GroupsAccumulator`] for `covar_samp` and `covar_pop`: one
/// [`CovarianceState`] per group, updated with the same arithmetic as
/// [`CovarianceAccumulator`].
#[derive(Debug)]
struct CovarianceGroupsAccumulator {
    states: Vec<CovarianceState>,
    stats_type: StatsType,
}

impl CovarianceGroupsAccumulator {
    fn new(stats_type: StatsType) -> Self {
        Self {
            states: vec![],
            stats_type,
        }
    }
}

/// The state columns, in the order of `state_fields`.
fn states_to_arrays(states: &[CovarianceState]) -> Vec<ArrayRef> {
    let column = |f: fn(&CovarianceState) -> f64| -> ArrayRef {
        Arc::new(Float64Array::from_iter_values(states.iter().map(f)))
    };
    vec![
        Arc::new(UInt64Array::from_iter_values(
            states.iter().map(|s| s.count),
        )),
        column(|s| s.mean1),
        column(|s| s.mean2),
        column(|s| s.algo_const),
    ]
}

/// Rows where both values and the filter are valid and the filter is true,
/// or `None` when every row is.
fn valid_rows(
    values1: &Float64Array,
    values2: &Float64Array,
    opt_filter: Option<&BooleanArray>,
) -> Option<BooleanBuffer> {
    let valid = arrow::buffer::NullBuffer::union(values1.nulls(), values2.nulls());
    let filter = opt_filter.map(|filter| match filter.nulls() {
        Some(nulls) => filter.values() & nulls.inner(),
        None => filter.values().clone(),
    });
    match (valid, filter) {
        (None, None) => None,
        (Some(valid), None) => Some(valid.into_inner()),
        (None, Some(filter)) => Some(filter),
        (Some(valid), Some(filter)) => Some(valid.inner() & &filter),
    }
}

impl GroupsAccumulator for CovarianceGroupsAccumulator {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        self.states
            .resize(total_num_groups, CovarianceState::default());
        let values1 = as_float64_array(&values[0])?;
        let values2 = as_float64_array(&values[1])?;
        let (v1, v2) = (values1.values(), values2.values());
        match valid_rows(values1, values2, opt_filter) {
            None => {
                for (row, &group) in group_indices.iter().enumerate() {
                    self.states[group].update(v1[row], v2[row]);
                }
            }
            Some(valid) => {
                for row in valid.set_indices() {
                    self.states[group_indices[row]].update(v1[row], v2[row]);
                }
            }
        }
        Ok(())
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        total_num_groups: usize,
    ) -> Result<()> {
        self.states
            .resize(total_num_groups, CovarianceState::default());
        let counts = as_uint64_array(&values[0])?;
        let means1 = as_float64_array(&values[1])?;
        let means2 = as_float64_array(&values[2])?;
        let cs = as_float64_array(&values[3])?;
        for (row, &group) in group_indices.iter().enumerate() {
            self.states[group].merge(
                counts.value(row),
                means1.value(row),
                means2.value(row),
                cs.value(row),
            );
        }
        Ok(())
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        let states = emit_to.take_needed(&mut self.states);
        Ok(Arc::new(Float64Array::from_iter(
            states.iter().map(|s| s.evaluate(&self.stats_type)),
        )))
    }

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        let states = emit_to.take_needed(&mut self.states);
        Ok(states_to_arrays(&states))
    }

    fn convert_to_state(
        &self,
        values: &[ArrayRef],
        opt_filter: Option<&BooleanArray>,
    ) -> Result<Vec<ArrayRef>> {
        let values1 = as_float64_array(&values[0])?;
        let values2 = as_float64_array(&values[1])?;
        let valid = valid_rows(values1, values2, opt_filter);
        let (v1, v2) = (values1.values(), values2.values());
        // Each valid row is a group with that one row; others stay empty,
        // which `merge` skips.
        let states: Vec<CovarianceState> = (0..values1.len())
            .map(|row| {
                let mut state = CovarianceState::default();
                if valid.as_ref().is_none_or(|v| v.value(row)) {
                    state.update(v1[row], v2[row]);
                }
                state
            })
            .collect();
        Ok(states_to_arrays(&states))
    }

    fn size(&self) -> usize {
        self.states.capacity() * size_of::<CovarianceState>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_functions_aggregate_common::aggregate::groups_accumulator::GroupsAccumulatorAdapter;

    type Batch = (Vec<ArrayRef>, Vec<usize>, BooleanArray);

    /// Batches with NULLs in both columns and the filter, and a group with
    /// a single row.
    fn batches() -> Vec<Batch> {
        (0..4)
            .map(|b| {
                let n = 200;
                let col = |m: usize, k: usize| -> ArrayRef {
                    Arc::new(Float64Array::from(
                        (0..n)
                            .map(|i| {
                                ((i + b) % m != 0)
                                    .then_some(((i * k + b) % 97) as f64 / 7.0)
                            })
                            .collect::<Vec<_>>(),
                    ))
                };
                let groups = (0..n)
                    .map(|i| if i == 5 { 9 } else { (i * 13 + b) % 9 })
                    .collect();
                let filter = (0..n)
                    .map(|i| (i % 19 != 0).then_some(i % 7 != 0))
                    .collect();
                (vec![col(11, 37), col(13, 17)], groups, filter)
            })
            .collect()
    }

    fn run(
        acc: &mut dyn GroupsAccumulator,
        partial: &mut dyn GroupsAccumulator,
    ) -> Result<Vec<Option<u64>>> {
        let batches = batches();
        let num_groups = 10;
        for (values, groups, filter) in &batches[..2] {
            acc.update_batch(values, groups, Some(filter), num_groups)?;
        }
        let (values, groups, filter) = &batches[2];
        partial.update_batch(values, groups, Some(filter), num_groups)?;
        let all_groups: Vec<usize> = (0..num_groups).collect();
        acc.merge_batch(&partial.state(EmitTo::All)?, &all_groups, num_groups)?;
        let (values, groups, filter) = &batches[3];
        acc.merge_batch(
            &partial.convert_to_state(values, Some(filter))?,
            groups,
            num_groups,
        )?;
        let mut out = vec![];
        for emit in [EmitTo::First(4), EmitTo::All] {
            let array = acc.evaluate(emit)?;
            out.extend(
                array
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .iter()
                    .map(|v| v.map(f64::to_bits)),
            );
        }
        Ok(out)
    }

    #[test]
    fn groups_accumulator_matches_row_accumulator() -> Result<()> {
        for stats_type in [StatsType::Sample, StatsType::Population] {
            let adapter = || {
                GroupsAccumulatorAdapter::new(move || {
                    Ok(Box::new(CovarianceAccumulator::try_new(stats_type)?))
                })
            };
            let expected = run(&mut adapter(), &mut adapter())?;
            let actual = run(
                &mut CovarianceGroupsAccumulator::new(stats_type),
                &mut CovarianceGroupsAccumulator::new(stats_type),
            )?;
            assert_eq!(actual, expected, "{stats_type:?}");
        }
        Ok(())
    }
}
