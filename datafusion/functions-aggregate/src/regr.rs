// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Defines physical expressions that can evaluated at runtime during query execution

use arrow::array::{Array, BooleanArray, Float64Array, UInt64Array};
use arrow::buffer::{BooleanBuffer, NullBuffer};
use arrow::datatypes::FieldRef;
use arrow::{array::ArrayRef, datatypes::DataType, datatypes::Field};
use datafusion_common::cast::{as_float64_array, as_uint64_array};
use datafusion_common::{HashMap, Result, ScalarValue, not_impl_err};
use datafusion_doc::aggregate_doc_sections::DOC_SECTION_STATISTICAL;
use datafusion_expr::DistinctHandling;
use datafusion_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion_expr::utils::format_state_name;
use datafusion_expr::{
    Accumulator, AggregateUDFImpl, Documentation, Signature, Volatility,
};
use datafusion_expr::{EmitTo, GroupsAccumulator};
use std::fmt::Debug;
use std::hash::Hash;
use std::mem::size_of_val;
use std::sync::{Arc, LazyLock};

macro_rules! make_regr_udaf_expr_and_func {
    ($EXPR_FN:ident, $AGGREGATE_UDF_FN:ident, $REGR_TYPE:expr) => {
        make_udaf_expr!($EXPR_FN, expr_y expr_x, concat!("Compute a linear regression of type [", stringify!($REGR_TYPE), "]"), $AGGREGATE_UDF_FN);
        create_func!($EXPR_FN, $AGGREGATE_UDF_FN, Regr::new($REGR_TYPE, stringify!($EXPR_FN)));
    }
}

make_regr_udaf_expr_and_func!(regr_slope, regr_slope_udaf, RegrType::Slope);
make_regr_udaf_expr_and_func!(regr_intercept, regr_intercept_udaf, RegrType::Intercept);
make_regr_udaf_expr_and_func!(regr_count, regr_count_udaf, RegrType::Count);
make_regr_udaf_expr_and_func!(regr_r2, regr_r2_udaf, RegrType::R2);
make_regr_udaf_expr_and_func!(regr_avgx, regr_avgx_udaf, RegrType::AvgX);
make_regr_udaf_expr_and_func!(regr_avgy, regr_avgy_udaf, RegrType::AvgY);
make_regr_udaf_expr_and_func!(regr_sxx, regr_sxx_udaf, RegrType::SXX);
make_regr_udaf_expr_and_func!(regr_syy, regr_syy_udaf, RegrType::SYY);
make_regr_udaf_expr_and_func!(regr_sxy, regr_sxy_udaf, RegrType::SXY);

#[derive(PartialEq, Eq, Hash, Debug)]
pub struct Regr {
    signature: Signature,
    regr_type: RegrType,
    func_name: &'static str,
}

impl Regr {
    pub fn new(regr_type: RegrType, func_name: &'static str) -> Self {
        Self {
            signature: Signature::exact(
                vec![DataType::Float64, DataType::Float64],
                Volatility::Immutable,
            ),
            regr_type,
            func_name,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub enum RegrType {
    /// Variant for `regr_slope` aggregate expression
    /// Returns the slope of the linear regression line for non-null pairs in aggregate columns.
    /// Given input column Y and X: `regr_slope(Y, X)` returns the slope (k in Y = k*X + b) using minimal
    /// RSS (Residual Sum of Squares) fitting.
    Slope,
    /// Variant for `regr_intercept` aggregate expression
    /// Returns the intercept of the linear regression line for non-null pairs in aggregate columns.
    /// Given input column Y and X: `regr_intercept(Y, X)` returns the intercept (b in Y = k*X + b) using minimal
    /// RSS fitting.
    Intercept,
    /// Variant for `regr_count` aggregate expression
    /// Returns the number of input rows for which both expressions are not null.
    /// Given input column Y and X: `regr_count(Y, X)` returns the count of non-null pairs.
    Count,
    /// Variant for `regr_r2` aggregate expression
    /// Returns the coefficient of determination (R-squared value) of the linear regression line for non-null pairs in aggregate columns.
    /// The R-squared value represents the proportion of variance in Y that is predictable from X.
    R2,
    /// Variant for `regr_avgx` aggregate expression
    /// Returns the average of the independent variable for non-null pairs in aggregate columns.
    /// Given input column X: `regr_avgx(Y, X)` returns the average of X values.
    AvgX,
    /// Variant for `regr_avgy` aggregate expression
    /// Returns the average of the dependent variable for non-null pairs in aggregate columns.
    /// Given input column Y: `regr_avgy(Y, X)` returns the average of Y values.
    AvgY,
    /// Variant for `regr_sxx` aggregate expression
    /// Returns the sum of squares of the independent variable for non-null pairs in aggregate columns.
    /// Given input column X: `regr_sxx(Y, X)` returns the sum of squares of deviations of X from its mean.
    SXX,
    /// Variant for `regr_syy` aggregate expression
    /// Returns the sum of squares of the dependent variable for non-null pairs in aggregate columns.
    /// Given input column Y: `regr_syy(Y, X)` returns the sum of squares of deviations of Y from its mean.
    SYY,
    /// Variant for `regr_sxy` aggregate expression
    /// Returns the sum of products of pairs of numbers for non-null pairs in aggregate columns.
    /// Given input column Y and X: `regr_sxy(Y, X)` returns the sum of products of the deviations of Y and X from their respective means.
    SXY,
}

impl RegrType {
    /// return the documentation for the `RegrType`
    fn documentation(&self) -> Option<&Documentation> {
        get_regr_docs().get(self)
    }
}

static DOCUMENTATION: LazyLock<HashMap<RegrType, Documentation>> = LazyLock::new(|| {
    let mut hash_map = HashMap::new();
    hash_map.insert(
            RegrType::Slope,
            Documentation::builder(
                DOC_SECTION_STATISTICAL,
                    "Returns the slope of the linear regression line for non-null pairs in aggregate columns. \
                    Given input column Y and X: regr_slope(Y, X) returns the slope (k in Y = k*X + b) using minimal RSS fitting.",

                "regr_slope(expression_y, expression_x)")
                .with_sql_example(
                    r#"```sql
create table weekly_performance(day int, user_signups int) as values (1,60), (2,65), (3, 70), (4,75), (5,80);
select * from weekly_performance;
+-----+--------------+
| day | user_signups |
+-----+--------------+
| 1   | 60           |
| 2   | 65           |
| 3   | 70           |
| 4   | 75           |
| 5   | 80           |
+-----+--------------+

SELECT regr_slope(user_signups, day) AS slope FROM weekly_performance;
+--------+
| slope  |
+--------+
| 5.0    |
+--------+
```
"#
                )
                .with_standard_argument("expression_y", Some("Dependent variable"))
                .with_standard_argument("expression_x", Some("Independent variable"))
                .build()
        );

    hash_map.insert(
            RegrType::Intercept,
            Documentation::builder(
                DOC_SECTION_STATISTICAL,
                    "Computes the y-intercept of the linear regression line. For the equation (y = kx + b), \
                    this function returns b.",

                "regr_intercept(expression_y, expression_x)")
                .with_sql_example(
                    r#"```sql
create table weekly_performance(week int, productivity_score int) as values (1,60), (2,65), (3, 70), (4,75), (5,80);
select * from weekly_performance;
+------+---------------------+
| week | productivity_score  |
+------+---------------------+
| 1    | 60                  |
| 2    | 65                  |
| 3    | 70                  |
| 4    | 75                  |
| 5    | 80                  |
+------+---------------------+

SELECT regr_intercept(productivity_score, week) AS intercept FROM weekly_performance;
+-----------+
| intercept |
+-----------+
| 55        |
+-----------+
```
"#
                )
                .with_standard_argument("expression_y", Some("Dependent variable"))
                .with_standard_argument("expression_x", Some("Independent variable"))
                .build()
        );

    hash_map.insert(
        RegrType::Count,
        Documentation::builder(
            DOC_SECTION_STATISTICAL,
            "Counts the number of non-null paired data points.",
            "regr_count(expression_y, expression_x)",
        )
        .with_sql_example(
            r#"```sql
create table daily_metrics(day int, user_signups int) as values (1,100), (2,120), (3, NULL), (4,110), (5,NULL);
select * from daily_metrics;
+-----+---------------+
| day | user_signups  |
+-----+---------------+
| 1   | 100           |
| 2   | 120           |
| 3   | NULL          |
| 4   | 110           |
| 5   | NULL          |
+-----+---------------+

SELECT regr_count(user_signups, day) AS valid_pairs FROM daily_metrics;
+-------------+
| valid_pairs |
+-------------+
| 3           |
+-------------+
```
"#
        )
        .with_standard_argument("expression_y", Some("Dependent variable"))
        .with_standard_argument("expression_x", Some("Independent variable"))
        .build(),
    );

    hash_map.insert(
            RegrType::R2,
            Documentation::builder(
                DOC_SECTION_STATISTICAL,
                    "Computes the square of the correlation coefficient between the independent and dependent variables.",

                "regr_r2(expression_y, expression_x)")
                .with_sql_example(
                    r#"```sql
create table weekly_performance(day int ,user_signups int) as values (1,60), (2,65), (3, 70), (4,75), (5,80);
select * from weekly_performance;
+-----+--------------+
| day | user_signups |
+-----+--------------+
| 1   | 60           |
| 2   | 65           |
| 3   | 70           |
| 4   | 75           |
| 5   | 80           |
+-----+--------------+

SELECT regr_r2(user_signups, day) AS r_squared FROM weekly_performance;
+-----------+
| r_squared |
+-----------+
| 1.0       |
+-----------+
```
"#
                )
                .with_standard_argument("expression_y", Some("Dependent variable"))
                .with_standard_argument("expression_x", Some("Independent variable"))
                .build()
        );

    hash_map.insert(
            RegrType::AvgX,
            Documentation::builder(
                DOC_SECTION_STATISTICAL,
                    "Computes the average of the independent variable (input) expression_x for the non-null paired data points.",

                "regr_avgx(expression_y, expression_x)")
                .with_sql_example(
                    r#"```sql
create table daily_sales(day int, total_sales int) as values (1,100), (2,150), (3,200), (4,NULL), (5,250);
select * from daily_sales;
+-----+-------------+
| day | total_sales |
+-----+-------------+
| 1   | 100         |
| 2   | 150         |
| 3   | 200         |
| 4   | NULL        |
| 5   | 250         |
+-----+-------------+

SELECT regr_avgx(total_sales, day) AS avg_day FROM daily_sales;
+----------+
| avg_day  |
+----------+
|   2.75   |
+----------+
```
"#
                )
                .with_standard_argument("expression_y", Some("Dependent variable"))
                .with_standard_argument("expression_x", Some("Independent variable"))
                .build()
        );

    hash_map.insert(
            RegrType::AvgY,
            Documentation::builder(
                DOC_SECTION_STATISTICAL,
                    "Computes the average of the dependent variable (output) expression_y for the non-null paired data points.",

                "regr_avgy(expression_y, expression_x)")
                .with_sql_example(
                    r#"```sql
create table daily_temperature(day int, temperature int) as values (1,30), (2,32), (3, NULL), (4,35), (5,36);
select * from daily_temperature;
+-----+-------------+
| day | temperature |
+-----+-------------+
| 1   | 30          |
| 2   | 32          |
| 3   | NULL        |
| 4   | 35          |
| 5   | 36          |
+-----+-------------+

-- temperature as Dependent Variable(Y), day as Independent Variable(X)
SELECT regr_avgy(temperature, day) AS avg_temperature FROM daily_temperature;
+-----------------+
| avg_temperature |
+-----------------+
| 33.25           |
+-----------------+
```
"#
                )
                .with_standard_argument("expression_y", Some("Dependent variable"))
                .with_standard_argument("expression_x", Some("Independent variable"))
                .build()
        );

    hash_map.insert(
        RegrType::SXX,
        Documentation::builder(
            DOC_SECTION_STATISTICAL,
            "Computes the sum of squares of the independent variable.",
            "regr_sxx(expression_y, expression_x)",
        )
        .with_sql_example(
            r#"```sql
create table study_hours(student_id int, hours int, test_score int) as values (1,2,55), (2,4,65), (3,6,75), (4,8,85), (5,10,95);
select * from study_hours;
+------------+-------+------------+
| student_id | hours | test_score |
+------------+-------+------------+
| 1          | 2     | 55         |
| 2          | 4     | 65         |
| 3          | 6     | 75         |
| 4          | 8     | 85         |
| 5          | 10    | 95         |
+------------+-------+------------+

SELECT regr_sxx(test_score, hours) AS sxx FROM study_hours;
+------+
| sxx  |
+------+
| 40.0 |
+------+
```
"#
        )
        .with_standard_argument("expression_y", Some("Dependent variable"))
        .with_standard_argument("expression_x", Some("Independent variable"))
        .build(),
    );

    hash_map.insert(
        RegrType::SYY,
        Documentation::builder(
            DOC_SECTION_STATISTICAL,
            "Computes the sum of squares of the dependent variable.",
            "regr_syy(expression_y, expression_x)",
        )
        .with_sql_example(
            r#"```sql
create table employee_productivity(week int, productivity_score int) as values (1,60), (2,65), (3,70);
select * from employee_productivity;
+------+--------------------+
| week | productivity_score |
+------+--------------------+
| 1    | 60                 |
| 2    | 65                 |
| 3    | 70                 |
+------+--------------------+

SELECT regr_syy(productivity_score, week) AS sum_squares_y FROM employee_productivity;
+---------------+
| sum_squares_y |
+---------------+
|    50.0       |
+---------------+
```
"#
        )
        .with_standard_argument("expression_y", Some("Dependent variable"))
        .with_standard_argument("expression_x", Some("Independent variable"))
        .build(),
    );

    hash_map.insert(
        RegrType::SXY,
        Documentation::builder(
            DOC_SECTION_STATISTICAL,
            "Computes the sum of products of paired data points.",
            "regr_sxy(expression_y, expression_x)",
        )
        .with_sql_example(
            r#"```sql
create table employee_productivity(week int, productivity_score int) as values(1,60), (2,65), (3,70);
select * from employee_productivity;
+------+--------------------+
| week | productivity_score |
+------+--------------------+
| 1    | 60                 |
| 2    | 65                 |
| 3    | 70                 |
+------+--------------------+

SELECT regr_sxy(productivity_score, week) AS sum_product_deviations FROM employee_productivity;
+------------------------+
| sum_product_deviations |
+------------------------+
|       10.0             |
+------------------------+
```
"#
        )
        .with_standard_argument("expression_y", Some("Dependent variable"))
        .with_standard_argument("expression_x", Some("Independent variable"))
        .build(),
    );
    hash_map
});
fn get_regr_docs() -> &'static HashMap<RegrType, Documentation> {
    &DOCUMENTATION
}

impl AggregateUDFImpl for Regr {
    fn name(&self) -> &str {
        self.func_name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        if self.regr_type == RegrType::Count {
            Ok(DataType::UInt64)
        } else {
            Ok(DataType::Float64)
        }
    }

    fn default_value(&self, _data_type: &DataType) -> Result<ScalarValue> {
        if self.regr_type == RegrType::Count {
            Ok(ScalarValue::UInt64(Some(0)))
        } else {
            Ok(ScalarValue::Float64(None))
        }
    }

    fn return_field(&self, arg_fields: &[FieldRef]) -> Result<FieldRef> {
        let arg_types: Vec<_> =
            arg_fields.iter().map(|f| f.data_type()).cloned().collect();
        Ok(Arc::new(Field::new(
            self.name(),
            self.return_type(&arg_types)?,
            self.regr_type != RegrType::Count,
        )))
    }

    // Keep the deprecated method consistent with `return_field` until it is removed
    fn is_nullable(&self) -> bool {
        self.regr_type != RegrType::Count
    }

    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        if acc_args.is_distinct {
            return not_impl_err!(
                "{}(DISTINCT) aggregations are not available",
                self.func_name.to_uppercase()
            );
        }
        Ok(Box::new(RegrAccumulator::try_new(&self.regr_type)?))
    }

    fn groups_accumulator_supported(&self, args: AccumulatorArgs) -> bool {
        !args.is_distinct
    }

    fn create_groups_accumulator(
        &self,
        args: AccumulatorArgs,
    ) -> Result<Box<dyn GroupsAccumulator>> {
        if args.is_distinct {
            return not_impl_err!(
                "{}(DISTINCT) aggregations are not available",
                self.func_name.to_uppercase()
            );
        }
        Ok(Box::new(RegrGroupsAccumulator::new(self.regr_type.clone())))
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        Ok(vec![
            Field::new(
                format_state_name(args.name, "count"),
                DataType::UInt64,
                true,
            ),
            Field::new(
                format_state_name(args.name, "mean_x"),
                DataType::Float64,
                true,
            ),
            Field::new(
                format_state_name(args.name, "mean_y"),
                DataType::Float64,
                true,
            ),
            Field::new(
                format_state_name(args.name, "m2_x"),
                DataType::Float64,
                true,
            ),
            Field::new(
                format_state_name(args.name, "m2_y"),
                DataType::Float64,
                true,
            ),
            Field::new(
                format_state_name(args.name, "algo_const"),
                DataType::Float64,
                true,
            ),
        ]
        .into_iter()
        .map(Arc::new)
        .collect())
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.regr_type.documentation()
    }

    fn distinct_handling(&self) -> DistinctHandling {
        // The accumulator rejects `DISTINCT` with `not_impl_err!`. It only
        // works when the planner removes it first (`SingleDistinctToGroupBy`
        // for a single distinct field, e.g. `regr_slope(DISTINCT x, x)`).
        DistinctHandling::Unsupported
    }
}

/// `RegrAccumulator` is used to compute linear regression aggregate functions
/// by maintaining statistics needed to compute them in an online fashion.
///
/// This struct uses Welford's online algorithm for calculating variance and covariance:
/// <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm>
///
/// Given the statistics, the following aggregate functions can be calculated:
///
/// - `regr_slope(y, x)`: Slope of the linear regression line, calculated as:
///   cov_pop(x, y) / var_pop(x).
///   It represents the expected change in Y for a one-unit change in X.
///
/// - `regr_intercept(y, x)`: Intercept of the linear regression line, calculated as:
///   mean_y - (regr_slope(y, x) * mean_x).
///   It represents the expected value of Y when X is 0.
///
/// - `regr_count(y, x)`: Count of the non-null(both x and y) input rows.
///
/// - `regr_r2(y, x)`: R-squared value (coefficient of determination), calculated as:
///   (cov_pop(x, y) ^ 2) / (var_pop(x) * var_pop(y)).
///   It provides a measure of how well the model's predictions match the observed data.
///
/// - `regr_avgx(y, x)`: Average of the independent variable X, calculated as: mean_x.
///
/// - `regr_avgy(y, x)`: Average of the dependent variable Y, calculated as: mean_y.
///
/// - `regr_sxx(y, x)`: Sum of squares of the independent variable X, calculated as:
///   m2_x.
///
/// - `regr_syy(y, x)`: Sum of squares of the dependent variable Y, calculated as:
///   m2_y.
///
/// - `regr_sxy(y, x)`: Sum of products of paired values, calculated as:
///   algo_const.
///
/// Here's how the statistics maintained in this struct are calculated:
/// - `cov_pop(x, y)`: algo_const / count.
/// - `var_pop(x)`: m2_x / count.
/// - `var_pop(y)`: m2_y / count.
#[derive(Debug)]
pub struct RegrAccumulator {
    state: RegrState,
    regr_type: RegrType,
}

/// The statistics of one group, shared by [`RegrAccumulator`] and
/// [`RegrGroupsAccumulator`].
#[derive(Debug, Default, Clone, Copy)]
struct RegrState {
    count: u64,
    mean_x: f64,
    mean_y: f64,
    m2_x: f64,
    m2_y: f64,
    algo_const: f64,
}

impl RegrState {
    /// Adds one row where both `x` and `y` are not NULL.
    #[inline]
    fn update(&mut self, value_y: f64, value_x: f64) {
        // Update states for regr_slope(y,x) [using cov_pop(x,y)/var_pop(x)]
        self.count += 1;
        let delta_x = value_x - self.mean_x;
        let delta_y = value_y - self.mean_y;
        self.mean_x += delta_x / self.count as f64;
        self.mean_y += delta_y / self.count as f64;
        let delta_x_2 = value_x - self.mean_x;
        let delta_y_2 = value_y - self.mean_y;
        self.m2_x += delta_x * delta_x_2;
        self.m2_y += delta_y * delta_y_2;
        self.algo_const += delta_x * (value_y - self.mean_y);
    }

    /// Merges the statistics of another partial state into this one.
    #[inline]
    fn merge(&mut self, other: &RegrState) {
        if other.count == 0_u64 {
            return;
        }
        let (count_a, mean_x_a, mean_y_a, m2_x_a, m2_y_a, algo_const_a) = (
            self.count,
            self.mean_x,
            self.mean_y,
            self.m2_x,
            self.m2_y,
            self.algo_const,
        );
        let RegrState {
            count: count_b,
            mean_x: mean_x_b,
            mean_y: mean_y_b,
            m2_x: m2_x_b,
            m2_y: m2_y_b,
            algo_const: algo_const_b,
        } = *other;

        // Assuming two different batches of input have calculated the states:
        // batch A of Y, X -> {count_a, mean_x_a, mean_y_a, m2_x_a, algo_const_a}
        // batch B of Y, X -> {count_b, mean_x_b, mean_y_b, m2_x_b, algo_const_b}
        // The merged states from A and B are {count_ab, mean_x_ab, mean_y_ab, m2_x_ab,
        // algo_const_ab}
        //
        // Reference for the algorithm to merge states:
        // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        let count_ab = count_a + count_b;
        let (count_a, count_b) = (count_a as f64, count_b as f64);
        let d_x = mean_x_b - mean_x_a;
        let d_y = mean_y_b - mean_y_a;
        let mean_x_ab = mean_x_a + d_x * count_b / count_ab as f64;
        let mean_y_ab = mean_y_a + d_y * count_b / count_ab as f64;
        let m2_x_ab = m2_x_a + m2_x_b + d_x * d_x * count_a * count_b / count_ab as f64;
        let m2_y_ab = m2_y_a + m2_y_b + d_y * d_y * count_a * count_b / count_ab as f64;
        let algo_const_ab =
            algo_const_a + algo_const_b + d_x * d_y * count_a * count_b / count_ab as f64;

        self.count = count_ab;
        self.mean_x = mean_x_ab;
        self.mean_y = mean_y_ab;
        self.m2_x = m2_x_ab;
        self.m2_y = m2_y_ab;
        self.algo_const = algo_const_ab;
    }

    /// The result for every type but [`RegrType::Count`], or `None` for NULL.
    fn evaluate(&self, regr_type: &RegrType) -> Option<f64> {
        let cov_pop_x_y = self.algo_const / self.count as f64;
        let var_pop_x = self.m2_x / self.count as f64;
        let var_pop_y = self.m2_y / self.count as f64;

        let nullif_or_stat = |cond: bool, stat: f64| (!cond).then_some(stat);

        match regr_type {
            RegrType::Slope => {
                // Only 0/1 point or slope is infinite
                let nullif_cond = self.count <= 1 || var_pop_x == 0.0;
                nullif_or_stat(nullif_cond, cov_pop_x_y / var_pop_x)
            }
            RegrType::Intercept => {
                let slope = cov_pop_x_y / var_pop_x;
                // Only 0/1 point or slope is infinite
                let nullif_cond = self.count <= 1 || var_pop_x == 0.0;
                nullif_or_stat(nullif_cond, self.mean_y - slope * self.mean_x)
            }
            RegrType::Count => Some(self.count as f64),
            RegrType::R2 => {
                // Only 0/1 point or all x(or y) is the same
                let nullif_cond = self.count <= 1 || var_pop_x == 0.0 || var_pop_y == 0.0;
                nullif_or_stat(
                    nullif_cond,
                    (cov_pop_x_y * cov_pop_x_y) / (var_pop_x * var_pop_y),
                )
            }
            RegrType::AvgX => nullif_or_stat(self.count < 1, self.mean_x),
            RegrType::AvgY => nullif_or_stat(self.count < 1, self.mean_y),
            RegrType::SXX => nullif_or_stat(self.count < 1, self.m2_x),
            RegrType::SYY => nullif_or_stat(self.count < 1, self.m2_y),
            RegrType::SXY => nullif_or_stat(self.count < 1, self.algo_const),
        }
    }
}

/// Reads row `i` of the state arrays written by `state`.
struct RegrStateArrays<'a> {
    count: &'a UInt64Array,
    mean_x: &'a Float64Array,
    mean_y: &'a Float64Array,
    m2_x: &'a Float64Array,
    m2_y: &'a Float64Array,
    algo_const: &'a Float64Array,
}

impl<'a> RegrStateArrays<'a> {
    fn try_new(states: &'a [ArrayRef]) -> Result<Self> {
        Ok(Self {
            count: as_uint64_array(&states[0])?,
            mean_x: as_float64_array(&states[1])?,
            mean_y: as_float64_array(&states[2])?,
            m2_x: as_float64_array(&states[3])?,
            m2_y: as_float64_array(&states[4])?,
            algo_const: as_float64_array(&states[5])?,
        })
    }

    fn len(&self) -> usize {
        self.count.len()
    }

    #[inline]
    fn get(&self, i: usize) -> RegrState {
        RegrState {
            count: self.count.value(i),
            mean_x: self.mean_x.value(i),
            mean_y: self.mean_y.value(i),
            m2_x: self.m2_x.value(i),
            m2_y: self.m2_y.value(i),
            algo_const: self.algo_const.value(i),
        }
    }
}

impl RegrAccumulator {
    /// Creates a new `RegrAccumulator`
    pub fn try_new(regr_type: &RegrType) -> Result<Self> {
        Ok(Self {
            state: RegrState::default(),
            regr_type: regr_type.clone(),
        })
    }
}

impl Accumulator for RegrAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let s = &self.state;
        Ok(vec![
            ScalarValue::from(s.count),
            ScalarValue::from(s.mean_x),
            ScalarValue::from(s.mean_y),
            ScalarValue::from(s.m2_x),
            ScalarValue::from(s.m2_y),
            ScalarValue::from(s.algo_const),
        ])
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        // regr_slope(Y, X) calculates k in y = k*x + b
        let values_y = as_float64_array(&values[0])?;
        let values_x = as_float64_array(&values[1])?;

        for (value_y, value_x) in values_y.iter().zip(values_x) {
            // skip either x or y is NULL
            let (Some(value_y), Some(value_x)) = (value_y, value_x) else {
                continue;
            };
            self.state.update(value_y, value_x);
        }

        Ok(())
    }

    fn supports_retract_batch(&self) -> bool {
        true
    }

    fn retract_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let values_y = as_float64_array(&values[0])?;
        let values_x = as_float64_array(&values[1])?;
        let s = &mut self.state;

        for (value_y, value_x) in values_y.iter().zip(values_x) {
            // skip either x or y is NULL
            let (Some(value_y), Some(value_x)) = (value_y, value_x) else {
                continue;
            };

            // Update states for regr_slope(y,x) [using cov_pop(x,y)/var_pop(x)]
            if s.count > 1 {
                s.count -= 1;
                let delta_x = value_x - s.mean_x;
                let delta_y = value_y - s.mean_y;
                s.mean_x -= delta_x / s.count as f64;
                s.mean_y -= delta_y / s.count as f64;
                let delta_x_2 = value_x - s.mean_x;
                let delta_y_2 = value_y - s.mean_y;
                s.m2_x -= delta_x * delta_x_2;
                s.m2_y -= delta_y * delta_y_2;
                s.algo_const -= delta_x * (value_y - s.mean_y);
            } else {
                *s = RegrState::default();
            }
        }

        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let states = RegrStateArrays::try_new(states)?;
        for i in 0..states.len() {
            self.state.merge(&states.get(i));
        }
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        Ok(match self.regr_type {
            RegrType::Count => ScalarValue::UInt64(Some(self.state.count)),
            _ => ScalarValue::Float64(self.state.evaluate(&self.regr_type)),
        })
    }

    fn size(&self) -> usize {
        size_of_val(self)
    }
}

/// [`GroupsAccumulator`] for the `regr_*` functions: one [`RegrState`] per
/// group, updated with the same arithmetic as [`RegrAccumulator`].
#[derive(Debug)]
struct RegrGroupsAccumulator {
    states: Vec<RegrState>,
    regr_type: RegrType,
}

impl RegrGroupsAccumulator {
    fn new(regr_type: RegrType) -> Self {
        Self {
            states: Vec::new(),
            regr_type,
        }
    }
}

impl GroupsAccumulator for RegrGroupsAccumulator {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        self.states.resize(total_num_groups, RegrState::default());
        let values_y = as_float64_array(&values[0])?;
        let values_x = as_float64_array(&values[1])?;
        let valid = valid_rows(values_y, values_x, opt_filter);
        let (ys, xs) = (values_y.values(), values_x.values());
        match valid {
            None => {
                for (row, &group) in group_indices.iter().enumerate() {
                    self.states[group].update(ys[row], xs[row]);
                }
            }
            Some(valid) => {
                for row in valid.set_indices() {
                    self.states[group_indices[row]].update(ys[row], xs[row]);
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
        self.states.resize(total_num_groups, RegrState::default());
        let states = RegrStateArrays::try_new(values)?;
        for (row, &group) in group_indices.iter().enumerate() {
            self.states[group].merge(&states.get(row));
        }
        Ok(())
    }

    fn convert_to_state(
        &self,
        values: &[ArrayRef],
        opt_filter: Option<&BooleanArray>,
    ) -> Result<Vec<ArrayRef>> {
        let values_y = as_float64_array(&values[0])?;
        let values_x = as_float64_array(&values[1])?;
        let valid = valid_rows(values_y, values_x, opt_filter);
        let (ys, xs) = (values_y.values(), values_x.values());
        // Each valid row is a group with that one row; others stay empty,
        // which `merge` skips.
        let states: Vec<RegrState> = (0..values_y.len())
            .map(|row| {
                let mut state = RegrState::default();
                if valid.as_ref().is_none_or(|v| v.value(row)) {
                    state.update(ys[row], xs[row]);
                }
                state
            })
            .collect();
        Ok(states_to_arrays(&states))
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        let states = emit_to.take_needed(&mut self.states);
        Ok(match self.regr_type {
            RegrType::Count => Arc::new(UInt64Array::from_iter_values(
                states.iter().map(|s| s.count),
            )),
            _ => Arc::new(Float64Array::from_iter(
                states.iter().map(|s| s.evaluate(&self.regr_type)),
            )),
        })
    }

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        let states = emit_to.take_needed(&mut self.states);
        Ok(states_to_arrays(&states))
    }

    fn size(&self) -> usize {
        self.states.capacity() * size_of::<RegrState>()
    }
}

/// The state columns, in the order of `Regr::state_fields`.
fn states_to_arrays(states: &[RegrState]) -> Vec<ArrayRef> {
    let column = |f: fn(&RegrState) -> f64| -> ArrayRef {
        Arc::new(Float64Array::from_iter_values(states.iter().map(f)))
    };
    vec![
        Arc::new(UInt64Array::from_iter_values(
            states.iter().map(|s| s.count),
        )),
        column(|s| s.mean_x),
        column(|s| s.mean_y),
        column(|s| s.m2_x),
        column(|s| s.m2_y),
        column(|s| s.algo_const),
    ]
}

/// Rows where `y`, `x` and the filter are all valid and the filter is true,
/// or `None` when every row is.
fn valid_rows(
    values_y: &Float64Array,
    values_x: &Float64Array,
    opt_filter: Option<&BooleanArray>,
) -> Option<BooleanBuffer> {
    let valid = NullBuffer::union(values_y.nulls(), values_x.nulls());
    match opt_filter {
        Some(filter) => {
            let filter = match filter.nulls() {
                Some(nulls) => filter.values() & nulls.inner(),
                None => filter.values().clone(),
            };
            Some(match valid {
                Some(valid) => valid.inner() & &filter,
                None => filter,
            })
        }
        None => valid.map(|v| v.inner().clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (group, y, x, filter)
    type Row = (usize, Option<f64>, Option<f64>, Option<bool>);

    const ALL_TYPES: [RegrType; 9] = [
        RegrType::Slope,
        RegrType::Intercept,
        RegrType::Count,
        RegrType::R2,
        RegrType::AvgX,
        RegrType::AvgY,
        RegrType::SXX,
        RegrType::SYY,
        RegrType::SXY,
    ];

    /// Rows of (group, y, x, filter) with NULLs in every column, a group with
    /// one row, a group with constant x, and a group with only NULL rows.
    fn rows() -> Vec<Row> {
        let mut rows = vec![];
        for i in 0..200 {
            let group = i % 7;
            let y = (i % 11 != 0).then_some((i * 37 % 101) as f64 / 3.0);
            let x = (i % 13 != 0).then_some(if group == 3 {
                2.0
            } else {
                (i * 17 % 89) as f64
            });
            let filter = (i % 17 != 0).then_some(i % 5 != 0);
            rows.push((group, y, x, filter));
        }
        rows.push((7, Some(1.0), Some(2.0), Some(true)));
        rows.push((8, None, Some(2.0), Some(true)));
        rows
    }

    /// Row accumulator per group, used as the reference.
    fn expected(
        regr_type: &RegrType,
        rows: &[Row],
        num_groups: usize,
    ) -> Vec<ScalarValue> {
        (0..num_groups)
            .map(|g| {
                let mut acc = RegrAccumulator::try_new(regr_type).unwrap();
                let (ys, xs): (Vec<_>, Vec<_>) = rows
                    .iter()
                    .filter(|r| r.0 == g && r.3 == Some(true))
                    .map(|r| (r.1, r.2))
                    .unzip();
                let values: Vec<ArrayRef> = vec![
                    Arc::new(Float64Array::from(ys)),
                    Arc::new(Float64Array::from(xs)),
                ];
                acc.update_batch(&values).unwrap();
                acc.evaluate().unwrap()
            })
            .collect()
    }

    fn batch(rows: &[Row]) -> (Vec<ArrayRef>, Vec<usize>, BooleanArray) {
        let values: Vec<ArrayRef> = vec![
            Arc::new(Float64Array::from(
                rows.iter().map(|r| r.1).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                rows.iter().map(|r| r.2).collect::<Vec<_>>(),
            )),
        ];
        let groups = rows.iter().map(|r| r.0).collect();
        let filter = BooleanArray::from(rows.iter().map(|r| r.3).collect::<Vec<_>>());
        (values, groups, filter)
    }

    fn to_scalars(array: &ArrayRef) -> Vec<ScalarValue> {
        (0..array.len())
            .map(|i| ScalarValue::try_from_array(array, i).unwrap())
            .collect()
    }

    #[test]
    fn groups_accumulator_matches_row_accumulator() -> Result<()> {
        let rows = rows();
        let num_groups = 9;
        let (half_a, half_b) = rows.split_at(rows.len() / 2);
        for regr_type in ALL_TYPES {
            let expected = expected(&regr_type, &rows, num_groups);

            // update_batch in two batches, then evaluate everything.
            let mut acc = RegrGroupsAccumulator::new(regr_type.clone());
            for half in [half_a, half_b] {
                let (values, groups, filter) = batch(half);
                acc.update_batch(&values, &groups, Some(&filter), num_groups)?;
            }
            assert_eq!(
                to_scalars(&acc.evaluate(EmitTo::All)?),
                expected,
                "{regr_type:?}"
            );

            // Partial states from each half merged by a final accumulator,
            // one half going through convert_to_state.
            let mut partial = RegrGroupsAccumulator::new(regr_type.clone());
            let (values, groups, filter) = batch(half_a);
            partial.update_batch(&values, &groups, Some(&filter), num_groups)?;
            let mut final_acc = RegrGroupsAccumulator::new(regr_type.clone());
            final_acc.merge_batch(
                &partial.state(EmitTo::All)?,
                &(0..num_groups).collect::<Vec<_>>(),
                num_groups,
            )?;
            let (values, groups, filter) = batch(half_b);
            let converted = partial.convert_to_state(&values, Some(&filter))?;
            final_acc.merge_batch(&converted, &groups, num_groups)?;

            // EmitTo::First, then the rest.
            let mut got = to_scalars(&final_acc.evaluate(EmitTo::First(4))?);
            got.extend(to_scalars(&final_acc.evaluate(EmitTo::All)?));
            assert_eq!(got.len(), num_groups);
            for (g, (got, expected)) in got.iter().zip(&expected).enumerate() {
                // Merging reorders the arithmetic, so allow rounding error.
                match (got, expected) {
                    (ScalarValue::Float64(Some(a)), ScalarValue::Float64(Some(b))) => {
                        assert!(
                            (a - b).abs() <= 1e-9 * b.abs().max(1.0),
                            "{regr_type:?} group {g}: {a} vs {b}"
                        )
                    }
                    _ => assert_eq!(got, expected, "{regr_type:?} group {g}"),
                }
            }
        }
        Ok(())
    }
}
