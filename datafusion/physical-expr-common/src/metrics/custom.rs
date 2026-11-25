// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Custom metric value type.

use std::{any::Any, fmt::Debug, fmt::Display, sync::Arc};

/// A trait for implementing custom metric values.
///
/// This trait enables defining application- or operator-specific metric types
/// that can be aggregated and displayed alongside standard metrics. These
/// custom metrics integrate with [`MetricValue::Custom`] and support
/// aggregation logic, introspection, and optional numeric representation.
///
/// # Requirements
/// Implementations of `CustomMetricValue` must satisfy the following:
///
/// 1. [`Self::aggregate`]: Defines how two metric values are combined
/// 2. [`Self::new_empty`]: Returns a new, zero-value instance for accumulation
/// 3. [`Self::as_any`]: Enables dynamic downcasting for type-specific operations
/// 4. [`Self::as_usize`]: Optionally maps the value to a `usize` (for sorting, display, etc.)
/// 5. [`Self::is_eq`]: Implements comparison between two values, this isn't reusing the std
///    PartialEq trait because this trait is used dynamically in the context of
///    [`MetricValue::Custom`]
///
/// Optionally, [`Self::aggregation`] describes how [`Self::aggregate`]
/// combines the numeric values, for code that only has those.
///
/// # Examples
/// ```
/// # use std::sync::Arc;
/// # use std::fmt::{Debug, Display};
/// # use std::any::Any;
/// # use std::sync::atomic::{AtomicUsize, Ordering};
///
/// # use datafusion_physical_expr_common::metrics::CustomMetricValue;
///
/// #[derive(Debug, Default)]
/// struct MyCounter {
///     count: AtomicUsize,
/// }
///
/// impl Display for MyCounter {
///     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
///         write!(f, "count: {}", self.count.load(Ordering::Relaxed))
///     }
/// }
///
/// impl CustomMetricValue for MyCounter {
///     fn new_empty(&self) -> Arc<dyn CustomMetricValue> {
///         Arc::new(Self::default())
///     }
///
///     fn aggregate(&self, other: Arc<dyn CustomMetricValue>) {
///         let other = other.as_any().downcast_ref::<Self>().unwrap();
///         self.count
///             .fetch_add(other.count.load(Ordering::Relaxed), Ordering::Relaxed);
///     }
///
///     fn as_any(&self) -> &dyn Any {
///         self
///     }
///
///     fn as_usize(&self) -> usize {
///         self.count.load(Ordering::Relaxed)
///     }
///
///     fn is_eq(&self, other: &Arc<dyn CustomMetricValue>) -> bool {
///         let Some(other) = other.as_any().downcast_ref::<Self>() else {
///             return false;
///         };
///
///         self.count.load(Ordering::Relaxed) == other.count.load(Ordering::Relaxed)
///     }
/// }
/// ```
///
/// [`MetricValue::Custom`]: super::MetricValue::Custom
pub trait CustomMetricValue: Display + Debug + Send + Sync {
    /// Returns a new, zero-initialized version of this metric value.
    ///
    /// This value is used during metric aggregation to accumulate results.
    fn new_empty(&self) -> Arc<dyn CustomMetricValue>;

    /// Merges another metric value into this one.
    ///
    /// The type of `other` could be of a different custom type as long as it's aggregatable into self.
    fn aggregate(&self, other: Arc<dyn CustomMetricValue + 'static>);

    /// Returns this value as a [`Any`] to support dynamic downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Optionally returns a numeric representation of the value, if meaningful.
    /// Otherwise will default to zero.
    ///
    /// This is used for sorting and summarizing metrics.
    fn as_usize(&self) -> usize {
        0
    }

    /// Compares this value with another custom value.
    fn is_eq(&self, other: &Arc<dyn CustomMetricValue>) -> bool;

    /// Describes what [`Self::aggregate`] does to [`Self::as_usize`].
    ///
    /// Code that has only the numeric value of each partition, and not the
    /// values themselves, uses this to combine them. For example, a
    /// distributed engine that sends metrics from workers to a coordinator
    /// can't call [`Self::aggregate`] there, since the coordinator doesn't
    /// have this type. The default, [`CustomMetricAggregation::Unknown`],
    /// tells it not to make up a total.
    fn aggregation(&self) -> CustomMetricAggregation {
        CustomMetricAggregation::Unknown
    }
}

/// What [`CustomMetricValue::aggregate`] does to
/// [`CustomMetricValue::as_usize`], as returned by
/// [`CustomMetricValue::aggregation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum CustomMetricAggregation {
    /// Not described, or not a function of `as_usize` alone. Don't combine
    /// the numeric values.
    #[default]
    Unknown,
    /// The aggregate's `as_usize` is the sum of the values' `as_usize`.
    Sum,
    /// The aggregate's `as_usize` is the largest of the values' `as_usize`.
    Max,
    /// The aggregate's `as_usize` is the smallest of the values' `as_usize`.
    Min,
}

impl CustomMetricAggregation {
    /// Combines the numeric values of several partitions as
    /// [`CustomMetricValue::aggregate`] would, or returns `None` when that
    /// isn't known.
    pub fn combine(self, values: impl IntoIterator<Item = usize>) -> Option<usize> {
        let values = values.into_iter();
        match self {
            Self::Unknown => None,
            Self::Sum => Some(values.fold(0, usize::saturating_add)),
            Self::Max => values.max(),
            Self::Min => values.min(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CustomMetricAggregation;

    #[test]
    fn combine_follows_the_aggregation() {
        let values = [12, 30, 5];
        assert_eq!(CustomMetricAggregation::Sum.combine(values), Some(47));
        assert_eq!(CustomMetricAggregation::Max.combine(values), Some(30));
        assert_eq!(CustomMetricAggregation::Min.combine(values), Some(5));
        assert_eq!(CustomMetricAggregation::Unknown.combine(values), None);
        assert_eq!(CustomMetricAggregation::Max.combine([]), None);
    }
}
