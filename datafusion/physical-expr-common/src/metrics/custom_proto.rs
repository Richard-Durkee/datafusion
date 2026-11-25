// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Sending [`CustomMetricValue`]s between processes.

use std::sync::Arc;

use datafusion_common::Result;
use datafusion_proto_models::ProtoDecoderRegistry;

use super::CustomMetricValue;

/// A [`CustomMetricValue`] encoded by [`CustomMetricValue::try_to_proto`].
///
/// Engines carry both fields in their own metrics message. On the receiving
/// side, [`decode_custom_metric`] rebuilds the value from them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedCustomMetric {
    /// The [`ExtensionMetricFromProto::NAME`] of the value's type.
    pub type_name: String,
    /// The value's own encoding.
    pub payload: Vec<u8>,
}

/// Builds the [`EncodedCustomMetric`] for a value of type `T`, stamping
/// `T`'s [`NAME`](ExtensionMetricFromProto::NAME) so the encoded name and the
/// registered one can't drift apart.
pub fn extension_metric<T: ExtensionMetricFromProto>(
    payload: Vec<u8>,
) -> EncodedCustomMetric {
    EncodedCustomMetric {
        type_name: T::NAME.to_string(),
        payload,
    }
}

/// Decode-side context for [`ExtensionMetricFromProto::try_from_proto`].
///
/// Empty today. It exists so that the decoder can later be given more, such
/// as the session, without changing the trait's signature.
#[derive(Debug, Default)]
#[non_exhaustive]
pub struct CustomMetricDecodeCtx {}

/// The wire name of a [`CustomMetricValue`] type, and the constructor that
/// rebuilds it.
///
/// One impl block carries both halves, together with
/// [`CustomMetricValue::try_to_proto`]:
///
/// ```ignore
/// impl CustomMetricValue for MaxMillis {
///     // ...
///     fn try_to_proto(&self) -> Result<Option<EncodedCustomMetric>> {
///         let millis = self.as_usize() as u64;
///         Ok(Some(extension_metric::<Self>(millis.to_le_bytes().to_vec())))
///     }
/// }
///
/// impl ExtensionMetricFromProto for MaxMillis {
///     const NAME: &'static str = "my-crate.MaxMillis";
///
///     fn try_from_proto(
///         payload: &[u8],
///         _ctx: &CustomMetricDecodeCtx,
///     ) -> Result<Arc<dyn CustomMetricValue>> {
///         let bytes: [u8; 8] = payload
///             .try_into()
///             .map_err(|_| internal_datafusion_err!("MaxMillis takes 8 bytes"))?;
///         Ok(Arc::new(MaxMillis::new(u64::from_le_bytes(bytes) as usize)))
///     }
/// }
///
/// config.update_extension(MaxMillis::register_proto)?;
/// ```
pub trait ExtensionMetricFromProto: CustomMetricValue + Sized + 'static {
    /// The name this metric type is written and registered under.
    ///
    /// Namespace it with the owning crate (`"my-crate.MaxMillis"`) so that a
    /// collision between two independent crates surfaces as a registration
    /// error rather than as a wrong decode.
    const NAME: &'static str;

    /// Rebuilds the value from the payload its
    /// [`CustomMetricValue::try_to_proto`] wrote.
    fn try_from_proto(
        payload: &[u8],
        ctx: &CustomMetricDecodeCtx,
    ) -> Result<Arc<dyn CustomMetricValue>>;

    /// Registers this type's decoder in `registry`, under [`Self::NAME`].
    ///
    /// Registering the same type twice is a no-op. A different type under a
    /// name that is already taken is an error. The name is scoped to custom
    /// metrics, so a plan or an expression can use the same name in the same
    /// registry.
    ///
    /// Add it to the session's registry with
    /// `config.update_extension(MaxMillis::register_proto)?`, which keeps the
    /// decoders that other libraries added.
    fn register_proto(registry: &mut ProtoDecoderRegistry) -> Result<()> {
        registry.register_decoder::<dyn CustomMetricValue, Self, CustomMetricDecoder>(
            Self::NAME,
            Self::try_from_proto,
        )
    }
}

type CustomMetricDecoder =
    fn(&[u8], &CustomMetricDecodeCtx) -> Result<Arc<dyn CustomMetricValue>>;

/// Rebuilds an encoded custom metric with the decoder registered for its
/// type, or returns `None` if no decoder is registered under its name.
///
/// Unlike the plan and expression decoders, which `datafusion-proto` calls,
/// this is public API: DataFusion doesn't serialize metrics itself, so the
/// engine that sends them calls it. On `None` the engine decides what to
/// show, for example the value's rendered text.
pub fn decode_custom_metric(
    registry: &ProtoDecoderRegistry,
    encoded: &EncodedCustomMetric,
) -> Option<Result<Arc<dyn CustomMetricValue>>> {
    let decoder = registry
        .decoder::<dyn CustomMetricValue, CustomMetricDecoder>(&encoded.type_name)?;
    Some(decoder(&encoded.payload, &CustomMetricDecodeCtx::default()))
}

/// The names of the custom metric types registered in `registry`.
pub fn custom_metric_names(
    registry: &ProtoDecoderRegistry,
) -> impl Iterator<Item = &str> {
    registry.names::<dyn CustomMetricValue>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::Any;
    use std::fmt;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use datafusion_common::internal_datafusion_err;

    /// Keeps the largest value, so a coordinator that only had the numbers
    /// would have to know that to aggregate them.
    #[derive(Debug, Default)]
    struct MaxMillis(AtomicUsize);

    impl fmt::Display for MaxMillis {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "max {} ms", self.0.load(Ordering::Relaxed))
        }
    }

    impl CustomMetricValue for MaxMillis {
        fn new_empty(&self) -> Arc<dyn CustomMetricValue> {
            Arc::new(Self::default())
        }

        fn aggregate(&self, other: Arc<dyn CustomMetricValue>) {
            self.0.fetch_max(other.as_usize(), Ordering::Relaxed);
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_usize(&self) -> usize {
            self.0.load(Ordering::Relaxed)
        }

        fn is_eq(&self, other: &Arc<dyn CustomMetricValue>) -> bool {
            other.as_usize() == self.as_usize()
        }

        fn try_to_proto(&self) -> Result<Option<EncodedCustomMetric>> {
            let millis = self.as_usize() as u64;
            Ok(Some(extension_metric::<Self>(
                millis.to_le_bytes().to_vec(),
            )))
        }
    }

    impl ExtensionMetricFromProto for MaxMillis {
        const NAME: &'static str = "test.MaxMillis";

        fn try_from_proto(
            payload: &[u8],
            _ctx: &CustomMetricDecodeCtx,
        ) -> Result<Arc<dyn CustomMetricValue>> {
            let bytes: [u8; 8] = payload
                .try_into()
                .map_err(|_| internal_datafusion_err!("MaxMillis takes 8 bytes"))?;
            Ok(Arc::new(MaxMillis(AtomicUsize::new(
                u64::from_le_bytes(bytes) as usize,
            ))))
        }
    }

    /// A different type that claims the same name.
    #[derive(Debug, Default)]
    struct OtherMaxMillis;

    impl fmt::Display for OtherMaxMillis {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("other")
        }
    }

    impl CustomMetricValue for OtherMaxMillis {
        fn new_empty(&self) -> Arc<dyn CustomMetricValue> {
            Arc::new(Self)
        }

        fn aggregate(&self, _other: Arc<dyn CustomMetricValue>) {}

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn is_eq(&self, _other: &Arc<dyn CustomMetricValue>) -> bool {
            true
        }
    }

    impl ExtensionMetricFromProto for OtherMaxMillis {
        const NAME: &'static str = MaxMillis::NAME;

        fn try_from_proto(
            _payload: &[u8],
            _ctx: &CustomMetricDecodeCtx,
        ) -> Result<Arc<dyn CustomMetricValue>> {
            Ok(Arc::new(Self))
        }
    }

    fn max_millis(value: usize) -> Arc<dyn CustomMetricValue> {
        Arc::new(MaxMillis(AtomicUsize::new(value)))
    }

    #[test]
    fn values_are_rebuilt_as_their_own_type() -> Result<()> {
        let mut registry = ProtoDecoderRegistry::new();
        MaxMillis::register_proto(&mut registry)?;

        // Each worker encodes its value, and the coordinator rebuilds them.
        let total = max_millis(0).new_empty();
        for value in [30, 12] {
            let encoded = max_millis(value).try_to_proto()?.unwrap();
            assert_eq!(encoded.type_name, MaxMillis::NAME);
            let decoded = decode_custom_metric(&registry, &encoded).unwrap()?;
            assert!(decoded.as_any().downcast_ref::<MaxMillis>().is_some());
            total.aggregate(decoded);
        }

        // The type's own aggregate keeps the maximum, and its own Display
        // keeps the unit.
        assert_eq!(total.to_string(), "max 30 ms");
        Ok(())
    }

    #[test]
    fn unregistered_types_are_not_decoded() -> Result<()> {
        let registry = ProtoDecoderRegistry::new();
        let encoded = max_millis(30).try_to_proto()?.unwrap();
        assert!(decode_custom_metric(&registry, &encoded).is_none());

        // Types that don't encode themselves keep the default.
        assert!(OtherMaxMillis.try_to_proto()?.is_none());
        Ok(())
    }

    #[test]
    fn a_corrupt_payload_is_a_decode_error() -> Result<()> {
        let mut registry = ProtoDecoderRegistry::new();
        MaxMillis::register_proto(&mut registry)?;
        let encoded = EncodedCustomMetric {
            type_name: MaxMillis::NAME.to_string(),
            payload: vec![1, 2, 3],
        };

        let err = decode_custom_metric(&registry, &encoded)
            .expect("the type is registered")
            .expect_err("three bytes can't be a MaxMillis");
        assert!(err.to_string().contains("MaxMillis takes 8 bytes"), "{err}");
        Ok(())
    }

    #[test]
    fn register_rejects_a_name_collision_between_types() -> Result<()> {
        let mut registry = ProtoDecoderRegistry::new();
        MaxMillis::register_proto(&mut registry)?;
        MaxMillis::register_proto(&mut registry)?;

        let err = OtherMaxMillis::register_proto(&mut registry)
            .expect_err("colliding registration must fail");
        assert!(err.to_string().contains("already registered"), "{err}");
        assert_eq!(
            custom_metric_names(&registry).collect::<Vec<_>>(),
            vec![MaxMillis::NAME]
        );
        Ok(())
    }
}
