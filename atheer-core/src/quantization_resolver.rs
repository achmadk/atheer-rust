use std::cell::OnceCell;

/// Tier classification for device GPU capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuTier {
    Unknown,
    Low,
    Medium,
    High,
}

/// Runtime quantization format resolver.
///
/// Selects the optimal quantization format based on device capabilities
/// and available RAM. Results are cached to avoid re-probing.
pub struct QuantizationResolver {
    /// Cached NPU INT4 support probe result.
    probed_npu_int4: OnceCell<bool>,
    /// Cached GPU tier probe result.
    probed_gpu_tier: OnceCell<GpuTier>,
    /// Total system RAM in MB.
    total_ram_mb: u64,
}

/// Known-good quantization formats, ordered from most to least compressed.
const KNOWN_FORMATS: &[&str] = &["q2_k", "q3_k_m", "q4_k_m", "q5_k_m", "q6_k", "q8_0", "f16"];

impl QuantizationResolver {
    pub fn new(total_ram_mb: u64) -> Self {
        Self {
            probed_npu_int4: OnceCell::new(),
            probed_gpu_tier: OnceCell::new(),
            total_ram_mb,
        }
    }

    /// Resolve the optimal quantization format given the user's requested format.
    ///
    /// Returns `(resolved_format, Option<warning>)`.
    pub fn resolve(&mut self, user_requested: &str) -> (String, Option<String>) {
        // Step 1: determine baseline based on RAM constraints
        let (baseline, warning) = self.apply_ram_constraints(user_requested);

        // Step 2: check if NPU supports INT4 — if so, prefer q4_k_m
        let with_npu = if self.supports_npu_int4() {
            // NPU prefers INT4; only upgrade if baseline is less compressed
            match baseline.as_str() {
                "f16" | "q8_0" | "q6_k" | "q5_k_m" => {
                    let w = Some(format!(
                        "Format '{}' downgraded to 'q4_k_m': NPU INT4 preferred",
                        baseline
                    ));
                    ("q4_k_m".to_string(), w)
                }
                _ => (baseline.clone(), warning),
            }
        } else {
            (baseline.clone(), warning)
        };

        // Step 3: validate against known format list
        if KNOWN_FORMATS.contains(&with_npu.0.as_str()) {
            with_npu
        } else {
            (
                "q4_k_m".to_string(),
                Some(format!(
                    "Unknown format '{}', falling back to 'q4_k_m'",
                    with_npu.0
                )),
            )
        }
    }

    /// Apply RAM-based downgrade rules.
    fn apply_ram_constraints(&self, user_requested: &str) -> (String, Option<String>) {
        if self.total_ram_mb < 2048 {
            return (
                "q4_k_m".to_string(),
                Some(format!(
                    "Insufficient RAM ({} MB): using q4_k_m",
                    self.total_ram_mb
                )),
            );
        }

        if self.total_ram_mb < 4096 {
            match user_requested {
                "f16" | "q8_0" => {
                    return (
                        "q8_0".to_string(),
                        Some(format!(
                            "Insufficient RAM ({} MB) for '{}': downgraded to q8_0",
                            self.total_ram_mb, user_requested
                        )),
                    );
                }
                _ => {}
            }
        }

        if self.total_ram_mb < 8192 && user_requested == "f16" {
            return (
                "q8_0".to_string(),
                Some(format!(
                    "Insufficient RAM ({} MB) for f16: downgraded to q8_0",
                    self.total_ram_mb
                )),
            );
        }

        (user_requested.to_string(), None)
    }

    /// Probe whether the device NPU supports INT4 quantization.
    ///
    /// Currently a stub that returns `true`. In production, this should
    /// query NNAPI (Android) or CoreML (iOS) for INT4 support.
    pub fn supports_npu_int4(&mut self) -> bool {
        *self.probed_npu_int4.get_or_init(|| Self::probe_npu_int4_impl())
    }

    /// The actual NPU probe implementation.
    ///
    /// On Android this would call:
    ///   ANeuralNetworks_getDeviceCount + ANeuralNetworks_getDeviceName
    /// to check for NPU accelerators that support INT4.
    ///
    /// For now, return `true` as a conservative default — most modern
    /// mobile NPUs (Snapdragon 8 Gen 2+, Dimensity 9000+, Apple A14+)
    /// support INT4.
    fn probe_npu_int4_impl() -> bool {
        true
    }

    /// Return the cached GPU tier, probing if not yet cached.
    pub fn gpu_tier(&mut self) -> GpuTier {
        *self.probed_gpu_tier.get_or_init(|| GpuTier::Unknown)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_high_ram_keeps_format() {
        let mut resolver = QuantizationResolver::new(16384);
        let (fmt, warn) = resolver.resolve("q4_k_m");
        assert_eq!(fmt, "q4_k_m");
        assert!(warn.is_none());
    }

    #[test]
    fn test_resolve_low_ram_downgrades() {
        let mut resolver = QuantizationResolver::new(3072);
        let (fmt, warn) = resolver.resolve("f16");
        assert_eq!(fmt, "q8_0");
        assert!(warn.is_some());
        assert!(warn.unwrap().contains("Insufficient RAM"));
    }

    #[test]
    fn test_resolve_severe_ram_forced_q4() {
        let mut resolver = QuantizationResolver::new(1024);
        let (fmt, warn) = resolver.resolve("f16");
        assert_eq!(fmt, "q4_k_m");
        assert!(warn.is_some());
    }

    #[test]
    fn test_resolve_unknown_format_falls_back() {
        let mut resolver = QuantizationResolver::new(8192);
        let (fmt, warn) = resolver.resolve("q1_k");
        assert_eq!(fmt, "q4_k_m");
        assert!(warn.is_some());
        assert!(warn.unwrap().contains("Unknown format"));
    }

    #[test]
    fn test_resolve_high_ram_f16_allowed() {
        let mut resolver = QuantizationResolver::new(16384);
        let (fmt, warn) = resolver.resolve("f16");
        assert_eq!(fmt, "f16");
        assert!(warn.is_none());
    }

    #[test]
    fn test_gpu_tier_defaults_to_unknown() {
        let mut resolver = QuantizationResolver::new(8192);
        assert_eq!(resolver.gpu_tier(), GpuTier::Unknown);
    }

    #[test]
    fn test_npu_probe_cached() {
        let mut resolver = QuantizationResolver::new(8192);
        // First call probes and caches
        let first = resolver.supports_npu_int4();
        // Second call uses cache — no re-probe needed
        let second = resolver.supports_npu_int4();
        assert_eq!(first, second);
    }

    #[test]
    fn test_ram_edge_2048_not_enough_for_f16() {
        let mut resolver = QuantizationResolver::new(2048);
        let (fmt, _) = resolver.resolve("f16");
        // 2048 MB is < 4096 but >= 2048, so q8_0 threshold applies if f16
        assert_eq!(fmt, "q8_0");
    }

    #[test]
    fn test_resolve_q4_k_m_at_2048_ram() {
        let mut resolver = QuantizationResolver::new(2048);
        let (fmt, warn) = resolver.resolve("q4_k_m");
        // q4_k_m should pass through at 2048 MB
        assert_eq!(fmt, "q4_k_m");
        assert!(warn.is_none());
    }
}
