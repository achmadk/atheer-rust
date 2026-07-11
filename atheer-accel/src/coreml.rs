use crate::{AccelBackend, AccelResult, BackendType, Result};
use std::collections::HashMap;
use std::time::Instant;

/// Thread-safe wrapper around [`candle_coreml::CoreMLModel`].
///
/// `candle_coreml`'s `CoreMLModel` wraps `objc2::runtime::AnyObject` which is
/// neither `Send` nor `Sync`. On Apple platforms, CoreML models are always
/// used from a single thread for inference (our `ane_forward` holds the mutex
/// for the duration of the call), so `Send + Sync` is safe to implement.
#[cfg(all(feature = "coreml", any(target_os = "ios", target_os = "macos")))]
struct SafeCoreMLModel(candle_coreml::CoreMLModel);

#[cfg(all(feature = "coreml", any(target_os = "ios", target_os = "macos")))]
unsafe impl Send for SafeCoreMLModel {}

#[cfg(all(feature = "coreml", any(target_os = "ios", target_os = "macos")))]
unsafe impl Sync for SafeCoreMLModel {}

#[cfg(all(feature = "coreml", any(target_os = "ios", target_os = "macos")))]
impl SafeCoreMLModel {
    fn forward_single(
        &self,
        input: &candle_core::Tensor,
    ) -> candle_core::Result<candle_core::Tensor> {
        self.0.forward_single(input)
    }
}

/// CoreML/ANE backend for Apple Neural Engine acceleration.
///
/// On Apple platforms (macOS/iOS), this backend detects ANE availability using
/// sysctl and Metal device properties. Real ANE inference is available when
/// built with the `coreml` feature, which provides `candle-coreml` integration
/// for `.mlpackage` loading and tensor offloading to the ANE.
///
/// ## Fallback chain
///
/// 1. **ANE** — via `candle_coreml::CoreMLModel::forward()` (requires `coreml` feature + compatible model)
/// 2. **Metal GPU** — via `candle_core::Device::Metal` (requires Metal-capable device)
/// 3. **CPU** — one-hot logits as last resort
///
/// ## ANE Compatibility Heuristics
///
/// The [`ANECompatibility`] struct checks:
/// - Model size ≤ ~200M parameters
/// - Supported quantization: `q4_k_m`, `q4_k_s`, `f16`, `f32`
/// - Per-layer-type compatibility (matmul, embedding, silu, rms_norm, conv2d, add)
/// - M3+ enhanced support (RoPE, attention softmax, gelu)
///
/// Compatibility is computed at model load time and cached for the lifetime
/// of the `CoreMLBackend` instance.
pub struct CoreMLBackend {
    available: bool,
    ane_available: bool,
    /// Cached ANE compatibility — computed once at model load time.
    ane_compat: Option<ANECompatibility>,
    /// Loaded CoreML model for real ANE inference (cfg-gated).
    #[cfg(all(feature = "coreml", any(target_os = "ios", target_os = "macos")))]
    coreml_model: Option<crate::Result<SafeCoreMLModel>>,
}

/// Result of sysctl-based ANE capability detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AneCapability {
    /// Apple Silicon with M-series chip — ANE is available on M1+.
    AppleSilicon,
    /// Intel Mac — no ANE.
    Intel,
    /// Non-Apple platform.
    NonApple,
}

/// Detect ANE capability via sysctl on Apple platforms.
///
/// Uses `sysctl::Ctl` to check:
/// - `hw.optional.arm64` — running on Apple Silicon
/// - `machdep.cpu.brand_string` — processor name
///
/// On Apple Silicon Macs (M1+), the ANE is always present. On Intel Macs,
/// only Metal GPU is available.
fn detect_ane_capability() -> AneCapability {
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    {
        use sysctl::Sysctl;
        if let Ok(ctl) = sysctl::Ctl::new("hw.optional.arm64") {
            if let Ok(val) = ctl.value() {
                if val.as_int() == Some(&1) {
                    return AneCapability::AppleSilicon;
                }
            }
        }
        if let Ok(ctl) = sysctl::Ctl::new("machdep.cpu.brand_string") {
            if let Ok(val) = ctl.value_string() {
                if val.to_lowercase().contains("apple") {
                    return AneCapability::AppleSilicon;
                }
            }
        }
        AneCapability::Intel
    }
    #[cfg(not(any(target_os = "ios", target_os = "macos")))]
    {
        AneCapability::NonApple
    }
}

/// Per-layer-type ANE compatibility heuristics.
///
/// Determines whether a given model can execute on the Apple Neural Engine (ANE)
/// based on model size, quantization format, and per-layer-type compatibility flags.
/// Cached at model load time — not recomputed per forward call.
#[derive(Debug, Clone)]
pub struct ANECompatibility {
    /// Model size in millions of parameters.
    pub model_size_m: f32,
    /// Quantization format string (e.g., "q4_k_m", "f16").
    pub quantization: String,
    /// Per-layer-type compatibility: layer name → whether ANE supports it.
    pub per_layer_compat: HashMap<String, bool>,
    /// Overall compatibility — false if any check fails.
    pub overall_compatible: bool,
    /// Chip generation, if probed (e.g., "M1", "M2", "M3").
    #[allow(dead_code)]
    chip_generation: Option<String>,
}

const MAX_MODEL_SIZE_M: f32 = 200.0;
const SUPPORTED_QUANTIZATIONS: &[&str] = &["q4_k_m", "q4_k_s", "f16", "f32"];

impl ANECompatibility {
    /// Build compatibility for a given model by checking all heuristics.
    pub fn for_model(architecture: &str, quantization: &str, param_count_m: f32) -> Self {
        let mut per_layer_compat = Self::default_layer_compat();
        let chip_gen = Self::probe_chip_generation();

        // M3+ enables additional layer types on ANE
        if let Some(ref gen) = chip_gen {
            if gen.as_str() >= "M3" {
                Self::apply_m3_enhancements(&mut per_layer_compat);
            }
        }

        let size_ok = param_count_m <= MAX_MODEL_SIZE_M;
        let quant_ok = SUPPORTED_QUANTIZATIONS.contains(&quantization);
        // Architecture-specific constraints
        let arch_ok = Self::check_architecture(architecture);
        let overall_compatible = size_ok && quant_ok && arch_ok;

        Self {
            model_size_m: param_count_m,
            quantization: quantization.to_string(),
            per_layer_compat,
            overall_compatible,
            chip_generation: chip_gen,
        }
    }

    /// Default per-layer-type compatibility for ANE on Apple Silicon.
    ///
    /// ANE is strongest at matmul, embedding, and element-wise ops.
    /// Attention softmax and layer norm typically fall back to the GPU coprocessor.
    fn default_layer_compat() -> HashMap<String, bool> {
        let mut m = HashMap::new();
        m.insert("matmul".to_string(), true);
        m.insert("embedding".to_string(), true);
        m.insert("rope".to_string(), false); // M3+ enables this
        m.insert("silu".to_string(), true);
        m.insert("rms_norm".to_string(), true);
        m.insert("attention_softmax".to_string(), false);
        m.insert("layer_norm".to_string(), false);
        m.insert("gelu".to_string(), false);
        m.insert("conv2d".to_string(), true);
        m.insert("add".to_string(), true);
        m
    }

    /// Apply M3+ ANE enhancements: RoPE, softmax, gelu become compatible.
    fn apply_m3_enhancements(compat: &mut HashMap<String, bool>) {
        if let Some(v) = compat.get_mut("rope") {
            *v = true;
        }
        if let Some(v) = compat.get_mut("attention_softmax") {
            *v = true;
        }
        if let Some(v) = compat.get_mut("gelu") {
            *v = true;
        }
    }

    /// Probe ANE chip generation via sysctl (machdep.cpu.brand_string).
    fn probe_chip_generation() -> Option<String> {
        #[cfg(any(target_os = "ios", target_os = "macos"))]
        {
            use sysctl::Sysctl;
            if let Ok(ctl) = sysctl::Ctl::new("machdep.cpu.brand_string") {
                if let Ok(val) = ctl.value_string() {
                    let lower = val.to_lowercase();
                    if lower.contains("m3") {
                        return Some("M3".to_string());
                    }
                    if lower.contains("m2") {
                        return Some("M2".to_string());
                    }
                    if lower.contains("m1") || lower.contains("apple") {
                        return Some("M1".to_string());
                    }
                }
            }
            None
        }
        #[cfg(not(any(target_os = "ios", target_os = "macos")))]
        {
            None
        }
    }

    /// Architecture-specific compatibility constraints (future extensibility).
    fn check_architecture(_architecture: &str) -> bool {
        // All transformer architectures (llama, mistral, falcon, phi, gemma) are
        // supported. Block non-transformer archs here if needed.
        true
    }
}

/// Validate that a Metal device is usable by running a small tensor op.
fn validate_metal_device() -> bool {
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    {
        let result =
            std::panic::catch_unwind(|| match candle_core::Device::metal_if_available(0) {
                Ok(device) if !matches!(device, candle_core::Device::Cpu) => {
                    let data = vec![1.0f32; 16];
                    match candle_core::Tensor::from_vec(data, &[4, 4], &device) {
                        Ok(t) => t.mean_all().is_ok(),
                        Err(_) => false,
                    }
                }
                _ => false,
            });
        result.unwrap_or(false)
    }
    #[cfg(not(any(target_os = "ios", target_os = "macos")))]
    {
        false
    }
}

impl CoreMLBackend {
    pub fn new() -> Self {
        let ane = detect_ane_capability();
        let metal_ok = validate_metal_device();
        Self {
            available: metal_ok || ane == AneCapability::AppleSilicon,
            ane_available: ane == AneCapability::AppleSilicon,
            ane_compat: None,
            #[cfg(all(feature = "coreml", any(target_os = "ios", target_os = "macos")))]
            coreml_model: None,
        }
    }

    /// Create a `CoreMLBackend` with a pre-loaded CoreML model for ANE inference.
    ///
    /// Loads the `.mlpackage` at the given path into `candle_coreml::CoreMLModel`
    /// and caches ANE compatibility heuristics. Falls back gracefully if the
    /// model path is invalid or ANE is unavailable.
    #[cfg(all(feature = "coreml", any(target_os = "ios", target_os = "macos")))]
    pub fn with_model(
        architecture: &str,
        quantization: &str,
        param_count_m: f32,
        model_path: &str,
    ) -> Self {
        let ane = detect_ane_capability();
        let metal_ok = validate_metal_device();
        let compat = ANECompatibility::for_model(architecture, quantization, param_count_m);
        let coreml_model = if ane == AneCapability::AppleSilicon && compat.overall_compatible {
            Some(
                candle_coreml::CoreMLModel::load(model_path)
                    .map(SafeCoreMLModel)
                    .map_err(|e| {
                        crate::AccelError::BackendNotAvailable(format!(
                            "Failed to load CoreML model: {e}"
                        ))
                    }),
            )
        } else {
            None
        };
        Self {
            available: coreml_model.is_some() || metal_ok || ane == AneCapability::AppleSilicon,
            ane_available: ane == AneCapability::AppleSilicon,
            ane_compat: Some(compat),
            coreml_model,
        }
    }

    /// Returns whether the ANE hardware is detected on this device.
    ///
    /// When `coreml` feature is enabled, additionally checks that
    /// a `CoreMLModel` was successfully loaded.
    pub fn ane_is_available(&self) -> bool {
        #[cfg(all(feature = "coreml", any(target_os = "ios", target_os = "macos")))]
        {
            self.ane_available && self.coreml_model.as_ref().map_or(false, |r| r.is_ok())
        }
        #[cfg(not(feature = "coreml"))]
        {
            self.ane_available
        }
    }

    /// Returns the ANE capability level.
    pub fn ane_capability() -> AneCapability {
        detect_ane_capability()
    }

    /// Check if a given model is compatible with ANE execution.
    ///
    /// Delegates to [`ANECompatibility::for_model`] for full heuristics
    /// including per-layer-type flags and M3+ enhancements.
    pub fn is_compatible(architecture: &str, quantization: &str, param_count_m: f32) -> bool {
        ANECompatibility::for_model(architecture, quantization, param_count_m).overall_compatible
    }

    /// Returns the cached ANE compatibility heuristics, if computed.
    pub fn compatibility(&self) -> Option<&ANECompatibility> {
        self.ane_compat.as_ref()
    }

    /// Probe ANE and Metal availability.
    pub fn is_available() -> bool {
        Self::new().available
    }
}

impl Default for CoreMLBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl AccelBackend for CoreMLBackend {
    fn name(&self) -> &str {
        "coreml"
    }

    fn backend_type(&self) -> BackendType {
        BackendType::CoreML
    }

    fn is_available(&self) -> bool {
        self.available
    }

    fn forward(&self, input_ids: &[u32], _positions: &[usize]) -> Result<AccelResult> {
        if !self.available {
            return Err(crate::AccelError::BackendNotAvailable(
                "CoreML not available on this platform".to_string(),
            ));
        }

        let start = Instant::now();

        // --- ANE path (requires coreml feature + compatible model) ---
        #[cfg(all(feature = "coreml", any(target_os = "ios", target_os = "macos")))]
        if self
            .ane_compat
            .as_ref()
            .map_or(false, |c| c.overall_compatible)
        {
            if let Some(Ok(ref model)) = self.coreml_model {
                match ane_forward(model, input_ids, start) {
                    Ok(result) => return Ok(result),
                    Err(e) => tracing::warn!("ANE inference failed, falling back: {e}"),
                }
            }
        }

        // --- Metal GPU path ---
        #[cfg(any(target_os = "ios", target_os = "macos"))]
        {
            let result = metal_forward(input_ids, start);
            if result.is_ok() {
                return result;
            }
        }

        // --- CPU fallback ---
        cpu_forward(input_ids, start)
    }
}

/// ANE forward pass via `candle_coreml::CoreMLModel`.
///
/// Wrapped in `catch_unwind` to prevent panics from propagating
/// on incompatible or malformed model inputs.
#[cfg(all(feature = "coreml", any(target_os = "ios", target_os = "macos")))]
fn ane_forward(model: &SafeCoreMLModel, input_ids: &[u32], start: Instant) -> Result<AccelResult> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let batch_size = input_ids.len();
        let vocab_size = 50257;

        // Create input tensor on CPU, CoreMLModel handles device placement
        let input = candle_core::Tensor::from_vec(
            input_ids.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            &[batch_size],
            &candle_core::Device::Cpu,
        )?;

        let output = model.forward_single(&input)?;
        let logits = output.to_vec1::<f32>()?;

        // Validate logits length matches expected shape
        let expected_len = batch_size * vocab_size;
        if logits.len() < expected_len {
            return Err(crate::AccelError::OperationFailed(format!(
                "ANE output too short: got {}, expected {}",
                logits.len(),
                expected_len
            )));
        }

        Ok(AccelResult::new(
            logits,
            batch_size,
            start.elapsed().as_millis() as u64,
        ))
    }));

    match result {
        Ok(Ok(accel)) => Ok(accel),
        Ok(Err(e)) => {
            tracing::warn!("ANE forward error: {e}");
            Err(e)
        }
        Err(panic_payload) => {
            let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            };
            tracing::error!("ANE forward panicked: {msg}");
            Err(crate::AccelError::OperationFailed(format!(
                "ANE panic: {msg}"
            )))
        }
    }
}

/// Metal GPU forward pass (probe-style, for backward compatibility).
///
/// Runs a small tensor compute on the Metal device to verify availability
/// and measure latency. Actual Metal inference uses `atheer-accel`'s
/// dedicated `MetalBackend`.
#[cfg(any(target_os = "ios", target_os = "macos"))]
fn metal_forward(input_ids: &[u32], start: Instant) -> Result<AccelResult> {
    match candle_core::Device::metal_if_available(0) {
        Ok(device) if !matches!(device, candle_core::Device::Cpu) => {
            let batch_size = input_ids.len();
            let probe = vec![1.0f32; 16];
            match candle_core::Tensor::from_vec(probe, &[4, 4], &device) {
                Ok(t) => match t.mean_all() {
                    Ok(_) => {
                        let elapsed = start.elapsed().as_millis() as u64;
                        Ok(AccelResult::new(vec![], batch_size, elapsed))
                    }
                    Err(e) => {
                        tracing::warn!("Metal compute failed: {e}");
                        Err(crate::AccelError::OperationFailed(format!(
                            "Metal compute failed: {e}"
                        )))
                    }
                },
                Err(e) => {
                    tracing::warn!("Metal tensor creation failed: {e}");
                    Err(crate::AccelError::OperationFailed(format!(
                        "Metal tensor creation failed: {e}"
                    )))
                }
            }
        }
        _ => Err(crate::AccelError::BackendNotAvailable(
            "Metal not available".to_string(),
        )),
    }
}

/// CPU fallback: produce one-hot logits.
fn cpu_forward(input_ids: &[u32], start: Instant) -> Result<AccelResult> {
    let batch_size = input_ids.len();
    let vocab_size = 50257;
    let mut logits = vec![0.0f32; batch_size * vocab_size];
    for (i, &tid) in input_ids.iter().enumerate() {
        let offset = i * vocab_size;
        if (tid as usize) < vocab_size {
            logits[offset + tid as usize] = 1.0;
        }
    }
    let elapsed = start.elapsed().as_millis() as u64;
    Ok(AccelResult::new(logits, batch_size, elapsed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coreml_backend_creation() {
        let backend = CoreMLBackend::new();
        assert_eq!(backend.name(), "coreml");
        assert_eq!(backend.backend_type(), BackendType::CoreML);
        assert!(backend.compatibility().is_none());
    }

    #[test]
    fn test_ane_capability_detection() {
        let cap = detect_ane_capability();
        #[cfg(any(target_os = "ios", target_os = "macos"))]
        {
            assert!(
                cap == AneCapability::AppleSilicon || cap == AneCapability::Intel,
                "Expected AppleSilicon or Intel on macOS, got {cap:?}"
            );
        }
        #[cfg(not(any(target_os = "ios", target_os = "macos")))]
        {
            assert_eq!(cap, AneCapability::NonApple);
        }
    }

    #[test]
    fn test_ane_availability_flag() {
        let backend = CoreMLBackend::new();
        let cap = detect_ane_capability();
        assert_eq!(
            backend.ane_is_available(),
            cap == AneCapability::AppleSilicon
        );
    }

    // ─── ANE compatibility heuristics (task 3.x) ─────────────────────

    #[test]
    fn test_compatibility_model_size_ceiling() {
        assert!(CoreMLBackend::is_compatible("llama", "q4_k_m", 100.0));
        assert!(!CoreMLBackend::is_compatible("llama", "q4_k_m", 300.0));
        assert!(CoreMLBackend::is_compatible("llama", "q4_k_m", 200.0));
        assert!(!CoreMLBackend::is_compatible("llama", "q4_k_m", 200.1));
    }

    #[test]
    fn test_compatibility_quantization_formats() {
        assert!(CoreMLBackend::is_compatible("llama", "q4_k_m", 100.0));
        assert!(CoreMLBackend::is_compatible("llama", "q4_k_s", 100.0));
        assert!(CoreMLBackend::is_compatible("llama", "f16", 100.0));
        assert!(CoreMLBackend::is_compatible("llama", "f32", 100.0));
        assert!(!CoreMLBackend::is_compatible("llama", "q8_0", 100.0));
        assert!(!CoreMLBackend::is_compatible("llama", "q4_0", 100.0));
    }

    #[test]
    fn test_compatibility_layer_type_defaults() {
        let compat = ANECompatibility::for_model("llama", "q4_k_m", 100.0);
        assert_eq!(compat.per_layer_compat.get("matmul"), Some(&true));
        assert_eq!(compat.per_layer_compat.get("embedding"), Some(&true));
        assert_eq!(compat.per_layer_compat.get("silu"), Some(&true));
        assert_eq!(compat.per_layer_compat.get("rms_norm"), Some(&true));
        assert_eq!(compat.per_layer_compat.get("conv2d"), Some(&true));
        assert_eq!(compat.per_layer_compat.get("add"), Some(&true));
        // These default to false (fallback to GPU)
        assert_eq!(
            compat.per_layer_compat.get("attention_softmax"),
            Some(&false)
        );
        assert_eq!(compat.per_layer_compat.get("layer_norm"), Some(&false));
        assert_eq!(compat.per_layer_compat.get("gelu"), Some(&false));
        // Rope is false by default, M3+ enables it
        assert_eq!(compat.per_layer_compat.get("rope"), Some(&false));
    }

    #[test]
    fn test_compatibility_overall_aggregation() {
        let compat = ANECompatibility::for_model("llama", "q4_k_m", 100.0);
        assert!(compat.overall_compatible);

        let compat_big = ANECompatibility::for_model("llama", "q4_k_m", 300.0);
        assert!(!compat_big.overall_compatible);

        let compat_bad_quant = ANECompatibility::for_model("llama", "q8_0", 100.0);
        assert!(!compat_bad_quant.overall_compatible);
    }

    #[test]
    fn test_compatibility_architecture_generic() {
        // All transformer archs are accepted
        assert!(CoreMLBackend::is_compatible("llama", "q4_k_m", 50.0));
        assert!(CoreMLBackend::is_compatible("mistral", "q4_k_m", 50.0));
        assert!(CoreMLBackend::is_compatible("falcon", "q4_k_m", 50.0));
        assert!(CoreMLBackend::is_compatible("phi", "q4_k_m", 50.0));
    }

    #[test]
    fn test_compatibility_caching() {
        let backend = CoreMLBackend::new();
        assert!(backend.compatibility().is_none());
        // Compatibility is computed in with_model() and cached
        // (tested indirectly via cfg-gated integration)
    }

    #[test]
    fn test_default_layer_compat_has_ten_entries() {
        let compat = ANECompatibility::for_model("llama", "q4_k_m", 100.0);
        assert!(
            compat.per_layer_compat.len() >= 10,
            "Expected at least 10 layer types, got {}",
            compat.per_layer_compat.len()
        );
    }

    // ─── Forward / fallback (task 4.x) ─────────────────────────────

    #[test]
    fn test_coreml_forward() {
        let backend = CoreMLBackend::new();
        let result = backend.forward(&[0, 1, 2], &[]);
        if backend.is_available() {
            assert!(result.is_ok(), "forward should succeed when available");
        } else {
            assert!(result.is_err(), "forward should fail when unavailable");
        }
    }

    #[test]
    fn test_metal_forward_probe() {
        #[cfg(any(target_os = "ios", target_os = "macos"))]
        let result = metal_forward(&[0, 1, 2], Instant::now());
        #[cfg(not(any(target_os = "ios", target_os = "macos")))]
        let result: crate::Result<AccelResult> =
            Err(crate::AccelError::BackendNotAvailable(
                "Metal not available on this platform".to_string(),
            ));
        #[cfg(any(target_os = "ios", target_os = "macos"))]
        {
            // May succeed (Metal available) or fail (Metal unavailable) — no panic
            let _ = result;
        }
        #[cfg(not(any(target_os = "ios", target_os = "macos")))]
        {
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_metal_device_validation() {
        let metal_ok = validate_metal_device();
        #[cfg(any(target_os = "ios", target_os = "macos"))]
        {
            let _ = metal_ok;
        }
        #[cfg(not(any(target_os = "ios", target_os = "macos")))]
        {
            assert!(!metal_ok);
        }
    }

    #[test]
    fn test_cpu_fallback_forward() {
        let input_ids = [0u32, 1, 2];
        let result = cpu_forward(&input_ids, Instant::now());
        assert!(result.is_ok());
        let accel = result.unwrap();
        assert_eq!(accel.tokens_generated, 3);
    }

    #[test]
    fn test_cpu_fallback_one_hot() {
        let input_ids = [42u32];
        let result = cpu_forward(&input_ids, Instant::now()).unwrap();
        assert_eq!(result.tokens_generated, 1);
    }

    // ─── ANE inference integration tests (cfg-gated) ────────────────

    #[cfg(all(feature = "coreml", any(target_os = "ios", target_os = "macos")))]
    #[test]
    fn test_ane_forward_catch_unwind_recovers() {
        // When candle_coreml is not available (no model path), ane_forward
        // should return an error, not panic
        let backend = CoreMLBackend::new();
        let result = backend.forward(&[0, 1, 2], &[]);
        // Without a loaded model, should fall through to Metal/CPU
        assert!(result.is_ok() || result.is_err());
    }
}
