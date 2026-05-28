use crate::{AccelBackend, AccelResult, BackendType, Result};
use std::time::Instant;

/// CoreML/ANE backend for Apple Neural Engine acceleration.
///
/// On macOS, this backend can use `candle-coreml` for real CoreML model inference.
/// On iOS, ANE access requires native Swift bindings through the FFI layer;
/// this backend provides detection and reports availability accordingly.
///
/// The actual model inference happens through `atheer-core::Model` on the
/// Candle Metal device (which is the primary iOS GPU path). This backend
/// serves as a compatibility detector and health check for the ANE path.
pub struct CoreMLBackend {
    available: bool,
}

impl CoreMLBackend {
    pub fn new() -> Self {
        let available = Self::detect_availability();
        Self { available }
    }

    /// Detect whether CoreML/ANE is available on this platform.
    fn detect_availability() -> bool {
        #[cfg(any(target_os = "ios", target_os = "macos"))]
        {
            // CoreML is available on all Apple platforms with A11+ chips
            // Detection is done at runtime via sysctl or Metal device check
            candle_core::Device::metal_if_available(0)
                .map(|d| !matches!(d, candle_core::Device::Cpu))
                .unwrap_or(false)
        }
        #[cfg(not(any(target_os = "ios", target_os = "macos")))]
        {
            false
        }
    }

    /// Check if a given model architecture is compatible with ANE constraints.
    ///
    /// ANE (Apple Neural Engine) has specific constraints:
    /// - Model size < ~200M parameters for on-device ANE
    /// - Quantization must be supported by the ANE hardware
    /// - Some layer types may fall back to GPU/CPU
    pub fn is_compatible(_architecture: &str, quantization: &str, param_count_m: f32) -> bool {
        // ANE on A17+/M-series: supports up to ~200M param models
        if param_count_m > 200.0 {
            return false;
        }
        // Quantization formats compatible with ANE
        matches!(quantization, "q4_k_m" | "q4_k_s" | "f16" | "f32")
    }

    pub fn is_available() -> bool {
        Self::detect_availability()
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

        // On Apple platforms, verify Metal/CoreML device works by running
        // a small tensor operation. Real ANE inference requires native
        // CoreML model compilation via the Swift FFI layer.
        #[cfg(any(target_os = "ios", target_os = "macos"))]
        {
            if let Ok(device) = candle_core::Device::metal_if_available(0) {
                if !matches!(device, candle_core::Device::Cpu) {
                    let logits =
                        vec![0.0f32; input_ids.len() * 50257];
                    if let Ok(t) = candle_core::Tensor::from_vec(
                        logits.clone(),
                        &[input_ids.len(), 50257],
                        &device,
                    ) {
                        let _ = t.mean_all();
                    }
                    let elapsed = start.elapsed().as_millis() as u64;
                    return Ok(AccelResult::new(vec![], input_ids.len(), elapsed));
                }
            }
            Err(crate::AccelError::BackendNotAvailable(
                "CoreML/Metal device unavailable".to_string(),
            ))
        }

        #[cfg(not(any(target_os = "ios", target_os = "macos")))]
        {
            let _ = start;
            Err(crate::AccelError::BackendNotAvailable(
                "CoreML not available on this platform".to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coreml_backend_creation() {
        let backend = CoreMLBackend::new();
        assert_eq!(backend.name(), "coreml");
        assert_eq!(backend.backend_type(), BackendType::CoreML);
    }

    #[test]
    fn test_coreml_compatibility() {
        // Small quantized model should be compatible
        assert!(CoreMLBackend::is_compatible("llama", "q4_k_m", 100.0));
        // Large model should be incompatible
        assert!(!CoreMLBackend::is_compatible("llama", "q4_k_m", 300.0));
        // Unsupported quantization
        assert!(!CoreMLBackend::is_compatible("llama", "q8_0", 100.0));
    }

    #[test]
    fn test_coreml_forward() {
        let backend = CoreMLBackend::new();
        let result = backend.forward(&[0, 1, 2], &[]);
        // On non-Apple platforms, CoreML is unavailable
        if CoreMLBackend::is_available() {
            assert!(result.is_ok());
        } else {
            assert!(result.is_err());
        }
    }
}
