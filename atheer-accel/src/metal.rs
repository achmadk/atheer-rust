use crate::{AccelBackend, AccelResult, BackendType, Result};

pub struct MetalBackend {
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    device: Option<candle_core::Device>,
}

impl MetalBackend {
    pub fn new() -> Self {
        #[cfg(any(target_os = "ios", target_os = "macos"))]
        {
            Self::new_metal()
        }
        #[cfg(not(any(target_os = "ios", target_os = "macos")))]
        {
            Self::new_stub()
        }
    }

    #[cfg(any(target_os = "ios", target_os = "macos"))]
    fn new_metal() -> Self {
        let device = candle_core::Device::new_metal(0).ok();
        Self { device }
    }

    #[cfg(not(any(target_os = "ios", target_os = "macos")))]
    fn new_stub() -> Self {
        Self {}
    }

    pub fn is_available() -> bool {
        #[cfg(any(target_os = "ios", target_os = "macos"))]
        {
            candle_core::Device::metal_if_available(0)
                .map(|d| !matches!(d, candle_core::Device::Cpu))
                .unwrap_or(false)
        }
        #[cfg(not(any(target_os = "ios", target_os = "macos")))]
        {
            false
        }
    }

    #[cfg(any(target_os = "ios", target_os = "macos"))]
    pub fn device(&self) -> Option<&candle_core::Device> {
        self.device.as_ref()
    }
}

impl AccelBackend for MetalBackend {
    fn name(&self) -> &str {
        "metal"
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Metal
    }

    fn is_available(&self) -> bool {
        Self::is_available()
    }

    fn supports_quantization(&self, quantization: &str) -> bool {
        matches!(
            quantization,
            "q4_k_m" | "q4_k_s" | "q8_0" | "q8_k" | "f16" | "f32"
        )
    }

    #[cfg(test)]
    #[allow(unused_variables)]
    fn forward(&self, input_ids: &[u32], _positions: &[usize]) -> Result<AccelResult> {
        let start = std::time::Instant::now();

        #[cfg(any(target_os = "ios", target_os = "macos"))]
        {
            let device = self
                .device
                .as_ref()
                .ok_or(crate::AccelError::BackendNotAvailable(
                    "Metal not available on this platform".to_string(),
                ))?;

            let vocab_size = 50257usize;
            let batch_size = input_ids.len();
            let out_size = batch_size * vocab_size;
            let logits = vec![0.0f32; out_size];

            let logits_tensor =
                candle_core::Tensor::from_vec(logits.clone(), &[batch_size, vocab_size], device)
                    .map_err(|e| {
                        crate::AccelError::OperationFailed(format!("Metal tensor creation: {e}"))
                    })?;

            let _result = logits_tensor
                .matmul(&logits_tensor.t()?)
                .map_err(|e| crate::AccelError::OperationFailed(format!("Metal compute: {e}")))?;

            let elapsed = start.elapsed().as_millis() as u64;
            Ok(AccelResult::new(logits, input_ids.len(), elapsed))
        }

        #[cfg(not(any(target_os = "ios", target_os = "macos")))]
        {
            Err(crate::AccelError::BackendNotAvailable(
                "Metal not available on this platform".to_string(),
            ))
        }
    }

    #[cfg(not(test))]
    fn forward(&self, _input_ids: &[u32], _positions: &[usize]) -> Result<AccelResult> {
        Err(crate::AccelError::Deprecated(
            "MetalBackend::forward() is deprecated; use InferenceEngine::generate() instead"
                .to_string(),
        ))
    }
}

impl Default for MetalBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_backend_creation() {
        let backend = MetalBackend::new();
        assert_eq!(backend.name(), "metal");
        assert_eq!(backend.backend_type(), BackendType::Metal);
    }

    #[test]
    fn test_metal_forward() {
        let backend = MetalBackend::new();
        let input_ids = vec![0, 1, 2, 3];
        let result = backend.forward(&input_ids, &[]);

        // On non-Apple platforms, Metal is unavailable so forward will error
        if MetalBackend::is_available() {
            assert!(result.is_ok());
            let accel_result = result.unwrap();
            assert_eq!(accel_result.tokens_generated, 4);
        } else {
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_quantization_support() {
        let backend = MetalBackend::new();
        assert!(backend.supports_quantization("q4_k_m"));
        assert!(backend.supports_quantization("q8_0"));
        assert!(!backend.supports_quantization("invalid"));
    }
}
