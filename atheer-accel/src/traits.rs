use crate::{BackendType, Result};

pub trait AccelBackend: Send + Sync {
    fn name(&self) -> &str;
    fn backend_type(&self) -> BackendType;

    /// Whether this backend is available on the current device.
    ///
    /// Returns `false` by default. Backends that can self-report availability
    /// (e.g. by probing for hardware or API support) should override this.
    fn is_available(&self) -> bool {
        false
    }

    fn supports_quantization(&self, _quantization: &str) -> bool {
        true
    }
    #[deprecated(since = "0.1.0", note = "use InferenceEngine::generate() instead")]
    fn forward(&self, input_ids: &[u32], positions: &[usize]) -> Result<AccelResult>;
}

#[derive(Debug, Clone)]
pub struct AccelResult {
    pub logits: Vec<f32>,
    pub tokens_generated: usize,
    pub inference_time_ms: u64,
}

impl AccelResult {
    pub fn new(logits: Vec<f32>, tokens: usize, time_ms: u64) -> Self {
        Self {
            logits,
            tokens_generated: tokens,
            inference_time_ms: time_ms,
        }
    }

    pub fn tokens_per_second(&self) -> f32 {
        if self.inference_time_ms == 0 {
            return 0.0;
        }
        (self.tokens_generated as f32 / self.inference_time_ms as f32) * 1000.0
    }
}
