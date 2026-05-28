use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};

/// Architecture-aware dispatch for model weights loading.
///
/// Reads `general.architecture` from GGUF metadata and selects the
/// appropriate `ModelWeights` implementation:
/// - `lfm2` → `candle_transformers::models::quantized_lfm2::ModelWeights`
/// - anything else → `candle_transformers::models::quantized_llama::ModelWeights`
pub enum WeightsVariant {
    Llama(candle_transformers::models::quantized_llama::ModelWeights),
    Lfm2(candle_transformers::models::quantized_lfm2::ModelWeights),
}

impl WeightsVariant {
    /// Load weights from a parsed GGUF `Content`, dispatching on architecture.
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> candle_core::Result<Self> {
        let arch = ct
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .map(|s| s.clone())
            .unwrap_or_else(|| "llama".to_string());

        match arch.as_str() {
            "lfm2" => {
                let w =
                    candle_transformers::models::quantized_lfm2::ModelWeights::from_gguf(
                        ct, reader, device,
                    )?;
                Ok(Self::Lfm2(w))
            }
            _ => {
                let w =
                    candle_transformers::models::quantized_llama::ModelWeights::from_gguf(
                        ct, reader, device,
                    )?;
                Ok(Self::Llama(w))
            }
        }
    }

    /// Forward pass: compute logits for the given token(s) at `index_pos`.
    ///
    /// Both model implementations return logits with shape `[1, vocab_size]`
    /// (batch=1, last-token-only). The batch dimension is squeezed so callers
    /// receive a 1D `[vocab_size]` tensor compatible with `Tensor::to_vec1()`.
    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
        let logits = match self {
            Self::Llama(w) => w.forward(x, index_pos)?,
            Self::Lfm2(w) => w.forward(x, index_pos)?,
        };
        // Squeeze the batch dimension — both models always return [1, vocab_size]
        logits.squeeze(0)
    }

    /// Drop all GPU-side KV cache tensors, freeing VRAM.
    pub fn kv_cache_clear(&mut self) {
        match self {
            Self::Llama(w) => w.kv_cache_clear(),
            Self::Lfm2(w) => w.kv_cache_clear()
        }
    }

    /// Snapshot per-layer KV cache to CPU memory.
    ///
    /// Returns an error for LFM2 models (not supported).
    pub fn kv_cache_snapshot(&self) -> candle_core::Result<Vec<(Vec<f32>, Vec<f32>)>> {
        match self {
            Self::Llama(w) => w.kv_cache_snapshot(),
            Self::Lfm2(_) => Err(candle_core::Error::Msg(
                "KV cache snapshot not supported for LFM2 models".to_string(),
            )),
        }
    }

    /// Restore per-layer KV cache from a CPU snapshot.
    ///
    /// Returns an error for LFM2 models (not supported).
    pub fn kv_cache_restore(
        &mut self,
        snapshot: &[(Vec<f32>, Vec<f32>)],
    ) -> candle_core::Result<()> {
        match self {
            Self::Llama(w) => w.kv_cache_restore(snapshot),
            Self::Lfm2(_) => Err(candle_core::Error::Msg(
                "KV cache restore not supported for LFM2 models".to_string(),
            )),
        }
    }
}
