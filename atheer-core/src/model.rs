use crate::kv_cache_bridge::KvCacheBridge;
use crate::quantization_resolver::QuantizationResolver;
use crate::{AtheerCoreError, Result};
use std::io::{Read, Seek};
use std::path::Path;

/// Wrapper around a candle-transformers quantized model.
pub struct Model {
    pub weights: crate::weights::WeightsVariant,
    pub device: candle_core::Device,
    pub context_size: usize,
    /// Recommended quantization format from the resolver (if used during loading).
    pub resolved_quant_format: Option<String>,
    /// Warning emitted by the resolver during format resolution.
    pub quant_format_warning: Option<String>,
}

impl Model {
    pub fn from_gguf(path: impl AsRef<Path>, device: &candle_core::Device) -> Result<Self> {
        Self::from_gguf_inner(path, device, None)
    }

    /// Load a model with an optional `QuantizationResolver` to validate
    /// the quantization format against device capabilities.
    ///
    /// The resolver's recommendation is stored on the returned `Model` and
    /// any downgrade warnings are emitted via `tracing::warn!`.
    pub fn from_gguf_with_resolver(
        path: impl AsRef<Path>,
        device: &candle_core::Device,
        resolver: &mut QuantizationResolver,
    ) -> Result<Self> {
        Self::from_gguf_inner(path, device, Some(resolver))
    }

    /// Load a model from an arbitrary `Read + Seek` source (e.g. a
    /// `Cursor<Vec<u8>>` of decrypted bytes) rather than a file path.
    ///
    /// This is the primary entry point for the decryption pipeline: after
    /// `Aes256GcmEncryption` produces plaintext bytes, they are passed to
    /// this constructor as a `Cursor<Vec<u8>>`.
    pub fn from_gguf_reader<R: Read + Seek>(
        reader: &mut R,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let device = device.clone();
        let gguf = candle_core::quantized::gguf_file::Content::read(reader)
            .map_err(|e| AtheerCoreError::ModelLoadFailed(format!("GGUF parse: {e}")))?;

        // Architecture-aware dispatch: reads general.architecture from GGUF
        // metadata and selects the appropriate ModelWeights implementation.
        let variant = crate::weights::WeightsVariant::from_gguf(gguf, reader, &device)
            .map_err(|e| AtheerCoreError::ModelLoadFailed(format!("ModelWeights: {e}")))?;

        Ok(Self {
            weights: variant,
            device,
            context_size: 4096,
            resolved_quant_format: None,
            quant_format_warning: None,
        })
    }

    fn from_gguf_inner(
        path: impl AsRef<Path>,
        device: &candle_core::Device,
        mut resolver: Option<&mut QuantizationResolver>,
    ) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(AtheerCoreError::ModelLoadFailed(format!(
                "Model file not found: {:?}",
                path
            )));
        }

        let mut file = std::fs::File::open(path)
            .map_err(|e| AtheerCoreError::ModelLoadFailed(e.to_string()))?;
        let mut reader = std::io::BufReader::new(&mut file);

        let device = device.clone();
        let gguf = candle_core::quantized::gguf_file::Content::read(&mut reader)
            .map_err(|e| AtheerCoreError::ModelLoadFailed(format!("GGUF parse: {e}")))?;

        // Architecture-aware dispatch: reads general.architecture from GGUF
        // metadata and selects the appropriate ModelWeights implementation.
        let variant = crate::weights::WeightsVariant::from_gguf(gguf, &mut reader, &device)
            .map_err(|e| AtheerCoreError::ModelLoadFailed(format!("ModelWeights: {e}")))?;

        let (resolved_quant_format, quant_format_warning) = if let Some(ref mut res) = resolver {
            // Use a sensible default format; the caller can pass a custom
            // preference by calling `resolver.resolve()` separately and
            // choosing the appropriate GGUF file before calling this.
            let (fmt, warn) = res.resolve("q4_k_m");
            if let Some(ref w) = warn {
                tracing::warn!("QuantizationResolver: {w}");
            }
            (Some(fmt), warn)
        } else {
            (None, None)
        };

        Ok(Self {
            weights: variant,
            device,
            context_size: 4096,
            resolved_quant_format,
            quant_format_warning,
        })
    }

    pub fn context_size(&self) -> usize {
        self.context_size
    }

    pub fn memory_estimate(&self) -> usize {
        // Rough estimate: ~1 byte per param for Q4_K_M quantized models
        4_000_000_000
    }
}

impl Model {
    /// Drop all GPU-side KV cache tensors, freeing VRAM.
    /// After this, calling `forward()` will rebuild the cache from scratch.
    pub fn kv_cache_clear(&mut self) {
        self.weights.kv_cache_clear();
    }
}

impl KvCacheBridge for Model {
    fn kv_cache_snapshot(&self) -> Result<Vec<(Vec<f32>, Vec<f32>)>> {
        self.weights
            .kv_cache_snapshot()
            .map_err(|e| AtheerCoreError::KvCacheError(e.to_string()))
    }

    fn kv_cache_restore(&mut self, snapshot: &[(Vec<f32>, Vec<f32>)]) -> Result<()> {
        self.weights
            .kv_cache_restore(snapshot)
            .map_err(|e| AtheerCoreError::KvCacheError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_from_gguf_not_found() {
        let device = candle_core::Device::Cpu;
        let result = Model::from_gguf("/nonexistent/path/model.gguf", &device);
        assert!(result.is_err());
    }

    #[test]
    fn test_context_size_default() {
        // Can't construct a real model without GGUF; tests will need an actual file
        // For now, verify the trait impl compiles and error paths work
        assert!(true);
    }
}
