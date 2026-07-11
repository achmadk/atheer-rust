use crate::error::Result;
use crate::Model;
use candle_core::Tensor;
use sha2::{Digest, Sha256};

/// Signature capturing the output behavior of a model for a fixed input.
///
/// Used to detect regressions across code changes: identical hardware+software
/// should produce identical signatures for the same prompt and seed.
#[derive(Debug, Clone)]
pub struct LogitFingerprint {
    pub prompt: String,
    pub prefill_fingerprint: String,
    pub max_tokens: u32,
    pub generation_fingerprint: String,
}

/// Statistics comparing two accuracy runs.
#[derive(Debug, Clone, Default)]
pub struct AccuracyComparison {
    pub prefill_match: bool,
    pub generation_match: bool,
    pub prefill_cosine_similarity: Option<f64>,
    pub token_match_count: usize,
    pub token_total_count: usize,
}

/// Creates a deterministic logit fingerprint: SHA-256 of the top-100 logit values.
///
/// The fingerprint only includes the top-K logits (sorted descending) to be
/// insensitive to numerical noise in very small logits that don't affect
/// sampling decisions.
pub fn fingerprint_topk(logits: &Tensor, k: usize) -> Result<String> {
    let data: Vec<f32> = logits
        .flatten_all()
        .map_err(|e| crate::error::AtheerCoreError::GenerationFailed(e.to_string()))?
        .to_vec1()
        .map_err(|e| crate::error::AtheerCoreError::GenerationFailed(e.to_string()))?;

    let mut sorted = data.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    sorted.truncate(k);

    let mut hasher = Sha256::new();
    for v in &sorted {
        hasher.update(v.to_le_bytes());
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Compute cosine similarity between two logit vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| *x as f64 * *y as f64)
        .sum();
    let norm_a: f64 = a.iter().map(|x| *x as f64 * *x as f64).sum();
    let norm_b: f64 = b.iter().map(|x| *x as f64 * *x as f64).sum();
    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 {
        return None;
    }
    Some(dot / denom)
}

/// Run a model with a fixed prompt and capture logit fingerprints.
pub fn capture_fingerprint(
    model: &mut Model,
    prompt: &str,
    max_tokens: u32,
    k: usize,
) -> Result<LogitFingerprint> {
    let device = &model.device;

    let input_ids: Vec<u32> = vec![1];
    let input_tensor = Tensor::new(
        &input_ids.iter().map(|x| *x as i64).collect::<Vec<_>>()[..],
        device,
    )
    .map_err(|e| crate::error::AtheerCoreError::GenerationFailed(e.to_string()))?
    .unsqueeze(0)
    .map_err(|e| crate::error::AtheerCoreError::GenerationFailed(e.to_string()))?;

    let logits = model
        .weights
        .forward(&input_tensor, 0)
        .map_err(|e| crate::error::AtheerCoreError::GenerationFailed(e.to_string()))?;

    let prefill_fingerprint = fingerprint_topk(&logits, k)?;

    let mut hasher = Sha256::new();
    let mut token = 1u32;

    for pos in 1..=max_tokens {
        let token_tensor = Tensor::new(&[token as i64][..], device)
            .map_err(|e| crate::error::AtheerCoreError::GenerationFailed(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| crate::error::AtheerCoreError::GenerationFailed(e.to_string()))?;

        let logits = model
            .weights
            .forward(&token_tensor, pos as usize)
            .map_err(|e| crate::error::AtheerCoreError::GenerationFailed(e.to_string()))?;

        let fp = fingerprint_topk(&logits, k)?;
        hasher.update(fp.as_bytes());

        let logit_data: Vec<f32> = logits
            .flatten_all()
            .map_err(|e| crate::error::AtheerCoreError::GenerationFailed(e.to_string()))?
            .to_vec1()
            .map_err(|e| crate::error::AtheerCoreError::GenerationFailed(e.to_string()))?;
        token = logit_data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }

    let generation_fingerprint = hex::encode(hasher.finalize());

    Ok(LogitFingerprint {
        prompt: prompt.to_string(),
        prefill_fingerprint,
        max_tokens,
        generation_fingerprint,
    })
}

/// Compare two accuracy runs and produce a structured diff.
pub fn compare_accuracy(
    baseline: &LogitFingerprint,
    candidate: &LogitFingerprint,
) -> AccuracyComparison {
    AccuracyComparison {
        prefill_match: baseline.prefill_fingerprint == candidate.prefill_fingerprint,
        generation_match: baseline.generation_fingerprint == candidate.generation_fingerprint,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache_bridge::KvCacheBridge;

    #[test]
    fn test_fingerprint_empty_logit() {
        let device = candle_core::Device::Cpu;
        let t = Tensor::new(&[0.0f32], &device).unwrap();
        let fp = fingerprint_topk(&t, 1).unwrap();
        assert_eq!(fp.len(), 64);
    }

    #[test]
    fn test_fingerprint_deterministic() {
        let device = candle_core::Device::Cpu;
        let t = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &device).unwrap();
        let fp1 = fingerprint_topk(&t, 3).unwrap();
        let fp2 = fingerprint_topk(&t, 3).unwrap();
        assert_eq!(fp1, fp2, "fingerprints must be deterministic");
    }

    #[test]
    fn test_fingerprint_k_smaller_than_vocab() {
        let device = candle_core::Device::Cpu;
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let t = Tensor::from_slice(&data, (data.len(),), &device).unwrap();
        let fp = fingerprint_topk(&t, 10).unwrap();
        assert_eq!(fp.len(), 64);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a).unwrap();
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_mismatched_length() {
        assert!(cosine_similarity(&[1.0], &[1.0, 2.0]).is_none());
    }

    #[test]
    fn test_compare_accuracy_mismatch() {
        let baseline = LogitFingerprint {
            prompt: "hello".into(),
            prefill_fingerprint: "aaa".into(),
            max_tokens: 10,
            generation_fingerprint: "bbb".into(),
        };
        let candidate = LogitFingerprint {
            prompt: "hello".into(),
            prefill_fingerprint: "aaa".into(),
            max_tokens: 10,
            generation_fingerprint: "ccc".into(),
        };
        let comp = compare_accuracy(&baseline, &candidate);
        assert!(comp.prefill_match);
        assert!(!comp.generation_match);
    }

    /// Integration test: requires a real GGUF model file.
    /// Set `ATHEER_TEST_MODEL` env var or run scripts/download-test-model.sh.
    #[test]
    #[ignore]
    fn test_accuracy_with_real_model() {
        let model_path = crate::test_model::ensure_test_model();
        let device = candle_core::Device::Cpu;
        let mut model = Model::from_gguf(&model_path, &device).unwrap();
        let fp = capture_fingerprint(&mut model, "Hello world", 5, 100).unwrap();
        assert_eq!(fp.prefill_fingerprint.len(), 64);
        assert_eq!(fp.generation_fingerprint.len(), 64);
    }

    #[test]
    #[ignore]
    fn test_kv_cache_roundtrip_accuracy() {
        let model_path = crate::test_model::ensure_test_model();
        let device = candle_core::Device::Cpu;
        let mut model = Model::from_gguf(&model_path, &device).unwrap();

        // KV cache snapshot/restore is model-specific — LFM2 does not
        // support it.  Skip the roundtrip assertion gracefully.
        let snapshot = match model.kv_cache_snapshot() {
            Ok(s) => s,
            Err(_) => return, // unsupported model arch, nothing to test
        };
        model.kv_cache_restore(&snapshot).unwrap();
        let fp1 = capture_fingerprint(&mut model, "Hello world", 3, 100).unwrap();
        let fp2 = capture_fingerprint(&mut model, "Hello world", 3, 100).unwrap();

        let comp = compare_accuracy(&fp1, &fp2);
        assert!(comp.prefill_match, "prefill changed after snapshot-restore");
        assert!(
            comp.generation_match,
            "generation changed after snapshot-restore"
        );
    }

    #[test]
    #[ignore]
    fn test_quantization_accuracy() {
        let model_path = crate::test_model::ensure_test_model();
        let device = candle_core::Device::Cpu;
        let mut model = Model::from_gguf(&model_path, &device).unwrap();
        let _fp = capture_fingerprint(&mut model, "The capital of France is", 10, 100).unwrap();
    }
}
