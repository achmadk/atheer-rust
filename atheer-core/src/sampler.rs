use candle_core::{Result as CandleResult, Tensor};
use std::collections::HashMap;

/// Sampling strategy configuration.
#[derive(Debug, Clone, Copy)]
pub struct SamplingConfig {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub seed: u64,
    /// Repetition penalty (range [1.0, 2.0]).
    /// Applied as: logits[t] /= repetition_penalty ^ count[t].
    /// Default 1.0 (disabled).
    pub repetition_penalty: f32,
    /// Frequency penalty (range [-2.0, 2.0]).
    /// Applied as: logits[t] -= frequency_penalty * count[t].
    /// Default 0.0 (disabled).
    pub frequency_penalty: f32,
    /// Presence penalty (range [-2.0, 2.0]).
    /// Applied as: logits[t] -= presence_penalty * (count[t] > 0).
    /// Default 0.0 (disabled).
    pub presence_penalty: f32,
    /// Min-p sampling threshold (range [0.0, 1.0]).
    /// Filters tokens with probability < max_prob * min_p.
    /// Default None (disabled).
    pub min_p: Option<f64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 40,
            seed: 42,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            min_p: None,
        }
    }
}

/// Trait for sampling a single token from logits.
///
/// `generated_tokens` contains the tokens produced so far in the current
/// generation pass, including the prompt tokens for repetition-aware
/// sampling strategies (repetition penalty, frequency/presence penalty).
pub trait Sampler: Send + Sync {
    fn sample(&mut self, logits: &Tensor, generated_tokens: &[u32]) -> CandleResult<u32>;
}

/// Default sampler supporting temperature, top-k, top-p, repetition penalty,
/// frequency/presence penalties, and min-p sampling.
pub struct DefaultSampler {
    config: SamplingConfig,
    rng: rand::rngs::StdRng,
}

impl DefaultSampler {
    pub fn new(config: SamplingConfig) -> Self {
        use rand::SeedableRng;
        Self {
            config,
            rng: rand::rngs::StdRng::seed_from_u64(config.seed),
        }
    }

    /// Count token frequencies in the generated_tokens slice.
    fn count_token_frequencies(tokens: &[u32]) -> HashMap<u32, usize> {
        let mut counts: HashMap<u32, usize> = HashMap::new();
        for &token in tokens {
            *counts.entry(token).or_insert(0) += 1;
        }
        counts
    }
}

impl Sampler for DefaultSampler {
    fn sample(&mut self, logits: &Tensor, generated_tokens: &[u32]) -> CandleResult<u32> {
        use candle_core::DType;
        use rand::distributions::{Distribution, WeightedIndex};

        let logits = logits.to_dtype(DType::F32)?;
        let mut logits_vec: Vec<f32> = logits.to_vec1()?;
        let vocab_size = logits_vec.len();

        // ── Pre-softmax penalties ──────────────────────────────────────────

        let has_penalties = self.config.repetition_penalty != 1.0
            || self.config.frequency_penalty != 0.0
            || self.config.presence_penalty != 0.0;

        if has_penalties && !generated_tokens.is_empty() {
            let counts = Self::count_token_frequencies(generated_tokens);

            let rp = self.config.repetition_penalty;
            let fp = self.config.frequency_penalty;
            let pp = self.config.presence_penalty;

            for t in 0..vocab_size {
                if let Some(&count) = counts.get(&(t as u32)) {
                    let count_f = count as f32;

                    // Repetition penalty: divide by rp^count
                    if rp != 1.0 && count > 0 {
                        logits_vec[t] /= rp.powi(count as i32);
                    }

                    // Frequency penalty: subtract fp * count
                    if fp != 0.0 {
                        logits_vec[t] -= fp * count_f;
                    }

                    // Presence penalty: subtract pp if count > 0
                    if pp != 0.0 {
                        logits_vec[t] -= pp;
                    }
                }
            }
        }

        // ── Temperature scaling ────────────────────────────────────────────

        if self.config.temperature != 1.0 && self.config.temperature > 0.0 {
            let inv_temp = 1.0 / self.config.temperature as f32;
            for v in &mut logits_vec {
                *v *= inv_temp;
            }
        }

        // ── Softmax ────────────────────────────────────────────────────────

        let max_logit = logits_vec
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in &mut logits_vec {
            *v = (*v - max_logit).exp();
            sum += *v;
        }
        for v in &mut logits_vec {
            *v /= sum;
        }

        // ── Min-p filtering (adaptive threshold) ───────────────────────────

        if let Some(min_p) = self.config.min_p {
            if min_p > 0.0 {
                let p_max = logits_vec
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let threshold = p_max * min_p as f32;
                for v in logits_vec.iter_mut() {
                    if *v < threshold {
                        *v = 0.0;
                    }
                }
            }
        }

        // ── Top-k filtering ────────────────────────────────────────────────

        if self.config.top_k > 0 && self.config.top_k < vocab_size {
            let mut indexed: Vec<(usize, f32)> =
                logits_vec.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let threshold = indexed[self.config.top_k - 1].1;
            for v in logits_vec.iter_mut() {
                if *v < threshold {
                    *v = 0.0;
                }
            }
        }

        // ── Top-p (nucleus) filtering ──────────────────────────────────────

        if self.config.top_p < 1.0 {
            let mut indexed: Vec<(usize, f32)> =
                logits_vec.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let mut cumsum = 0.0f32;
            let mut cutoff_idx = indexed.len();
            for (idx, (_, prob)) in indexed.iter().enumerate() {
                cumsum += prob;
                if cumsum > self.config.top_p as f32 {
                    cutoff_idx = idx + 1;
                    break;
                }
            }
            let threshold = indexed[cutoff_idx.saturating_sub(1)].1;
            for v in logits_vec.iter_mut() {
                if *v < threshold {
                    *v = 0.0;
                }
            }
        }

        // ── Re-normalize after filtering ───────────────────────────────────

        let sum: f32 = logits_vec.iter().sum();
        if sum > 0.0 {
            for v in &mut logits_vec {
                *v /= sum;
            }
        }

        let dist =
            WeightedIndex::new(&logits_vec).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let token_id = dist.sample(&mut self.rng);
        Ok(token_id as u32)
    }
}

/// Greedy sampler (always picks the highest logit).
pub struct GreedySampler;

impl Sampler for GreedySampler {
    fn sample(&mut self, logits: &Tensor, _generated_tokens: &[u32]) -> CandleResult<u32> {
        use candle_core::DType;
        let logits_f32 = logits.to_dtype(DType::F32)?;
        let logits_vec: Vec<f32> = logits_f32.to_vec1()?;
        let max_idx = logits_vec
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        Ok(max_idx as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn make_test_logits() -> Tensor {
        // Create logits where token 0 is most likely
        let values: Vec<f32> = (0..100).map(|i| (100 - i) as f32).collect();
        Tensor::from_slice(&values, values.len(), &Device::Cpu).unwrap()
    }

    #[test]
    fn test_default_sampler_produces_token() {
        let mut sampler = DefaultSampler::new(SamplingConfig::default());
        let logits = make_test_logits();
        let token = sampler.sample(&logits, &[]).unwrap();
        assert!(token < 100);
    }

    #[test]
    fn test_greedy_sampler_picks_highest() {
        let mut sampler = GreedySampler;
        let logits = make_test_logits();
        let token = sampler.sample(&logits, &[]).unwrap();
        assert_eq!(token, 0); // token 0 has the highest logit (100.0)
    }

    #[test]
    fn test_repetition_penalty_discourages_repeated() {
        let mut config = SamplingConfig::default();
        config.repetition_penalty = 2.0;
        config.temperature = 1.0;
        config.top_k = 0;
        config.top_p = 1.0;

        let mut sampler = DefaultSampler::new(config);
        let logits = make_test_logits();

        // Token 0 was repeated many times — penalty should make it unlikely
        let repeated: Vec<u32> = vec![0; 50];
        let token = sampler.sample(&logits, &repeated).unwrap();
        // Token 0 should NOT be selected due to heavy penalty
        assert_ne!(token, 0, "Repetition penalty should discourage token 0");
    }

    #[test]
    fn test_frequency_penalty_linear_scaling() {
        let mut config = SamplingConfig::default();
        config.frequency_penalty = 2.0;
        config.temperature = 1.0;
        config.top_k = 0;
        config.top_p = 1.0;

        let mut sampler = DefaultSampler::new(config);
        let logits = make_test_logits();
        let repeated: Vec<u32> = vec![0; 10]; // token 0 appears 10 times

        let token = sampler.sample(&logits, &repeated).unwrap();
        assert_ne!(token, 0, "Frequency penalty should discourage token 0");
    }

    #[test]
    fn test_min_p_does_not_filter_all() {
        let mut config = SamplingConfig::default();
        config.min_p = Some(0.1);
        config.top_k = 0;
        config.top_p = 1.0;

        let mut sampler = DefaultSampler::new(config);
        // Create logits with one very dominant token
        let mut values = vec![0.0f32; 100];
        values[0] = 100.0; // dominant
        values[1] = 1.0; // second best
        let logits = Tensor::from_slice(&values, values.len(), &Device::Cpu).unwrap();

        let token = sampler.sample(&logits, &[]).unwrap();
        assert!(token < 100);
    }

    #[test]
    fn test_defaults_config() {
        let config = SamplingConfig::default();
        assert_eq!(config.repetition_penalty, 1.0);
        assert_eq!(config.frequency_penalty, 0.0);
        assert_eq!(config.presence_penalty, 0.0);
        assert!(config.min_p.is_none());
    }

    #[test]
    fn test_repetition_penalty_one_is_noop() {
        let config = SamplingConfig::default();
        assert_eq!(config.repetition_penalty, 1.0);
        // At 1.0, the penalty should not change the logits
        let mut sampler = DefaultSampler::new(config);
        let logits = make_test_logits();
        // Should not crash with repeated tokens
        let token = sampler.sample(&logits, &[0, 0, 0]).unwrap();
        assert!(token < 100);
    }

    #[test]
    fn test_empty_generated_tokens_no_crash() {
        let mut sampler = DefaultSampler::new(SamplingConfig::default());
        let logits = make_test_logits();
        let token = sampler.sample(&logits, &[]).unwrap();
        assert!(token < 100);
    }
}
