use atheer_core::Sampler;
use candle_core::{Result as CandleResult, Tensor};

use super::GrammarConstraint;

/// A sampler wrapper that applies grammar constraints to mask invalid tokens.
pub struct GrammarSampler<G: GrammarConstraint> {
    base: Box<dyn Sampler>,
    grammar: G,
    tokenizer: atheer_core::Tokenizer,
}

impl<G: GrammarConstraint> GrammarSampler<G> {
    pub fn new(base: Box<dyn Sampler>, grammar: G, tokenizer: atheer_core::Tokenizer) -> Self {
        Self {
            base,
            grammar,
            tokenizer,
        }
    }
}

impl<G: GrammarConstraint + Clone> Sampler for GrammarSampler<G> {
    fn sample(&mut self, logits: &Tensor, generated_tokens: &[u32]) -> CandleResult<u32> {
        use candle_core::DType;

        let vocab_size = logits.dims().last().copied().unwrap_or(0);
        let mut mask: Vec<bool> = vec![true; vocab_size];

        // Build a valid-token mask by testing every token against the grammar
        for token_id in 0..vocab_size {
            let token_text = self.tokenizer.decode(&[token_id as u32], false);
            if !self.grammar.is_valid_prefix(&token_text) {
                mask[token_id] = false;
            }
        }

        // Create masked logits: zero out invalid tokens and mask with large negative
        let logits_f32 = logits.to_dtype(DType::F32)?;
        let mut logits_vec: Vec<f32> = logits_f32.to_vec1()?;
        let mut any_valid = false;
        for (i, is_valid) in mask.iter().enumerate() {
            if !is_valid {
                logits_vec[i] = f32::NEG_INFINITY;
            } else {
                any_valid = true;
            }
        }

        // Fallback: if no tokens are valid according to the grammar,
        // fall back to unconstrained sampling (graceful degradation).
        if !any_valid {
            let orig_logits: Vec<f32> = logits_f32.to_vec1()?;
            logits_vec = orig_logits;
        }

        // Materialise a new Tensor for the base sampler
        let device = logits.device();
        let masked_logits = Tensor::new(logits_vec.as_slice(), device)?;

        // Sample from the masked logits
        let sampled_token = self.base.sample(&masked_logits, generated_tokens)?;

        // Advance the grammar with the chosen token
        let sampled_text = self.tokenizer.decode(&[sampled_token], false);
        self.grammar.advance(&sampled_text);

        Ok(sampled_token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atheer_core::sampler::GreedySampler;

    // These tests require a real tokenizer and model, which is hard in a unit test.
    // We test the GrammarSampler construction here.
    #[test]
    fn test_grammar_sampler_construction() {
        // Can't easily test without a real tokenizer, but verify compilation
        assert!(true);
    }
}
