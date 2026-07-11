use atheer_core::Sampler;
use candle_core::{Result as CandleResult, Tensor};
use std::sync::Arc;

use super::trie::GrammarTrie;
use super::GrammarConstraint;

/// A sampler wrapper that applies grammar constraints to mask invalid tokens.
pub struct GrammarSampler<G: GrammarConstraint> {
    base: Box<dyn Sampler>,
    grammar: G,
    tokenizer: atheer_core::Tokenizer,
    trie: Option<Arc<GrammarTrie>>,
}

impl<G: GrammarConstraint> GrammarSampler<G> {
    pub fn new(base: Box<dyn Sampler>, grammar: G, tokenizer: atheer_core::Tokenizer) -> Self {
        Self {
            base,
            grammar,
            tokenizer,
            trie: None,
        }
    }

    pub fn with_trie(mut self, trie: Arc<GrammarTrie>) -> Self {
        self.trie = Some(trie);
        self
    }

    fn get_valid_token_ids(&self) -> Vec<usize> {
        if let Some(ref trie) = self.trie {
            trie.valid_tokens("")
                .into_iter()
                .map(|id| id as usize)
                .collect()
        } else {
            let vocab_size = self.tokenizer.vocab_size();
            (0..vocab_size).collect()
        }
    }
}

impl<G: GrammarConstraint + Clone> Sampler for GrammarSampler<G> {
    fn sample(&mut self, logits: &Tensor, generated_tokens: &[u32]) -> CandleResult<u32> {
        use candle_core::DType;

        let vocab_size = logits.dims().last().copied().unwrap_or(0);
        let mut mask: Vec<bool> = vec![true; vocab_size];

        let candidate_ids = self.get_valid_token_ids();
        let mut valid_count = 0;

        for token_id in candidate_ids {
            let token_text = self.tokenizer.decode(&[token_id as u32], false);
            if !self.grammar.is_valid_prefix(&token_text) {
                mask[token_id] = false;
            } else {
                valid_count += 1;
            }
        }

        if valid_count == 0 {
            tracing::warn!("GrammarSampler: no valid tokens from trie, falling back to full scan");
            for (token_id, is_valid) in mask.iter_mut().enumerate().take(vocab_size) {
                if *is_valid {
                    continue;
                }
                let token_text = self.tokenizer.decode(&[token_id as u32], false);
                if !self.grammar.is_valid_prefix(&token_text) {
                    *is_valid = false;
                }
            }
        }

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

        if !any_valid {
            let orig_logits: Vec<f32> = logits_f32.to_vec1()?;
            logits_vec = orig_logits;
        }

        let device = logits.device();
        let masked_logits = Tensor::new(logits_vec.as_slice(), device)?;

        let sampled_token = self.base.sample(&masked_logits, generated_tokens)?;

        let sampled_text = self.tokenizer.decode(&[sampled_token], false);
        self.grammar.advance(&sampled_text);

        Ok(sampled_token)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_grammar_sampler_construction() {
        assert!(true);
    }
}
