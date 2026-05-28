use std::collections::VecDeque;

use super::speculative::{SpeculativeDecoder, SpeculativeVerify};

pub struct TurboMode {
    draft_enabled: bool,
    speculation_depth: usize,
    min_speculation_depth: usize,
    max_speculation_depth: usize,
    acceptance_history: VecDeque<f32>,
    draft_model_loaded: bool,
    speculative_decoder: SpeculativeDecoder,
}

impl TurboMode {
    pub fn new() -> Self {
        Self {
            draft_enabled: true,
            speculation_depth: 4,
            min_speculation_depth: 1,
            max_speculation_depth: 8,
            acceptance_history: VecDeque::with_capacity(20),
            draft_model_loaded: false,
            speculative_decoder: SpeculativeDecoder::new(1, 8),
        }
    }

    pub fn speculation_depth(&self) -> usize {
        self.speculation_depth
    }

    pub fn draft_enabled(&self) -> bool {
        self.draft_enabled && self.draft_model_loaded
    }

    pub fn set_draft_enabled(&mut self, enabled: bool) {
        self.draft_enabled = enabled;
    }

    pub fn set_draft_model_loaded(&mut self, loaded: bool) {
        self.draft_model_loaded = loaded;
    }

    pub fn is_draft_loaded(&self) -> bool {
        self.draft_model_loaded
    }

    pub fn record_acceptance(&mut self, accepted: usize, total: usize) {
        if total == 0 {
            return;
        }
        let rate = accepted as f32 / total as f32;
        self.acceptance_history.push_back(rate);
        if self.acceptance_history.len() > 20 {
            self.acceptance_history.pop_front();
        }
        self.adjust_speculation_depth();
        self.speculative_decoder.adjust_depth();
    }

    fn adjust_speculation_depth(&mut self) {
        if self.acceptance_history.len() < 5 {
            return;
        }

        let avg_acceptance: f32 =
            self.acceptance_history.iter().sum::<f32>() / self.acceptance_history.len() as f32;

        if avg_acceptance > 0.8 && self.speculation_depth < self.max_speculation_depth {
            self.speculation_depth = (self.speculation_depth * 2).min(self.max_speculation_depth);
        } else if avg_acceptance < 0.5 && self.speculation_depth > self.min_speculation_depth {
            self.speculation_depth = (self.speculation_depth / 2).max(self.min_speculation_depth);
        }
    }

    pub fn avg_acceptance_rate(&self) -> f32 {
        if self.acceptance_history.is_empty() {
            return 0.0;
        }
        self.acceptance_history.iter().sum::<f32>() / self.acceptance_history.len() as f32
    }

    pub fn start_speculation(&mut self) -> usize {
        self.speculative_decoder.start_draft()
    }

    pub fn propose_speculative(&mut self, tokens: Vec<u32>, log_probs: Vec<f32>) {
        self.speculative_decoder.propose(tokens, log_probs);
    }

    pub fn verify_speculative(&mut self, target_tokens: &[u32]) -> SpeculativeVerify {
        let result = self.speculative_decoder.verify(target_tokens);
        if !result.accepted.is_empty() || !result.rejected.is_empty() {
            self.record_acceptance(
                result.accepted.len(),
                result.accepted.len() + result.rejected.len(),
            );
        }
        result
    }

    pub fn speculative_depth(&self) -> usize {
        self.speculative_decoder.current_max_depth()
    }
}

impl Default for TurboMode {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turbo_mode_defaults() {
        let mut mode = TurboMode::new();
        assert!(!mode.draft_enabled());
        mode.set_draft_model_loaded(true);
        assert!(mode.draft_enabled());
        assert_eq!(mode.speculation_depth(), 4);
    }

    #[test]
    fn test_draft_loading() {
        let mut mode = TurboMode::new();
        assert!(!mode.is_draft_loaded());

        mode.set_draft_model_loaded(true);
        assert!(mode.is_draft_loaded());
        assert!(mode.draft_enabled());
    }

    #[test]
    fn test_acceptance_tracking() {
        let mut mode = TurboMode::new();

        mode.record_acceptance(8, 10);
        mode.record_acceptance(7, 10);
        mode.record_acceptance(9, 10);

        assert!(mode.acceptance_history.len() == 3);
        assert!((mode.avg_acceptance_rate() - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_speculation_depth_adjustment() {
        let mut mode = TurboMode::new();
        mode.speculation_depth = 4;

        for _ in 0..10 {
            mode.record_acceptance(9, 10);
        }

        assert_eq!(mode.speculation_depth, 8);
    }

    #[test]
    fn test_speculation_depth_decrease() {
        let mut mode = TurboMode::new();
        mode.speculation_depth = 4;

        for _ in 0..5 {
            mode.record_acceptance(3, 10);
        }
        assert_eq!(mode.speculation_depth, 2);
    }

    #[test]
    fn test_speculative_decoder_integration() {
        let mut mode = TurboMode::new();

        mode.start_speculation();
        mode.propose_speculative(vec![1, 2, 3], vec![-0.1, -0.2, -0.3]);
        let result = mode.verify_speculative(&[1, 2, 3, 4]);

        assert_eq!(result.accepted, vec![1, 2, 3, 4]);
        assert!(result.rejected.is_empty());
    }

    #[test]
    fn test_speculative_rejection_tracking() {
        let mut mode = TurboMode::new();

        mode.start_speculation();
        mode.propose_speculative(vec![1, 2, 3, 4], vec![-0.1; 4]);
        let result = mode.verify_speculative(&[1, 99]);

        assert_eq!(result.accepted, vec![1]);
        assert_eq!(result.rejected, vec![2, 3, 4]);
        assert_eq!(result.rejected_at, 1);
    }
}
