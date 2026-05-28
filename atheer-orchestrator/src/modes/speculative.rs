use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct SpeculativeDraft {
    pub tokens: Vec<u32>,
    pub log_probs: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct SpeculativeVerify {
    pub accepted: Vec<u32>,
    pub rejected: Vec<u32>,
    pub rejected_at: usize,
}

#[allow(dead_code)]
pub struct SpeculativeDecoder {
    max_draft_depth: usize,
    min_draft_depth: usize,
    draft_history: VecDeque<SpeculativeDraft>,
    verify_history: VecDeque<SpeculativeVerify>,
    acceptance_threshold: f32,
    current_draft: Option<SpeculativeDraft>,
}

impl SpeculativeDecoder {
    pub fn new(min_draft_depth: usize, max_draft_depth: usize) -> Self {
        Self {
            max_draft_depth,
            min_draft_depth,
            draft_history: VecDeque::with_capacity(50),
            verify_history: VecDeque::with_capacity(50),
            acceptance_threshold: 0.5,
            current_draft: None,
        }
    }

    pub fn set_acceptance_threshold(&mut self, threshold: f32) {
        self.acceptance_threshold = threshold;
    }

    pub fn start_draft(&mut self) -> usize {
        self.max_draft_depth
    }

    pub fn propose(&mut self, tokens: Vec<u32>, log_probs: Vec<f32>) {
        self.current_draft = Some(SpeculativeDraft { tokens, log_probs });
    }

    pub fn verify(&mut self, target_tokens: &[u32]) -> SpeculativeVerify {
        let draft = match self.current_draft.take() {
            Some(d) => d,
            None => {
                return SpeculativeVerify {
                    accepted: vec![],
                    rejected: vec![],
                    rejected_at: 0,
                }
            }
        };

        let mut accepted = Vec::new();
        let mut rejected = Vec::new();
        let mut rejected_at = draft.tokens.len();

        for (i, (draft_tok, target_tok)) in
            draft.tokens.iter().zip(target_tokens.iter()).enumerate()
        {
            if draft_tok == target_tok {
                accepted.push(*draft_tok);
            } else {
                rejected.push(*draft_tok);
                rejected_at = i;
                rejected.extend_from_slice(&draft.tokens[i + 1..]);
                break;
            }
        }

        if rejected_at == draft.tokens.len() && target_tokens.len() > draft.tokens.len() {
            accepted.extend_from_slice(&target_tokens[draft.tokens.len()..]);
        }

        let verify = SpeculativeVerify {
            accepted,
            rejected,
            rejected_at,
        };

        self.draft_history.push_back(draft);
        self.verify_history.push_back(verify.clone());

        if self.draft_history.len() > 50 {
            self.draft_history.pop_front();
        }
        if self.verify_history.len() > 50 {
            self.verify_history.pop_front();
        }

        verify
    }

    pub fn acceptance_rate(&self) -> f32 {
        if self.verify_history.is_empty() {
            return 1.0;
        }
        let total: usize = self.verify_history.iter().map(|v| v.accepted.len()).sum();
        let drafts: usize = self.draft_history.iter().map(|d| d.tokens.len()).sum();
        if drafts == 0 {
            return 1.0;
        }
        total as f32 / drafts as f32
    }

    pub fn adjust_depth(&mut self) {
        let rate = self.acceptance_rate();
        if rate > 0.85 && self.max_draft_depth < 16 {
            self.max_draft_depth = (self.max_draft_depth * 2).min(16);
        } else if rate < 0.4 && self.max_draft_depth > 2 {
            self.max_draft_depth = (self.max_draft_depth / 2).max(2);
        }
    }

    pub fn current_max_depth(&self) -> usize {
        self.max_draft_depth
    }

    pub fn recent_acceptance_rates(&self) -> Vec<f32> {
        self.verify_history
            .iter()
            .rev()
            .take(10)
            .map(|v| {
                let total = v.accepted.len() + v.rejected.len();
                if total == 0 {
                    1.0
                } else {
                    v.accepted.len() as f32 / total as f32
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_all_accepted() {
        let mut decoder = SpeculativeDecoder::new(1, 8);
        decoder.propose(vec![1, 2, 3, 4], vec![-0.1; 4]);

        let result = decoder.verify(&[1, 2, 3, 4]);

        assert_eq!(result.accepted, vec![1, 2, 3, 4]);
        assert!(result.rejected.is_empty());
        assert_eq!(result.rejected_at, 4);
    }

    #[test]
    fn test_verify_partial_rejection() {
        let mut decoder = SpeculativeDecoder::new(1, 8);
        decoder.propose(vec![1, 2, 3, 4], vec![-0.1; 4]);

        let result = decoder.verify(&[1, 2, 99, 100]);

        assert_eq!(result.accepted, vec![1, 2]);
        assert_eq!(result.rejected, vec![3, 4]);
        assert_eq!(result.rejected_at, 2);
    }

    #[test]
    fn test_verify_early_rejection() {
        let mut decoder = SpeculativeDecoder::new(1, 8);
        decoder.propose(vec![1, 2, 3, 4], vec![-0.1; 4]);

        let result = decoder.verify(&[99, 100]);

        assert!(result.accepted.is_empty());
        assert_eq!(result.rejected, vec![1, 2, 3, 4]);
        assert_eq!(result.rejected_at, 0);
    }

    #[test]
    fn test_verify_extra_target_tokens() {
        let mut decoder = SpeculativeDecoder::new(1, 8);
        decoder.propose(vec![1, 2], vec![-0.1; 2]);

        let result = decoder.verify(&[1, 2, 3, 4]);

        assert_eq!(result.accepted, vec![1, 2, 3, 4]);
        assert!(result.rejected.is_empty());
    }

    #[test]
    fn test_acceptance_rate() {
        let mut decoder = SpeculativeDecoder::new(1, 8);

        decoder.propose(vec![1, 2, 3, 4], vec![-0.1; 4]);
        decoder.verify(&[1, 2, 3, 4]);

        decoder.propose(vec![1, 2, 3], vec![-0.1; 3]);
        decoder.verify(&[1, 2]);

        assert!((decoder.acceptance_rate() - 0.857).abs() < 0.01);
    }

    #[test]
    fn test_depth_adjustment_up() {
        let mut decoder = SpeculativeDecoder::new(1, 4);

        for _ in 0..20 {
            decoder.propose(vec![1, 2, 3], vec![-0.1; 3]);
            decoder.verify(&[1, 2, 3]);
        }

        decoder.adjust_depth();
        assert_eq!(decoder.current_max_depth(), 8);
    }

    #[test]
    fn test_depth_adjustment_down() {
        let mut decoder = SpeculativeDecoder::new(1, 4);

        for _ in 0..20 {
            decoder.propose(vec![1, 2, 3, 4], vec![-0.1; 4]);
            decoder.verify(&[1]);
        }

        decoder.adjust_depth();
        assert_eq!(decoder.current_max_depth(), 2);
    }
}
