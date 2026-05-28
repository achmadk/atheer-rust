use std::collections::VecDeque;

pub struct BalancedMode {
    self_speculative: bool,
    speculation_depth: usize,
    consistency_buffer: VecDeque<f32>,
    consistency_threshold: f32,
    last_logits: Option<Vec<f32>>,
}

impl BalancedMode {
    pub fn new() -> Self {
        Self {
            self_speculative: true,
            speculation_depth: 2,
            consistency_buffer: VecDeque::with_capacity(5),
            consistency_threshold: 0.85,
            last_logits: None,
        }
    }

    pub fn speculation_depth(&self) -> usize {
        self.speculation_depth
    }

    pub fn set_speculation_depth(&mut self, depth: usize) {
        self.speculation_depth = depth.clamp(1, 4);
    }

    pub fn is_self_speculative(&self) -> bool {
        self.self_speculative
    }

    pub fn set_self_speculative(&mut self, enabled: bool) {
        self.self_speculative = enabled;
    }

    pub fn update_logits(&mut self, logits: Vec<f32>) {
        if let Some(ref last) = self.last_logits {
            let similarity = self.cosine_similarity(last, &logits);
            self.consistency_buffer.push_back(similarity);
            if self.consistency_buffer.len() > 5 {
                self.consistency_buffer.pop_front();
            }

            if self.average_consistency() > self.consistency_threshold {
                self.speculation_depth = (self.speculation_depth + 1).min(4);
            }
        }
        self.last_logits = Some(logits);
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a = (a.iter().map(|x| x * x).sum::<f32>()).sqrt();
        let mag_b = (b.iter().map(|x| x * x).sum::<f32>()).sqrt();

        if mag_a == 0.0 || mag_b == 0.0 {
            return 0.0;
        }

        dot_product / (mag_a * mag_b)
    }

    pub fn average_consistency(&self) -> f32 {
        if self.consistency_buffer.is_empty() {
            return 1.0;
        }
        let sum: f32 = self.consistency_buffer.iter().sum();
        sum / self.consistency_buffer.len() as f32
    }

    pub fn should_use_self_speculation(&self) -> bool {
        if !self.self_speculative {
            return false;
        }
        self.consistency_buffer.len() >= 3
            && self.average_consistency() > self.consistency_threshold
    }

    pub fn reset(&mut self) {
        self.consistency_buffer.clear();
        self.last_logits = None;
    }
}

impl Default for BalancedMode {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_balanced_mode_defaults() {
        let mode = BalancedMode::new();
        assert!(mode.speculation_depth() <= 2);
        assert!(mode.is_self_speculative());
    }

    #[test]
    fn test_consistency_tracking() {
        let mut mode = BalancedMode::new();

        mode.update_logits(vec![0.1, 0.2, 0.3]);
        mode.update_logits(vec![0.1, 0.2, 0.3]);
        mode.update_logits(vec![0.1, 0.2, 0.3]);

        assert!(mode.consistency_buffer.len() == 2);
        assert!((mode.average_consistency() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_speculation_depth_adjustment() {
        let mut mode = BalancedMode::new();
        mode.speculation_depth = 2;

        for _ in 0..5 {
            mode.update_logits(vec![0.1, 0.2, 0.3]);
        }

        assert!(mode.speculation_depth >= 2);
    }

    #[test]
    fn test_self_speculation_decision() {
        let mut mode = BalancedMode::new();

        for _ in 0..5 {
            mode.update_logits(vec![0.1, 0.2, 0.3]);
        }

        assert!(mode.should_use_self_speculation());
    }
}
