use std::collections::VecDeque;

/// Configurable latency budget for P99-aware generation.
///
/// Each `LatencyBudget` holds three thresholds:
/// * `prefill_budget_ms` — max wall time allowed for the initial prompt forward pass.
/// * `decode_budget_ms` — max wall time allowed for a single decode step.
/// * `p99_target_ms`   — desired P99 per-step latency across the whole generation.
#[derive(Debug, Clone)]
pub struct LatencyBudget {
    pub prefill_budget_ms: f64,
    pub decode_budget_ms: f64,
    pub p99_target_ms: f64,
}

impl Default for LatencyBudget {
    /// Sensible defaults for a 3B-class model on a flagship mobile SoC.
    fn default() -> Self {
        Self {
            prefill_budget_ms: 500.0,
            decode_budget_ms: 100.0,
            p99_target_ms: 150.0,
        }
    }
}

impl LatencyBudget {
    pub fn new(prefill_budget_ms: f64, decode_budget_ms: f64, p99_target_ms: f64) -> Self {
        Self {
            prefill_budget_ms,
            decode_budget_ms,
            p99_target_ms,
        }
    }
}

/// Tracks per-step latencies and maintains a running P99 estimate.
pub struct LatencyTracker {
    /// Recent decode step durations (ms), for P99 estimation.
    recent_steps: VecDeque<f64>,
    max_history: usize,

    /// Running aggregate.
    pub total_decode_steps: u64,
    pub total_decode_ms: f64,
    pub prefill_ms: f64,

    /// Whether the last step exceeded the decode budget.
    pub last_step_exceeded_budget: bool,

    /// Whether any step so far exceeded the budget.
    pub any_step_exceeded_budget: bool,
}

impl LatencyTracker {
    pub fn new(max_history: usize) -> Self {
        Self {
            recent_steps: VecDeque::with_capacity(max_history + 1),
            max_history,
            total_decode_steps: 0,
            total_decode_ms: 0.0,
            prefill_ms: 0.0,
            last_step_exceeded_budget: false,
            any_step_exceeded_budget: false,
        }
    }

    /// Record the prefill duration.
    pub fn record_prefill(&mut self, ms: f64) {
        self.prefill_ms = ms;
    }

    /// Record a single decode step duration.
    /// `budget` is the per-step decode budget to compare against.
    pub fn record_decode_step(&mut self, ms: f64, budget_ms: f64) {
        self.total_decode_steps += 1;
        self.total_decode_ms += ms;
        self.recent_steps.push_back(ms);
        if self.recent_steps.len() > self.max_history {
            self.recent_steps.pop_front();
        }
        self.last_step_exceeded_budget = ms > budget_ms;
        if self.last_step_exceeded_budget {
            self.any_step_exceeded_budget = true;
        }
    }

    /// Average decode step latency (ms).
    pub fn avg_decode_ms(&self) -> f64 {
        if self.total_decode_steps == 0 {
            return 0.0;
        }
        self.total_decode_ms / self.total_decode_steps as f64
    }

    /// Running P99 latency estimate over the recent window.
    ///
    /// Returns `None` if fewer than 10 samples have been collected.
    pub fn p99_estimate_ms(&self) -> Option<f64> {
        if self.recent_steps.len() < 10 {
            return None;
        }
        let mut sorted: Vec<f64> = self.recent_steps.iter().copied().collect();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((sorted.len() as f64) * 0.99) as usize;
        let idx = idx.min(sorted.len() - 1);
        Some(sorted[idx])
    }

    /// Current P50 (median) latency estimate.
    pub fn p50_estimate_ms(&self) -> Option<f64> {
        if self.recent_steps.is_empty() {
            return None;
        }
        let mut sorted: Vec<f64> = self.recent_steps.iter().copied().collect();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Some(sorted[sorted.len() / 2])
    }

    /// Reset all tracking state.
    pub fn reset(&mut self) {
        self.recent_steps.clear();
        self.total_decode_steps = 0;
        self.total_decode_ms = 0.0;
        self.prefill_ms = 0.0;
        self.last_step_exceeded_budget = false;
        self.any_step_exceeded_budget = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_budget() {
        let budget = LatencyBudget::default();
        assert!((budget.prefill_budget_ms - 500.0).abs() < 0.001);
        assert!((budget.decode_budget_ms - 100.0).abs() < 0.001);
        assert!((budget.p99_target_ms - 150.0).abs() < 0.001);
    }

    #[test]
    fn test_custom_budget() {
        let budget = LatencyBudget::new(1000.0, 50.0, 80.0);
        assert!((budget.prefill_budget_ms - 1000.0).abs() < 0.001);
        assert!((budget.decode_budget_ms - 50.0).abs() < 0.001);
        assert!((budget.p99_target_ms - 80.0).abs() < 0.001);
    }

    #[test]
    fn test_empty_tracker() {
        let tracker = LatencyTracker::new(100);
        assert!((tracker.avg_decode_ms() - 0.0).abs() < 0.001);
        assert!(tracker.p99_estimate_ms().is_none());
        assert_eq!(tracker.total_decode_steps, 0);
    }

    #[test]
    fn test_record_prefill() {
        let mut tracker = LatencyTracker::new(100);
        tracker.record_prefill(42.5);
        assert!((tracker.prefill_ms - 42.5).abs() < 0.001);
    }

    #[test]
    fn test_record_decode_steps() {
        let mut tracker = LatencyTracker::new(100);
        tracker.record_decode_step(10.0, 100.0);
        tracker.record_decode_step(20.0, 100.0);
        tracker.record_decode_step(30.0, 100.0);
        assert_eq!(tracker.total_decode_steps, 3);
        assert!((tracker.total_decode_ms - 60.0).abs() < 0.001);
        assert!((tracker.avg_decode_ms() - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_budget_exceeded() {
        let mut tracker = LatencyTracker::new(100);
        tracker.record_decode_step(50.0, 100.0);
        assert!(!tracker.last_step_exceeded_budget);
        assert!(!tracker.any_step_exceeded_budget);

        tracker.record_decode_step(150.0, 100.0);
        assert!(tracker.last_step_exceeded_budget);
        assert!(tracker.any_step_exceeded_budget);
    }

    #[test]
    fn test_p99_estimate_needs_10_samples() {
        let mut tracker = LatencyTracker::new(100);
        for i in 1..9 {
            tracker.record_decode_step(i as f64, 100.0);
        }
        assert!(tracker.p99_estimate_ms().is_none());
    }

    #[test]
    fn test_p99_estimate() {
        let mut tracker = LatencyTracker::new(100);
        // 100 samples: 99 at 10ms, 1 at 500ms
        for _ in 0..99 {
            tracker.record_decode_step(10.0, 100.0);
        }
        tracker.record_decode_step(500.0, 100.0);
        let p99 = tracker.p99_estimate_ms().unwrap();
        // P99 should capture the tail: 500ms
        assert!((p99 - 500.0).abs() < 0.001);
    }

    #[test]
    fn test_p50_estimate() {
        let mut tracker = LatencyTracker::new(100);
        for i in 1..=11 {
            tracker.record_decode_step(i as f64, 100.0);
        }
        let p50 = tracker.p50_estimate_ms().unwrap();
        // Median of [1..11] is 6
        assert!((p50 - 6.0).abs() < 0.001);
    }

    #[test]
    fn test_reset() {
        let mut tracker = LatencyTracker::new(100);
        tracker.record_prefill(50.0);
        tracker.record_decode_step(10.0, 100.0);
        tracker.record_decode_step(200.0, 100.0);
        assert!(tracker.any_step_exceeded_budget);

        tracker.reset();
        assert_eq!(tracker.total_decode_steps, 0);
        assert!(!tracker.any_step_exceeded_budget);
        assert!(tracker.p99_estimate_ms().is_none());
    }

    #[test]
    fn test_sliding_window_eviction() {
        let mut tracker = LatencyTracker::new(5);
        for i in 0..10 {
            tracker.record_decode_step(i as f64, 100.0);
        }
        // Only last 5 should be in the window
        assert_eq!(tracker.recent_steps.len(), 5);
    }
}
