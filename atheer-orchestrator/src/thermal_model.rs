/// Predictive thermal scheduler for edge-device inference.
///
/// Maintains a moving-average window of temperature samples, computes
/// a least-squares trend over the last N points, and surfaces a
/// recommended action (Stable / Rising / Falling) plus a predicted
/// temperature for the next sampling interval.
///
/// The model drives pre-emptive mode downgrades so the orchestrator
/// can react *before* the device hits the hard thermal throttle.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// ThermalTrend
// ---------------------------------------------------------------------------

/// Classified temperature trend direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThermalTrend {
    /// Temperature is stable or falling — no action needed.
    Stable,
    /// Temperature is rising — may warrant a pre-emptive downgrade.
    Rising,
    /// Temperature is falling — can consider upgrading again.
    Falling,
}

impl ThermalTrend {
    /// True if the trend suggests the device is heating up.
    pub fn is_concerning(&self) -> bool {
        matches!(self, Self::Rising)
    }
}

// ---------------------------------------------------------------------------
// ThermalModel
// ---------------------------------------------------------------------------

/// Simple predictive thermal model.
///
/// Samples are pushed via [`feed`](Self::feed). Call [`analyze`](Self::analyze)
/// to retrieve the current trend, raw slope, and predicted next temperature.
pub struct ThermalModel {
    /// Sliding window of raw temperature samples (most recent first is easiest
    /// but we use oldest-first because we convert to Vec for regression).
    samples: VecDeque<f32>,
    /// Maximum number of samples retained.
    window_size: usize,
    /// Number of most-recent points used for slope calculation (≤ window_size).
    trend_window: usize,
}

impl ThermalModel {
    /// Create a new model.
    ///
    /// * `window_size` — how many raw samples to retain (moving average window).
    /// * `trend_window` — how many of the most recent samples to use for the
    ///   least-squares slope.  Must be ≥ 2 for a meaningful slope.
    pub fn new(window_size: usize, trend_window: usize) -> Self {
        let trend_window = trend_window.max(2).min(window_size);
        Self {
            samples: VecDeque::with_capacity(window_size),
            window_size,
            trend_window,
        }
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Push a new temperature sample (degrees Celsius).
    ///
    /// If the window is full the oldest sample is evicted (FIFO).
    pub fn feed(&mut self, temp_c: f32) {
        if self.samples.len() >= self.window_size {
            self.samples.pop_front();
        }
        self.samples.push_back(temp_c);
    }

    /// Number of samples currently in the window.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Simple moving-average over the full window (NaN when empty).
    pub fn average(&self) -> Option<f32> {
        let n = self.samples.len();
        if n == 0 {
            return None;
        }
        Some(self.samples.iter().sum::<f32>() / n as f32)
    }

    /// Maximum temperature in the current window (NaN when empty).
    pub fn max_temp(&self) -> Option<f32> {
        self.samples.iter().copied().reduce(f32::max)
    }

    /// Clear all samples.
    pub fn reset(&mut self) {
        self.samples.clear();
    }

    // ------------------------------------------------------------------
    // Trend analysis
    // ------------------------------------------------------------------

    /// Run trend analysis and return the full analysis result.
    pub fn analyze(&self) -> ThermalAnalysis {
        let n = self.samples.len();

        if n < 2 {
            return ThermalAnalysis {
                trend: ThermalTrend::Stable,
                slope_c_per_s: 0.0,
                predicted_next_c: self.average().unwrap_or(35.0),
            };
        }

        // Use the most recent `trend_window` samples for the least-squares fit.
        let window_len = self.trend_window.min(n);
        let tail: Vec<f32> = self.samples.iter().rev().take(window_len).copied().collect();
        // tail[0] = most recent, tail[last] = oldest within window.
        // For the regression we want x = 0, 1, 2, ... (index within window).
        // Since we want a slope that makes physical sense we keep natural order:
        // x = 0 is the *oldest* point in the window, x = window_len-1 is newest.

        let oldest_first: Vec<f32> = tail.into_iter().rev().collect();
        let m = oldest_first.len() as f32;

        // Least-squares slope:  slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
        let sum_x: f32 = (0..oldest_first.len()).map(|i| i as f32).sum();
        let sum_y: f32 = oldest_first.iter().sum();
        let sum_xy: f32 = oldest_first
            .iter()
            .enumerate()
            .map(|(i, &y)| (i as f32) * y)
            .sum();
        let sum_xx: f32 = (0..oldest_first.len()).map(|i| (i as f32) * (i as f32)).sum();

        let denom = m * sum_xx - sum_x * sum_x;
        let slope = if denom.abs() > f32::EPSILON {
            (m * sum_xy - sum_x * sum_y) / denom
        } else {
            0.0
        };

        // Predict next temp = last value + slope (one step ahead)
        let predicted_next_c = oldest_first.last().copied().unwrap_or(35.0) + slope;

        // Classify trend
        // A slope > 0.05 °C per sample-interval is "rising".
        // Below -0.05 is "falling".  In between → stable.
        const RISING_THRESHOLD: f32 = 0.05;
        const FALLING_THRESHOLD: f32 = -0.05;

        let trend = if slope > RISING_THRESHOLD {
            ThermalTrend::Rising
        } else if slope < FALLING_THRESHOLD {
            ThermalTrend::Falling
        } else {
            ThermalTrend::Stable
        };

        ThermalAnalysis {
            trend,
            slope_c_per_s: slope,
            predicted_next_c,
        }
    }

    /// Convenience: returns `true` when the model recommends a pre-emptive
    /// downgrade *before* hitting the hard throttle.
    ///
    /// Arguments:
    /// * `thermal_threshold_c` — the hard throttle temperature.
    /// * `margin_c` — how many degrees below the threshold to pre-downgrade.
    /// * `current_temp` — the latest real temperature (will be fed first).
    ///
    /// The logic: if the trend is Rising AND the *predicted* temperature
    /// exceeds `(threshold - margin)`, downgrade is recommended.
    pub fn should_pre_downgrade(
        &mut self,
        current_temp: f32,
        thermal_threshold_c: f32,
        margin_c: f32,
    ) -> bool {
        self.feed(current_temp);
        let analysis = self.analyze();

        if !analysis.trend.is_concerning() {
            return false;
        }

        let trigger_at = thermal_threshold_c - margin_c;
        analysis.predicted_next_c >= trigger_at
            || current_temp >= trigger_at
    }

    /// Convenience: should the system consider upgrading (trend is Falling
    /// and predicted temp is well below threshold)?
    pub fn should_upgrade(
        &mut self,
        current_temp: f32,
        thermal_threshold_c: f32,
        safety_margin_c: f32,
    ) -> bool {
        self.feed(current_temp);
        let analysis = self.analyze();

        if analysis.trend != ThermalTrend::Falling {
            return false;
        }

        // Only suggest upgrade if we're comfortably below threshold with margin
        let safe_zone = thermal_threshold_c - safety_margin_c;
        analysis.predicted_next_c <= safe_zone && current_temp <= safe_zone
    }

    /// Feed a batch of historical samples to warm up the model.
    pub fn feed_batch(&mut self, temps: &[f32]) {
        for &t in temps {
            self.feed(t);
        }
    }
}

// ---------------------------------------------------------------------------
// ThermalAnalysis
// ---------------------------------------------------------------------------

/// Result of a trend analysis.
#[derive(Debug, Clone)]
pub struct ThermalAnalysis {
    /// Classified trend direction.
    pub trend: ThermalTrend,
    /// Slope in °C per sample interval.
    pub slope_c_per_s: f32,
    /// Predicted temperature at the next sample interval.
    pub predicted_next_c: f32,
}

impl Default for ThermalAnalysis {
    fn default() -> Self {
        Self {
            trend: ThermalTrend::Stable,
            slope_c_per_s: 0.0,
            predicted_next_c: 35.0,
        }
    }
}

// ---------------------------------------------------------------------------
// PerfModel — energy-per-token estimation
// ---------------------------------------------------------------------------

/// Lightweight per-operator energy model for edge inference.
///
/// Tracks estimated energy consumption per token across different
/// inference modes and provides a token budget based on remaining
/// battery capacity.
pub struct PerfModel {
    /// Estimated mJ per token in each mode (Turbo, Balanced, Eco).
    mj_per_token_turbo: f32,
    mj_per_token_balanced: f32,
    mj_per_token_eco: f32,
}

impl PerfModel {
    /// Default energy costs based on typical edge hardware (Snapdragon 8 Gen 2 / Apple M-series).
    ///
    /// These are placeholder values — real-world numbers depend on model size,
    /// chipset, and quantization.  Calibrate via benchmark (Section 9).
    pub const DEFAULT_MJ_TURBO: f32 = 120.0;
    pub const DEFAULT_MJ_BALANCED: f32 = 80.0;
    pub const DEFAULT_MJ_ECO: f32 = 45.0;

    /// Create a new model with custom energy estimates.
    pub fn new(mj_per_token_turbo: f32, mj_per_token_balanced: f32, mj_per_token_eco: f32) -> Self {
        Self {
            mj_per_token_turbo,
            mj_per_token_balanced,
            mj_per_token_eco,
        }
    }

    /// Create with default estimates.
    pub fn default_calibrated() -> Self {
        Self::new(
            Self::DEFAULT_MJ_TURBO,
            Self::DEFAULT_MJ_BALANCED,
            Self::DEFAULT_MJ_ECO,
        )
    }

    /// Estimated mJ per token for the given mode.
    pub fn mj_per_token(&self, mode: &crate::InferenceMode) -> f32 {
        match mode {
            crate::InferenceMode::Turbo => self.mj_per_token_turbo,
            crate::InferenceMode::Balanced => self.mj_per_token_balanced,
            crate::InferenceMode::Eco => self.mj_per_token_eco,
        }
    }

    /// Maximum number of tokens that can be generated given the remaining
    /// battery capacity (in milliJoules).
    pub fn token_budget(&self, remaining_mj: f32, mode: &crate::InferenceMode) -> u64 {
        let cost = self.mj_per_token(mode);
        if cost <= 0.0 {
            return u64::MAX;
        }
        (remaining_mj / cost) as u64
    }

    /// Update energy estimates (e.g., from calibration benchmarks).
    pub fn calibrate(&mut self, turbo: f32, balanced: f32, eco: f32) {
        self.mj_per_token_turbo = turbo;
        self.mj_per_token_balanced = balanced;
        self.mj_per_token_eco = eco;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- ThermalModel basics -------------------------------------------------

    #[test]
    fn test_empty_model() {
        let model = ThermalModel::new(10, 3);
        assert_eq!(model.sample_count(), 0);
        assert!(model.average().is_none());
        let analysis = model.analyze();
        assert_eq!(analysis.trend, ThermalTrend::Stable);
    }

    #[test]
    fn test_single_sample_no_trend() {
        let mut model = ThermalModel::new(10, 3);
        model.feed(40.0);
        assert_eq!(model.sample_count(), 1);
        assert!((model.average().unwrap() - 40.0).abs() < 0.001);
        let analysis = model.analyze();
        assert_eq!(analysis.trend, ThermalTrend::Stable);
    }

    #[test]
    fn test_moving_average() {
        let mut model = ThermalModel::new(5, 3);
        for t in [30.0, 40.0, 50.0] {
            model.feed(t);
        }
        let avg = model.average().unwrap();
        assert!((avg - 40.0).abs() < 0.001);
    }

    #[test]
    fn test_window_eviction() {
        let mut model = ThermalModel::new(3, 2);
        for t in [10.0, 20.0, 30.0, 40.0] {
            model.feed(t);
        }
        // Only last 3: 20, 30, 40
        assert_eq!(model.sample_count(), 3);
        let avg = model.average().unwrap();
        assert!((avg - 30.0).abs() < 0.001);
    }

    // -- Trend detection -----------------------------------------------------

    #[test]
    fn test_rising_trend() {
        let mut model = ThermalModel::new(10, 5);
        // Monotonically increasing: 35, 37, 39, 41, 43
        for t in [35.0, 37.0, 39.0, 41.0, 43.0] {
            model.feed(t);
        }
        let analysis = model.analyze();
        assert_eq!(analysis.trend, ThermalTrend::Rising);
        assert!(analysis.slope_c_per_s > 0.0);
        // Predicted should be > last value
        assert!(analysis.predicted_next_c > 43.0);
    }

    #[test]
    fn test_falling_trend() {
        let mut model = ThermalModel::new(10, 5);
        for t in [50.0, 47.0, 44.0, 41.0, 38.0] {
            model.feed(t);
        }
        let analysis = model.analyze();
        assert_eq!(analysis.trend, ThermalTrend::Falling);
        assert!(analysis.slope_c_per_s < 0.0);
    }

    #[test]
    fn test_stable_trend() {
        let mut model = ThermalModel::new(10, 5);
        // Small fluctuations around 40 without directional bias.
        for t in [39.9, 40.1, 40.0, 39.9, 40.0] {
            model.feed(t);
        }
        let analysis = model.analyze();
        assert_eq!(analysis.trend, ThermalTrend::Stable);
    }

    #[test]
    fn test_flat_line_stable() {
        let mut model = ThermalModel::new(10, 3);
        for _ in 0..5 {
            model.feed(45.0);
        }
        let analysis = model.analyze();
        assert_eq!(analysis.trend, ThermalTrend::Stable);
        assert!(analysis.slope_c_per_s.abs() < 0.001);
    }

    // -- Predictive pre-downgrade --------------------------------------------

    #[test]
    fn test_should_pre_downgrade_rising_towards_threshold() {
        let mut model = ThermalModel::new(10, 4);
        // Rising towards 80 °C threshold with 5 °C margin.
        // Pre-downgrade should fire when predicted > 75.
        for t in [40.0, 50.0, 60.0, 70.0, 78.0] {
            model.feed(t);
        }
        // The slope is ~9.5, predicted ~87.5 > 75
        assert!(model.should_pre_downgrade(78.0, 80.0, 5.0));
    }

    #[test]
    fn test_should_pre_downgrade_stable_no_action() {
        let mut model = ThermalModel::new(10, 4);
        for t in [45.0, 44.0, 45.0, 44.0, 45.0] {
            model.feed(t);
        }
        // Stable, no downgrade needed despite being close to threshold
        assert!(!model.should_pre_downgrade(45.0, 80.0, 10.0));
    }

    #[test]
    fn test_should_pre_downgrade_falling_no_action() {
        let mut model = ThermalModel::new(10, 4);
        for t in [75.0, 70.0, 65.0, 60.0, 55.0] {
            model.feed(t);
        }
        assert!(!model.should_pre_downgrade(55.0, 80.0, 10.0));
    }

    // -- should_upgrade ------------------------------------------------------

    #[test]
    fn test_should_upgrade_when_cooling() {
        let mut model = ThermalModel::new(10, 4);
        for t in [70.0, 65.0, 60.0, 55.0, 50.0] {
            model.feed(t);
        }
        assert!(model.should_upgrade(50.0, 80.0, 15.0));
    }

    #[test]
    fn test_should_upgrade_denied_if_still_hot() {
        let mut model = ThermalModel::new(10, 4);
        for t in [75.0, 74.0, 73.0, 72.0, 71.0] {
            model.feed(t);
        }
        // Trend is falling but still near threshold
        assert!(!model.should_upgrade(71.0, 80.0, 15.0));
    }

    // -- PerfModel -----------------------------------------------------------

    #[test]
    fn test_perf_model_defaults() {
        let pm = PerfModel::default_calibrated();
        assert!((pm.mj_per_token(&crate::InferenceMode::Turbo) - 120.0).abs() < 0.001);
        assert!((pm.mj_per_token(&crate::InferenceMode::Balanced) - 80.0).abs() < 0.001);
        assert!((pm.mj_per_token(&crate::InferenceMode::Eco) - 45.0).abs() < 0.001);
    }

    #[test]
    fn test_token_budget() {
        let pm = PerfModel::default_calibrated();
        let remaining_mj = 10_000.0; // 10 J
        let budget = pm.token_budget(remaining_mj, &crate::InferenceMode::Eco);
        // 10_000 / 45 ≈ 222
        assert!(budget > 200 && budget < 250);
    }

    #[test]
    fn test_token_budget_turbo_vs_eco() {
        let pm = PerfModel::default_calibrated();
        let remaining_mj = 12_000.0;
        let turbo_budget = pm.token_budget(remaining_mj, &crate::InferenceMode::Turbo);
        let eco_budget = pm.token_budget(remaining_mj, &crate::InferenceMode::Eco);
        assert!(turbo_budget < eco_budget); // Turbo costs more
    }

    #[test]
    fn test_calibrate() {
        let mut pm = PerfModel::default_calibrated();
        pm.calibrate(100.0, 60.0, 30.0);
        assert!((pm.mj_per_token(&crate::InferenceMode::Turbo) - 100.0).abs() < 0.001);
        assert!((pm.mj_per_token(&crate::InferenceMode::Balanced) - 60.0).abs() < 0.001);
        assert!((pm.mj_per_token(&crate::InferenceMode::Eco) - 30.0).abs() < 0.001);
    }

    // -- Integration: feed_batch + reset -------------------------------------

    #[test]
    fn test_feed_batch() {
        let mut model = ThermalModel::new(5, 3);
        model.feed_batch(&[30.0, 40.0, 50.0, 60.0, 70.0]);
        assert_eq!(model.sample_count(), 5);
        assert!((model.average().unwrap() - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_reset() {
        let mut model = ThermalModel::new(5, 3);
        model.feed_batch(&[30.0, 40.0, 50.0]);
        assert_eq!(model.sample_count(), 3);
        model.reset();
        assert_eq!(model.sample_count(), 0);
        assert!(model.average().is_none());
    }

    #[test]
    fn test_max_temp() {
        let mut model = ThermalModel::new(5, 3);
        model.feed_batch(&[30.0, 50.0, 40.0]);
        assert!((model.max_temp().unwrap() - 50.0).abs() < 0.001);
    }
}
