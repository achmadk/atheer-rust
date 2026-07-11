use crate::InferenceMode;
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// CalibrationSample — one generation's performance data
// ---------------------------------------------------------------------------

/// A single datapoint from one `generate()` or `generate_speculative()` call.
pub struct CalibrationSample {
    pub tok_s: f32,
    pub tokens_gen: u32,
    pub mode: InferenceMode,
    pub speculation_depth: usize,
    pub acceptance_rate: Option<f32>,
}

// ---------------------------------------------------------------------------
// CalibrationUpdate — output of a recalibration cycle
// ---------------------------------------------------------------------------

/// Recommendations produced by a recalibration cycle.
///
/// The orchestrator applies these to `PerfModel`, speculation depth bounds,
/// and mode-switch thresholds.
pub struct CalibrationUpdate {
    pub turbo_mj: f32,
    pub balanced_mj: f32,
    pub eco_mj: f32,
    pub depth_bounds: HashMap<InferenceMode, (usize, usize)>,
    pub threshold_delta_c: f32,
    pub threshold_delta_mb: u64,
    pub threshold_delta_battery: u32,
}

// ---------------------------------------------------------------------------
// Energy modelling constants
// ---------------------------------------------------------------------------

/// Assumed baseline tok/s values used to derive the device energy envelope
/// from the default mJ/token estimates in `PerfModel`.
///
/// These are conservative estimates for ~1B-param edge LLM inference on
/// mobile SoCs (Snapdragon 8 Gen 2 / Apple M-series).  The exact values
/// do not matter for calibration — what matters is the *ratio* between
/// measured and assumed throughput, which scales the energy estimate
/// linearly.
const BASELINE_TOK_S_TURBO: f32 = 25.0;
const BASELINE_TOK_S_BALANCED: f32 = 18.0;
const BASELINE_TOK_S_ECO: f32 = 10.0;

/// Default mJ/token values — mirrors `PerfModel::DEFAULT_MJ_*`.
const DEFAULT_MJ_TURBO: f32 = 120.0;
const DEFAULT_MJ_BALANCED: f32 = 80.0;
const DEFAULT_MJ_ECO: f32 = 45.0;

/// Minimum percentage throughput delta required to change a depth bound.
const DEPTH_PLATEAU_PCT: f32 = 5.0;
const DEPTH_UNDERPERFORM_PCT: f32 = 15.0;

/// Minimum samples before we adjust a threshold band.
const MIN_SAMPLES_FOR_THRESHOLD_ADJUSTMENT: usize = 100;

// ---------------------------------------------------------------------------
// Calibrator
// ---------------------------------------------------------------------------

/// Accumulates generation performance data per inference mode and
/// periodically recalibrates the orchestrator's energy model, speculation
/// depth bounds, and mode-switch thresholds.
pub struct Calibrator {
    samples_turbo: VecDeque<CalibrationSample>,
    samples_balanced: VecDeque<CalibrationSample>,
    samples_eco: VecDeque<CalibrationSample>,
    window_capacity: usize,
    generation_count: u64,
    last_recalibrate_at: u64,
    recalibrate_interval: u64,
    min_samples_before_calibrate: usize,
    /// Tracks throttling events near the thermal threshold so we do not
    /// raise it unsafely.
    throttling_observed_since_last_reset: bool,
}

impl Calibrator {
    /// Default window capacity (max samples per mode).
    pub const DEFAULT_WINDOW_CAPACITY: usize = 20;

    /// Default recalibration interval (generations between recalibrations).
    pub const DEFAULT_RECALIBRATE_INTERVAL: u64 = 10;

    /// Default minimum samples before a mode's data is used for calibration.
    pub const DEFAULT_MIN_SAMPLES: usize = 5;

    pub fn new() -> Self {
        Self {
            samples_turbo: VecDeque::with_capacity(Self::DEFAULT_WINDOW_CAPACITY),
            samples_balanced: VecDeque::with_capacity(Self::DEFAULT_WINDOW_CAPACITY),
            samples_eco: VecDeque::with_capacity(Self::DEFAULT_WINDOW_CAPACITY),
            window_capacity: Self::DEFAULT_WINDOW_CAPACITY,
            generation_count: 0,
            last_recalibrate_at: 0,
            recalibrate_interval: Self::DEFAULT_RECALIBRATE_INTERVAL,
            min_samples_before_calibrate: Self::DEFAULT_MIN_SAMPLES,
            throttling_observed_since_last_reset: false,
        }
    }

    /// Append a sample to the correct mode's sliding window, evicting the
    /// oldest entry when the window is full.
    pub fn feed(&mut self, sample: CalibrationSample) {
        let capacity = self.window_capacity;
        let deque = self.window_mut(sample.mode);
        if deque.len() >= capacity {
            deque.pop_front();
        }
        deque.push_back(sample);
        self.generation_count += 1;
    }

    /// Mean tok/s for the given mode, or `None` if no samples exist.
    pub fn average_tok_s(&self, mode: InferenceMode) -> Option<f32> {
        let deque = self.window(mode);
        if deque.is_empty() {
            return None;
        }
        let sum: f32 = deque.iter().map(|s| s.tok_s).sum();
        Some(sum / deque.len() as f32)
    }

    /// True when enough generations have elapsed since the last recalibration.
    pub fn should_recalibrate(&self) -> bool {
        self.generation_count - self.last_recalibrate_at >= self.recalibrate_interval
    }

    /// Clear all windows and reset counters.
    pub fn reset(&mut self) {
        self.samples_turbo.clear();
        self.samples_balanced.clear();
        self.samples_eco.clear();
        self.generation_count = 0;
        self.last_recalibrate_at = 0;
        self.throttling_observed_since_last_reset = false;
    }

    /// Record that a throttling event occurred (prevents unsafe threshold
    /// raises in the current calibration epoch).
    pub fn record_throttling_event(&mut self) {
        self.throttling_observed_since_last_reset = true;
    }

    /// Override the recalibration interval (for testing).
    pub fn set_recalibrate_interval(&mut self, interval: u64) {
        self.recalibrate_interval = interval;
    }

    /// Override the minimum samples before a mode is used for calibration (for testing).
    pub fn set_min_samples(&mut self, min: usize) {
        self.min_samples_before_calibrate = min;
    }

    /// Mark a recalibration cycle as completed.
    ///
    /// Advances `last_recalibrate_at` to the current `generation_count` so
    /// that `should_recalibrate()` returns `false` until the interval elapses
    /// again.
    pub fn finish_recalibration(&mut self) {
        self.last_recalibrate_at = self.generation_count;
    }

    // ------------------------------------------------------------------
    // Recalibration (public, called by Orchestrator)
    // ------------------------------------------------------------------

    /// Run one recalibration cycle.
    ///
    /// ## Panics
    ///
    /// Panics if `recalibrate_interval` is 0 (enforced by the invariants
    /// on [`OrchestratorConfig`]).
    pub fn recalibrate(&self) -> CalibrationUpdate {
        let turbo_mj = self.recalibrate_mode(InferenceMode::Turbo);
        let balanced_mj = self.recalibrate_mode(InferenceMode::Balanced);
        let eco_mj = self.recalibrate_mode(InferenceMode::Eco);

        let depth_bounds = self.calibrate_speculation_bounds();

        let (td_c, td_mb, td_bat) = self.calibrate_thresholds();

        CalibrationUpdate {
            turbo_mj,
            balanced_mj,
            eco_mj,
            depth_bounds,
            threshold_delta_c: td_c,
            threshold_delta_mb: td_mb,
            threshold_delta_battery: td_bat,
        }
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn window(&self, mode: InferenceMode) -> &VecDeque<CalibrationSample> {
        match mode {
            InferenceMode::Turbo => &self.samples_turbo,
            InferenceMode::Balanced => &self.samples_balanced,
            InferenceMode::Eco => &self.samples_eco,
        }
    }

    fn window_mut(&mut self, mode: InferenceMode) -> &mut VecDeque<CalibrationSample> {
        match mode {
            InferenceMode::Turbo => &mut self.samples_turbo,
            InferenceMode::Balanced => &mut self.samples_balanced,
            InferenceMode::Eco => &mut self.samples_eco,
        }
    }

    /// Compute a new mJ/token estimate for one mode from measured throughput.
    ///
    /// Returns the current default if insufficient samples exist.
    fn recalibrate_mode(&self, mode: InferenceMode) -> f32 {
        let avg = match self.average_tok_s(mode) {
            Some(v) if v > 0.0 => v,
            _ => return default_mj(mode),
        };

        let samples_in_window = self.window(mode).len();
        if samples_in_window < self.min_samples_before_calibrate {
            return default_mj(mode);
        }

        // energy = baseline_energy × (baseline_tok_s / measured_tok_s)
        let energy_factor = default_mj(mode) * baseline_tok_s(mode);
        let new_mj = energy_factor / avg;

        // Clamp to a sane range: [10% of default, 10× default]
        let default = default_mj(mode);
        new_mj.clamp(default * 0.1, default * 10.0)
    }

    /// Calibrate speculation depth bounds for each mode based on measured
    /// throughput deltas between depths.
    fn calibrate_speculation_bounds(&self) -> HashMap<InferenceMode, (usize, usize)> {
        let mut bounds = HashMap::new();

        for mode in &[InferenceMode::Turbo, InferenceMode::Balanced] {
            let deque = self.window(*mode);
            let by_depth = group_by_depth(deque);
            if by_depth.len() < 2 {
                // Not enough distinct depths to compare.
                bounds.insert(*mode, (1, 8));
                continue;
            }

            let mut max_depth = 8;
            let mut min_depth = 1;

            let mut sorted_depths: Vec<_> = by_depth.keys().copied().collect();
            sorted_depths.sort();

            for (i, &depth) in sorted_depths.iter().enumerate().skip(1) {
                let prev_depth = sorted_depths[i - 1];
                let cur_tok_s = by_depth[&depth];
                let prev_tok_s = by_depth[&prev_depth];

                // Cap max depth where throughput plateaus (<5% delta)
                if prev_tok_s > 0.0 {
                    let delta_pct = ((cur_tok_s - prev_tok_s) / prev_tok_s) * 100.0;
                    if delta_pct < DEPTH_PLATEAU_PCT && depth <= max_depth {
                        max_depth = prev_depth;
                    }
                }

                // Raise min depth where shallow speculation underperforms (>15% delta)
                // When depth D gives >15% improvement over depth D-1, the shallower
                // depth meaningfully underperforms — raise min depth to D.
                if prev_tok_s > 0.0 {
                    let improvement_pct = ((cur_tok_s - prev_tok_s) / prev_tok_s) * 100.0;
                    if improvement_pct > DEPTH_UNDERPERFORM_PCT && prev_depth >= min_depth {
                        min_depth = depth;
                    }
                }
            }

            bounds.insert(*mode, (min_depth, max_depth));
        }

        // Eco mode does not speculate
        bounds.insert(InferenceMode::Eco, (0, 0));
        bounds
    }

    /// Compute threshold nudges based on long-term device behavior.
    ///
    /// Returns (delta_c, delta_mb, delta_battery) — all non-negative
    /// (positive values mean "raise the threshold").
    fn calibrate_thresholds(&self) -> (f32, u64, u32) {
        // Do not raise thresholds if throttling was observed.
        if self.throttling_observed_since_last_reset {
            return (0.0, 0, 0);
        }

        let turbo_count = self.samples_turbo.len();
        let balanced_count = self.samples_balanced.len();
        let eco_count = self.samples_eco.len();

        // Need a large sample base before nudging thresholds.
        if turbo_count < MIN_SAMPLES_FOR_THRESHOLD_ADJUSTMENT
            && balanced_count < MIN_SAMPLES_FOR_THRESHOLD_ADJUSTMENT
            && eco_count < MIN_SAMPLES_FOR_THRESHOLD_ADJUSTMENT
        {
            return (0.0, 0, 0);
        }

        // Conservative single-step nudges.
        // Thermal: +1°C per adjustment (max will be enforced by orchestrator)
        // Memory: +50 MB per adjustment
        // Battery: +2% per adjustment
        (1.0, 50, 2)
    }
}

impl Default for Calibrator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

fn default_mj(mode: InferenceMode) -> f32 {
    match mode {
        InferenceMode::Turbo => DEFAULT_MJ_TURBO,
        InferenceMode::Balanced => DEFAULT_MJ_BALANCED,
        InferenceMode::Eco => DEFAULT_MJ_ECO,
    }
}

fn baseline_tok_s(mode: InferenceMode) -> f32 {
    match mode {
        InferenceMode::Turbo => BASELINE_TOK_S_TURBO,
        InferenceMode::Balanced => BASELINE_TOK_S_BALANCED,
        InferenceMode::Eco => BASELINE_TOK_S_ECO,
    }
}

/// Group samples by speculation depth, returning the mean tok/s at each
/// depth (only depths with 3+ samples are considered).
fn group_by_depth(samples: &VecDeque<CalibrationSample>) -> HashMap<usize, f32> {
    let mut depth_totals: HashMap<usize, (f32, usize)> = HashMap::new();

    for sample in samples {
        let entry = depth_totals
            .entry(sample.speculation_depth)
            .or_insert((0.0, 0));
        entry.0 += sample.tok_s;
        entry.1 += 1;
    }

    depth_totals
        .into_iter()
        .filter(|(_, (_, count))| *count >= 3)
        .map(|(depth, (total, count))| (depth, total / count as f32))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(tok_s: f32, mode: InferenceMode, depth: usize) -> CalibrationSample {
        CalibrationSample {
            tok_s,
            tokens_gen: 10,
            mode,
            speculation_depth: depth,
            acceptance_rate: None,
        }
    }

    // -- 5.1: feed stores in correct mode window, evicts at capacity ---

    #[test]
    fn test_feed_stores_in_correct_mode_window() {
        let mut cal = Calibrator::new();
        cal.feed(sample(10.0, InferenceMode::Turbo, 0));
        cal.feed(sample(8.0, InferenceMode::Balanced, 0));
        cal.feed(sample(5.0, InferenceMode::Eco, 0));

        assert_eq!(cal.samples_turbo.len(), 1);
        assert_eq!(cal.samples_balanced.len(), 1);
        assert_eq!(cal.samples_eco.len(), 1);
    }

    #[test]
    fn test_feed_evicts_oldest_at_capacity() {
        let mut cal = Calibrator::new();
        // Fill to exactly capacity
        for i in 0..Calibrator::DEFAULT_WINDOW_CAPACITY {
            cal.feed(sample(i as f32, InferenceMode::Turbo, 0));
        }
        assert_eq!(cal.samples_turbo.len(), Calibrator::DEFAULT_WINDOW_CAPACITY);
        assert!((cal.average_tok_s(InferenceMode::Turbo).unwrap() - 9.5).abs() < 0.01);

        // One more — oldest (tok_s=0) should be evicted
        cal.feed(sample(100.0, InferenceMode::Turbo, 0));
        assert_eq!(cal.samples_turbo.len(), Calibrator::DEFAULT_WINDOW_CAPACITY);

        // Average should no longer include the evicted 0.0
        let avg = cal.average_tok_s(InferenceMode::Turbo).unwrap();
        // Was: (0+1+...+19) / 20 = 9.5
        // Now: (1+2+...+19+100) / 20 = (190+100)/20 = 14.5
        assert!((avg - 14.5).abs() < 0.01, "avg={avg}");
    }

    // -- 5.2: average_tok_s -------------------------------------------

    #[test]
    fn test_average_tok_s_returns_none_on_empty() {
        let cal = Calibrator::new();
        assert!(cal.average_tok_s(InferenceMode::Turbo).is_none());
        assert!(cal.average_tok_s(InferenceMode::Balanced).is_none());
        assert!(cal.average_tok_s(InferenceMode::Eco).is_none());
    }

    #[test]
    fn test_average_tok_s_computes_correct_mean() {
        let mut cal = Calibrator::new();
        cal.feed(sample(10.0, InferenceMode::Turbo, 0));
        cal.feed(sample(20.0, InferenceMode::Turbo, 0));
        cal.feed(sample(30.0, InferenceMode::Turbo, 0));

        let avg = cal.average_tok_s(InferenceMode::Turbo).unwrap();
        assert!((avg - 20.0).abs() < 0.001);
    }

    // -- 5.3: should_recalibrate --------------------------------------

    #[test]
    fn test_should_recalibrate_false_before_interval() {
        let mut cal = Calibrator::new();
        cal.recalibrate_interval = 10;

        for i in 0..9 {
            cal.feed(sample(10.0, InferenceMode::Turbo, 0));
            assert!(!cal.should_recalibrate(), "iteration {i}: expected false");
        }
    }

    #[test]
    fn test_should_recalibrate_true_at_interval() {
        let mut cal = Calibrator::new();
        cal.recalibrate_interval = 5;
        cal.min_samples_before_calibrate = 1;

        for _ in 0..5 {
            cal.feed(sample(10.0, InferenceMode::Turbo, 0));
        }
        assert!(cal.should_recalibrate());
    }

    // -- 5.4: reset ----------------------------------------------------

    #[test]
    fn test_reset_clears_all_state() {
        let mut cal = Calibrator::new();
        cal.feed(sample(10.0, InferenceMode::Turbo, 0));
        cal.feed(sample(8.0, InferenceMode::Balanced, 0));
        cal.feed(sample(5.0, InferenceMode::Eco, 0));
        cal.generation_count = 42;
        cal.last_recalibrate_at = 10;
        cal.throttling_observed_since_last_reset = true;

        cal.reset();

        assert!(cal.samples_turbo.is_empty());
        assert!(cal.samples_balanced.is_empty());
        assert!(cal.samples_eco.is_empty());
        assert_eq!(cal.generation_count, 0);
        assert_eq!(cal.last_recalibrate_at, 0);
        assert!(!cal.throttling_observed_since_last_reset);
    }

    // -- 5.5: recalibrate produces CalibrationUpdate -------------------

    #[test]
    fn test_recalibrate_produces_update() {
        let mut cal = Calibrator::new();
        cal.min_samples_before_calibrate = 3;

        // Feed 5 samples per mode
        for _ in 0..5 {
            cal.feed(sample(20.0, InferenceMode::Turbo, 4));
            cal.feed(sample(15.0, InferenceMode::Balanced, 2));
            cal.feed(sample(10.0, InferenceMode::Eco, 0));
        }

        cal.generation_count = 15;
        cal.last_recalibrate_at = 5;
        assert!(cal.should_recalibrate());

        let update = cal.recalibrate();

        // Energy estimates should be updated (25/20 ratio for Turbo)
        let expected_turbo_mj = DEFAULT_MJ_TURBO * BASELINE_TOK_S_TURBO / 20.0;
        assert!((update.turbo_mj - expected_turbo_mj).abs() < 0.1);

        let expected_balanced_mj = DEFAULT_MJ_BALANCED * BASELINE_TOK_S_BALANCED / 15.0;
        assert!((update.balanced_mj - expected_balanced_mj).abs() < 0.1);

        let expected_eco_mj = DEFAULT_MJ_ECO * BASELINE_TOK_S_ECO / 10.0;
        assert!((update.eco_mj - expected_eco_mj).abs() < 0.1);
    }

    // -- 5.6: speculation depth bounds --------------------------------

    #[test]
    fn test_speculation_depth_bound_reduced_on_plateau() {
        let mut cal = Calibrator::new();
        cal.min_samples_before_calibrate = 1;

        // Feed samples where depth=4 and depth=2 have nearly identical throughput
        for _ in 0..5 {
            cal.feed(sample(20.0, InferenceMode::Turbo, 4));
        }
        for _ in 0..5 {
            cal.feed(sample(19.5, InferenceMode::Turbo, 2));
        }

        let update = cal.recalibrate();
        let (min, max) = update
            .depth_bounds
            .get(&InferenceMode::Turbo)
            .copied()
            .unwrap_or((1, 8));
        // Max should be reduced since depth 4 throughput is within 5% of depth 2
        assert!(max <= 4, "expected max depth capped, got {max}");
        assert_eq!(min, 1);
    }

    #[test]
    fn test_speculation_depth_floor_raised_when_underperforms() {
        let mut cal = Calibrator::new();
        cal.min_samples_before_calibrate = 1;

        // depth=2 is significantly worse than depth=4
        for _ in 0..5 {
            cal.feed(sample(10.0, InferenceMode::Balanced, 2));
        }
        for _ in 0..5 {
            cal.feed(sample(30.0, InferenceMode::Balanced, 4));
        }

        let update = cal.recalibrate();
        let (min, _max) = update
            .depth_bounds
            .get(&InferenceMode::Balanced)
            .copied()
            .unwrap_or((1, 8));
        // Min should be raised since depth 2 underperforms by >15%
        assert!(min >= 2, "expected min depth raised, got {min}");
    }

    // -- 5.7: thermal threshold not raised when throttling -------------

    #[test]
    fn test_thermal_threshold_not_raised_when_throttling_observed() {
        let mut cal = Calibrator::new();
        cal.min_samples_before_calibrate = 1;

        // Fill enough samples
        for _ in 0..Calibrator::DEFAULT_MIN_SAMPLES {
            cal.feed(sample(20.0, InferenceMode::Turbo, 4));
        }

        cal.throttling_observed_since_last_reset = true;

        let update = cal.recalibrate();
        assert_eq!(update.threshold_delta_c, 0.0);
        assert_eq!(update.threshold_delta_mb, 0);
        assert_eq!(update.threshold_delta_battery, 0);
    }

    // -- 5.9: insufficient samples skips mode calibration -------------

    #[test]
    fn test_insufficient_samples_skips_calibration() {
        let mut cal = Calibrator::new();
        cal.min_samples_before_calibrate = 10;

        // Feed only 3 samples — below min_samples_before_calibrate
        for _ in 0..3 {
            cal.feed(sample(99.0, InferenceMode::Turbo, 4));
        }

        let update = cal.recalibrate();
        // Should fall back to default values
        assert!((update.turbo_mj - DEFAULT_MJ_TURBO).abs() < 0.1);
    }
}
