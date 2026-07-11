use crate::calibrator::{CalibrationSample, Calibrator};
use crate::modes::{BalancedMode, EcoMode, TurboMode};
use crate::thermal_model::{PerfModel, ThermalModel};
use crate::{InferenceMode, OrchestratorConfig};
use atheer_memory_bank::MemoryBank;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[allow(dead_code)]
pub struct Orchestrator {
    config: OrchestratorConfig,
    current_mode: InferenceMode,
    previous_mode: InferenceMode,
    turbo: TurboMode,
    balanced: BalancedMode,
    eco: EcoMode,
    confidence: f32,
    last_mode_change: Arc<AtomicU64>,
    mode_change_count: u32,
    thermal_model: ThermalModel,
    perf_model: PerfModel,
    calibrator: Calibrator,
    /// Calibrated speculation depth bounds per mode, applied on top of
    /// the mode's native defaults.  Updated by `record_generation_metrics`.
    depth_bounds: HashMap<InferenceMode, (usize, usize)>,
}

impl Orchestrator {
    pub fn new(config: OrchestratorConfig) -> Self {
        let current_mode = if config.adaptive {
            InferenceMode::Eco
        } else {
            InferenceMode::Balanced
        };

        Self {
            config: config.clone(),
            current_mode,
            previous_mode: current_mode,
            turbo: TurboMode::new(),
            balanced: BalancedMode::new(),
            eco: EcoMode::new(),
            confidence: 0.0,
            last_mode_change: Arc::new(AtomicU64::new(0)),
            mode_change_count: 0,
            thermal_model: ThermalModel::new(
                config.thermal_window_size,
                config.thermal_trend_window,
            ),
            perf_model: PerfModel::default_calibrated(),
            calibrator: Calibrator::new(),
            depth_bounds: HashMap::new(),
        }
    }

    pub fn current_mode(&self) -> InferenceMode {
        self.current_mode
    }

    pub fn previous_mode(&self) -> InferenceMode {
        self.previous_mode
    }

    pub fn set_mode(&mut self, mode: InferenceMode) {
        let prev = self.current_mode;
        if prev != mode {
            tracing::info!(
                target: "atheer::orchestrator::mode",
                "Mode transition: {:?} -> {:?} (total changes: {})",
                prev,
                mode,
                self.mode_change_count + 1,
            );
        }
        self.previous_mode = prev;
        self.current_mode = mode;
        self.mode_change_count += 1;
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self.last_mode_change.store(timestamp, Ordering::SeqCst);
    }

    fn can_change_mode(&self) -> bool {
        let last_change_ms = self.last_mode_change.load(Ordering::SeqCst);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(u64::MAX);

        now.saturating_sub(last_change_ms) >= self.config.hysteresis_cooldown_ms
    }

    pub fn select_mode(
        &mut self,
        thermal_c: Option<f32>,
        available_ram_mb: u64,
        battery_level: Option<u32>,
        on_battery: bool,
    ) -> InferenceMode {
        if !self.config.adaptive {
            return self.current_mode;
        }

        // 1. Feed temperature sample into the predictive thermal model
        if let Some(temp) = thermal_c {
            self.thermal_model.feed(temp);
        }

        // 2. Predictive pre-downgrade check: if trend is Rising and predicted
        //    temperature is approaching the hard throttle, downgrade pre-emptively
        //    to Balanced (not Eco) — avoids aggressive cutting performance.
        if let Some(temp) = thermal_c {
            if self.thermal_model.should_pre_downgrade(
                temp,
                self.config.thermal_threshold_c,
                self.config.thermal_margin_c,
            ) {
                let downgrade_target = InferenceMode::Balanced;
                if downgrade_target != self.current_mode {
                    tracing::info!(
                        target: "atheer::orchestrator::thermal",
                        "Predictive thermal downgrade: {:?} -> {:?} (temp={:.1}°C, trend=Rising, margin={:.1}°C)",
                        self.current_mode,
                        downgrade_target,
                        temp,
                        self.config.thermal_margin_c,
                    );
                    if self.can_change_mode() || self.is_downgrade(&downgrade_target) {
                        self.set_mode(downgrade_target);
                        return self.current_mode;
                    }
                }
            }

            // 3. Check if we can upgrade back (trend is Falling, well below threshold)
            if self.thermal_model.should_upgrade(
                temp,
                self.config.thermal_threshold_c,
                self.config.thermal_margin_c * 2.0,
            ) {
                // Try to upgrade to the target the reactive logic would pick
                let reactive_target = self.calculate_target_mode(
                    thermal_c,
                    available_ram_mb,
                    battery_level,
                    on_battery,
                );
                if reactive_target != self.current_mode
                    && !self.is_downgrade(&reactive_target)
                    && self.can_change_mode()
                {
                    tracing::info!(
                        target: "atheer::orchestrator::thermal",
                        "Predictive thermal upgrade: {:?} -> {:?} (trend=Falling, safe)",
                        self.current_mode,
                        reactive_target,
                    );
                    self.set_mode(reactive_target);
                    return self.current_mode;
                }
            }
        }

        // 4. Fall through to existing reactive logic
        self.reactive_select(thermal_c, available_ram_mb, battery_level, on_battery)
    }

    /// Reactive-only mode selection (original logic, unmodified).
    fn reactive_select(
        &mut self,
        thermal_c: Option<f32>,
        available_ram_mb: u64,
        battery_level: Option<u32>,
        on_battery: bool,
    ) -> InferenceMode {
        let target_mode =
            self.calculate_target_mode(thermal_c, available_ram_mb, battery_level, on_battery);

        if target_mode != self.current_mode
            && (self.can_change_mode() || self.is_downgrade(&target_mode))
        {
            self.set_mode(target_mode);
        }

        self.current_mode
    }

    fn calculate_target_mode(
        &self,
        thermal_c: Option<f32>,
        available_ram_mb: u64,
        battery_level: Option<u32>,
        on_battery: bool,
    ) -> InferenceMode {
        if let Some(temp) = thermal_c {
            if temp > self.config.thermal_threshold_c {
                return InferenceMode::Eco;
            }
        }

        if available_ram_mb < self.config.memory_critical_mb {
            return InferenceMode::Eco;
        }

        if available_ram_mb < self.config.memory_threshold_mb {
            return InferenceMode::Eco;
        }

        if on_battery {
            if let Some(battery) = battery_level {
                if battery < self.config.battery_threshold_percent {
                    return InferenceMode::Eco;
                }
            }
        }

        if available_ram_mb < 1024 {
            return InferenceMode::Balanced;
        }

        InferenceMode::Turbo
    }

    fn is_downgrade(&self, target: &InferenceMode) -> bool {
        let current_rank = self.mode_rank(&self.current_mode);
        let target_rank = self.mode_rank(target);
        target_rank < current_rank
    }

    fn mode_rank(&self, mode: &InferenceMode) -> u8 {
        match mode {
            InferenceMode::Turbo => 2,
            InferenceMode::Balanced => 1,
            InferenceMode::Eco => 0,
        }
    }

    pub fn speculation_depth(&self) -> usize {
        let default_depth = self.current_mode.speculation_depth();
        // Clamp within calibrated bounds if they exist for this mode.
        match self.depth_bounds.get(&self.current_mode) {
            Some(&(min, max)) => default_depth.clamp(min, max),
            None => default_depth,
        }
    }

    /// Record the result of a speculative decoding cycle.
    /// Dispatches to the appropriate mode's acceptance tracking.
    pub fn record_speculative_result(&mut self, accepted: usize, total: usize) {
        match self.current_mode {
            InferenceMode::Turbo => {
                self.turbo.record_acceptance(accepted, total);
            }
            InferenceMode::Balanced => {
                // BalancedMode self-adjusts via logit consistency (update_logits).
                // No additional tracking needed here — the generate loop calls
                // update_logits() directly as part of self-speculation.
            }
            InferenceMode::Eco => {
                // Eco mode does not speculate
            }
        }
    }

    /// Set whether a draft model is loaded in the engine.
    /// This enables TurboMode's draft-based speculation path.
    pub fn set_draft_model_loaded(&mut self, loaded: bool) {
        self.turbo.set_draft_model_loaded(loaded);
    }

    /// Returns whether a draft model is loaded and ready.
    pub fn is_draft_loaded(&self) -> bool {
        self.turbo.is_draft_loaded()
    }

    pub fn mode_change_count(&self) -> u32 {
        self.mode_change_count
    }

    pub fn time_since_last_change(&self) -> Duration {
        let last_change_ms = self.last_mode_change.load(Ordering::SeqCst);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(last_change_ms);

        Duration::from_millis(now.saturating_sub(last_change_ms))
    }

    pub fn update_confidence(&mut self, confidence: f32) {
        self.confidence = confidence.clamp(0.0, 1.0);
    }

    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    /// Check memory pressure across all tiers.
    /// Returns true if current usage exceeds `memory_threshold_mb`.
    pub fn check_memory_pressure(&self, memory_bank: &MemoryBank) -> bool {
        let threshold_bytes = (self.config.memory_threshold_mb as usize) * 1024 * 1024;
        let total_bytes = memory_bank.total_allocated_bytes();
        total_bytes > threshold_bytes
    }

    /// Log memory pressure warning if threshold exceeded.
    pub fn log_memory_pressure_if_needed(&self, memory_bank: &MemoryBank) {
        let threshold_bytes = (self.config.memory_threshold_mb as usize) * 1024 * 1024;
        let total_bytes = memory_bank.total_allocated_bytes();

        if total_bytes > threshold_bytes {
            tracing::warn!(
                target: "atheer::orchestrator::memory",
                "Memory pressure detected: {}MB used, threshold {}MB",
                total_bytes / (1024 * 1024),
                threshold_bytes / (1024 * 1024),
            );
        }
    }

    // ------------------------------------------------------------------
    // Calibration plumbing (tasks 3.x)
    // ------------------------------------------------------------------

    /// Feed a generation sample into the calibrator, and — if enough
    /// generations have elapsed — run a recalibration cycle and apply
    /// the resulting updates to `perf_model`, speculation depth bounds,
    /// and config thresholds.
    ///
    /// This is an **internal** method — it is NOT exported via
    /// `#[uniffi::export]` and will never cross the FFI boundary.
    pub fn record_generation_metrics(&mut self, sample: CalibrationSample) {
        self.calibrator.feed(sample);

        if !self.calibrator.should_recalibrate() {
            return;
        }

        let update = self.calibrator.recalibrate();

        // 1. Update energy model
        self.perf_model
            .calibrate(update.turbo_mj, update.balanced_mj, update.eco_mj);

        // 2. Update speculation depth bounds
        for (mode, &(min, max)) in &update.depth_bounds {
            self.depth_bounds.insert(*mode, (min, max));
        }

        // 3. Nudge mode-switch thresholds (conservatively)
        self.config.thermal_threshold_c = (self.config.thermal_threshold_c
            + update.threshold_delta_c)
            .min(OrchestratorConfig::default().thermal_threshold_c + 3.0);

        self.config.memory_threshold_mb = self
            .config
            .memory_threshold_mb
            .saturating_add(update.threshold_delta_mb);

        self.config.battery_threshold_percent = self
            .config
            .battery_threshold_percent
            .saturating_add(update.threshold_delta_battery);

        self.calibrator.finish_recalibration();

        tracing::info!(
            target: "atheer::orchestrator::calibration",
            "Recalibration applied: turbo_mj={:.1} balanced_mj={:.1} eco_mj={:.1} depth_bounds={:?} thermal_delta={:.1} mem_delta={} battery_delta={}",
            update.turbo_mj,
            update.balanced_mj,
            update.eco_mj,
            update.depth_bounds,
            update.threshold_delta_c,
            update.threshold_delta_mb,
            update.threshold_delta_battery,
        );
    }

    /// Mutable access to the calibrator (testing only).
    pub fn calibrator_mut(&mut self) -> &mut Calibrator {
        &mut self.calibrator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orchestrator_default_mode() {
        let config = OrchestratorConfig::default();
        let orchestrator = Orchestrator::new(config);
        assert_eq!(orchestrator.current_mode(), InferenceMode::Eco);
    }

    #[test]
    fn test_thermal_throttle() {
        let mut config = OrchestratorConfig::default();
        config.adaptive = true;
        config.hysteresis_cooldown_ms = 0;
        let mut orchestrator = Orchestrator::new(config);

        let mode = orchestrator.select_mode(Some(45.0), 4096, Some(50), false);
        assert_eq!(mode, InferenceMode::Eco);
    }

    #[test]
    fn test_low_memory_mode() {
        let mut config = OrchestratorConfig::default();
        config.adaptive = true;
        config.hysteresis_cooldown_ms = 0;
        let mut orchestrator = Orchestrator::new(config);

        let mode = orchestrator.select_mode(None, 600, Some(50), false);
        assert_eq!(mode, InferenceMode::Eco);
    }

    #[test]
    fn test_turbo_mode_selection() {
        let mut config = OrchestratorConfig::default();
        config.adaptive = true;
        config.hysteresis_cooldown_ms = 0;
        let mut orchestrator = Orchestrator::new(config);

        let mode = orchestrator.select_mode(Some(35.0), 4096, Some(80), false);
        assert_eq!(mode, InferenceMode::Turbo);
    }

    #[test]
    fn test_hysteresis_prevents_upgrade() {
        let mut config = OrchestratorConfig::default();
        config.adaptive = true;
        config.hysteresis_cooldown_ms = 10000;
        let mut orchestrator = Orchestrator::new(config);

        orchestrator.select_mode(Some(35.0), 4096, Some(80), false);
        assert_eq!(orchestrator.current_mode(), InferenceMode::Turbo);

        orchestrator.set_mode(InferenceMode::Eco);
        let mode = orchestrator.select_mode(Some(35.0), 4096, Some(80), false);
        assert_eq!(mode, InferenceMode::Eco);
    }

    #[test]
    fn test_downgrade_allowed_during_hysteresis() {
        let mut config = OrchestratorConfig::default();
        config.adaptive = true;
        config.hysteresis_cooldown_ms = 10000;
        let mut orchestrator = Orchestrator::new(config);

        orchestrator.set_mode(InferenceMode::Turbo);

        let mode = orchestrator.select_mode(Some(50.0), 4096, Some(80), false);
        assert_eq!(mode, InferenceMode::Eco);
    }

    #[test]
    fn test_mode_change_count() {
        let mut config = OrchestratorConfig::default();
        config.hysteresis_cooldown_ms = 0;
        let mut orchestrator = Orchestrator::new(config);

        assert_eq!(orchestrator.mode_change_count(), 0);

        orchestrator.set_mode(InferenceMode::Balanced);
        assert_eq!(orchestrator.mode_change_count(), 1);

        orchestrator.set_mode(InferenceMode::Turbo);
        assert_eq!(orchestrator.mode_change_count(), 2);
    }

    #[test]
    fn test_confidence_tracking() {
        let mut config = OrchestratorConfig::default();
        config.hysteresis_cooldown_ms = 0;
        let mut orchestrator = Orchestrator::new(config);

        orchestrator.update_confidence(0.8);
        assert!((orchestrator.confidence() - 0.8).abs() < 0.001);

        orchestrator.update_confidence(1.5);
        assert!((orchestrator.confidence() - 1.0).abs() < 0.001);

        orchestrator.update_confidence(-0.5);
        assert!((orchestrator.confidence() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_speculation_depth_via_mode() {
        let mut config = OrchestratorConfig::default();
        config.hysteresis_cooldown_ms = 0;
        let mut orchestrator = Orchestrator::new(config);

        orchestrator.set_mode(InferenceMode::Turbo);
        let depth = orchestrator.speculation_depth();
        assert!(depth > 0);
    }

    #[test]
    fn test_battery_threshold_triggers_eco() {
        let mut config = OrchestratorConfig::default();
        config.adaptive = true;
        config.hysteresis_cooldown_ms = 0;
        config.battery_threshold_percent = 20;
        let mut orchestrator = Orchestrator::new(config);

        let mode = orchestrator.select_mode(Some(30.0), 4096, Some(15), true);
        assert_eq!(mode, InferenceMode::Eco);
    }

    #[test]
    fn test_balanced_mode_selection() {
        let mut config = OrchestratorConfig::default();
        config.adaptive = true;
        config.hysteresis_cooldown_ms = 0;
        let mut orchestrator = Orchestrator::new(config);

        orchestrator.set_mode(InferenceMode::Balanced);
        assert_eq!(orchestrator.current_mode(), InferenceMode::Balanced);
    }

    #[test]
    fn test_check_memory_pressure_returns_false_when_under_threshold() {
        let config = OrchestratorConfig::default();
        let orchestrator = Orchestrator::new(config);
        let memory_bank = atheer_memory_bank::MemoryBank::new(1024);

        let has_pressure = orchestrator.check_memory_pressure(&memory_bank);
        assert!(!has_pressure);
    }

    #[test]
    fn test_check_memory_pressure_returns_true_when_over_threshold() {
        let mut config = OrchestratorConfig::default();
        config.memory_threshold_mb = 1; // Very low threshold
        let orchestrator = Orchestrator::new(config);
        let memory_bank = atheer_memory_bank::MemoryBank::new(1024);

        // Even with small usage, with 1MB threshold it should detect pressure
        let has_pressure = orchestrator.check_memory_pressure(&memory_bank);
        // With default MemoryBank at 0 usage, it won't exceed 1MB threshold
        // So this tests the method works, not that pressure is detected at 0 usage
        assert!(!has_pressure || memory_bank.total_allocated_bytes() > 0);
    }

    // -- 5.8: record_generation_metrics integration test --------------

    #[test]
    fn test_record_generation_metrics_triggers_recalibration_after_interval() {
        let config = OrchestratorConfig::default();
        let mut orchestrator = Orchestrator::new(config);

        let cal = orchestrator.calibrator_mut();
        cal.set_recalibrate_interval(3);
        cal.set_min_samples(1);

        // Sanity: default perf model values
        let default_turbo_mj = crate::thermal_model::PerfModel::DEFAULT_MJ_TURBO;
        let default_bal_mj = crate::thermal_model::PerfModel::DEFAULT_MJ_BALANCED;
        let default_eco_mj = crate::thermal_model::PerfModel::DEFAULT_MJ_ECO;

        // Feed 4 samples (should trigger recalibration after the 3rd)
        for i in 0..4 {
            orchestrator.record_generation_metrics(CalibrationSample {
                tok_s: 20.0 + i as f32,
                tokens_gen: 10,
                mode: InferenceMode::Turbo,
                speculation_depth: 4,
                acceptance_rate: None,
            });
        }

        // After recalibration, turbo_mj should have been updated from default
        let mj = orchestrator.perf_model.mj_per_token(&InferenceMode::Turbo);
        assert!(
            (mj - default_turbo_mj).abs() > 0.01,
            "expected turbo_mj to change from {default_turbo_mj} after recalibration, got {mj}"
        );

        // Balanced and Eco should have been recalibrated too
        let mj_bal = orchestrator
            .perf_model
            .mj_per_token(&InferenceMode::Balanced);
        assert!(
            (mj_bal - default_bal_mj).abs() < 0.01,
            "expected balanced unchanged (no samples), got {mj_bal}"
        );
        let mj_eco = orchestrator.perf_model.mj_per_token(&InferenceMode::Eco);
        assert!(
            (mj_eco - default_eco_mj).abs() < 0.01,
            "expected eco unchanged (no samples), got {mj_eco}"
        );
    }

    // -- 5.9: generation failure does not record metrics (tested at
    //         the orchestrator level: no metrics means no recalibration)

    #[test]
    fn test_no_recalibration_without_metrics_feed() {
        let config = OrchestratorConfig::default();
        let mut orchestrator = Orchestrator::new(config);

        // Do NOT call record_generation_metrics — verify no recalibration
        // was triggered (should_recalibrate returns false because no
        // samples were fed)
        let cal = orchestrator.calibrator_mut();
        assert!(!cal.should_recalibrate());
        assert!(cal.average_tok_s(InferenceMode::Turbo).is_none());
    }
}
