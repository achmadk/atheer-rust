use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorConfig {
    pub max_iterations: usize,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub adaptive: bool,
    pub thermal_threshold_c: f32,
    pub memory_threshold_mb: u64,
    pub memory_critical_mb: u64,
    pub battery_threshold_percent: u32,
    pub hysteresis_cooldown_ms: u64,
    pub polling_interval_ms: u64,
    /// Predictive thermal margin (°C): downgrade pre-emptively when predicted
    /// temperature exceeds `thermal_threshold_c - thermal_margin_c`.
    pub thermal_margin_c: f32,
    /// Number of temperature samples retained by the thermal model.
    pub thermal_window_size: usize,
    /// Number of most-recent samples used for least-squares slope calculation.
    pub thermal_trend_window: usize,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            repeat_penalty: 1.1,
            adaptive: true,
            thermal_threshold_c: 42.0,
            memory_threshold_mb: 800,
            memory_critical_mb: 600,
            battery_threshold_percent: 20,
            hysteresis_cooldown_ms: 5000,
            polling_interval_ms: 1000,
            thermal_margin_c: 5.0,
            thermal_window_size: 20,
            thermal_trend_window: 5,
        }
    }
}
