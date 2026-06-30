use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerSample {
    pub timestamp: i64,
    pub elapsed_secs: f64,
    pub tokens_generated: u32,
    pub generation_time_ms: u64,
    pub thermal_state: String,
    pub available_ram_mb: u64,
    pub battery_level: u32,
    pub on_battery: bool,
}
