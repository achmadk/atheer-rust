use crate::ThermalState;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub thermal: ThermalState,
    pub available_ram_mb: u64,
    pub total_ram_mb: u64,
    pub battery_level: u32,
    pub on_battery: bool,
    pub timestamp: i64,
}

impl HealthStatus {
    pub fn new() -> Self {
        Self {
            thermal: ThermalState::Nominal,
            available_ram_mb: 0,
            total_ram_mb: 0,
            battery_level: 100,
            on_battery: false,
            timestamp: chrono::Utc::now().timestamp(),
        }
    }

    pub fn is_critical(&self) -> bool {
        matches!(self.thermal, ThermalState::Critical)
            || self.available_ram_mb < 800
            || (self.on_battery && self.battery_level < 10)
    }

    pub fn memory_pressure_percent(&self) -> f32 {
        if self.total_ram_mb == 0 {
            return 0.0;
        }
        let used = self.total_ram_mb - self.available_ram_mb;
        (used as f32 / self.total_ram_mb as f32) * 100.0
    }
}

impl Default for HealthStatus {
    fn default() -> Self {
        Self::new()
    }
}
