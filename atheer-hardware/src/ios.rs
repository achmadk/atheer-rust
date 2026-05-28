use crate::{HardwareMonitor, HealthStatus, MemoryStatus, PowerState, ThermalState};

pub struct IosMonitor;

impl IosMonitor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for IosMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl HardwareMonitor for IosMonitor {
    fn health(&self) -> HealthStatus {
        HealthStatus {
            thermal: self.thermal_state(),
            available_ram_mb: self.memory_status().available_mb,
            total_ram_mb: self.memory_status().total_mb,
            battery_level: self.battery_level(),
            on_battery: self.is_on_battery(),
            timestamp: chrono::Utc::now().timestamp(),
        }
    }

    fn thermal_state(&self) -> ThermalState {
        ThermalState::Nominal
    }

    fn memory_status(&self) -> MemoryStatus {
        MemoryStatus {
            available_mb: 2048,
            total_mb: 6144,
            low_memory_threshold_mb: 800,
        }
    }

    fn power_state(&self) -> PowerState {
        PowerState::Unknown
    }

    fn battery_level(&self) -> u32 {
        100
    }

    fn is_on_battery(&self) -> bool {
        false
    }
}
