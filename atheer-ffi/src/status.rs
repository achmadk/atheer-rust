use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, uniffi::Record)]
pub struct HardwareHealth {
    pub thermal: String,
    pub available_ram_mb: u64,
    pub battery_level: u32,
    pub on_battery: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, uniffi::Record)]
pub struct MemoryBankStatus {
    pub l1_active: Option<String>,
    pub l2_warm: Option<String>,
    pub alignment_score: f32,
    pub is_handoff: bool,
    pub handoff_phase: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, uniffi::Record)]
pub struct EngineStatus {
    pub mode: String,
    pub tokens_per_second: f32,
    pub draft_loaded: bool,
    pub hardware_health: HardwareHealth,
    pub memory_bank: MemoryBankStatus,
}

impl Default for EngineStatus {
    fn default() -> Self {
        Self {
            mode: "eco".to_string(),
            tokens_per_second: 0.0,
            draft_loaded: false,
            hardware_health: HardwareHealth {
                thermal: "nominal".to_string(),
                available_ram_mb: 4096,
                battery_level: 100,
                on_battery: false,
            },
            memory_bank: MemoryBankStatus {
                l1_active: None,
                l2_warm: None,
                alignment_score: 0.0,
                is_handoff: false,
                handoff_phase: "idle".to_string(),
            },
        }
    }
}
