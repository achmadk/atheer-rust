use atheer_hardware::ThermalState;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, uniffi::Enum)]
pub enum AtheerThermalState {
    Nominal,
    Fair,
    Serious,
    Critical,
}

impl From<ThermalState> for AtheerThermalState {
    fn from(state: ThermalState) -> Self {
        match state {
            ThermalState::Nominal => AtheerThermalState::Nominal,
            ThermalState::Fair => AtheerThermalState::Fair,
            ThermalState::Serious => AtheerThermalState::Serious,
            ThermalState::Critical => AtheerThermalState::Critical,
        }
    }
}

impl From<AtheerThermalState> for ThermalState {
    fn from(state: AtheerThermalState) -> Self {
        match state {
            AtheerThermalState::Nominal => ThermalState::Nominal,
            AtheerThermalState::Fair => ThermalState::Fair,
            AtheerThermalState::Serious => ThermalState::Serious,
            AtheerThermalState::Critical => ThermalState::Critical,
        }
    }
}
