use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ThermalState {
    #[default]
    Nominal,
    Fair,
    Serious,
    Critical,
}

impl ThermalState {
    pub fn as_str(&self) -> &'static str {
        match self {
            ThermalState::Nominal => "nominal",
            ThermalState::Fair => "fair",
            ThermalState::Serious => "serious",
            ThermalState::Critical => "critical",
        }
    }

    pub fn from_degrees_celsius(temp: f32) -> Self {
        if temp < 35.0 {
            ThermalState::Nominal
        } else if temp < 40.0 {
            ThermalState::Fair
        } else if temp < 45.0 {
            ThermalState::Serious
        } else {
            ThermalState::Critical
        }
    }
}

