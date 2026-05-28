use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PowerState {
    Charging,
    Discharging,
    Full,
    Unknown,
}

impl PowerState {
    pub fn as_str(&self) -> &'static str {
        match self {
            PowerState::Charging => "charging",
            PowerState::Discharging => "discharging",
            PowerState::Full => "full",
            PowerState::Unknown => "unknown",
        }
    }
}

impl Default for PowerState {
    fn default() -> Self {
        PowerState::Unknown
    }
}
