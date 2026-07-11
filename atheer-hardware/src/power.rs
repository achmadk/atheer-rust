use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PowerState {
    #[default]
    Unknown,
    Charging,
    Discharging,
    Full,
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

