use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferenceMode {
    Turbo,
    Balanced,
    Eco,
}

impl InferenceMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            InferenceMode::Turbo => "turbo",
            InferenceMode::Balanced => "balanced",
            InferenceMode::Eco => "eco",
        }
    }

    pub fn speculation_depth(&self) -> usize {
        match self {
            InferenceMode::Turbo => 4,
            InferenceMode::Balanced => 2,
            InferenceMode::Eco => 0,
        }
    }
}

impl Default for InferenceMode {
    fn default() -> Self {
        InferenceMode::Eco
    }
}
