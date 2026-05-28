use atheer_orchestrator::InferenceMode;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, uniffi::Enum)]
pub enum AtheerInferenceMode {
    Turbo,
    Balanced,
    Eco,
}

impl From<InferenceMode> for AtheerInferenceMode {
    fn from(mode: InferenceMode) -> Self {
        match mode {
            InferenceMode::Turbo => AtheerInferenceMode::Turbo,
            InferenceMode::Balanced => AtheerInferenceMode::Balanced,
            InferenceMode::Eco => AtheerInferenceMode::Eco,
        }
    }
}

impl From<AtheerInferenceMode> for InferenceMode {
    fn from(mode: AtheerInferenceMode) -> Self {
        match mode {
            AtheerInferenceMode::Turbo => InferenceMode::Turbo,
            AtheerInferenceMode::Balanced => InferenceMode::Balanced,
            AtheerInferenceMode::Eco => InferenceMode::Eco,
        }
    }
}
