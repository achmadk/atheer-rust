use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum InferenceMode {
    Turbo,
    Balanced,
    #[default]
    Eco,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpType {
    Prefill,
    Decode,
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

    pub fn is_eco(&self) -> bool {
        matches!(self, InferenceMode::Eco)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_op_type_variants() {
        let prefill = OpType::Prefill;
        let decode = OpType::Decode;
        assert_ne!(prefill, decode);
    }

    #[test]
    fn test_inference_mode_is_eco() {
        assert!(InferenceMode::Eco.is_eco());
        assert!(!InferenceMode::Turbo.is_eco());
        assert!(!InferenceMode::Balanced.is_eco());
    }

    #[test]
    fn test_inference_mode_default() {
        let mode = InferenceMode::default();
        assert_eq!(mode, InferenceMode::Eco);
    }
}
