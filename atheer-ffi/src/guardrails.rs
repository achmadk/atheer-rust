use atheer_core::GuardrailLevel;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, uniffi::Enum)]
pub enum AtheerGuardrailLevel {
    None,
    Basic,
    Balanced,
    Strict,
}

impl From<GuardrailLevel> for AtheerGuardrailLevel {
    fn from(level: GuardrailLevel) -> Self {
        match level {
            GuardrailLevel::None => AtheerGuardrailLevel::None,
            GuardrailLevel::Basic => AtheerGuardrailLevel::Basic,
            GuardrailLevel::Balanced => AtheerGuardrailLevel::Balanced,
            GuardrailLevel::Strict => AtheerGuardrailLevel::Strict,
        }
    }
}

impl From<AtheerGuardrailLevel> for GuardrailLevel {
    fn from(level: AtheerGuardrailLevel) -> Self {
        match level {
            AtheerGuardrailLevel::None => GuardrailLevel::None,
            AtheerGuardrailLevel::Basic => GuardrailLevel::Basic,
            AtheerGuardrailLevel::Balanced => GuardrailLevel::Balanced,
            AtheerGuardrailLevel::Strict => GuardrailLevel::Strict,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_none() {
        let core = GuardrailLevel::None;
        let ffi: AtheerGuardrailLevel = core.into();
        let back: GuardrailLevel = ffi.into();
        assert_eq!(back, GuardrailLevel::None);
    }

    #[test]
    fn test_roundtrip_strict() {
        let core = GuardrailLevel::Strict;
        let ffi: AtheerGuardrailLevel = core.into();
        let back: GuardrailLevel = ffi.into();
        assert_eq!(back, GuardrailLevel::Strict);
    }

    #[test]
    fn test_all_variants_roundtrip() {
        let variants = [
            GuardrailLevel::None,
            GuardrailLevel::Basic,
            GuardrailLevel::Balanced,
            GuardrailLevel::Strict,
        ];
        for v in variants {
            let ffi: AtheerGuardrailLevel = v.into();
            let back: GuardrailLevel = ffi.into();
            assert_eq!(back, v);
        }
    }
}
