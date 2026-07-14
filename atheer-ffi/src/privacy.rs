use atheer_core::PrivacyMode;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, uniffi::Enum)]
pub enum AtheerPrivacyMode {
    Normal,
    Ephemeral,
    Audited,
}

impl From<PrivacyMode> for AtheerPrivacyMode {
    fn from(mode: PrivacyMode) -> Self {
        match mode {
            PrivacyMode::Normal => AtheerPrivacyMode::Normal,
            PrivacyMode::Ephemeral => AtheerPrivacyMode::Ephemeral,
            PrivacyMode::Audited => AtheerPrivacyMode::Audited,
        }
    }
}

impl From<AtheerPrivacyMode> for PrivacyMode {
    fn from(mode: AtheerPrivacyMode) -> Self {
        match mode {
            AtheerPrivacyMode::Normal => PrivacyMode::Normal,
            AtheerPrivacyMode::Ephemeral => PrivacyMode::Ephemeral,
            AtheerPrivacyMode::Audited => PrivacyMode::Audited,
        }
    }
}
