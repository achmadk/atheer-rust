use crate::AtheerBackendType;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, uniffi::Record)]
pub struct AtheerConfig {
    pub model_path: Option<String>,
    pub tokenizer_path: Option<String>,
    pub model_id: Option<String>,
    pub draft_model_path: Option<String>,
    pub adaptive: bool,
    pub max_tokens: u32,
    pub temperature: f32,
    pub quantization: String,
    pub adaptive_handoff: bool,
    pub memory_bank_size_mb: u32,
    pub standby_draft_path: Option<String>,
    pub backend_type: Option<AtheerBackendType>,
    pub coreml_model_path: Option<String>,
}

impl Default for AtheerConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            tokenizer_path: None,
            model_id: None,
            draft_model_path: None,
            adaptive: true,
            max_tokens: 512,
            temperature: 0.7,
            quantization: "q4_k_m".to_string(),
            adaptive_handoff: true,
            memory_bank_size_mb: 512,
            standby_draft_path: None,
            backend_type: None,
            coreml_model_path: None,
        }
    }
}

impl AtheerConfig {
    pub fn new(model_path: String) -> Self {
        Self {
            model_path: Some(model_path),
            ..Default::default()
        }
    }
}
