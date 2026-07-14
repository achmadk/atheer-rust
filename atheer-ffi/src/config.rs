use crate::{AtheerBackendType, AtheerGuardrailLevel, AtheerPrivacyMode};
use atheer_core::model_credential::ModelCredential;
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
    /// Directory for KV cache checkpoint files. When set, enables checkpoint persistence.
    pub checkpoint_dir: Option<String>,
    /// Maximum number of checkpoint generations to retain (default: 3).
    pub max_checkpoints: u32,
    /// Checkpoint TTL in hours (0 = no TTL-based expiry).
    pub checkpoint_ttl_hours: u32,
    /// Enable checkpoint save on app background.
    pub checkpoint_on_background: bool,
    /// Enable KV cache restore on app foreground.
    pub restore_on_foreground: bool,
    /// Enable LZ4-compressed checkpoint on low memory.
    pub checkpoint_on_low_memory: bool,
    /// Enable final checkpoint on app terminate.
    pub checkpoint_on_terminate: bool,
    /// Clear GPU-side KV cache after low-memory checkpoint.
    pub clear_on_low_memory: bool,
    /// 32-byte AES-256 key for cache/checkpoint encryption.
    /// When set alongside `checkpoint_dir`, enables encrypted L3 persistence.
    /// If `checkpoint_dir` is set but this is `None`, an ephemeral key is
    /// generated per session (cache is still encrypted but unrecoverable
    /// after process exit — same security level for V1).
    pub cache_encryption_key: Option<Vec<u8>>,
    #[serde(skip)]
    pub model_credential: Option<ModelCredential>,
    /// Runtime privacy mode. When set to `Ephemeral`, the engine skips crash
    /// report writes, L3 cache persistence, and suppresses tracing output.
    /// When set to `Audited`, additional decision-point logging is enabled.
    /// Default (`None`) means normal operation with no restrictions.
    pub privacy_mode: Option<AtheerPrivacyMode>,
    /// Raw Ed25519 public key bytes (32 bytes) for model signature verification.
    /// When set, `SecurityAudit.enable_signature_verify` is enabled and the
    /// engine verifies the model file against a `.gguf.sig` detached signature
    /// before loading.
    pub model_signature_public_key: Option<Vec<u8>>,
    /// Expected SHA-256 hash of the model file, hex-encoded (64 hex chars).
    /// When set, the engine verifies the model file hash before loading.
    /// Computed via streaming SHA-256 at load time.
    pub model_expected_sha256: Option<String>,
    /// Guardrail detection level. Default (`None`) means disabled.
    pub guardrail_level: Option<AtheerGuardrailLevel>,
    /// Path to a sidecar JSON pattern file for guardrail detection.
    pub guardrail_patterns_path: Option<String>,
    /// Additional custom patterns appended to the guardrail pattern set.
    pub guardrail_custom_patterns: Vec<String>,
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
            checkpoint_dir: None,
            max_checkpoints: 3,
            checkpoint_ttl_hours: 0,
            checkpoint_on_background: true,
            restore_on_foreground: true,
            checkpoint_on_low_memory: true,
            checkpoint_on_terminate: true,
            clear_on_low_memory: true,
            cache_encryption_key: None,
            model_credential: None,
            privacy_mode: None,
            model_signature_public_key: None,
            model_expected_sha256: None,
            guardrail_level: None,
            guardrail_patterns_path: None,
            guardrail_custom_patterns: Vec::new(),
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
