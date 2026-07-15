use thiserror::Error;

#[derive(Error, Debug)]
pub enum AtheerCoreError {
    #[error("Failed to load model: {0}")]
    ModelLoadFailed(String),

    #[error("Failed to load tokenizer: {0}")]
    TokenizerLoadFailed(String),

    #[error("Failed to generate tokens: {0}")]
    GenerationFailed(String),

    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),

    #[error("Session error: {0}")]
    SessionError(String),

    #[error("Device mismatch: expected {expected}, got {actual}")]
    DeviceMismatch { expected: String, actual: String },

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("KV cache error: {0}")]
    KvCacheError(String),

    #[error("Download failed: {0}")]
    DownloadFailed(String),

    #[error("Checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },

    #[error("Cache error: {0}")]
    CacheError(String),

    #[error("Generation timeout after {elapsed_ms}ms ({tokens_generated} tokens generated)")]
    Timeout {
        elapsed_ms: u64,
        tokens_generated: usize,
    },

    #[error("Model decryption failed: {0}")]
    ModelDecryptionFailed(String),

    #[error("GGUF header magic mismatch: expected GGUF, got {actual:?}")]
    InvalidMagic { actual: [u8; 4] },

    #[cfg(feature = "model-registry")]
    #[error("TLS certificate pinning failed for {hostname}: peer key hash {peer_hash} does not match any pinned key [{pinned_hashes}]")]
    TlsPinningFailed {
        hostname: String,
        peer_hash: String,
        pinned_hashes: String,
    },

    #[error("GGUF version {version} is not supported (expected 1, 2, or 3)")]
    InvalidVersion { version: u32 },

    #[error("GGUF counts invalid: tensor_count={tensor_count}, metadata_kv_count={metadata_kv_count}, requested_tensor_bytes={requested_tensor_bytes}, max_tensor_bytes={max_tensor_bytes}")]
    InvalidCounts {
        tensor_count: u64,
        metadata_kv_count: u64,
        max_tensor_bytes: u64,
        requested_tensor_bytes: u64,
    },

    #[error("GGUF general.alignment invalid: {value}")]
    InvalidAlignment { value: i64 },

    #[error("GGUF tensor '{tensor_name}' out of bounds: offset={offset}, size={size}, file_size={file_size}")]
    InvalidTensorBounds {
        tensor_name: String,
        offset: u64,
        size: u64,
        file_size: u64,
    },

    #[error("GGUF duplicate tensor name: {name}")]
    DuplicateTensorName { name: String },
}

pub type Result<T> = std::result::Result<T, AtheerCoreError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_mismatch_display() {
        let err = AtheerCoreError::DeviceMismatch {
            expected: "Cpu".to_string(),
            actual: "Metal (unavailable)".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Device mismatch"));
        assert!(msg.contains("Cpu"));
        assert!(msg.contains("Metal (unavailable)"));
    }

    #[test]
    fn test_timeout_error_display() {
        let err = AtheerCoreError::Timeout {
            elapsed_ms: 1234,
            tokens_generated: 5,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("1234ms"));
        assert!(msg.contains("5"));
    }

    #[test]
    fn test_invalid_magic_display() {
        let err = AtheerCoreError::InvalidMagic {
            actual: [0xDE, 0xAD, 0xBE, 0xEF],
        };
        let msg = format!("{}", err);
        assert!(msg.contains("magic"));
    }

    #[test]
    fn test_invalid_tensor_bounds_display() {
        let err = AtheerCoreError::InvalidTensorBounds {
            tensor_name: "attn.q".to_string(),
            offset: 1_000_000_000,
            size: 4096,
            file_size: 134_217_728,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("attn.q"));
        assert!(msg.contains("offset"));
    }

    #[test]
    fn test_invalid_alignment_display() {
        let err = AtheerCoreError::InvalidAlignment { value: 100 };
        let msg = format!("{}", err);
        assert!(msg.contains("alignment"));
        assert!(msg.contains("100"));
    }
}
