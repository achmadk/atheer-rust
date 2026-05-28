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
}
