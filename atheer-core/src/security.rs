use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fmt;
use std::io::Read;
use std::path::Path;
use tracing;

pub struct SecurityAudit {
    allowed_paths: HashSet<String>,
    max_model_size_mb: usize,
    enable_signature_verify: bool,
}

impl SecurityAudit {
    pub fn new() -> Self {
        Self {
            allowed_paths: HashSet::new(),
            max_model_size_mb: 10_000,
            enable_signature_verify: false,
        }
    }

    pub fn with_allowed_paths(mut self, paths: Vec<String>) -> Self {
        self.allowed_paths = paths.into_iter().collect();
        self
    }

    pub fn with_max_model_size_mb(mut self, size_mb: usize) -> Self {
        self.max_model_size_mb = size_mb;
        self
    }

    pub fn with_signature_verification(mut self, enabled: bool) -> Self {
        self.enable_signature_verify = enabled;
        self
    }

    pub fn validate_model_path(&self, path: &str) -> Result<(), SecurityError> {
        if self.allowed_paths.is_empty() {
            return Ok(());
        }

        if !self.allowed_paths.contains(path) {
            return Err(SecurityError::PathNotAllowed(path.to_string()));
        }

        Ok(())
    }

    pub fn validate_model_size(&self, size_mb: usize) -> Result<(), SecurityError> {
        if size_mb > self.max_model_size_mb {
            return Err(SecurityError::ModelTooLarge {
                size_mb,
                max_mb: self.max_model_size_mb,
            });
        }
        Ok(())
    }

    pub fn enable_signature_verify(&self) -> bool {
        self.enable_signature_verify
    }

    /// Verify the SHA-256 hash of a model file against an expected value.
    ///
    /// Uses streaming reads (64 KB chunks) to avoid loading large files
    /// into memory. Returns `HashMismatch` on failure.
    pub fn verify_model_hash(&self, path: &Path, expected: &[u8; 32]) -> Result<(), SecurityError> {
        let mut hasher = Sha256::new();
        let mut file = std::fs::File::open(path).map_err(|e| SecurityError::HashMismatch {
            expected: hex::encode(expected),
            actual: format!("cannot open file: {e}"),
        })?;
        let mut buf = [0u8; 65536];
        loop {
            let n = file
                .read(&mut buf)
                .map_err(|e| SecurityError::HashMismatch {
                    expected: hex::encode(expected),
                    actual: format!("read error: {e}"),
                })?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        let actual_hash = hasher.finalize();
        if actual_hash.as_slice() != expected {
            return Err(SecurityError::HashMismatch {
                expected: hex::encode(expected),
                actual: hex::encode(actual_hash),
            });
        }
        Ok(())
    }

    pub fn sanitize_prompt(&self, prompt: &str) -> String {
        let max_chars = 32_000;
        let char_count = prompt.chars().count();
        if char_count > max_chars {
            let truncated: String = prompt.chars().take(max_chars).collect();
            tracing::warn!(
                target: "atheer::core::security",
                "Prompt truncated from {} to {} characters",
                char_count,
                max_chars,
            );
            return truncated;
        }
        prompt.to_string()
    }
}

impl Default for SecurityAudit {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub enum SecurityError {
    PathNotAllowed(String),
    ModelTooLarge { size_mb: usize, max_mb: usize },
    SignatureInvalid,
    SignatureVerificationFailed(String),
    KeyParseFailed(String),
    HashMismatch { expected: String, actual: String },
}

impl fmt::Display for SecurityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SecurityError::PathNotAllowed(p) => write!(f, "Model path not allowed: {p}"),
            SecurityError::ModelTooLarge { size_mb, max_mb } => {
                write!(f, "Model size {size_mb} MB exceeds limit of {max_mb} MB")
            }
            SecurityError::SignatureInvalid => {
                write!(f, "Model signature verification failed")
            }
            SecurityError::SignatureVerificationFailed(msg) => {
                write!(f, "Signature verification error: {msg}")
            }
            SecurityError::KeyParseFailed(msg) => {
                write!(f, "Failed to parse public key: {msg}")
            }
            SecurityError::HashMismatch { expected, actual } => {
                write!(f, "Model hash mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_audit_creation() {
        let audit = SecurityAudit::new();
        assert_eq!(audit.max_model_size_mb, 10_000);
    }

    #[test]
    fn test_allowed_paths() {
        let audit = SecurityAudit::new().with_allowed_paths(vec!["/models".to_string()]);

        let result = audit.validate_model_path("/models");
        assert!(result.is_ok());

        let result = audit.validate_model_path("/other");
        assert!(result.is_err());
    }

    #[test]
    fn test_model_size_validation() {
        let audit = SecurityAudit::new().with_max_model_size_mb(1000);

        let result = audit.validate_model_size(500);
        assert!(result.is_ok());

        let result = audit.validate_model_size(2000);
        assert!(result.is_err());
    }

    #[test]
    fn test_prompt_sanitization() {
        let audit = SecurityAudit::new();
        let long_prompt = "a".repeat(50_000);
        let sanitized = audit.sanitize_prompt(&long_prompt);
        assert_eq!(sanitized.len(), 32_000);
    }

    #[test]
    fn test_sanitize_cjk_multi_byte() {
        let audit = SecurityAudit::new();
        // Each CJK character is 3 bytes in UTF-8, so 50,000 chars = 150,000 bytes
        // Old code would panic at prompt[..32000] since 32000 falls mid-CJK
        let long_prompt = "中".repeat(50_000);
        let sanitized = audit.sanitize_prompt(&long_prompt);
        assert_eq!(sanitized.chars().count(), 32_000);
        // All chars should be valid CJK (no partial character)
        assert!(sanitized.chars().all(|c| c == '中'));
    }

    #[test]
    fn test_verify_model_hash_match() {
        let audit = SecurityAudit::new();
        let dir = std::env::temp_dir().join(format!("atheer_hash_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_hash.gguf");
        let data = b"test model data for hash check";
        std::fs::write(&path, data).unwrap();

        let expected = sha2::Sha256::digest(data);
        let hash_array: [u8; 32] = expected.into();

        let result = audit.verify_model_hash(&path, &hash_array);
        assert!(result.is_ok(), "Correct hash should pass: {:?}", result);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_verify_model_hash_mismatch() {
        let audit = SecurityAudit::new();
        let dir = std::env::temp_dir().join(format!("atheer_hash_mismatch_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_mismatch.gguf");
        let data = b"test data";
        std::fs::write(&path, data).unwrap();

        let wrong_hash = [0u8; 32];
        let result = audit.verify_model_hash(&path, &wrong_hash);
        assert!(result.is_err(), "Wrong hash should fail");
        match result.unwrap_err() {
            SecurityError::HashMismatch { expected, actual } => {
                assert_eq!(
                    expected,
                    "0000000000000000000000000000000000000000000000000000000000000000"
                );
                assert_eq!(actual.len(), 64, "Actual hash should be 64 hex chars");
            }
            e => panic!("Expected HashMismatch, got {:?}", e),
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_verify_model_hash_nonexistent_file() {
        let audit = SecurityAudit::new();
        let path = std::path::Path::new("/nonexistent/file.gguf");
        let hash = [0u8; 32];
        let result = audit.verify_model_hash(path, &hash);
        assert!(result.is_err(), "Non-existent file should fail");
    }

    #[test]
    fn test_hash_message_format() {
        let audit = SecurityAudit::new();
        let dir = std::env::temp_dir().join(format!("atheer_hash_fmt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_fmt.gguf");
        let data = b"test";
        std::fs::write(&path, data).unwrap();

        let expected = [0xabu8; 32];
        let result = audit.verify_model_hash(&path, &expected);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match &err {
            SecurityError::HashMismatch {
                expected: exp,
                actual: _,
            } => {
                assert_eq!(
                    exp,
                    "abababababababababababababababababababababababababababababababab"
                );
            }
            _ => panic!("Wrong error variant"),
        }
        let msg = format!("{err}");
        assert!(
            msg.contains("abababab"),
            "Message should contain expected hash: {msg}"
        );
        assert!(
            msg.contains("mismatch"),
            "Message should mention mismatch: {msg}"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_sanitize_emoji_mixed() {
        let audit = SecurityAudit::new();
        // Emoji (4 bytes) + accented Latin (2 bytes) + ASCII — mixed multi-byte
        let mut prompt = String::with_capacity(60_000);
        // Build a prompt with alternating multi-byte sequences
        for _ in 0..20_000 {
            prompt.push('é'); // 2-byte UTF-8
            prompt.push('a'); // 1-byte
            prompt.push('🔥'); // 4-byte UTF-8 (fire emoji)
        }
        // Total chars = 60,000, well over the 32,000 limit
        let sanitized = audit.sanitize_prompt(&prompt);
        assert_eq!(sanitized.chars().count(), 32_000);
        // Verify no partial characters at the end — all chars must be valid
        assert!(sanitized
            .chars()
            .all(|c| c.is_alphabetic() || c.is_ascii() || c == '🔥'));
    }
}
