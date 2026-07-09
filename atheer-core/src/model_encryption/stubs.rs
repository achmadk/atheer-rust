use crate::model_encryption::ModelEncryption;
use crate::AtheerCoreError;

/// Placeholder scaffold for a future streaming decryption scheme.
///
/// Streaming decryption would process model data in chunks, enabling
/// inference to begin before the entire file is decrypted. Not yet
/// implemented.
pub struct StreamingDecryption;

impl ModelEncryption for StreamingDecryption {
    fn decrypt_reader(&self, _path: &str) -> Result<Vec<u8>, AtheerCoreError> {
        Err(AtheerCoreError::ModelDecryptionFailed(
            "StreamingDecryption: not yet implemented".into(),
        ))
    }

    fn decrypt_mlpackage(&self, _path: &str) -> Result<String, AtheerCoreError> {
        Err(AtheerCoreError::ModelDecryptionFailed(
            "StreamingDecryption: not yet implemented".into(),
        ))
    }

    fn scrub(&self) {}
}

/// Placeholder scaffold for a future per-chunk encryption scheme.
///
/// Per-chunk encryption would encrypt each GGUF tensor chunk independently,
/// enabling selective decryption and streaming. Not yet implemented.
pub struct PerChunkEncryption;

impl ModelEncryption for PerChunkEncryption {
    fn decrypt_reader(&self, _path: &str) -> Result<Vec<u8>, AtheerCoreError> {
        Err(AtheerCoreError::ModelDecryptionFailed(
            "PerChunkEncryption: not yet implemented".into(),
        ))
    }

    fn decrypt_mlpackage(&self, _path: &str) -> Result<String, AtheerCoreError> {
        Err(AtheerCoreError::ModelDecryptionFailed(
            "PerChunkEncryption: not yet implemented".into(),
        ))
    }

    fn scrub(&self) {}
}
