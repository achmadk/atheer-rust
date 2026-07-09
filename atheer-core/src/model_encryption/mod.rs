pub mod aes256_gcm;
pub mod stubs;

use std::io::Cursor;

/// Trait for decrypting model files at load time.
///
/// Implementations handle decryption of GGUF files and `.mlpackage`
/// weight-file bundles. The trait is `Send + Sync` so it can be shared
/// across threads and exposed over UniFFI.
pub trait ModelEncryption: Send + Sync {
    /// Decrypt a single model file (GGUF) and return the plaintext bytes.
    ///
    /// `path` is the filesystem path to the encrypted file.
    fn decrypt_reader(&self, path: &str) -> Result<Vec<u8>, crate::AtheerCoreError>;

    /// Decrypt weight files within an `.mlpackage` bundle.
    ///
    /// `path` is the filesystem path to the `.mlpackage` directory.
    /// Implementations MUST decrypt each `.bin` weight file independently
    /// and return the path to a temporary directory containing the
    /// decrypted bundle (or modify in-place if the source is writable).
    fn decrypt_mlpackage(&self, path: &str) -> Result<String, crate::AtheerCoreError>;

    /// Scrub any sensitive material (keys, plaintext buffers) from memory.
    fn scrub(&self);
}

/// Helper: wrap decrypted bytes into a `Cursor<Vec<u8>>` for use with
/// `Model::from_gguf_reader()`.
pub fn decrypted_cursor(bytes: Vec<u8>) -> Cursor<Vec<u8>> {
    Cursor::new(bytes)
}
