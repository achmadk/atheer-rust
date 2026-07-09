use thiserror::Error;

#[derive(Debug, Error, uniffi::Error)]
pub enum AtheerError {
    #[error("Model load failed: {message}")]
    ModelLoadFailed { message: String },
    #[error("Tokenizer load failed: {message}")]
    TokenizerLoadFailed { message: String },
    #[error("Generation failed: {message}")]
    GenerationFailed { message: String },
    #[error("Invalid parameters: {message}")]
    InvalidParameters { message: String },
    #[error("Engine not initialized")]
    NotInitialized,
    #[error("Invalid mode: {0}")]
    InvalidMode(String),

    #[error("Model decryption failed: {message}")]
    ModelDecryptionFailed { message: String },
}
