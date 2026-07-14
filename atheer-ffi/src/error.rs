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

/// Convert a typed `AtheerCoreError` to the FFI-layer `AtheerError`.
///
/// The six S6 typed variants (InvalidMagic, InvalidVersion, InvalidCounts,
/// InvalidAlignment, InvalidTensorBounds, DuplicateTensorName) preserve their
/// structured fields in the message string so consumers can still classify
/// load failures without parsing. All other variants collapse into
/// `ModelLoadFailed` with the core-layer `Display` message.
pub fn map_core_error(err: atheer_core::AtheerCoreError) -> AtheerError {
    use atheer_core::AtheerCoreError as Core;
    match err {
        Core::ModelLoadFailed(m) => AtheerError::ModelLoadFailed { message: m },
        Core::TokenizerLoadFailed(m) => AtheerError::TokenizerLoadFailed { message: m },
        Core::GenerationFailed(m) => AtheerError::GenerationFailed { message: m },
        Core::InvalidParameters(m) => AtheerError::InvalidParameters { message: m },
        Core::ModelDecryptionFailed(m) => AtheerError::ModelDecryptionFailed { message: m },
        Core::InvalidMagic { actual } => AtheerError::ModelLoadFailed {
            message: format!("InvalidMagic {{ actual: {:?} }}", actual),
        },
        Core::InvalidVersion { version } => AtheerError::ModelLoadFailed {
            message: format!("InvalidVersion {{ version: {version} }}"),
        },
        Core::InvalidCounts {
            tensor_count,
            metadata_kv_count,
            max_tensor_bytes,
            requested_tensor_bytes,
        } => AtheerError::ModelLoadFailed {
            message: format!(
                "InvalidCounts {{ tensor_count: {tensor_count}, metadata_kv_count: {metadata_kv_count}, max_tensor_bytes: {max_tensor_bytes}, requested_tensor_bytes: {requested_tensor_bytes} }}"
            ),
        },
        Core::InvalidAlignment { value } => AtheerError::ModelLoadFailed {
            message: format!("InvalidAlignment {{ value: {value} }}"),
        },
        Core::InvalidTensorBounds {
            tensor_name,
            offset,
            size,
            file_size,
        } => AtheerError::ModelLoadFailed {
            message: format!(
                "InvalidTensorBounds {{ tensor: {tensor_name:?}, offset: {offset}, size: {size}, file_size: {file_size} }}"
            ),
        },
        Core::DuplicateTensorName { name } => AtheerError::ModelLoadFailed {
            message: format!("DuplicateTensorName {{ name: {name:?} }}"),
        },
        other => AtheerError::ModelLoadFailed {
            message: format!("{other}"),
        },
    }
}
