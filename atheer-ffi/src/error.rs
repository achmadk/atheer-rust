use thiserror::Error;

#[derive(Debug, Error, uniffi::Error)]
pub enum AtheerError {
    #[error("Model load failed: {msg}")]
    ModelLoadFailed { msg: String },
    #[error("Tokenizer load failed: {msg}")]
    TokenizerLoadFailed { msg: String },
    #[error("Generation failed: {msg}")]
    GenerationFailed { msg: String },
    #[error("Invalid parameters: {msg}")]
    InvalidParameters { msg: String },
    #[error("Engine not initialized")]
    NotInitialized,
    #[error("Invalid mode: {0}")]
    InvalidMode(String),

    #[error("Model decryption failed: {msg}")]
    ModelDecryptionFailed { msg: String },
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
        Core::ModelLoadFailed(m) => AtheerError::ModelLoadFailed { msg: m },
        Core::TokenizerLoadFailed(m) => AtheerError::TokenizerLoadFailed { msg: m },
        Core::GenerationFailed(m) => AtheerError::GenerationFailed { msg: m },
        Core::InvalidParameters(m) => AtheerError::InvalidParameters { msg: m },
        Core::ModelDecryptionFailed(m) => AtheerError::ModelDecryptionFailed { msg: m },
        Core::InvalidMagic { actual } => AtheerError::ModelLoadFailed {
            msg: format!("InvalidMagic {{ actual: {:?} }}", actual),
        },
        Core::InvalidVersion { version } => AtheerError::ModelLoadFailed {
            msg: format!("InvalidVersion {{ version: {version} }}"),
        },
        Core::InvalidCounts {
            tensor_count,
            metadata_kv_count,
            max_tensor_bytes,
            requested_tensor_bytes,
        } => AtheerError::ModelLoadFailed {
            msg: format!(
                "InvalidCounts {{ tensor_count: {tensor_count}, metadata_kv_count: {metadata_kv_count}, max_tensor_bytes: {max_tensor_bytes}, requested_tensor_bytes: {requested_tensor_bytes} }}"
            ),
        },
        Core::InvalidAlignment { value } => AtheerError::ModelLoadFailed {
            msg: format!("InvalidAlignment {{ value: {value} }}"),
        },
        Core::InvalidTensorBounds {
            tensor_name,
            offset,
            size,
            file_size,
        } => AtheerError::ModelLoadFailed {
            msg: format!(
                "InvalidTensorBounds {{ tensor: {tensor_name:?}, offset: {offset}, size: {size}, file_size: {file_size} }}"
            ),
        },
        Core::DuplicateTensorName { name } => AtheerError::ModelLoadFailed {
            msg: format!("DuplicateTensorName {{ name: {name:?} }}"),
        },
        other => AtheerError::ModelLoadFailed {
            msg: format!("{other}"),
        },
    }
}
