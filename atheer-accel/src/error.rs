use thiserror::Error;

#[derive(Error, Debug)]
pub enum AccelError {
    #[error("Backend not available: {0}")]
    BackendNotAvailable(String),

    #[error("Backend unavailable: {0}")]
    BackendUnavailable(String),

    #[error("Operation failed: {0}")]
    OperationFailed(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Memory allocation failed: {0}")]
    MemoryAllocationFailed(String),

    #[error("Model compilation failed: {0}")]
    ModelCompilationFailed(String),

    #[deprecated(since = "0.1.0", note = "use InferenceEngine::generate() instead")]
    #[error("Deprecated: {0}")]
    Deprecated(String),
}

pub type Result<T> = std::result::Result<T, AccelError>;

impl From<candle_core::Error> for AccelError {
    fn from(e: candle_core::Error) -> Self {
        AccelError::OperationFailed(format!("Candle core error: {e}"))
    }
}
