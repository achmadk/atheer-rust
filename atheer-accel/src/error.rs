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

    #[error("GPU operation timed out after {0}ms")]
    GpuTimeout(u64),

    #[error("Tensor validation failed: {0}")]
    TensorValidationFailed(String),

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_timeout_display() {
        let err = AccelError::GpuTimeout(5000);
        let msg = format!("{}", err);
        assert_eq!(msg, "GPU operation timed out after 5000ms");
    }

    #[test]
    fn test_tensor_validation_display() {
        let err = AccelError::TensorValidationFailed("dims mismatch".to_string());
        let msg = format!("{}", err);
        assert_eq!(msg, "Tensor validation failed: dims mismatch");
    }
}
