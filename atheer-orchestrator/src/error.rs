use thiserror::Error;

#[derive(Error, Debug)]
pub enum OrchestratorError {
    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Mode switch failed: {0}")]
    ModeSwitchFailed(String),

    #[error("Hardware monitor unavailable: {0}")]
    HardwareMonitorError(String),

    #[error("Model loading failed: {0}")]
    ModelLoadError(String),

    #[error("Generation error: {0}")]
    GenerationError(String),
}

pub type Result<T> = std::result::Result<T, OrchestratorError>;
