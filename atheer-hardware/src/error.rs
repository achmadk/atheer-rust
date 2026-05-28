use thiserror::Error;

#[derive(Error, Debug)]
pub enum HardwareError {
    #[error("Platform not supported: {0}")]
    UnsupportedPlatform(String),

    #[error("JNI error: {0}")]
    JniError(String),

    #[error("Failed to get hardware info: {0}")]
    InfoError(String),
}

pub type Result<T> = std::result::Result<T, HardwareError>;
