use thiserror::Error;

#[derive(Error, Debug)]
pub enum MemoryBankError {
    #[error("Cache error: {0}")]
    CacheError(String),

    #[error("Sync error: {0}")]
    SyncError(String),

    #[error("Compression error: {0}")]
    CompressionError(String),

    #[error("Handoff error: {0}")]
    HandoffError(String),

    #[error("L1/L2 cache is empty")]
    CacheEmpty,

    #[error("L3 compressed storage not initialized")]
    StorageNotInitialized,

    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, MemoryBankError>;
