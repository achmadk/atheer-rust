pub mod backend_type;
pub mod config;
pub mod engine;
pub mod error;
pub mod ffi;
pub mod mode;
pub mod privacy;
pub mod status;
pub mod streaming;
pub mod thermal;
pub mod types;

pub use backend_type::AtheerBackendType;
pub use config::AtheerConfig;
pub use engine::AtheerEngine;
pub use error::AtheerError;
pub use ffi::*;
pub use mode::AtheerInferenceMode;
pub use privacy::AtheerPrivacyMode;
pub use status::{EngineStatus, HardwareHealth, MemoryBankStatus};
pub use streaming::{StreamingCallback, StreamingResult};
pub use thermal::AtheerThermalState;
pub use types::{GenerationRequest, GenerationResponse};

uniffi::setup_scaffolding!();
