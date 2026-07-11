uniffi::setup_scaffolding!();

pub mod accuracy;
pub mod block_manager;
pub mod crash;
pub mod error;
pub mod inference;
pub mod kv_cache;
pub mod kv_cache_bridge;
pub mod kv_cache_quantizer;
pub mod latency_budget;
pub mod lifecycle;
#[cfg(feature = "mmap")]
pub mod mmap_model;
pub mod model;
pub mod model_credential;
pub mod model_encryption;
#[cfg(feature = "model-registry")]
pub mod model_registry;
pub mod production;
pub mod quantization_resolver;
pub mod safety;
pub mod sampler;
pub mod security;
pub mod session;
pub mod streaming;
#[cfg(test)]
pub mod test_model;
pub mod tokenizer;
pub mod weights;

pub use block_manager::{BlockId, BlockManager, DEFAULT_BLOCK_SIZE, NULL_BLOCK};
pub use crash::CrashReporter;
pub use error::{AtheerCoreError, Result};
pub use inference::InferenceEngine;
pub use latency_budget::{LatencyBudget, LatencyTracker};
pub use lifecycle::{
    CheckpointHeader, EngineLifecycle, IncrementalCheckpoint, LifecycleConfig, LifecycleObserver,
};
pub use model::Model;
pub use model_credential::ModelCredential;
pub use model_encryption::{aes256_gcm::Aes256GcmEncryption, ModelEncryption};
pub use production::{ConfigError, ProductionConfig};
pub use quantization_resolver::{GpuTier, QuantizationResolver};
pub use safety::{
    ContentModeration, ContentModerationBuilder, InjectionDetector, ModerationStage,
    ModerationVerdict, OutputFilter, PiiRedactor, Severity, TopicBlocker,
};
pub use sampler::{Sampler, SamplingConfig};
pub use security::{SecurityAudit, SecurityError};
pub use session::Session;
pub use streaming::{
    callback_from_fn, GenerationState, NullCallback, SharedCallback, StreamingCallback,
};
pub use tokenizer::Tokenizer;
