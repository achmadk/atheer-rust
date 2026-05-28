pub mod accuracy;
pub mod block_manager;
pub mod crash;
pub mod error;
pub mod inference;
pub mod kv_cache_bridge;
pub mod kv_cache;
pub mod kv_cache_quantizer;
pub mod latency_budget;
pub mod lifecycle;
pub mod model;
#[cfg(feature = "mmap")]
pub mod mmap_model;
#[cfg(feature = "model-registry")]
pub mod model_registry;
pub mod production;
pub mod quantization_resolver;
pub mod safety;
pub mod weights;
pub mod sampler;
pub mod security;
pub mod session;
pub mod streaming;
#[cfg(test)]
pub mod test_model;
pub mod tokenizer;

pub use block_manager::{BlockManager, BlockId, NULL_BLOCK, DEFAULT_BLOCK_SIZE};
pub use crash::CrashReporter;
pub use error::{AtheerCoreError, Result};
pub use inference::InferenceEngine;
pub use latency_budget::{LatencyBudget, LatencyTracker};
pub use lifecycle::{
    CheckpointHeader, EngineLifecycle, IncrementalCheckpoint, LifecycleConfig, LifecycleObserver,
};
pub use model::Model;
pub use production::{ConfigError, ProductionConfig};
pub use quantization_resolver::{GpuTier, QuantizationResolver};
pub use safety::{
    ContentModeration, ContentModerationBuilder, InjectionDetector, ModerationStage,
    ModerationVerdict, OutputFilter, PiiRedactor, Severity, TopicBlocker,
};
pub use sampler::{Sampler, SamplingConfig};
pub use security::{SecurityAudit, SecurityError};
pub use session::Session;
pub use streaming::{callback_from_fn, GenerationState, NullCallback, SharedCallback, StreamingCallback};
pub use tokenizer::Tokenizer;
