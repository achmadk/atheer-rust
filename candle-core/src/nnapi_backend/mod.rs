//! NNAPI backend module for candle-core.
//!
//! This module provides NNAPI (Neural Networks API) support for Android devices,
//! enabling acceleration via NPUs, GPUs, and DSPs.
//!
//! # Architecture
//!
//! - [`NnapiDevice`](device::NnapiDevice) - Device handle for NNAPI acceleration
//! - [`NnapiStorage`](storage::NnapiStorage) - Storage wrapper for NNAPI tensors
//! - [`NnapiExecutor`](executor::NnapiExecutor) - Device discovery and execution engine
//! - [`NnapiGraphBuilder`](graph::NnapiGraphBuilder) - Graph construction API
//! - [`NnapiCompiledModel`](graph::NnapiCompiledModel) - Compiled model handle
//!
//! # Usage
//!
//! ```ignore
//! use candle_core::{Device, Tensor, Result};
//!
//! fn main() -> Result<()> {
//!     // Create NNAPI device (Android only)
//!     let device = Device::new_nnapi(0)?;
//!
//!     // Create tensors on NNAPI device
//!     let a = Tensor::randn(0.0, 1.0, (128, 512), &device)?;
//!     let b = Tensor::randn(0.0, 1.0, (512, 256), &device)?;
//!
//!     // Matrix multiplication via NNAPI FullyConnected
//!     let c = a.matmul(&b)?;
//!
//!     // Binary operations via NNAPI
//!     let d = (&a + &b)?;
//!     let e = (&a * &b)?;
//!
//!     Ok(())
//! }
//! ```
//!
//! # Supported Operations
//!
//! - FullyConnected (matmul)
//! - Add, Mul (element-wise binary)
//! - Relu, Tanh, Logistic, Softmax (unary)
//! - Conv2d (via CPU fallback for complex configurations)
//!
//! # Requirements
//!
//! - Rust 1.75+
//! - Android API 27+ (NNAPI level 1)
//! - `target_os = "android"`
//! - `feature = "nnapi"`

#[cfg(all(feature = "nnapi", target_os = "android"))]
mod device;
#[cfg(not(all(feature = "nnapi", target_os = "android")))]
mod device;

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub use device::NnapiDevice;
#[cfg(not(all(feature = "nnapi", target_os = "android")))]
pub use device::NnapiDevice;

#[cfg(all(feature = "nnapi", target_os = "android"))]
mod storage;
#[cfg(not(all(feature = "nnapi", target_os = "android")))]
mod storage;

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub use storage::NnapiStorage;
#[cfg(not(all(feature = "nnapi", target_os = "android")))]
pub use storage::NnapiStorage;

#[cfg(all(feature = "nnapi", target_os = "android"))]
mod executor;
#[cfg(not(all(feature = "nnapi", target_os = "android")))]
mod executor;

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub use executor::{
    create_shared_executor, BinaryOp, NnapiDeviceKind, NnapiExecutor, NnapiExecutorDevice,
    SharedExecutor, UnaryOp,
};
#[cfg(not(all(feature = "nnapi", target_os = "android")))]
pub use executor::{
    create_shared_executor, BinaryOp, NnapiDeviceKind, NnapiExecutor, NnapiExecutorDevice,
    SharedExecutor, UnaryOp,
};

#[cfg(all(feature = "nnapi", target_os = "android"))]
mod graph;
#[cfg(not(all(feature = "nnapi", target_os = "android")))]
mod graph;

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub use graph::{
    tensor_f32_type, BinaryOp as GraphBinaryOp, ExecutionPreference, NnapiCompiledModel,
    NnapiGraphBuilder, NnapiOperation, UnaryOp as GraphUnaryOp,
};
#[cfg(not(all(feature = "nnapi", target_os = "android")))]
pub use graph::{
    tensor_f32_type, BinaryOp as GraphBinaryOp, ExecutionPreference, NnapiCompiledModel,
    NnapiGraphBuilder, NnapiOperation, UnaryOp as GraphUnaryOp,
};

#[cfg(all(feature = "nnapi", target_os = "android"))]
mod nnapi_ndk;
#[cfg(not(all(feature = "nnapi", target_os = "android")))]
mod nnapi_ndk;

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub use nnapi_ndk::NnapiError;
#[cfg(not(all(feature = "nnapi", target_os = "android")))]
pub use nnapi_ndk::NnapiError;

#[derive(thiserror::Error, Debug)]
pub enum NnapiError {
    #[error("{0}")]
    Message(String),
    #[error("NNAPI error: {0}")]
    Nnapi(String),
    #[error("buffer error: {0}")]
    Buffer(String),
}

impl From<String> for NnapiError {
    fn from(e: String) -> Self {
        NnapiError::Message(e)
    }
}
