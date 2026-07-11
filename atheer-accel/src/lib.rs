#![allow(deprecated)]

pub mod backend;
pub mod coreml;
pub mod cpu;
pub mod error;
pub mod manager;
pub mod metal;
#[cfg_attr(not(target_os = "android"), allow(dead_code))]
pub mod nnapi;
#[cfg(target_os = "android")]
pub mod nnapi_ndk;
pub mod traits;
#[cfg_attr(not(target_os = "android"), allow(dead_code))]
pub mod vulkan;

pub use backend::Backend;
pub use backend::BackendType;
pub use coreml::CoreMLBackend;
pub use cpu::CpuBackend;
pub use error::{AccelError, Result};
pub use manager::BackendManager;
pub use metal::MetalBackend;
pub use nnapi::NnapiBackend;
pub use traits::{AccelBackend, AccelResult};
pub use vulkan::VulkanBackend;
