//! NNAPI device implementation for candle-core.
//!
//! # Overview
//!
//! `NnapiDevice` represents a handle to an NNAPI acceleration device on Android.
//! NNAPI provides access to NPUs, GPUs, and DSPs for ML inference.
//!
//! # Creating a Device
//!
//! ```ignore
//! use candle_core::{Device, Result};
//!
//! fn main() -> Result<()> {
//!     // Create device for the first available NNAPI accelerator
//!     let device = Device::new_nnapi(0)?;
//!     Ok(())
//! }
//! ```
//!
//! # Device Discovery
//!
//! The executor probes available devices at startup and selects the best available
//! accelerator (NPU > GPU > CPU) for inference.

use crate::backend::BackendDevice;
use crate::{DType, DeviceLocation, Result, Shape};
use std::sync::Arc;

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub struct NnapiDevice {
    ordinal: usize,
    inner: Arc<NnapiDeviceInner>,
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
struct NnapiDeviceInner {
    location: DeviceLocation,
}

#[cfg(not(all(feature = "nnapi", target_os = "android")))]
pub struct NnapiDevice {
    ordinal: usize,
}

#[cfg(not(all(feature = "nnapi", target_os = "android")))]
impl NnapiDevice {
    pub fn new(_ordinal: usize) -> Result<Self> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
impl NnapiDevice {
    pub fn new(ordinal: usize) -> Result<Self> {
        if ordinal != 0 {
            crate::bail!("NNAPI only supports ordinal 0 on Android")
        }
        Ok(Self::new_internal(ordinal))
    }

    fn new_internal(ordinal: usize) -> Self {
        Self {
            ordinal,
            inner: Arc::new(NnapiDeviceInner {
                location: DeviceLocation::Nnapi { gpu_id: ordinal },
            }),
        }
    }

    pub fn location(&self) -> DeviceLocation {
        self.inner.location
    }

    pub fn set_seed(&self, _seed: u64) -> Result<()> {
        Ok(())
    }

    pub fn get_current_seed(&self) -> Result<u64> {
        Ok(0)
    }

    pub fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<crate::Storage> {
        crate::Storage::zeros_nnapi(self, shape, dtype)
    }

    pub fn rand_uniform_impl(
        &self,
        shape: &Shape,
        dtype: DType,
        lo: f64,
        up: f64,
    ) -> Result<crate::Storage> {
        crate::Storage::rand_uniform_nnapi(self, shape, dtype, lo, up)
    }

    pub fn rand_normal_impl(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<crate::Storage> {
        crate::Storage::rand_normal_nnapi(self, shape, dtype, mean, std)
    }

    pub fn storage_from_slice<D: crate::WithDType>(&self, data: &[D]) -> Result<crate::Storage> {
        crate::Storage::from_slice_nnapi(self, data)
    }

    pub fn storage_from_cpu_storage_owned(&self, cpu: crate::CpuStorage) -> Result<crate::Storage> {
        crate::Storage::from_cpu_storage_nnapi(self, cpu)
    }

    pub fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

impl std::fmt::Debug for NnapiDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NnapiDevice({})", self.ordinal)
    }
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
impl BackendDevice for NnapiDevice {
    fn location(&self) -> DeviceLocation {
        self.location()
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        self.set_seed(seed)
    }

    fn get_current_seed(&self) -> Result<u64> {
        self.get_current_seed()
    }
}
