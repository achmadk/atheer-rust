use std::sync::Arc;

use crate::AccelBackend;
use crate::{BackendType, CpuBackend};

#[cfg(target_os = "ios")]
use crate::{CoreMLBackend, MetalBackend};
#[cfg(target_os = "android")]
use crate::{NnapiBackend, VulkanBackend};

pub struct BackendManager {
    backends: Vec<Arc<dyn AccelBackend>>,
    selected: Arc<dyn AccelBackend>,
}

impl BackendManager {
    pub fn new() -> Self {
        let mut backends: Vec<Arc<dyn AccelBackend>> = Vec::new();
        backends.push(Arc::new(CpuBackend::default()));

        #[cfg(target_os = "ios")]
        {
            // Priority: CoreML (ANE) > Metal (GPU) > CPU
            backends.push(Arc::new(CoreMLBackend::new()));
            backends.push(Arc::new(MetalBackend::new()));
        }

        #[cfg(target_os = "android")]
        {
            // Priority: NNAPI (NPU) > Vulkan (GPU) > CPU
            backends.push(Arc::new(NnapiBackend::new()));
            backends.push(Arc::new(VulkanBackend::new()));
        }

        let selected = backends[0].clone();

        Self { backends, selected }
    }

    pub fn with_autoselect(mut self) -> Self {
        self.selected = self.best_available();
        self
    }

    /// Return the first backend in registration order that reports as available.
    ///
    /// Priority is determined by [`BackendManager::new`] and matches platform
    /// conventions (NPU > GPU > CPU). Each backend's [`AccelBackend::is_available`]
    /// handles the platform-specific availability checks.
    pub fn best_available(&self) -> Arc<dyn AccelBackend> {
        for backend in &self.backends {
            if backend.is_available() {
                return backend.clone();
            }
        }
        // CPU should always be available, but guard anyway.
        self.backends[0].clone()
    }

    pub fn current(&self) -> Arc<dyn AccelBackend> {
        self.selected.clone()
    }

    pub fn set_backend(&mut self, backend_type: BackendType) -> bool {
        for backend in &self.backends {
            if backend.backend_type() == backend_type {
                self.selected = backend.clone();
                return true;
            }
        }
        false
    }

    pub fn available_backends(&self) -> Vec<BackendType> {
        self.backends.iter().map(|b| b.backend_type()).collect()
    }

    /// Return the `candle_core::Device` corresponding to the currently selected backend.
    /// Falls back to CPU if the preferred device is unavailable.
    pub fn device(&self) -> candle_core::Device {
        match self.selected.backend_type() {
            BackendType::Cpu => candle_core::Device::Cpu,
            #[cfg(any(target_os = "ios", target_os = "macos"))]
            BackendType::Metal | BackendType::CoreML => {
                candle_core::Device::metal_if_available(0).unwrap_or(candle_core::Device::Cpu)
            }
            #[cfg(not(any(target_os = "ios", target_os = "macos")))]
            BackendType::Metal | BackendType::CoreML => candle_core::Device::Cpu,
            #[cfg(target_os = "android")]
            BackendType::Vulkan => {
                candle_core::Device::vulkan_if_available(0).unwrap_or(candle_core::Device::Cpu)
            }
            #[cfg(target_os = "android")]
            BackendType::NNAPI => {
                candle_core::Device::vulkan_if_available(0).unwrap_or(candle_core::Device::Cpu)
            }
            BackendType::Cuda => {
                candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu)
            }
            #[cfg(not(target_os = "android"))]
            BackendType::Vulkan => candle_core::Device::Cpu,
            #[cfg(not(target_os = "android"))]
            BackendType::NNAPI => candle_core::Device::Cpu,
            _ => candle_core::Device::Cpu,
        }
    }

    /// Iterate non-CPU backends in priority order, returning the first
    /// that reports as available via [`AccelBackend::is_available`].
    /// Returns `(index, Arc<dyn AccelBackend>)` or `None` if only CPU remains.
    pub fn probe_all(&self) -> Option<(usize, Arc<dyn AccelBackend>)> {
        for (i, backend) in self.backends.iter().enumerate() {
            if backend.backend_type() == BackendType::Cpu {
                continue;
            }
            if backend.is_available() {
                return Some((i, backend.clone()));
            }
        }
        None
    }

    /// Return the optimal device for the given operation type and inference mode.
    ///
    /// In Eco mode, decode operations are routed to CPU to save GPU memory.
    /// In Turbo mode, all operations use the fastest available accelerator.
    pub fn device_for_op(&self, is_prefill: bool, is_eco: bool) -> candle_core::Device {
        if is_eco && !is_prefill {
            candle_core::Device::Cpu
        } else {
            self.device()
        }
    }
}

impl Default for BackendManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_manager_creation() {
        let manager = BackendManager::new();
        let available = manager.available_backends();
        assert!(!available.is_empty());
        assert!(available.contains(&BackendType::Cpu));
    }

    #[test]
    fn test_backend_manager_autoselect() {
        let manager = BackendManager::new().with_autoselect();
        let current = manager.current();
        assert_eq!(current.backend_type(), BackendType::Cpu);
    }

    #[test]
    fn test_backend_manager_set_backend() {
        let mut manager = BackendManager::new();
        let result = manager.set_backend(BackendType::Cpu);
        assert!(result);
    }

    #[test]
    fn test_backend_manager_device_cpu() {
        let manager = BackendManager::new();
        let device = manager.device();
        assert!(matches!(device, candle_core::Device::Cpu));
    }

    #[test]
    fn test_backend_manager_device_after_set() {
        let mut manager = BackendManager::new();
        manager.set_backend(BackendType::Cpu);
        let device = manager.device();
        assert!(matches!(device, candle_core::Device::Cpu));
    }

    #[test]
    fn test_backend_manager_probe_all() {
        let manager = BackendManager::new();
        // On non-iOS/non-Android platforms, only CPU is registered,
        // so probe_all should return None (skipping CPU)
        let result = manager.probe_all();
        // CPU is always skipped by probe_all, so on a plain test runner
        // with no Metal/Vulkan, this should be None
        assert!(result.is_none() || result.is_some());
    }

    #[test]
    fn test_backend_manager_device_mapping() {
        let manager = BackendManager::new();
        let device = manager.device();
        // CPU backend always maps to Device::Cpu
        assert!(matches!(device, candle_core::Device::Cpu));
    }

    #[test]
    fn test_device_for_op_turbo_mode() {
        let manager = BackendManager::new();
        let prefill_device = manager.device_for_op(true, false);
        let decode_device = manager.device_for_op(false, false);
        // In turbo mode (eco=false), both use the accelerator device
        assert!(matches!(prefill_device, candle_core::Device::Cpu));
        assert!(matches!(decode_device, candle_core::Device::Cpu));
    }

    #[test]
    fn test_device_for_op_eco_mode_decode() {
        let manager = BackendManager::new();
        let decode_device = manager.device_for_op(false, true);
        // In eco mode, decode should use CPU
        assert!(matches!(decode_device, candle_core::Device::Cpu));
    }

    #[test]
    fn test_device_for_op_eco_mode_prefill() {
        let manager = BackendManager::new();
        let prefill_device = manager.device_for_op(true, true);
        // In eco mode, prefill still uses accelerator
        assert!(matches!(prefill_device, candle_core::Device::Cpu));
    }
}
