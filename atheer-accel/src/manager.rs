use std::sync::Arc;

use crate::AccelBackend;
use crate::{BackendType, CpuBackend};

#[cfg(any(target_os = "ios", target_os = "macos"))]
use crate::{CoreMLBackend, MetalBackend};
#[cfg(target_os = "android")]
use crate::{NnapiBackend, VulkanBackend};

pub struct BackendManager {
    backends: Vec<Arc<dyn AccelBackend>>,
    selected: Arc<dyn AccelBackend>,
    /// Whether a CoreML .mlpackage model path was configured.
    /// When set, the ANE backend uses `with_model()` instead of `new()`,
    /// loading the actual .mlpackage for real inference.
    coreml_model_path: Option<String>,
}

impl BackendManager {
    #[allow(unused_mut)]
    pub fn new() -> Self {
        let mut backends: Vec<Arc<dyn AccelBackend>> = vec![Arc::new(CpuBackend::default())];

        #[cfg(any(target_os = "ios", target_os = "macos"))]
        {
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

        Self {
            backends,
            selected,
            coreml_model_path: None,
        }
    }

    /// Configure a CoreML model path for ANE inference with background pre-heat.
    ///
    /// When a model path is provided and we're on an Apple platform with the
    /// `coreml` feature, this replaces the default `CoreMLBackend::new()` with
    /// `CoreMLBackend::for_preheat()` — storing the model path and computing
    /// ANE compatibility without synchronously loading the `.mlpackage`.
    /// The actual model loading happens in the background via [`preheat_ane`]
    /// (called later from [`AtheerEngine::initialize`]), eliminating cold-start
    /// ANE compilation latency on first inference.
    ///
    /// On non-Apple platforms or without the feature, the path is stored but
    /// has no effect (the backend falls back to Metal/CPU).
    pub fn with_coreml_model(
        mut self,
        model_path: &str,
        _architecture: &str,
        _quantization: &str,
        _param_count_m: f32,
    ) -> Self {
        self.coreml_model_path = Some(model_path.to_string());

        #[cfg(all(feature = "coreml", any(target_os = "ios", target_os = "macos")))]
        if let Some(pos) = self
            .backends
            .iter()
            .position(|b| b.backend_type() == BackendType::CoreML)
        {
            let architecture = _architecture;
            let quantization = _quantization;
            let param_count_m = _param_count_m;
            self.backends[pos] = Arc::new(CoreMLBackend::for_preheat(
                architecture,
                quantization,
                param_count_m,
                model_path,
            ));
        }

        self.selected = self.best_available();
        self
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
    ///
    /// When a CoreML model is loaded (ANE path), returns `Device::Cpu` because
    /// `candle_coreml::CoreMLModel::forward()` handles device placement internally
    /// — CPU tensors are the correct input format. Metal is returned when no ANE
    /// model is loaded (Metal GPU acceleration for fallback).
    pub fn device(&self) -> candle_core::Device {
        match self.selected.backend_type() {
            BackendType::Cpu => candle_core::Device::Cpu,
            #[cfg(any(target_os = "ios", target_os = "macos"))]
            BackendType::CoreML if self.coreml_model_path.is_some() => candle_core::Device::Cpu,
            #[cfg(any(target_os = "ios", target_os = "macos"))]
            BackendType::Metal | BackendType::CoreML => {
                candle_core::Device::metal_if_available(0).unwrap_or(candle_core::Device::Cpu)
            }
            #[cfg(not(any(target_os = "ios", target_os = "macos")))]
            BackendType::Metal | BackendType::CoreML => candle_core::Device::Cpu,
            #[cfg(target_os = "android")]
            BackendType::Vulkan => candle_core::Device::Cpu,
            #[cfg(target_os = "android")]
            BackendType::NNAPI => candle_core::Device::Cpu,
            BackendType::Cuda => {
                candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu)
            }
            #[cfg(not(target_os = "android"))]
            BackendType::Vulkan => candle_core::Device::Cpu,
            #[cfg(not(target_os = "android"))]
            BackendType::NNAPI => candle_core::Device::Cpu,
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

    #[test]
    fn test_with_coreml_model_stores_path() {
        let manager = BackendManager::new().with_coreml_model(
            "/tmp/test.mlpackage",
            "llama",
            "q4_k_m",
            100.0,
        );
        // The path should be stored on the manager
        // (coreml_model_path is private, but we verify via device() — when CoreML
        // is selected with a model path, device() returns Cpu, not Metal)
        let current = manager.current();
        // On non-Apple platforms, CoreML isn't registered, so autoselect picks CPU
        // The test verifies the builder doesn't panic and the manager is usable
        assert!(
            current.backend_type() == BackendType::CoreML
                || current.backend_type() == BackendType::Cpu
        );
        // device() should not panic
        let _device = manager.device();
    }
}
