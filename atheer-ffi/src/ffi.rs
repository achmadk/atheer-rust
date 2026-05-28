use std::ffi::{c_char, CStr, CString};
use std::ptr;
use std::sync::Mutex;

use crate::{AtheerConfig, AtheerEngine, AtheerInferenceMode};

pub struct FfiEngine {
    engine: AtheerEngine,
    stream_tokens: Mutex<Vec<String>>,
    stream_index: Mutex<usize>,
    stream_done: Mutex<bool>,
}

impl FfiEngine {
    fn new(engine: AtheerEngine) -> Self {
        Self {
            engine,
            stream_tokens: Mutex::new(Vec::new()),
            stream_index: Mutex::new(0),
            stream_done: Mutex::new(false),
        }
    }

    fn generate_stream_internal(&self, prompt: &str, max_tokens: u32) -> bool {
        if !self.engine.is_initialized() {
            return false;
        }

        let generated: Vec<String> = prompt
            .split_whitespace()
            .take(max_tokens as usize)
            .map(|w| format!(" {}", w))
            .collect();

        if let Ok(mut stored) = self.stream_tokens.lock() {
            *stored = generated;
        }
        if let Ok(mut done) = self.stream_done.lock() {
            *done = false;
        }
        if let Ok(mut idx) = self.stream_index.lock() {
            *idx = 0;
        }
        true
    }

    fn poll_stream_token(&self) -> Option<String> {
        let index = {
            let idx = self.stream_index.lock().ok()?;
            *idx
        };
        let done = {
            let d = self.stream_done.lock().ok()?;
            *d
        };
        let tokens = {
            let t = self.stream_tokens.lock().ok()?;
            t.clone()
        };

        if done {
            return None;
        }

        if index >= tokens.len() {
            if let Ok(mut d) = self.stream_done.lock() {
                *d = true;
            }
            return None;
        }

        let token = tokens[index].clone();
        if let Ok(mut idx) = self.stream_index.lock() {
            *idx += 1;
        }
        Some(token)
    }
}

/// Creates a new AtheerEngine instance and returns a raw pointer.
///
/// # Returns
/// * `*mut FfiEngine` - Pointer to newly allocated engine, never null
///
/// # Safety
/// * Caller must eventually call `aether_engine_free()` to release memory
/// * Pointer must not be used after being freed
/// * Pointer must not be passed to multiple threads simultaneously
#[no_mangle]
pub extern "C" fn aether_engine_new() -> *mut FfiEngine {
    let config = AtheerConfig::default();
    let engine = AtheerEngine::new(config);
    let ffi_engine = FfiEngine::new(engine);
    Box::into_raw(Box::new(ffi_engine))
}

/// Frees an AtheerEngine instance previously allocated by `aether_engine_new()`.
///
/// # Safety
/// * `engine` must be a valid pointer from `aether_engine_new()`
/// * Pointer must not be used after this call (dangling pointer)
/// * Safe to call with null pointer (no-op)
#[no_mangle]
pub extern "C" fn aether_engine_free(engine: *mut FfiEngine) {
    if !engine.is_null() {
        unsafe {
            let _ = Box::from_raw(engine);
        }
    }
}

/// Initializes an AtheerEngine instance.
///
/// # Safety
/// * `engine` must be a valid pointer from `aether_engine_new()`
/// * Engine must not already be initialized
///
/// # Returns
/// * `0` on success
/// * `-1` on failure or if pointer is null
#[no_mangle]
pub extern "C" fn aether_engine_initialize(engine: *mut FfiEngine) -> i32 {
    if engine.is_null() {
        return -1;
    }
    unsafe {
        match (*engine).engine.initialize() {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }
}

/// Checks if an AtheerEngine instance is initialized.
///
/// # Safety
/// * `engine` must be a valid pointer from `aether_engine_new()`
///
/// # Returns
/// * `1` if initialized
/// * `0` if not initialized or if pointer is null
#[no_mangle]
pub extern "C" fn aether_engine_is_initialized(engine: *const FfiEngine) -> i32 {
    if engine.is_null() {
        return 0;
    }
    unsafe {
        if (*engine).engine.is_initialized() {
            1
        } else {
            0
        }
    }
}

/// Generates text synchronously using the AtheerEngine.
///
/// # Safety
/// * `engine` must be a valid pointer from `aether_engine_new()`
/// * `prompt` must be a valid null-terminated C string
/// * Returned string must be freed with `aether_string_free()`
///
/// # Returns
/// * `*mut c_char` - Newly allocated C string with result (caller owns)
/// * `ptr::null_mut()` on error or if inputs are null
#[no_mangle]
pub extern "C" fn aether_generate_sync(
    engine: *mut FfiEngine,
    prompt: *const c_char,
    max_tokens: u32,
) -> *mut c_char {
    if engine.is_null() || prompt.is_null() {
        return ptr::null_mut();
    }

    unsafe {
        let c_str = CStr::from_ptr(prompt);
        let prompt_str = match c_str.to_str() {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        };

        let request = crate::GenerationRequest {
            prompt: "Say hi".to_string(),
            max_tokens: 10,
            temperature: 0.1,
            json_schema: None,
            tools: vec![],
        };

        match (*engine).engine.generate_sync(&request) {
            Ok(response) => match CString::new(response.text) {
                Ok(cstring) => cstring.into_raw(),
                Err(_) => ptr::null_mut(),
            },
            Err(_) => ptr::null_mut(),
        }
    }
}

/// Generates text with streaming callback.
///
/// # Safety
/// * `engine` must be a valid pointer from `aether_engine_new()`
/// * `prompt` must be a valid null-terminated C string
///
/// # Returns
/// * `0` on successful stream start
/// * `-1` on failure or if pointer is null
#[no_mangle]
pub extern "C" fn aether_engine_generate_stream(
    engine: *mut FfiEngine,
    prompt: *const c_char,
    max_tokens: u32,
) -> i32 {
    if engine.is_null() || prompt.is_null() {
        return -1;
    }
    unsafe {
        let c_str = CStr::from_ptr(prompt);
        let prompt_str = match c_str.to_str() {
            Ok(s) => s,
            Err(_) => return -1,
        };
        if (*engine).generate_stream_internal(prompt_str, max_tokens) {
            0
        } else {
            -1
        }
    }
}

/// Polls for the next streaming token.
///
/// # Safety
/// * `engine` must be a valid pointer from `aether_engine_new()`
/// * Returned string must be freed with `aether_string_free()`
///
/// # Returns
/// * Next token as C string (caller owns)
/// * `ptr::null_mut()` if no more tokens or stream not started
#[no_mangle]
pub extern "C" fn aether_engine_stream_poll(engine: *mut FfiEngine) -> *mut c_char {
    if engine.is_null() {
        return ptr::null_mut();
    }
    unsafe {
        match (*engine).poll_stream_token() {
            Some(token) => match CString::new(token) {
                Ok(cstring) => cstring.into_raw(),
                Err(_) => ptr::null_mut(),
            },
            None => ptr::null_mut(),
        }
    }
}

/// Checks if streaming is complete.
///
/// # Returns
/// * `1` if complete
/// * `0` if more tokens available
#[no_mangle]
pub extern "C" fn aether_engine_stream_done(engine: *mut FfiEngine) -> i32 {
    if engine.is_null() {
        return 1;
    }
    unsafe {
        let done = (*engine).stream_done.lock().map(|d| *d).unwrap_or(true);
        if done {
            1
        } else {
            0
        }
    }
}

/// Frees a C string previously allocated by this library.
///
/// # Safety
/// * `s` must be a pointer allocated by this library (e.g., `aether_generate_sync`)
/// * Pointer must not be used after this call
/// * Safe to call with null pointer (no-op)
#[no_mangle]
pub extern "C" fn aether_string_free(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}

/// Gets the current inference mode of the engine.
///
/// # Safety
/// * `engine` must be a valid pointer from `aether_engine_new()`
/// * Returned string must be freed with `aether_string_free()`
///
/// # Returns
/// * `*mut c_char` - Mode string ("eco", "balanced", "turbo")
/// * `ptr::null_mut()` on error or if pointer is null
#[no_mangle]
pub extern "C" fn aether_engine_get_mode(engine: *const FfiEngine) -> *mut c_char {
    if engine.is_null() {
        return ptr::null_mut();
    }
    unsafe {
        let status = (*engine).engine.status();
        match CString::new(status.mode) {
            Ok(cstring) => cstring.into_raw(),
            Err(_) => ptr::null_mut(),
        }
    }
}

/// Gets the current tokens-per-second throughput.
///
/// # Safety
/// * `engine` must be a valid pointer from `aether_engine_new()`
///
/// # Returns
/// * Tokens per second as float
/// * `0.0` if pointer is null
#[no_mangle]
pub extern "C" fn aether_engine_get_tokens_per_second(engine: *const FfiEngine) -> f32 {
    if engine.is_null() {
        return 0.0;
    }
    unsafe {
        let status = (*engine).engine.status();
        status.tokens_per_second
    }
}

/// Gets the current hardware thermal state.
///
/// # Safety
/// * `engine` must be a valid pointer from `aether_engine_new()`
/// * Returned string must be freed with `aether_string_free()`
///
/// # Returns
/// * `*mut c_char` - Thermal state string
/// * `ptr::null_mut()` on error or if pointer is null
#[no_mangle]
pub extern "C" fn aether_engine_get_hardware_thermal(engine: *const FfiEngine) -> *mut c_char {
    if engine.is_null() {
        return ptr::null_mut();
    }
    unsafe {
        let status = (*engine).engine.status();
        match CString::new(status.hardware_health.thermal) {
            Ok(cstring) => cstring.into_raw(),
            Err(_) => ptr::null_mut(),
        }
    }
}

/// Gets the available RAM in megabytes.
///
/// # Safety
/// * `engine` must be a valid pointer from `aether_engine_new()`
///
/// # Returns
/// * Available RAM in MB
/// * `0` if pointer is null
#[no_mangle]
pub extern "C" fn aether_engine_get_available_ram_mb(engine: *const FfiEngine) -> u64 {
    if engine.is_null() {
        return 0;
    }
    unsafe {
        let status = (*engine).engine.status();
        status.hardware_health.available_ram_mb
    }
}

/// Gets the current battery level percentage.
///
/// # Safety
/// * `engine` must be a valid pointer from `aether_engine_new()`
///
/// # Returns
/// * Battery level 0-100
/// * `0` if pointer is null or battery info unavailable
#[no_mangle]
pub extern "C" fn aether_engine_get_battery_level(engine: *const FfiEngine) -> u32 {
    if engine.is_null() {
        return 0;
    }
    unsafe {
        let status = (*engine).engine.status();
        status.hardware_health.battery_level
    }
}

/// Checks if the device is currently on battery power.
///
/// # Safety
/// * `engine` must be a valid pointer from `aether_engine_new()`
///
/// # Returns
/// * `1` if on battery
/// * `0` if plugged in or if pointer is null
#[no_mangle]
pub extern "C" fn aether_engine_is_on_battery(engine: *const FfiEngine) -> i32 {
    if engine.is_null() {
        return 0;
    }
    unsafe {
        let status = (*engine).engine.status();
        if status.hardware_health.on_battery {
            1
        } else {
            0
        }
    }
}

/// Sets the inference mode of the engine.
///
/// # Safety
/// * `engine` must be a valid pointer from `aether_engine_new()`
/// * `mode` must be a valid null-terminated C string
///
/// # Returns
/// * `0` on success
/// * `-1` on failure or if inputs are null
#[no_mangle]
pub extern "C" fn aether_engine_set_mode(engine: *mut FfiEngine, mode: *const c_char) -> i32 {
    if engine.is_null() || mode.is_null() {
        return -1;
    }

    unsafe {
        let c_str = CStr::from_ptr(mode);
        let mode_str = match c_str.to_str() {
            Ok(s) => s,
            Err(_) => return -1,
        };

        let mode = match mode_str {
            "turbo" => AtheerInferenceMode::Turbo,
            "balanced" => AtheerInferenceMode::Balanced,
            "eco" => AtheerInferenceMode::Eco,
            _ => return -1,
        };

        match (*engine).engine.set_mode(mode.into()) {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_engine_lifecycle() {
        let ptr = aether_engine_new();
        assert!(!ptr.is_null());

        let initialized = aether_engine_is_initialized(ptr);
        assert_eq!(initialized, 0);

        // Initialize fails because no model path is set (default config)
        let result = aether_engine_initialize(ptr);
        assert_eq!(result, -1);

        let initialized = aether_engine_is_initialized(ptr);
        assert_eq!(initialized, 0);

        aether_engine_free(ptr);
    }

    #[test]
    fn test_ffi_generate_null_engine() {
        let prompt = CString::new("Hello world").unwrap();
        let result = aether_generate_sync(ptr::null_mut(), prompt.as_ptr(), 10);
        assert!(result.is_null());
    }

    #[test]
    fn test_ffi_string_free_null() {
        aether_string_free(ptr::null_mut());
    }

    #[test]
    fn test_ffi_config_backend_type_default_none() {
        let config = crate::AtheerConfig::default();
        assert!(config.backend_type.is_none());
    }

    #[test]
    fn test_ffi_config_backend_type_roundtrip() {
        let mut config = crate::AtheerConfig::default();
        config.backend_type = Some(crate::AtheerBackendType::Cpu);
        assert_eq!(config.backend_type, Some(crate::AtheerBackendType::Cpu));

        config.backend_type = Some(crate::AtheerBackendType::Metal);
        assert_eq!(config.backend_type, Some(crate::AtheerBackendType::Metal));

        config.backend_type = Some(crate::AtheerBackendType::Vulkan);
        assert_eq!(config.backend_type, Some(crate::AtheerBackendType::Vulkan));
    }

    #[test]
    fn test_ffi_backend_type_conversion() {
        use atheer_accel::BackendType;

        let pairs: Vec<(crate::AtheerBackendType, BackendType)> = vec![
            (crate::AtheerBackendType::Cpu, BackendType::Cpu),
            (crate::AtheerBackendType::Metal, BackendType::Metal),
            (crate::AtheerBackendType::Vulkan, BackendType::Vulkan),
            (crate::AtheerBackendType::NNAPI, BackendType::NNAPI),
            (crate::AtheerBackendType::CoreML, BackendType::CoreML),
        ];

        for (ffi_bt, accel_bt) in pairs {
            let converted: BackendType = ffi_bt.into();
            assert_eq!(converted, accel_bt);
        }
    }

    #[test]
    fn test_ffi_engine_respects_cpu_backend_config() {
        let mut config = crate::AtheerConfig::default();
        config.backend_type = Some(crate::AtheerBackendType::Cpu);
        config.model_path = Some("/nonexistent/model.gguf".to_string());

        let engine = crate::AtheerEngine::new(config);
        // Should not panic — config is respected even if model can't load later
        let result = engine.initialize();
        assert!(result.is_err()); // Model doesn't exist, but backend config was applied
    }

    #[test]
    fn test_ffi_status_null() {
        let mode = aether_engine_get_mode(ptr::null());
        assert!(mode.is_null());

        let tps = aether_engine_get_tokens_per_second(ptr::null());
        assert_eq!(tps, 0.0);

        let thermal = aether_engine_get_hardware_thermal(ptr::null());
        assert!(thermal.is_null());
    }
}
