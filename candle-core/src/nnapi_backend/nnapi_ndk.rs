//! Raw FFI bindings to the Android NNAPI NDK (NeuralNetworks.h).
//!
//! NNAPI is loaded at runtime via dlopen/dlsym because libneuralnetworks.so
//! is a device-only system library — it's NOT available in the NDK sysroot
//! for compile-time linking.
#![allow(dead_code)]

use std::os::raw::{c_char, c_void};

// ── Opaque NNAPI type handles ──────────────────────────────────────────────

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub enum ANeuralNetworksMemory {}

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub enum ANeuralNetworksModel {}

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub enum ANeuralNetworksCompilation {}

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub enum ANeuralNetworksExecution {}

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub enum ANeuralNetworksDevice {}

#[cfg(all(feature = "nnapi", target_os = "android"))]
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ANeuralNetworksOperandType {
    pub type_: i32,
    pub dimension_count: u32,
    pub dimensions: *const u32,
    pub scale: f32,
    pub zero_point: i32,
}

// ── AHardwareBuffer types (part of Android NDK, separate library) ──────────

#[cfg(all(feature = "nnapi", target_os = "android"))]
#[repr(C)]
pub struct AHardwareBuffer {}

#[cfg(all(feature = "nnapi", target_os = "android"))]
#[repr(C)]
pub struct AHardwareBuffer_Desc {
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub format: u32,
    pub usage: u64,
    pub stride: u32,
    _rfu0: u32,
    _rfu1: u64,
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
impl AHardwareBuffer_Desc {
    /// Create a new descriptor with default reserved fields.
    pub const fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            layers: 0,
            format: 0,
            usage: 0,
            stride: 0,
            _rfu0: 0,
            _rfu1: 0,
        }
    }
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const AHARDWAREBUFFER_FORMAT_BLOB: u32 = 33;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN: u64 = 3;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN: u64 = 2;

// ── NNAPI Constants ────────────────────────────────────────────────────────

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_FLOAT32: i32 = 0;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_INT32: i32 = 1;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_UINT32: i32 = 2;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_TENSOR_FLOAT32: i32 = 3;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_TENSOR_QUANT8_ASYMM: i32 = 5;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_TENSOR_FLOAT16: i32 = 8;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED: i32 = 14;

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_ADD: i32 = 0;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_MUL: i32 = 1;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_CONV_2D: i32 = 2;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_CONCATENATION: i32 = 3;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_FULLY_CONNECTED: i32 = 9;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_LOGISTIC: i32 = 14;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_RELU: i32 = 15;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_TANH: i32 = 16;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_RESHAPE: i32 = 22;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_SOFTMAX: i32 = 25;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_TRANSPOSE: i32 = 32;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_BATCH_TO_SPACE_ND: i32 = 27;

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_FUSED_NONE: i32 = 0;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_FUSED_RELU: i32 = 1;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_FUSED_RELU1: i32 = 2;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_FUSED_RELU6: i32 = 3;

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_PREFER_LOW_POWER: i32 = 0;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER: i32 = 1;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_PREFER_SUSTAINED_SPEED: i32 = 2;

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_NO_ERROR: i32 = 0;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_OUT_OF_MEMORY: i32 = 1;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_INCOMPLETE: i32 = 2;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_UNEXPECTED_NULL: i32 = 3;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_BAD_DATA: i32 = 4;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_OP_FAILED: i32 = 5;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_BAD_STATE: i32 = 6;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_UNMAPPABLE: i32 = 7;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_OUTPUT_INSUFFICIENT_SIZE: i32 = 8;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_UNAVAILABLE_DEVICE: i32 = 9;

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_DEVICE_UNKNOWN: i32 = 0;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_DEVICE_OTHER: i32 = 1;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_DEVICE_CPU: i32 = 2;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_DEVICE_GPU: i32 = 3;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_DEVICE_ACCELERATOR: i32 = 4;

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_FEATURE_LEVEL_1: i32 = 27;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_FEATURE_LEVEL_2: i32 = 28;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_FEATURE_LEVEL_3: i32 = 29;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_FEATURE_LEVEL_4: i32 = 30;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const ANEURALNETWORKS_FEATURE_LEVEL_5: i32 = 31;

// ── Runtime-loaded NNAPI function pointers ─────────────────────────────────
// libneuralnetworks.so is only available on-device, not in the NDK sysroot,
// so all symbols must be resolved at runtime via dlopen + dlsym.

#[cfg(all(feature = "nnapi", target_os = "android"))]
mod runtime {
    use super::*;
    use std::ffi::CStr;
    use std::sync::OnceLock;

    struct NnapiHandle {
        _lib: *mut std::ffi::c_void,
        // ── Device discovery ────────────────────────────────────────────
        get_device_count: unsafe extern "C" fn(*mut u32) -> i32,
        get_device:
            unsafe extern "C" fn(u32, *mut *mut ANeuralNetworksDevice) -> i32,
        device_get_name:
            unsafe extern "C" fn(*const ANeuralNetworksDevice, *mut *const c_char) -> i32,
        device_get_type: unsafe extern "C" fn(*const ANeuralNetworksDevice, *mut i32) -> i32,
        device_get_version:
            unsafe extern "C" fn(*const ANeuralNetworksDevice, *mut *const c_char) -> i32,
        device_get_feature_level:
            unsafe extern "C" fn(*const ANeuralNetworksDevice, *mut i32) -> i32,
        // ── Model ───────────────────────────────────────────────────────
        model_create: unsafe extern "C" fn(*mut *mut ANeuralNetworksModel) -> i32,
        model_free: unsafe extern "C" fn(*mut ANeuralNetworksModel),
        model_finish: unsafe extern "C" fn(*mut ANeuralNetworksModel) -> i32,
        model_add_operand:
            unsafe extern "C" fn(*mut ANeuralNetworksModel, *const ANeuralNetworksOperandType) -> i32,
        model_set_operand_value:
            unsafe extern "C" fn(*mut ANeuralNetworksModel, i32, *const c_void, usize) -> i32,
        model_add_operation:
            unsafe extern "C" fn(*mut ANeuralNetworksModel, i32, u32, *const u32, u32, *const u32) -> i32,
        model_identify_inputs_and_outputs:
            unsafe extern "C" fn(*mut ANeuralNetworksModel, u32, *const u32, u32, *const u32) -> i32,
        model_get_supported_operations_for_devices:
            unsafe extern "C" fn(*const ANeuralNetworksModel, *const *const ANeuralNetworksDevice, u32, *mut bool) -> i32,
        model_set_operand_value_from_memory:
            unsafe extern "C" fn(*mut ANeuralNetworksModel, i32, *const ANeuralNetworksMemory, usize, usize) -> i32,
        // ── Compilation ─────────────────────────────────────────────────
        compilation_create:
            unsafe extern "C" fn(*mut ANeuralNetworksModel, *mut *mut ANeuralNetworksCompilation) -> i32,
        compilation_free: unsafe extern "C" fn(*mut ANeuralNetworksCompilation),
        compilation_set_preference:
            unsafe extern "C" fn(*mut ANeuralNetworksCompilation, i32) -> i32,
        compilation_finish: unsafe extern "C" fn(*mut ANeuralNetworksCompilation) -> i32,
        compilation_create_for_devices:
            unsafe extern "C" fn(*mut ANeuralNetworksModel, *const *const ANeuralNetworksDevice, u32, *mut *mut ANeuralNetworksCompilation) -> i32,
        // ── Execution ───────────────────────────────────────────────────
        execution_create:
            unsafe extern "C" fn(*mut ANeuralNetworksCompilation, *mut *mut ANeuralNetworksExecution) -> i32,
        execution_free: unsafe extern "C" fn(*mut ANeuralNetworksExecution),
        execution_set_input:
            unsafe extern "C" fn(*mut ANeuralNetworksExecution, i32, *const ANeuralNetworksOperandType, *const c_void, usize) -> i32,
        execution_set_output:
            unsafe extern "C" fn(*mut ANeuralNetworksExecution, i32, *const ANeuralNetworksOperandType, *mut c_void, usize) -> i32,
        execution_compute: unsafe extern "C" fn(*mut ANeuralNetworksExecution) -> i32,
        execution_set_input_from_memory:
            unsafe extern "C" fn(*mut ANeuralNetworksExecution, i32, *const ANeuralNetworksOperandType, *const ANeuralNetworksMemory, usize, usize) -> i32,
        execution_set_output_from_memory:
            unsafe extern "C" fn(*mut ANeuralNetworksExecution, i32, *const ANeuralNetworksOperandType, *const ANeuralNetworksMemory, usize, usize) -> i32,
        // ── Memory ──────────────────────────────────────────────────────
        memory_create_from_fd:
            unsafe extern "C" fn(usize, i32, i32, usize, *mut *mut ANeuralNetworksMemory) -> i32,
        memory_free: unsafe extern "C" fn(*mut ANeuralNetworksMemory),
    }

    static NNAPI: OnceLock<Result<NnapiHandle, ()>> = OnceLock::new();

    /// SAFETY: `_lib` is only stored for the library handle lifetime;
    /// the handle is never accessed after drop and is only used within
    /// wrapper functions that call through loaded function pointers.
    unsafe impl Send for NnapiHandle {}
    unsafe impl Sync for NnapiHandle {}

    fn load_symbol<T>(lib: *mut std::ffi::c_void, name: &CStr) -> Option<T> {
        let sym = unsafe { libc::dlsym(lib, name.as_ptr()) };
        if sym.is_null() {
            None
        } else {
            Some(unsafe { std::mem::transmute_copy(&sym) })
        }
    }

    fn init() -> Result<&'static NnapiHandle, ()> {
        match NNAPI.get_or_init(|| {
            let lib_name = CStr::from_bytes_with_nul(b"libneuralnetworks.so\0").unwrap();
            let lib = unsafe { libc::dlopen(lib_name.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL) };
            if lib.is_null() {
                return Err(());
            }

            macro_rules! load {
                ($name:literal) => {{
                    let sym_name = CStr::from_bytes_with_nul(concat!($name, "\0").as_bytes()).unwrap();
                    load_symbol(lib, sym_name).ok_or(())?
                }};
            }

            Ok(NnapiHandle {
                _lib: lib as *mut std::ffi::c_void,
                get_device_count: load!("ANeuralNetworks_getDeviceCount"),
                get_device: load!("ANeuralNetworks_getDevice"),
                device_get_name: load!("ANeuralNetworksDevice_getName"),
                device_get_type: load!("ANeuralNetworksDevice_getType"),
                device_get_version: load!("ANeuralNetworksDevice_getVersion"),
                device_get_feature_level: load!("ANeuralNetworksDevice_getFeatureLevel"),
                model_create: load!("ANeuralNetworksModel_create"),
                model_free: load!("ANeuralNetworksModel_free"),
                model_finish: load!("ANeuralNetworksModel_finish"),
                model_add_operand: load!("ANeuralNetworksModel_addOperand"),
                model_set_operand_value: load!("ANeuralNetworksModel_setOperandValue"),
                model_add_operation: load!("ANeuralNetworksModel_addOperation"),
                model_identify_inputs_and_outputs: load!("ANeuralNetworksModel_identifyInputsAndOutputs"),
                model_get_supported_operations_for_devices: load!("ANeuralNetworksModel_getSupportedOperationsForDevices"),
                model_set_operand_value_from_memory: load!("ANeuralNetworksModel_setOperandValueFromMemory"),
                compilation_create: load!("ANeuralNetworksCompilation_create"),
                compilation_free: load!("ANeuralNetworksCompilation_free"),
                compilation_set_preference: load!("ANeuralNetworksCompilation_setPreference"),
                compilation_finish: load!("ANeuralNetworksCompilation_finish"),
                compilation_create_for_devices: load!("ANeuralNetworksCompilation_createForDevices"),
                execution_create: load!("ANeuralNetworksExecution_create"),
                execution_free: load!("ANeuralNetworksExecution_free"),
                execution_set_input: load!("ANeuralNetworksExecution_setInput"),
                execution_set_output: load!("ANeuralNetworksExecution_setOutput"),
                execution_compute: load!("ANeuralNetworksExecution_compute"),
                execution_set_input_from_memory: load!("ANeuralNetworksExecution_setInputFromMemory"),
                execution_set_output_from_memory: load!("ANeuralNetworksExecution_setOutputFromMemory"),
                memory_create_from_fd: load!("ANeuralNetworksMemory_createFromFd"),
                memory_free: load!("ANeuralNetworksMemory_free"),
            })
        }) {
            Ok(ref handle) => Ok(handle),
            Err(_) => Err(()),
        }
    }

    // ── Public wrapper functions ───────────────────────────────────────────

    pub unsafe fn ANeuralNetworks_getDeviceCount(numDevices: *mut u32) -> i32 {
        (init().unwrap().get_device_count)(numDevices)
    }
    pub unsafe fn ANeuralNetworks_getDevice(devIndex: u32, device: *mut *mut ANeuralNetworksDevice) -> i32 {
        (init().unwrap().get_device)(devIndex, device)
    }
    pub unsafe fn ANeuralNetworksDevice_getName(device: *const ANeuralNetworksDevice, name: *mut *const c_char) -> i32 {
        (init().unwrap().device_get_name)(device, name)
    }
    pub unsafe fn ANeuralNetworksDevice_getType(device: *const ANeuralNetworksDevice, type_: *mut i32) -> i32 {
        (init().unwrap().device_get_type)(device, type_)
    }
    pub unsafe fn ANeuralNetworksDevice_getVersion(device: *const ANeuralNetworksDevice, version: *mut *const c_char) -> i32 {
        (init().unwrap().device_get_version)(device, version)
    }
    pub unsafe fn ANeuralNetworksDevice_getFeatureLevel(device: *const ANeuralNetworksDevice, featureLevel: *mut i32) -> i32 {
        (init().unwrap().device_get_feature_level)(device, featureLevel)
    }
    pub unsafe fn ANeuralNetworksModel_create(model: *mut *mut ANeuralNetworksModel) -> i32 {
        (init().unwrap().model_create)(model)
    }
    pub unsafe fn ANeuralNetworksModel_free(model: *mut ANeuralNetworksModel) {
        (init().unwrap().model_free)(model)
    }
    pub unsafe fn ANeuralNetworksModel_finish(model: *mut ANeuralNetworksModel) -> i32 {
        (init().unwrap().model_finish)(model)
    }
    pub unsafe fn ANeuralNetworksModel_addOperand(model: *mut ANeuralNetworksModel, type_: *const ANeuralNetworksOperandType) -> i32 {
        (init().unwrap().model_add_operand)(model, type_)
    }
    pub unsafe fn ANeuralNetworksModel_setOperandValue(model: *mut ANeuralNetworksModel, index: i32, buffer: *const c_void, length: usize) -> i32 {
        (init().unwrap().model_set_operand_value)(model, index, buffer, length)
    }
    pub unsafe fn ANeuralNetworksModel_addOperation(model: *mut ANeuralNetworksModel, operation_type: i32, inputCount: u32, inputs: *const u32, outputCount: u32, outputs: *const u32) -> i32 {
        (init().unwrap().model_add_operation)(model, operation_type, inputCount, inputs, outputCount, outputs)
    }
    pub unsafe fn ANeuralNetworksModel_identifyInputsAndOutputs(model: *mut ANeuralNetworksModel, inputCount: u32, inputs: *const u32, outputCount: u32, outputs: *const u32) -> i32 {
        (init().unwrap().model_identify_inputs_and_outputs)(model, inputCount, inputs, outputCount, outputs)
    }
    pub unsafe fn ANeuralNetworksModel_getSupportedOperationsForDevices(model: *const ANeuralNetworksModel, devices: *const *const ANeuralNetworksDevice, numDevices: u32, supportedOps: *mut bool) -> i32 {
        (init().unwrap().model_get_supported_operations_for_devices)(model, devices, numDevices, supportedOps)
    }
    pub unsafe fn ANeuralNetworksModel_setOperandValueFromMemory(model: *mut ANeuralNetworksModel, index: i32, memory: *const ANeuralNetworksMemory, offset: usize, length: usize) -> i32 {
        (init().unwrap().model_set_operand_value_from_memory)(model, index, memory, offset, length)
    }
    pub unsafe fn ANeuralNetworksCompilation_create(model: *mut ANeuralNetworksModel, compilation: *mut *mut ANeuralNetworksCompilation) -> i32 {
        (init().unwrap().compilation_create)(model, compilation)
    }
    pub unsafe fn ANeuralNetworksCompilation_free(compilation: *mut ANeuralNetworksCompilation) {
        (init().unwrap().compilation_free)(compilation)
    }
    pub unsafe fn ANeuralNetworksCompilation_setPreference(compilation: *mut ANeuralNetworksCompilation, preference: i32) -> i32 {
        (init().unwrap().compilation_set_preference)(compilation, preference)
    }
    pub unsafe fn ANeuralNetworksCompilation_finish(compilation: *mut ANeuralNetworksCompilation) -> i32 {
        (init().unwrap().compilation_finish)(compilation)
    }
    pub unsafe fn ANeuralNetworksCompilation_createForDevices(model: *mut ANeuralNetworksModel, devices: *const *const ANeuralNetworksDevice, numDevices: u32, compilation: *mut *mut ANeuralNetworksCompilation) -> i32 {
        (init().unwrap().compilation_create_for_devices)(model, devices, numDevices, compilation)
    }
    pub unsafe fn ANeuralNetworksExecution_create(compilation: *mut ANeuralNetworksCompilation, execution: *mut *mut ANeuralNetworksExecution) -> i32 {
        (init().unwrap().execution_create)(compilation, execution)
    }
    pub unsafe fn ANeuralNetworksExecution_free(execution: *mut ANeuralNetworksExecution) {
        (init().unwrap().execution_free)(execution)
    }
    pub unsafe fn ANeuralNetworksExecution_setInput(execution: *mut ANeuralNetworksExecution, index: i32, type_: *const ANeuralNetworksOperandType, buffer: *const c_void, length: usize) -> i32 {
        (init().unwrap().execution_set_input)(execution, index, type_, buffer, length)
    }
    pub unsafe fn ANeuralNetworksExecution_setOutput(execution: *mut ANeuralNetworksExecution, index: i32, type_: *const ANeuralNetworksOperandType, buffer: *mut c_void, length: usize) -> i32 {
        (init().unwrap().execution_set_output)(execution, index, type_, buffer, length)
    }
    pub unsafe fn ANeuralNetworksExecution_compute(execution: *mut ANeuralNetworksExecution) -> i32 {
        (init().unwrap().execution_compute)(execution)
    }
    pub unsafe fn ANeuralNetworksExecution_setInputFromMemory(execution: *mut ANeuralNetworksExecution, index: i32, type_: *const ANeuralNetworksOperandType, memory: *const ANeuralNetworksMemory, offset: usize, length: usize) -> i32 {
        (init().unwrap().execution_set_input_from_memory)(execution, index, type_, memory, offset, length)
    }
    pub unsafe fn ANeuralNetworksExecution_setOutputFromMemory(execution: *mut ANeuralNetworksExecution, index: i32, type_: *const ANeuralNetworksOperandType, memory: *const ANeuralNetworksMemory, offset: usize, length: usize) -> i32 {
        (init().unwrap().execution_set_output_from_memory)(execution, index, type_, memory, offset, length)
    }
    pub unsafe fn ANeuralNetworksMemory_createFromFd(size: usize, protect: i32, fd: i32, offset: usize, memory: *mut *mut ANeuralNetworksMemory) -> i32 {
        (init().unwrap().memory_create_from_fd)(size, protect, fd, offset, memory)
    }
    pub unsafe fn ANeuralNetworksMemory_free(memory: *mut ANeuralNetworksMemory) {
        (init().unwrap().memory_free)(memory)
    }
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub use runtime::*;

// ── AHardwareBuffer FFI (separate library: libandroid.so) ──────────────────
// These are also loaded at runtime to avoid compile-time linking issues.

#[cfg(all(feature = "nnapi", target_os = "android"))]
mod ahb_runtime {
    use super::*;

    struct AhbHandle {
        _lib: *mut std::ffi::c_void,
        allocate: unsafe extern "C" fn(*const AHardwareBuffer_Desc, *mut *mut AHardwareBuffer) -> i32,
        release: unsafe extern "C" fn(*mut AHardwareBuffer),
        describe: unsafe extern "C" fn(*const AHardwareBuffer, *mut AHardwareBuffer_Desc),
        memory_create_from_hardware_buffer:
            unsafe extern "C" fn(*const ANeuralNetworksDevice, *const AHardwareBuffer, *mut *mut ANeuralNetworksMemory) -> i32,
    }

    unsafe impl Send for AhbHandle {}
    unsafe impl Sync for AhbHandle {}

    static AHB: std::sync::OnceLock<Result<AhbHandle, ()>> = std::sync::OnceLock::new();

    fn load_symbol<T>(lib: *mut std::ffi::c_void, name: &std::ffi::CStr) -> Option<T> {
        let sym = unsafe { libc::dlsym(lib, name.as_ptr()) };
        if sym.is_null() { None }
        else { Some(unsafe { std::mem::transmute_copy(&sym) }) }
    }

    fn init() -> Result<&'static AhbHandle, ()> {
        match AHB.get_or_init(|| {
            let lib_name = std::ffi::CStr::from_bytes_with_nul(b"libandroid.so\0").unwrap();
            let lib = unsafe { libc::dlopen(lib_name.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL) };
            if lib.is_null() { return Err(()); }

            macro_rules! load {
                ($name:literal) => {{
                    let sym_name = std::ffi::CStr::from_bytes_with_nul(concat!($name, "\0").as_bytes()).unwrap();
                    load_symbol(lib, sym_name).ok_or(())?
                }};
            }

            Ok(AhbHandle {
                _lib: lib as *mut std::ffi::c_void,
                allocate: load!("AHardwareBuffer_allocate"),
                release: load!("AHardwareBuffer_release"),
                describe: load!("AHardwareBuffer_describe"),
                memory_create_from_hardware_buffer: load!("ANeuralNetworksMemory_createFromHardwareBuffer"),
            })
        }) {
            Ok(ref handle) => Ok(handle),
            Err(_) => Err(()),
        }
    }

    pub unsafe fn AHardwareBuffer_allocate(desc: *const AHardwareBuffer_Desc, out_buffer: *mut *mut AHardwareBuffer) -> i32 {
        (init().unwrap().allocate)(desc, out_buffer)
    }
    pub unsafe fn AHardwareBuffer_release(buffer: *mut AHardwareBuffer) {
        (init().unwrap().release)(buffer)
    }
    pub unsafe fn AHardwareBuffer_describe(buffer: *const AHardwareBuffer, desc: *mut AHardwareBuffer_Desc) {
        (init().unwrap().describe)(buffer, desc)
    }
    pub unsafe fn ANeuralNetworksMemory_createFromHardwareBuffer(
        device: *const ANeuralNetworksDevice,
        buffer: *const AHardwareBuffer,
        memory: *mut *mut ANeuralNetworksMemory,
    ) -> i32 {
        (init().unwrap().memory_create_from_hardware_buffer)(device, buffer, memory)
    }
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub use ahb_runtime::*;

// ── NnapiError ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NnapiError {
    Message(String),
    NoError,
    OutOfMemory,
    Incomplete,
    UnexpectedNull,
    BadData,
    OperationFailed,
    BadState,
    Unmappable,
    OutputInsufficientSize,
    UnavailableDevice,
    Code(i32),
    Nnapi(String),
    LibraryNotLoaded,
}

impl std::error::Error for NnapiError {}

impl std::fmt::Display for NnapiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NnapiError::Message(s) => write!(f, "NNAPI: {}", s),
            NnapiError::NoError => write!(f, "NNAPI: no error"),
            NnapiError::OutOfMemory => write!(f, "NNAPI: out of memory"),
            NnapiError::Incomplete => write!(f, "NNAPI: incomplete"),
            NnapiError::UnexpectedNull => write!(f, "NNAPI: unexpected null"),
            NnapiError::BadData => write!(f, "NNAPI: bad data"),
            NnapiError::OperationFailed => write!(f, "NNAPI: operation failed"),
            NnapiError::BadState => write!(f, "NNAPI: bad state"),
            NnapiError::Unmappable => write!(f, "NNAPI: unmappable"),
            NnapiError::OutputInsufficientSize => write!(f, "NNAPI: output insufficient size"),
            NnapiError::UnavailableDevice => write!(f, "NNAPI: unavailable device"),
            NnapiError::Code(c) => write!(f, "NNAPI error code: {}", c),
            NnapiError::Nnapi(s) => write!(f, "NNAPI: {}", s),
            NnapiError::LibraryNotLoaded => write!(f, "NNAPI: library not loaded"),
        }
    }
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
impl NnapiError {
    pub fn from_code(code: i32) -> Self {
        match code {
            0 => NnapiError::NoError,
            1 => NnapiError::OutOfMemory,
            2 => NnapiError::Incomplete,
            3 => NnapiError::UnexpectedNull,
            4 => NnapiError::BadData,
            5 => NnapiError::OperationFailed,
            6 => NnapiError::BadState,
            7 => NnapiError::Unmappable,
            8 => NnapiError::OutputInsufficientSize,
            9 => NnapiError::UnavailableDevice,
            other => NnapiError::Code(other),
        }
    }
}

#[cfg(not(all(feature = "nnapi", target_os = "android")))]
impl NnapiError {
    pub fn from_code(_code: i32) -> Self {
        NnapiError::Code(-1)
    }
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub fn nnapi_result(code: i32) -> Result<(), NnapiError> {
    if code == ANEURALNETWORKS_NO_ERROR {
        Ok(())
    } else {
        Err(NnapiError::from_code(code))
    }
}

#[cfg(not(all(feature = "nnapi", target_os = "android")))]
pub fn nnapi_result(_code: i32) -> Result<(), NnapiError> {
    Err(NnapiError::LibraryNotLoaded)
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub fn get_devices() -> Result<Vec<NnapiDeviceInfo>, NnapiError> {
    let mut count: u32 = 0;
    let rc = unsafe { ANeuralNetworks_getDeviceCount(&mut count as *mut u32) };
    nnapi_result(rc)?;

    let mut devices = Vec::new();
    for i in 0..count {
        let mut device: *mut ANeuralNetworksDevice = std::ptr::null_mut();
        let rc = unsafe { ANeuralNetworks_getDevice(i, &mut device) };
        nnapi_result(rc)?;

        let mut name_ptr: *const c_char = std::ptr::null();
        let mut type_code: i32 = 0;
        let mut version_ptr: *const c_char = std::ptr::null();
        let mut feature_level: i32 = 0;

        unsafe {
            nnapi_result(ANeuralNetworksDevice_getName(device, &mut name_ptr))?;
            nnapi_result(ANeuralNetworksDevice_getType(device, &mut type_code))?;
            nnapi_result(ANeuralNetworksDevice_getVersion(device, &mut version_ptr))?;
            nnapi_result(ANeuralNetworksDevice_getFeatureLevel(device, &mut feature_level))?;
        }

        let name = if name_ptr.is_null() {
            "unknown".to_string()
        } else {
            unsafe { std::ffi::CStr::from_ptr(name_ptr).to_string_lossy().into_owned() }
        };
        let version = if version_ptr.is_null() {
            "unknown".to_string()
        } else {
            unsafe { std::ffi::CStr::from_ptr(version_ptr).to_string_lossy().into_owned() }
        };

        devices.push(NnapiDeviceInfo {
            name,
            device_type: type_code,
            version,
            feature_level,
        });
    }
    Ok(devices)
}

#[cfg(not(all(feature = "nnapi", target_os = "android")))]
pub fn get_devices() -> Result<Vec<NnapiDeviceInfo>, NnapiError> {
    Ok(Vec::new())
}

#[derive(Debug, Clone)]
pub struct NnapiDeviceInfo {
    pub name: String,
    pub device_type: i32,
    pub version: String,
    pub feature_level: i32,
}