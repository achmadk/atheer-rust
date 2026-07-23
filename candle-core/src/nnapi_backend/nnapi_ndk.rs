//! Raw FFI bindings to the Android NNAPI NDK (NeuralNetworks.h).

use std::os::raw::{c_char, c_void};

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

#[cfg(all(feature = "nnapi", target_os = "android"))]
extern "C" {
    pub fn ANeuralNetworks_getDeviceCount(numDevices: *mut u32) -> i32;
    pub fn ANeuralNetworks_getDevice(devIndex: u32, device: *mut *mut ANeuralNetworksDevice)
        -> i32;
    pub fn ANeuralNetworksDevice_getName(
        device: *const ANeuralNetworksDevice,
        name: *mut *const c_char,
    ) -> i32;
    pub fn ANeuralNetworksDevice_getType(
        device: *const ANeuralNetworksDevice,
        type_: *mut i32,
    ) -> i32;
    pub fn ANeuralNetworksDevice_getVersion(
        device: *const ANeuralNetworksDevice,
        version: *mut *const c_char,
    ) -> i32;
    pub fn ANeuralNetworksDevice_getFeatureLevel(
        device: *const ANeuralNetworksDevice,
        featureLevel: *mut i32,
    ) -> i32;

    pub fn ANeuralNetworksModel_create(model: *mut *mut ANeuralNetworksModel) -> i32;
    pub fn ANeuralNetworksModel_free(model: *mut ANeuralNetworksModel);
    pub fn ANeuralNetworksModel_finish(model: *mut ANeuralNetworksModel) -> i32;

    pub fn ANeuralNetworksModel_addOperand(
        model: *mut ANeuralNetworksModel,
        type_: *const ANeuralNetworksOperandType,
    ) -> i32;

    pub fn ANeuralNetworksModel_setOperandValue(
        model: *mut ANeuralNetworksModel,
        index: i32,
        buffer: *const c_void,
        length: usize,
    ) -> i32;

    pub fn ANeuralNetworksModel_addOperation(
        model: *mut ANeuralNetworksModel,
        operation_type: i32,
        inputCount: u32,
        inputs: *const u32,
        outputCount: u32,
        outputs: *const u32,
    ) -> i32;

    pub fn ANeuralNetworksModel_identifyInputsAndOutputs(
        model: *mut ANeuralNetworksModel,
        inputCount: u32,
        inputs: *const u32,
        outputCount: u32,
        outputs: *const u32,
    ) -> i32;

    pub fn ANeuralNetworksModel_getSupportedOperationsForDevices(
        model: *const ANeuralNetworksModel,
        devices: *const *const ANeuralNetworksDevice,
        numDevices: u32,
        supportedOps: *mut bool,
    ) -> i32;

    pub fn ANeuralNetworksCompilation_create(
        model: *mut ANeuralNetworksModel,
        compilation: *mut *mut ANeuralNetworksCompilation,
    ) -> i32;

    pub fn ANeuralNetworksCompilation_free(compilation: *mut ANeuralNetworksCompilation);

    pub fn ANeuralNetworksCompilation_setPreference(
        compilation: *mut ANeuralNetworksCompilation,
        preference: i32,
    ) -> i32;

    pub fn ANeuralNetworksCompilation_finish(compilation: *mut ANeuralNetworksCompilation) -> i32;

    pub fn ANeuralNetworksCompilation_createForDevices(
        model: *mut ANeuralNetworksModel,
        devices: *const *const ANeuralNetworksDevice,
        numDevices: u32,
        compilation: *mut *mut ANeuralNetworksCompilation,
    ) -> i32;

    pub fn ANeuralNetworksExecution_create(
        compilation: *mut ANeuralNetworksCompilation,
        execution: *mut *mut ANeuralNetworksExecution,
    ) -> i32;

    pub fn ANeuralNetworksExecution_free(execution: *mut ANeuralNetworksExecution);

    pub fn ANeuralNetworksExecution_setInput(
        execution: *mut ANeuralNetworksExecution,
        index: i32,
        type_: *const ANeuralNetworksOperandType,
        buffer: *const c_void,
        length: usize,
    ) -> i32;

    pub fn ANeuralNetworksExecution_setOutput(
        execution: *mut ANeuralNetworksExecution,
        index: i32,
        type_: *const ANeuralNetworksOperandType,
        buffer: *mut c_void,
        length: usize,
    ) -> i32;

    pub fn ANeuralNetworksExecution_compute(execution: *mut ANeuralNetworksExecution) -> i32;

    pub fn ANeuralNetworksExecution_setInputFromMemory(
        execution: *mut ANeuralNetworksExecution,
        index: i32,
        type_: *const ANeuralNetworksOperandType,
        memory: *const ANeuralNetworksMemory,
        offset: usize,
        length: usize,
    ) -> i32;

    pub fn ANeuralNetworksExecution_setOutputFromMemory(
        execution: *mut ANeuralNetworksExecution,
        index: i32,
        type_: *const ANeuralNetworksOperandType,
        memory: *const ANeuralNetworksMemory,
        offset: usize,
        length: usize,
    ) -> i32;

    pub fn ANeuralNetworksMemory_createFromFd(
        size: usize,
        protect: i32,
        fd: i32,
        offset: usize,
        memory: *mut *mut ANeuralNetworksMemory,
    ) -> i32;

    pub fn ANeuralNetworksMemory_free(memory: *mut ANeuralNetworksMemory);

    pub fn ANeuralNetworksModel_setOperandValueFromMemory(
        model: *mut ANeuralNetworksModel,
        index: i32,
        memory: *const ANeuralNetworksMemory,
        offset: usize,
        length: usize,
    ) -> i32;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NnapiError {
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
    Unknown(i32),
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
            other => NnapiError::Unknown(other),
        }
    }
}

#[cfg(not(all(feature = "nnapi", target_os = "android")))]
impl NnapiError {
    pub fn from_code(_code: i32) -> Self {
        NnapiError::Unknown(-1)
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
    Err(NnapiError::Unknown(-1))
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
#[derive(Debug, Clone)]
pub struct NnapiDeviceInfo {
    pub name: String,
    pub version: String,
    pub device_type: i32,
    pub feature_level: i32,
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub fn get_devices() -> Result<Vec<NnapiDeviceInfo>, NnapiError> {
    let mut count: u32 = 0;
    let rc = unsafe { ANeuralNetworks_getDeviceCount(&mut count as *mut u32) };
    nnapi_result(rc)?;

    let mut devices = Vec::with_capacity(count as usize);
    for i in 0..count {
        let mut raw: *mut ANeuralNetworksDevice = std::ptr::null_mut();
        let rc =
            unsafe { ANeuralNetworks_getDevice(i, &mut raw as *mut *mut ANeuralNetworksDevice) };
        nnapi_result(rc)?;

        let info = unsafe { get_device_info(raw)? };
        devices.push(info);
    }
    Ok(devices)
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
unsafe fn get_device_info(raw: *mut ANeuralNetworksDevice) -> Result<NnapiDeviceInfo, NnapiError> {
    let mut name_ptr: *const c_char = std::ptr::null();
    let rc = ANeuralNetworksDevice_getName(
        raw as *const ANeuralNetworksDevice,
        &mut name_ptr as *mut *const c_char,
    );
    nnapi_result(rc)?;
    let name = if name_ptr.is_null() {
        String::new()
    } else {
        std::ffi::CStr::from_ptr(name_ptr)
            .to_string_lossy()
            .into_owned()
    };

    let mut ver_ptr: *const c_char = std::ptr::null();
    let rc = ANeuralNetworksDevice_getVersion(
        raw as *const ANeuralNetworksDevice,
        &mut ver_ptr as *mut *const c_char,
    );
    nnapi_result(rc)?;
    let version = if ver_ptr.is_null() {
        String::new()
    } else {
        std::ffi::CStr::from_ptr(ver_ptr)
            .to_string_lossy()
            .into_owned()
    };

    let mut device_type: i32 = 0;
    let rc = ANeuralNetworksDevice_getType(
        raw as *const ANeuralNetworksDevice,
        &mut device_type as *mut i32,
    );
    nnapi_result(rc)?;

    let mut feature_level: i32 = 0;
    let rc = ANeuralNetworksDevice_getFeatureLevel(
        raw as *const ANeuralNetworksDevice,
        &mut feature_level as *mut i32,
    );
    nnapi_result(rc)?;

    Ok(NnapiDeviceInfo {
        name,
        version,
        device_type,
        feature_level,
    })
}

// ---------------------------------------------------------------------------
// AHardwareBuffer types and constants
// ---------------------------------------------------------------------------

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub enum AHardwareBuffer {}

#[cfg(all(feature = "nnapi", target_os = "android"))]
#[repr(C)]
#[derive(Debug, Clone)]
pub struct AHardwareBuffer_Desc {
    pub width: u64,
    pub height: u64,
    pub layers: u32,
    pub format: u32,
    pub usage: u64,
    pub stride: u32,
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const AHARDWAREBUFFER_FORMAT_BLOB: u32 = 1;

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN: u64 = 0x0002_0001;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN: u64 = 0x0000_0002;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER: u64 = 0x0000_4000;
#[cfg(all(feature = "nnapi", target_os = "android"))]
pub const AHARDWAREBUFFER_USAGE_VIDEO_ENCODE: u64 = 0x0000_0300;

#[cfg(all(feature = "nnapi", target_os = "android"))]
extern "C" {
    pub fn AHardwareBuffer_allocate(
        desc: *const AHardwareBuffer_Desc,
        out_buffer: *mut *mut AHardwareBuffer,
    ) -> i32;
    pub fn AHardwareBuffer_release(buffer: *mut AHardwareBuffer);
    pub fn AHardwareBuffer_describe(
        buffer: *const AHardwareBuffer,
        desc: *mut AHardwareBuffer_Desc,
    );
    pub fn ANeuralNetworksMemory_createFromHardwareBuffer(
        device: *const ANeuralNetworksDevice,
        buffer: *const AHardwareBuffer,
        memory: *mut *mut ANeuralNetworksMemory,
    ) -> i32;
    pub fn ANeuralNetworksMemory_free(memory: *mut ANeuralNetworksMemory);
}
