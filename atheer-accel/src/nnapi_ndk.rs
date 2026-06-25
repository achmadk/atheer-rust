//! Raw FFI bindings to the Android NNAPI NDK (NeuralNetworks.h).
//!
//! These are handwritten extern declarations for the subset of NNAPI NDK
//! functions needed to discover devices, construct model graphs, compile,
//! and execute inference on Android NPU/GPU/DSP accelerators.
//!
//! NNAPI feature level 1 (API 27) or higher is required minimum. Device
//! discovery (getDevice*) requires API 29.  All functions are gated behind
//! `#[cfg(target_os = "android")]` because the NDK linker script is only
//! present on Android targets.

#![cfg(target_os = "android")]
#![allow(non_camel_case_types, dead_code)]

use std::os::raw::{c_char, c_int, c_uint, c_void};

// ---------------------------------------------------------------------------
// Opaque handle types
// ---------------------------------------------------------------------------

pub enum ANeuralNetworksMemory {}
pub enum ANeuralNetworksModel {}
pub enum ANeuralNetworksCompilation {}
pub enum ANeuralNetworksExecution {}
pub enum ANeuralNetworksDevice {}
pub enum ANeuralNetworksEvent {}
pub enum ANeuralNetworksMemoryDesc {}

// ---------------------------------------------------------------------------
// OperandType struct  (NeuralNetworksTypes.h)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone)]
pub struct ANeuralNetworksOperandType {
    pub type_: i32,               // OperandCode
    pub dimension_count: u32,
    pub dimensions: *const u32,
    pub scale: f32,
    pub zero_point: i32,
}

// ---------------------------------------------------------------------------
// Operand codes
// ---------------------------------------------------------------------------

pub const ANEURALNETWORKS_FLOAT32: i32 = 0;
pub const ANEURALNETWORKS_INT32: i32 = 1;
pub const ANEURALNETWORKS_UINT32: i32 = 2;
pub const ANEURALNETWORKS_TENSOR_FLOAT32: i32 = 3;
pub const ANEURALNETWORKS_TENSOR_QUANT8_ASYMM: i32 = 5;
pub const ANEURALNETWORKS_TENSOR_FLOAT16: i32 = 8;
pub const ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED: i32 = 14;

// ---------------------------------------------------------------------------
// Operation codes
// ---------------------------------------------------------------------------

pub const ANEURALNETWORKS_ADD: i32 = 0;
pub const ANEURALNETWORKS_MUL: i32 = 1;
pub const ANEURALNETWORKS_CONCATENATION: i32 = 3;
pub const ANEURALNETWORKS_FULLY_CONNECTED: i32 = 9;
pub const ANEURALNETWORKS_LOGISTIC: i32 = 14;
pub const ANEURALNETWORKS_RELU: i32 = 15;
pub const ANEURALNETWORKS_TANH: i32 = 16;
pub const ANEURALNETWORKS_RESHAPE: i32 = 22;
pub const ANEURALNETWORKS_SOFTMAX: i32 = 25;
pub const ANEURALNETWORKS_BATCH_TO_SPACE_ND: i32 = 27;
pub const ANEURALNETWORKS_TRANSPOSE: i32 = 32;

// ---------------------------------------------------------------------------
// FuseCode
// ---------------------------------------------------------------------------

pub const ANEURALNETWORKS_FUSED_NONE: i32 = 0;
pub const ANEURALNETWORKS_FUSED_RELU: i32 = 1;
pub const ANEURALNETWORKS_FUSED_RELU1: i32 = 2;
pub const ANEURALNETWORKS_FUSED_RELU6: i32 = 3;

// ---------------------------------------------------------------------------
// PreferenceCode
// ---------------------------------------------------------------------------

pub const ANEURALNETWORKS_PREFER_LOW_POWER: i32 = 0;
pub const ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER: i32 = 1;
pub const ANEURALNETWORKS_PREFER_SUSTAINED_SPEED: i32 = 2;

// ---------------------------------------------------------------------------
// Result codes
// ---------------------------------------------------------------------------

pub const ANEURALNETWORKS_NO_ERROR: i32 = 0;
pub const ANEURALNETWORKS_OUT_OF_MEMORY: i32 = 1;
pub const ANEURALNETWORKS_INCOMPLETE: i32 = 2;
pub const ANEURALNETWORKS_UNEXPECTED_NULL: i32 = 3;
pub const ANEURALNETWORKS_BAD_DATA: i32 = 4;
pub const ANEURALNETWORKS_OP_FAILED: i32 = 5;
pub const ANEURALNETWORKS_BAD_STATE: i32 = 6;
pub const ANEURALNETWORKS_UNMAPPABLE: i32 = 7;
pub const ANEURALNETWORKS_OUTPUT_INSUFFICIENT_SIZE: i32 = 8;
pub const ANEURALNETWORKS_UNAVAILABLE_DEVICE: i32 = 9;

// ---------------------------------------------------------------------------
// Device type codes
// ---------------------------------------------------------------------------

pub const ANEURALNETWORKS_DEVICE_UNKNOWN: i32 = 0;
pub const ANEURALNETWORKS_DEVICE_OTHER: i32 = 1;
pub const ANEURALNETWORKS_DEVICE_CPU: i32 = 2;
pub const ANEURALNETWORKS_DEVICE_GPU: i32 = 3;
pub const ANEURALNETWORKS_DEVICE_ACCELERATOR: i32 = 4;

// ---------------------------------------------------------------------------
// Feature levels
// ---------------------------------------------------------------------------

pub const ANEURALNETWORKS_FEATURE_LEVEL_1: i32 = 27;
pub const ANEURALNETWORKS_FEATURE_LEVEL_2: i32 = 28;
pub const ANEURALNETWORKS_FEATURE_LEVEL_3: i32 = 29;
pub const ANEURALNETWORKS_FEATURE_LEVEL_4: i32 = 30;
pub const ANEURALNETWORKS_FEATURE_LEVEL_5: i32 = 31;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES: usize = 128;

// ---------------------------------------------------------------------------
// FFI function declarations  (NeuralNetworks.h)
// ---------------------------------------------------------------------------

extern "C" {

    // -- Device discovery (API 29+) --

    pub fn ANeuralNetworks_getDeviceCount(numDevices: *mut u32) -> i32;
    pub fn ANeuralNetworks_getDevice(
        devIndex: u32,
        device: *mut *mut ANeuralNetworksDevice,
    ) -> i32;
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

    // -- Model construction (API 27+) --

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
        operation_type: i32, // ANeuralNetworksOperationType
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

    pub fn ANeuralNetworksModel_relaxComputationFloat32toFloat16(
        model: *mut ANeuralNetworksModel,
        allow: bool,
    ) -> i32;

    pub fn ANeuralNetworksModel_getSupportedOperationsForDevices(
        model: *const ANeuralNetworksModel,
        devices: *const *const ANeuralNetworksDevice,
        numDevices: u32,
        supportedOps: *mut bool,
    ) -> i32;

    // -- Compilation (API 27+) --

    pub fn ANeuralNetworksCompilation_create(
        model: *mut ANeuralNetworksModel,
        compilation: *mut *mut ANeuralNetworksCompilation,
    ) -> i32;

    pub fn ANeuralNetworksCompilation_free(
        compilation: *mut ANeuralNetworksCompilation,
    );

    pub fn ANeuralNetworksCompilation_setPreference(
        compilation: *mut ANeuralNetworksCompilation,
        preference: i32,
    ) -> i32;

    pub fn ANeuralNetworksCompilation_finish(
        compilation: *mut ANeuralNetworksCompilation,
    ) -> i32;

    pub fn ANeuralNetworksCompilation_createForDevices(
        model: *mut ANeuralNetworksModel,
        devices: *const *const ANeuralNetworksDevice,
        numDevices: u32,
        compilation: *mut *mut ANeuralNetworksCompilation,
    ) -> i32;

    // -- Execution (API 27+) --

    pub fn ANeuralNetworksExecution_create(
        compilation: *mut ANeuralNetworksCompilation,
        execution: *mut *mut ANeuralNetworksExecution,
    ) -> i32;

    pub fn ANeuralNetworksExecution_free(
        execution: *mut ANeuralNetworksExecution,
    );

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

    pub fn ANeuralNetworksExecution_compute(
        execution: *mut ANeuralNetworksExecution,
    ) -> i32;

    pub fn ANeuralNetworksExecution_setMeasureTiming(
        execution: *mut ANeuralNetworksExecution,
        measure: bool,
    ) -> i32;

    pub fn ANeuralNetworksExecution_getDuration(
        execution: *const ANeuralNetworksExecution,
        durationCode: i32,
        duration: *mut u64,
    ) -> i32;

    pub fn ANeuralNetworksExecution_getOutputOperandRank(
        execution: *mut ANeuralNetworksExecution,
        index: i32,
        rank: *mut u32,
    ) -> i32;

    pub fn ANeuralNetworksExecution_getOutputOperandDimensions(
        execution: *mut ANeuralNetworksExecution,
        index: i32,
        dimensions: *mut u32,
    ) -> i32;

    // -- Memory (API 27+) --

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

// ---------------------------------------------------------------------------
// Safe Rust wrappers
// ---------------------------------------------------------------------------

/// Convert an NNAPI result code to a Rust `Result`.
pub fn nnapi_result(code: i32) -> Result<(), NnapiError> {
    if code == ANEURALNETWORKS_NO_ERROR {
        Ok(())
    } else {
        Err(NnapiError::from_code(code))
    }
}

/// Errors produced by NNAPI NDK operations.
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

    pub fn as_str(&self) -> &'static str {
        match self {
            NnapiError::NoError => "NO_ERROR",
            NnapiError::OutOfMemory => "OUT_OF_MEMORY",
            NnapiError::Incomplete => "INCOMPLETE",
            NnapiError::UnexpectedNull => "UNEXPECTED_NULL",
            NnapiError::BadData => "BAD_DATA",
            NnapiError::OperationFailed => "OP_FAILED",
            NnapiError::BadState => "BAD_STATE",
            NnapiError::Unmappable => "UNMAPPABLE",
            NnapiError::OutputInsufficientSize => "OUTPUT_INSUFFICIENT_SIZE",
            NnapiError::UnavailableDevice => "UNAVAILABLE_DEVICE",
            NnapiError::Unknown(c) => "UNKNOWN",
        }
    }
}

impl std::fmt::Display for NnapiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NNAPI error {} ({})", self.as_str(), self.code())
    }
}

impl NnapiError {
    pub fn code(&self) -> i32 {
        match self {
            NnapiError::NoError => 0,
            NnapiError::OutOfMemory => 1,
            NnapiError::Incomplete => 2,
            NnapiError::UnexpectedNull => 3,
            NnapiError::BadData => 4,
            NnapiError::OperationFailed => 5,
            NnapiError::BadState => 6,
            NnapiError::Unmappable => 7,
            NnapiError::OutputInsufficientSize => 8,
            NnapiError::UnavailableDevice => 9,
            NnapiError::Unknown(c) => *c,
        }
    }
}

/// A helper for turning NNAPI result codes into `crate::AccelError`.
impl From<NnapiError> for crate::AccelError {
    fn from(e: NnapiError) -> Self {
        match e {
            NnapiError::OperationFailed => {
                crate::AccelError::OperationFailed(e.to_string())
            }
            NnapiError::BadData => {
                crate::AccelError::OperationFailed(format!("NNAPI bad data: {e}"))
            }
            NnapiError::OutOfMemory => {
                crate::AccelError::MemoryAllocationFailed(e.to_string())
            }
            NnapiError::UnavailableDevice
            | NnapiError::UnexpectedNull
            | NnapiError::BadState => {
                crate::AccelError::BackendNotAvailable(e.to_string())
            }
            _ => crate::AccelError::OperationFailed(e.to_string()),
        }
    }
}

/// Information about a discovered NNAPI device.
#[derive(Debug, Clone)]
pub struct NnapiDeviceInfo {
    pub name: String,
    pub version: String,
    pub device_type: i32,
    pub feature_level: i32,
}

/// Discover all available NNAPI accelerators on the device.
///
/// Returns `Ok(vec)` with one entry per device. Returns an empty
/// vector (not an error) when NNAPI is not available.
pub fn get_devices() -> Result<Vec<NnapiDeviceInfo>, NnapiError> {
    let mut count: u32 = 0;
    // SAFETY: safe because we pass a valid stack pointer
    let rc = unsafe { ANeuralNetworks_getDeviceCount(&mut count as *mut u32) };
    nnapi_result(rc)?;

    let mut devices = Vec::with_capacity(count as usize);
    for i in 0..count {
        let mut raw: *mut ANeuralNetworksDevice = std::ptr::null_mut();
        let rc = unsafe { ANeuralNetworks_getDevice(i, &mut raw as *mut *mut ANeuralNetworksDevice) };
        nnapi_result(rc)?;

        let info = unsafe { get_device_info(raw)? };
        devices.push(info);
    }
    Ok(devices)
}

/// Read device metadata from a raw device pointer.
///
/// # Safety
/// `raw` must be a valid, non-null `ANeuralNetworksDevice*` returned by
/// `ANeuralNetworks_getDevice`.
unsafe fn get_device_info(raw: *mut ANeuralNetworksDevice) -> Result<NnapiDeviceInfo, NnapiError> {
    // Name
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

    // Version
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

    // Type
    let mut device_type: i32 = 0;
    let rc = ANeuralNetworksDevice_getType(
        raw as *const ANeuralNetworksDevice,
        &mut device_type as *mut i32,
    );
    nnapi_result(rc)?;

    // Feature level
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nnapi_error_conversion() {
        let err = NnapiError::from_code(0);
        assert_eq!(err, NnapiError::NoError);

        let err = NnapiError::from_code(5);
        assert_eq!(err, NnapiError::OperationFailed);
        assert_eq!(err.as_str(), "OP_FAILED");

        let err = NnapiError::from_code(99);
        assert_eq!(err, NnapiError::Unknown(99));
    }

    #[test]
    fn test_nnapi_error_to_acceleerror() {
        let op_fail: crate::AccelError = NnapiError::OperationFailed.into();
        assert!(matches!(op_fail, crate::AccelError::OperationFailed(_)));

        let oom: crate::AccelError = NnapiError::OutOfMemory.into();
        assert!(matches!(oom, crate::AccelError::MemoryAllocationFailed(_)));
    }

    #[test]
    fn test_device_type_constants() {
        assert_eq!(ANEURALNETWORKS_DEVICE_CPU, 2);
        assert_eq!(ANEURALNETWORKS_DEVICE_GPU, 3);
        assert_eq!(ANEURALNETWORKS_DEVICE_ACCELERATOR, 4);
    }

    #[test]
    fn test_operand_type_layout() {
        let ot = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_TENSOR_FLOAT32,
            dimension_count: 2,
            dimensions: std::ptr::null(),
            scale: 0.0,
            zero_point: 0,
        };
        assert_eq!(ot.type_, 3);
    }
}
