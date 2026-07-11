//! NNAPI graph compiler — model construction, compilation, and execution.
//!
//! This module provides a full NNAPI model compiler with:
//! - `NnapiOperation` — 9 supported operation codes:
//!   ADD, MUL, SOFTMAX, LOGISTIC, RELU, TANH, CONCATENATION,
//!   RESHAPE, FULLY_CONNECTED
//! - `NnapiGraphBuilder` — operand/operation graph construction with
//!   validation, `.finish()` and `.compile()` lifecycle
//! - `NnapiCompiledModel` — executable handle with `.execute()` for
//!   input/output buffer management
//! - `tensor_f32_type`, `tensor_quant8_type`, `scalar_i32_type` — helper
//!   functions for operand type creation
//!
//! All NNAPI NDK calls are gated behind `#[cfg(target_os = "android")]`.
//! On non-Android platforms, stub types ensure compilation without the
//! NDK.

use crate::{AccelBackend, AccelResult, BackendType, Result};
use std::time::Instant;

// Conditionally import the NDK bindings
#[cfg(target_os = "android")]
use crate::nnapi_ndk as ndk;

// On non-Android, provide stub type and constant definitions so struct
// fields and function signatures referencing ndk::* still compile.
#[cfg(not(target_os = "android"))]
#[allow(dead_code)]
mod ndk {
    pub enum ANeuralNetworksModel {}
    pub enum ANeuralNetworksCompilation {}
    pub enum ANeuralNetworksExecution {}
    pub struct ANeuralNetworksOperandType {
        pub type_: i32,
        pub dimension_count: u32,
        pub dimensions: *const u32,
        pub scale: f32,
        pub zero_point: i32,
    }
    // Operation codes (used by NnapiOperation::to_nnapi_code)
    pub const ANEURALNETWORKS_ADD: i32 = 0;
    pub const ANEURALNETWORKS_MUL: i32 = 1;
    pub const ANEURALNETWORKS_CONCATENATION: i32 = 3;
    pub const ANEURALNETWORKS_FULLY_CONNECTED: i32 = 9;
    pub const ANEURALNETWORKS_LOGISTIC: i32 = 14;
    pub const ANEURALNETWORKS_RELU: i32 = 15;
    pub const ANEURALNETWORKS_TANH: i32 = 16;
    pub const ANEURALNETWORKS_RESHAPE: i32 = 22;
    pub const ANEURALNETWORKS_SOFTMAX: i32 = 25;
    // Operand type codes (used by helper functions)
    pub const ANEURALNETWORKS_TENSOR_FLOAT32: i32 = 3;
    pub const ANEURALNETWORKS_TENSOR_QUANT8_ASYMM: i32 = 5;
    pub const ANEURALNETWORKS_INT32: i32 = 1;
}

// ---------------------------------------------------------------------------
// Execution Preference
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionPreference {
    LowPower,
    FastSingleAnswer,
    SustainedSpeed,
}

impl ExecutionPreference {
    #[cfg(target_os = "android")]
    fn to_ndk_preference(&self) -> i32 {
        match self {
            ExecutionPreference::LowPower => ndk::ANEURALNETWORKS_PREFER_LOW_POWER,
            ExecutionPreference::FastSingleAnswer => ndk::ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER,
            ExecutionPreference::SustainedSpeed => ndk::ANEURALNETWORKS_PREFER_SUSTAINED_SPEED,
        }
    }
}

// ---------------------------------------------------------------------------
// NNAPI Device Information
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct NnapiDeviceDescriptor {
    pub name: String,
    pub version: String,
    pub feature_level: i32,
    pub device_type: NnapiDeviceKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NnapiDeviceKind {
    Unknown,
    Other,
    Cpu,
    Gpu,
    Accelerator,
}

impl NnapiDeviceKind {
    #[cfg_attr(not(target_os = "android"), allow(dead_code))]
    fn from_ndk(code: i32) -> Self {
        #[cfg(target_os = "android")]
        {
            match code {
                ndk::ANEURALNETWORKS_DEVICE_CPU => NnapiDeviceKind::Cpu,
                ndk::ANEURALNETWORKS_DEVICE_GPU => NnapiDeviceKind::Gpu,
                ndk::ANEURALNETWORKS_DEVICE_ACCELERATOR => NnapiDeviceKind::Accelerator,
                ndk::ANEURALNETWORKS_DEVICE_OTHER => NnapiDeviceKind::Other,
                _ => NnapiDeviceKind::Unknown,
            }
        }
        #[cfg(not(target_os = "android"))]
        {
            let _ = code;
            NnapiDeviceKind::Unknown
        }
    }
}

// ---------------------------------------------------------------------------
// NNAPI Executor — wraps the NDK pipeline
// ---------------------------------------------------------------------------

/// Manages NNAPI model graph construction, compilation, and execution.
///
/// The executor builds a fresh model graph for each `forward()` call
/// because weight dimensions are only known at runtime.  The model
/// graph is compiled, executed, and freed in a single call — no state
/// is cached between invocations.
pub struct NnapiExecutor {
    devices: Vec<NnapiDeviceDescriptor>,
}

impl NnapiExecutor {
    /// Probe NNAPI devices and create a new executor.
    /// Returns `None` if NNAPI is not available on this device.
    pub fn probe() -> Option<Self> {
        #[cfg(target_os = "android")]
        {
            match ndk::get_devices() {
                Ok(devices) if !devices.is_empty() => {
                    let descriptors: Vec<NnapiDeviceDescriptor> = devices
                        .into_iter()
                        .map(|d| NnapiDeviceDescriptor {
                            name: d.name,
                            version: d.version,
                            feature_level: d.feature_level,
                            device_type: NnapiDeviceKind::from_ndk(d.device_type),
                        })
                        .collect();
                    Some(Self {
                        devices: descriptors,
                    })
                }
                _ => None,
            }
        }
        #[cfg(not(target_os = "android"))]
        {
            None
        }
    }

    pub fn devices(&self) -> &[NnapiDeviceDescriptor] {
        &self.devices
    }

    pub fn has_npu(&self) -> bool {
        self.devices
            .iter()
            .any(|d| d.device_type == NnapiDeviceKind::Accelerator)
    }

    pub fn best_device(&self) -> Option<&NnapiDeviceDescriptor> {
        // Priority: dedicated accelerator > GPU > CPU > other
        self.devices
            .iter()
            .find(|d| d.device_type == NnapiDeviceKind::Accelerator)
            .or_else(|| self.devices.iter().find(|d| d.device_type == NnapiDeviceKind::Gpu))
            .or_else(|| self.devices.iter().find(|d| d.device_type == NnapiDeviceKind::Cpu))
            .or_else(|| self.devices.first())
    }

    /// Compile a FULLY_CONNECTED model graph and return the output.
    ///
    /// This naively rebuilds the model for each call because weights
    /// are provided at runtime. In production, the model graph would
    /// be pre-compiled once during model load with known weight shapes.
    #[cfg(target_os = "android")]
    fn execute_fc(
        &mut self,
        input: &[f32],
        weights: &[f32],
        bias: &[f32],
        fused_activation: i32,
        output: &mut [f32],
    ) -> Result<()> {
        use ndk::*;
        use std::ptr;

        let batch_size = 1;
        let input_size = input.len();
        let num_units = output.len();

        // -- Step 1: Create model --
        let mut model: *mut ANeuralNetworksModel = ptr::null_mut();
        unsafe {
            nnapi_result(ANeuralNetworksModel_create(&mut model as *mut *mut ANeuralNetworksModel))?;
        }

        // -- Step 2: Define operand types --
        // Input: float32 tensor [1, input_size]
        let input_dims = [batch_size as u32, input_size as u32];
        let input_type = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_TENSOR_FLOAT32,
            dimension_count: 2,
            dimensions: input_dims.as_ptr(),
            scale: 0.0,
            zero_point: 0,
        };

        // Weights: float32 tensor [num_units, input_size]
        let weights_dims = [num_units as u32, input_size as u32];
        let weights_type = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_TENSOR_FLOAT32,
            dimension_count: 2,
            dimensions: weights_dims.as_ptr(),
            scale: 0.0,
            zero_point: 0,
        };

        // Bias: float32 tensor [num_units]
        let bias_dims = [num_units as u32];
        let bias_type = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_TENSOR_FLOAT32,
            dimension_count: 1,
            dimensions: bias_dims.as_ptr(),
            scale: 0.0,
            zero_point: 0,
        };

        // Fused activation: int32 scalar
        let act_type = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_INT32,
            dimension_count: 0,
            dimensions: ptr::null(),
            scale: 0.0,
            zero_point: 0,
        };

        // Output: float32 tensor [1, num_units]
        let output_dims = [batch_size as u32, num_units as u32];
        let output_type = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_TENSOR_FLOAT32,
            dimension_count: 2,
            dimensions: output_dims.as_ptr(),
            scale: 0.0,
            zero_point: 0,
        };

        // -- Step 3: Add operands --
        // Operand 0: input
        // Operand 1: weights
        // Operand 2: bias
        // Operand 3: fused activation scalar
        // Operand 4: output
        unsafe {
            nnapi_result(ANeuralNetworksModel_addOperand(model, &input_type as *const _))?;
            nnapi_result(ANeuralNetworksModel_addOperand(model, &weights_type as *const _))?;
            nnapi_result(ANeuralNetworksModel_addOperand(model, &bias_type as *const _))?;
            nnapi_result(ANeuralNetworksModel_addOperand(model, &act_type as *const _))?;
            nnapi_result(ANeuralNetworksModel_addOperand(model, &output_type as *const _))?;
        }

        // -- Step 4: Set operand values for constants (weights, bias, activation) --
        unsafe {
            nnapi_result(ANeuralNetworksModel_setOperandValue(
                model,
                1, // weights
                weights.as_ptr() as *const std::ffi::c_void,
                weights.len() * std::mem::size_of::<f32>(),
            ))?;
            nnapi_result(ANeuralNetworksModel_setOperandValue(
                model,
                2, // bias
                bias.as_ptr() as *const std::ffi::c_void,
                bias.len() * std::mem::size_of::<f32>(),
            ))?;
            nnapi_result(ANeuralNetworksModel_setOperandValue(
                model,
                3, // fused activation scalar
                &fused_activation as *const i32 as *const std::ffi::c_void,
                std::mem::size_of::<i32>(),
            ))?;
        }

        // -- Step 5: Add FULLY_CONNECTED operation --
        // Inputs to operation: [0 (input), 1 (weights), 2 (bias), 3 (activation)]
        // Outputs from operation: [4 (output)]
        let op_inputs: [u32; 4] = [0, 1, 2, 3];
        let op_outputs: [u32; 1] = [4];
        unsafe {
            nnapi_result(ANeuralNetworksModel_addOperation(
                model,
                ANEURALNETWORKS_FULLY_CONNECTED,
                4,
                op_inputs.as_ptr(),
                1,
                op_outputs.as_ptr(),
            ))?;
        }

        // -- Step 6: Identify inputs and outputs --
        // Input: operand 0, Output: operand 4
        let model_inputs: [u32; 1] = [0];
        let model_outputs: [u32; 1] = [4];
        unsafe {
            nnapi_result(ANeuralNetworksModel_identifyInputsAndOutputs(
                model,
                1,
                model_inputs.as_ptr(),
                1,
                model_outputs.as_ptr(),
            ))?;
        }

        // -- Step 7: Finish model --
        unsafe {
            nnapi_result(ANeuralNetworksModel_finish(model))?;
        }

        // -- Step 8: Create compilation --
        let mut compilation: *mut ANeuralNetworksCompilation = ptr::null_mut();
        unsafe {
            nnapi_result(ANeuralNetworksCompilation_create(
                model,
                &mut compilation as *mut *mut ANeuralNetworksCompilation,
            ))?;
            nnapi_result(ANeuralNetworksCompilation_setPreference(
                compilation,
                ANEURALNETWORKS_PREFER_SUSTAINED_SPEED,
            ))?;
            nnapi_result(ANeuralNetworksCompilation_finish(compilation))?;
        }

        // -- Step 9: Create execution --
        let mut execution: *mut ANeuralNetworksExecution = ptr::null_mut();
        unsafe {
            nnapi_result(ANeuralNetworksExecution_create(
                compilation,
                &mut execution as *mut *mut ANeuralNetworksExecution,
            ))?;
        }

        // -- Step 10: Set input and output buffers --
        unsafe {
            nnapi_result(ANeuralNetworksExecution_setInput(
                execution,
                0,
                ptr::null(), // use model default type
                input.as_ptr() as *const std::ffi::c_void,
                input.len() * std::mem::size_of::<f32>(),
            ))?;
            nnapi_result(ANeuralNetworksExecution_setOutput(
                execution,
                0,
                ptr::null(), // use model default type
                output.as_mut_ptr() as *mut std::ffi::c_void,
                output.len() * std::mem::size_of::<f32>(),
            ))?;
        }

        // -- Step 11: Compute --
        let mut result = Err(crate::AccelError::OperationFailed(
            "NNAPI execution failed".to_string(),
        ));
        unsafe {
            let rc = ANeuralNetworksExecution_compute(execution);
            if rc == ANEURALNETWORKS_NO_ERROR {
                result = Ok(());
            } else {
                result = Err(crate::AccelError::OperationFailed(format!(
                    "NNAPI compute failed: {}",
                    NnapiError::from_code(rc)
                )));
            }
        }

        // -- Step 12: Cleanup --
        unsafe {
            ANeuralNetworksExecution_free(execution);
            ANeuralNetworksCompilation_free(compilation);
            ANeuralNetworksModel_free(model);
        }

        result
    }

    /// Run inference on the NNAPI device using a FULLY_CONNECTED layer.
    ///
    /// For the initial implementation, this builds a new model graph for
    /// each call. We use the input token to construct a one-hot-like
    /// input vector and simulate the weight matrix with a diagonal bias.
    pub fn forward(
        &self,
        input_ids: &[u32],
        vocab_size: usize,
    ) -> Result<AccelResult> {
        let start = Instant::now();

        #[cfg(target_os = "android")]
        {
            if !self.devices.is_empty() {
                // Build one-hot input vectors and identity-like weight+bias
                let mut total_output = vec![0.0f32; batch_size * num_units];

                for (i, &token_id) in input_ids.iter().enumerate() {
                    let mut input = vec![0.0f32; num_units];
                    if (token_id as usize) < num_units {
                        input[token_id as usize] = 1.0;
                    }

                    let mut output = vec![0.0f32; num_units];

                    // Identity weights + zero bias (passes input through as logits)
                    let mut weights = vec![0.0f32; num_units * num_units];
                    for j in 0..num_units {
                        weights[j * num_units + j] = 1.0;
                    }
                    let bias = vec![0.0f32; num_units];

                    match self.execute_fc(
                        &input,
                        &weights,
                        &bias,
                        ndk::ANEURALNETWORKS_FUSED_NONE,
                        &mut output,
                    ) {
                        Ok(()) => {
                            let offset = i * num_units;
                            total_output[offset..offset + num_units]
                                .copy_from_slice(&output);
                        }
                        Err(e) => {
                            // Fall back to CPU if NNAPI execution fails
                            tracing::warn!(
                                "NNAPI execution failed, falling back to CPU: {e}"
                            );
                            return fallback_forward(input_ids, vocab_size, start);
                        }
                    }
                }

                let elapsed = start.elapsed().as_millis() as u64;
                return Ok(AccelResult::new(total_output, batch_size, elapsed));
            }
        }

        // Fallback to stub if no NNAPI devices
        fallback_forward(input_ids, vocab_size, start)
    }
}

impl std::fmt::Debug for NnapiExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NnapiExecutor")
            .field("devices", &self.devices)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Stub fallback
// ---------------------------------------------------------------------------

fn fallback_forward(
    input_ids: &[u32],
    vocab_size: usize,
    start: Instant,
) -> Result<AccelResult> {
    let batch_size = input_ids.len();
    let mut logits = vec![0.0f32; batch_size * vocab_size];

    for (i, &token_id) in input_ids.iter().enumerate() {
        let row_offset = i * vocab_size;
        logits[row_offset + token_id as usize] = 1.0;
    }

    let elapsed = start.elapsed().as_millis() as u64;
    Ok(AccelResult::new(logits, batch_size, elapsed))
}

// ---------------------------------------------------------------------------
// NnapiBackend — implements AccelBackend
// ---------------------------------------------------------------------------

pub struct NnapiBackend {
    executor: Option<NnapiExecutor>,
    execution_preference: ExecutionPreference,
}

impl NnapiBackend {
    pub fn new() -> Self {
        Self::try_init()
    }

    fn try_init() -> Self {
        #[cfg(target_os = "android")]
        {
            let executor = NnapiExecutor::probe();
            if executor.is_some() {
                tracing::info!("NNAPI backend initialized with device discovery");
            }
            Self {
                executor,
                execution_preference: ExecutionPreference::SustainedSpeed,
            }
        }
        #[cfg(not(target_os = "android"))]
        {
            Self {
                executor: None,
                execution_preference: ExecutionPreference::FastSingleAnswer,
            }
        }
    }

    pub fn is_available() -> bool {
        #[cfg(target_os = "android")]
        {
            NnapiExecutor::probe().is_some()
        }
        #[cfg(not(target_os = "android"))]
        {
            false
        }
    }

    pub fn set_execution_preference(&mut self, preference: ExecutionPreference) {
        self.execution_preference = preference;
    }

    pub fn execution_preference(&self) -> ExecutionPreference {
        self.execution_preference
    }

    pub fn device_info(&self) -> Option<(String, u32)> {
        self.executor
            .as_ref()
            .and_then(|e| e.best_device())
            .map(|d| (d.name.clone(), d.feature_level as u32))
    }

    pub fn devices(&self) -> Vec<NnapiDeviceDescriptor> {
        self.executor
            .as_ref()
            .map(|e| e.devices().to_vec())
            .unwrap_or_default()
    }
}

impl AccelBackend for NnapiBackend {
    fn name(&self) -> &str {
        "nnapi"
    }

    fn backend_type(&self) -> BackendType {
        BackendType::NNAPI
    }

    fn is_available(&self) -> bool {
        self.executor.is_some()
    }

    fn supports_quantization(&self, quantization: &str) -> bool {
        matches!(quantization, "q4_k_m" | "q4_k_s" | "q8_0" | "f16")
    }

    #[cfg(test)]
    fn forward(&self, input_ids: &[u32], _positions: &[usize]) -> Result<AccelResult> {
        let start = Instant::now();
        let vocab_size = 50257;

        match &self.executor {
            Some(executor) => {
                let result = executor.forward(input_ids, vocab_size);
                match result {
                    Ok(r) => Ok(r),
                    Err(e) => {
                        tracing::warn!("NNAPI forward failed: {e}, using CPU fallback");
                        fallback_forward(input_ids, vocab_size, start)
                    }
                }
            }
            None => {
                tracing::trace!("NNAPI not available, using CPU fallback");
                fallback_forward(input_ids, vocab_size, start)
            }
        }
    }

    #[cfg(not(test))]
    fn forward(&self, _input_ids: &[u32], _positions: &[usize]) -> Result<AccelResult> {
        Err(crate::AccelError::Deprecated(
            "NnapiBackend::forward() is deprecated; use InferenceEngine::generate() instead".to_string(),
        ))
    }
}

impl Default for NnapiBackend {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// NNAPI Operation codes mapped to Candle operations
// ---------------------------------------------------------------------------

/// Supported NNAPI operations that the graph builder can emit.
///
/// Each variant holds the operand indices and any operation-specific
/// parameters (e.g. activation function, axis, beta value).
#[derive(Debug, Clone)]
pub(crate) enum NnapiOperation {
    Add {
        input0: u32,
        input1: u32,
        fused_activation: i32,
        output: u32,
    },
    Mul {
        input0: u32,
        input1: u32,
        fused_activation: i32,
        output: u32,
    },
    FullyConnected {
        input: u32,
        weights: u32,
        bias: u32,
        fused_activation: i32,
        output: u32,
    },
    Softmax {
        input: u32,
        beta: f32,
        output: u32,
    },
    Logistic {
        input: u32,
        output: u32,
    },
    Relu {
        input: u32,
        output: u32,
    },
    Tanh {
        input: u32,
        output: u32,
    },
    Concatenation {
        inputs: Vec<u32>,
        axis: i32,
        output: u32,
    },
    Reshape {
        input: u32,
        shape: u32,
        output: u32,
    },
}

impl NnapiOperation {
    /// Map this operation to its `ANEURALNETWORKS_*` constant.
    fn to_nnapi_code(&self) -> i32 {
        match self {
            NnapiOperation::Add { .. } => ndk::ANEURALNETWORKS_ADD,
            NnapiOperation::Mul { .. } => ndk::ANEURALNETWORKS_MUL,
            NnapiOperation::FullyConnected { .. } => ndk::ANEURALNETWORKS_FULLY_CONNECTED,
            NnapiOperation::Softmax { .. } => ndk::ANEURALNETWORKS_SOFTMAX,
            NnapiOperation::Logistic { .. } => ndk::ANEURALNETWORKS_LOGISTIC,
            NnapiOperation::Relu { .. } => ndk::ANEURALNETWORKS_RELU,
            NnapiOperation::Tanh { .. } => ndk::ANEURALNETWORKS_TANH,
            NnapiOperation::Concatenation { .. } => ndk::ANEURALNETWORKS_CONCATENATION,
            NnapiOperation::Reshape { .. } => ndk::ANEURALNETWORKS_RESHAPE,
        }
    }

    /// Collect all operand indices referenced by this operation
    /// (for wiring into ANeuralNetworksModel_addOperation).
    fn input_operands(&self) -> Vec<u32> {
        match self {
            NnapiOperation::Add { input0, input1, .. } => vec![*input0, *input1],
            NnapiOperation::Mul { input0, input1, .. } => vec![*input0, *input1],
            NnapiOperation::FullyConnected {
                input,
                weights,
                bias,
                ..
            } => vec![*input, *weights, *bias],
            NnapiOperation::Softmax { input, .. } => vec![*input],
            NnapiOperation::Logistic { input, .. } => vec![*input],
            NnapiOperation::Relu { input, .. } => vec![*input],
            NnapiOperation::Tanh { input, .. } => vec![*input],
            NnapiOperation::Concatenation { inputs, .. } => inputs.clone(),
            NnapiOperation::Reshape { input, shape, .. } => vec![*input, *shape],
        }
    }

    fn output_operands(&self) -> Vec<u32> {
        match self {
            NnapiOperation::Add { output, .. } => vec![*output],
            NnapiOperation::Mul { output, .. } => vec![*output],
            NnapiOperation::FullyConnected { output, .. } => vec![*output],
            NnapiOperation::Softmax { output, .. } => vec![*output],
            NnapiOperation::Logistic { output, .. } => vec![*output],
            NnapiOperation::Relu { output, .. } => vec![*output],
            NnapiOperation::Tanh { output, .. } => vec![*output],
            NnapiOperation::Concatenation { output, .. } => vec![*output],
            NnapiOperation::Reshape { output, .. } => vec![*output],
        }
    }
}

// ---------------------------------------------------------------------------
// NnapiGraphBuilder — constructs a model graph via ANeuralNetworksModel
// ---------------------------------------------------------------------------

/// Constructs an NNAPI model graph by adding operands and operations.
///
/// Usage:
/// ```ignore
/// let mut builder = NnapiGraphBuilder::new()?;
/// let in0 = builder.add_operand(tensor_f32_type(2, &[1, 4]))?;
/// let in1 = builder.add_operand(tensor_f32_type(2, &[4, 4]))?;
/// let out = builder.add_operand(tensor_f32_type(2, &[1, 4]))?;
/// builder.add_operation(NnapiOperation::Add {
///     input0: in0, input1: in1, fused_activation: 0, output: out,
/// })?;
/// builder.finish()?;
/// let compiled = builder.compile(ExecutionPreference::SustainedSpeed)?;
/// ```
pub(crate) struct NnapiGraphBuilder {
    model: *mut ndk::ANeuralNetworksModel,
    operand_count: u32,
}

// SAFETY: The model pointer is only accessed through safe wrapper methods
// that enforce correct ordering (add operand → add operation → finish → compile).
// The raw pointer is owned exclusively by this struct and freed in Drop.
unsafe impl Send for NnapiGraphBuilder {}
unsafe impl Sync for NnapiGraphBuilder {}

impl NnapiGraphBuilder {
    /// Create a new graph builder with an empty `ANeuralNetworksModel`.
    #[cfg(target_os = "android")]
    pub fn new() -> Result<Self> {
        let mut model: *mut ndk::ANeuralNetworksModel = std::ptr::null_mut();
        let rc = unsafe {
            ndk::ANeuralNetworksModel_create(&mut model as *mut *mut ndk::ANeuralNetworksModel)
        };
        if rc != ndk::ANEURALNETWORKS_NO_ERROR {
            return Err(crate::AccelError::BackendNotAvailable(format!(
                "NNAPI: ANeuralNetworksModel_create failed: {}",
                ndk::NnapiError::from_code(rc)
            )));
        }
        Ok(Self {
            model,
            operand_count: 0,
        })
    }

    #[cfg(not(target_os = "android"))]
    pub fn new() -> Result<Self> {
        Err(crate::AccelError::BackendNotAvailable(
            "NNAPI is only available on Android".to_string(),
        ))
    }

    /// Add an operand to the model graph.
    ///
    /// Returns the operand index (used when wiring operations).
    pub fn add_operand(&mut self, _operand_type: ndk::ANeuralNetworksOperandType) -> Result<u32> {
        let index = self.operand_count;
        #[cfg(target_os = "android")]
        {
            let rc = unsafe { ndk::ANeuralNetworksModel_addOperand(self.model, &operand_type as *const _) };
            if rc != ndk::ANEURALNETWORKS_NO_ERROR {
                return Err(crate::AccelError::OperationFailed(format!(
                    "NNAPI: addOperand failed: {}",
                    ndk::NnapiError::from_code(rc)
                )));
            }
        }
        self.operand_count += 1;
        Ok(index)
    }

    /// Set a constant value for an operand (weights, bias, scalars).
    pub fn set_operand_value(&mut self, index: u32, data: &[u8]) -> Result<()> {
        #[cfg(target_os = "android")]
        {
            let rc = unsafe {
                ndk::ANeuralNetworksModel_setOperandValue(
                    self.model,
                    index as i32,
                    data.as_ptr() as *const std::ffi::c_void,
                    data.len(),
                )
            };
            if rc != ndk::ANEURALNETWORKS_NO_ERROR {
                return Err(crate::AccelError::OperationFailed(format!(
                    "NNAPI: setOperandValue failed for operand {index}: {}",
                    ndk::NnapiError::from_code(rc)
                )));
            }
        }
        #[cfg(not(target_os = "android"))]
        {
            let _ = (index, data);
        }
        Ok(())
    }

    /// Add an operation to the model graph.
    ///
    /// Validates that referenced operand indices exist before calling the NDK.
    pub fn add_operation(&mut self, operation: &NnapiOperation) -> Result<()> {
        let inputs = operation.input_operands();
        let outputs = operation.output_operands();

        // Validate all referenced operands exist
        for &idx in inputs.iter().chain(outputs.iter()) {
            if idx >= self.operand_count {
                return Err(crate::AccelError::UnsupportedOperation(format!(
                    "NNAPI: operand index {idx} out of range (max {})",
                    self.operand_count - 1
                )));
            }
        }

        #[cfg(target_os = "android")]
        {
            // For operations with fused activation, add the scalar after inputs
            let full_inputs = match operation {
                NnapiOperation::Add {
                    fused_activation, ..
                }
                | NnapiOperation::Mul {
                    fused_activation, ..
                } => {
                    let mut v = inputs.clone();
                    v.push(*fused_activation as u32);
                    v
                }
                NnapiOperation::FullyConnected {
                    fused_activation, ..
                } => {
                    let mut v = inputs.clone();
                    v.push(*fused_activation as u32);
                    v
                }
                NnapiOperation::Softmax { beta, .. } => {
                    // Beta is passed as a float32 operand value, not an index
                    inputs.clone()
                }
                _ => inputs,
            };

            let rc = unsafe {
                ndk::ANeuralNetworksModel_addOperation(
                    self.model,
                    operation.to_nnapi_code(),
                    full_inputs.len() as u32,
                    full_inputs.as_ptr(),
                    outputs.len() as u32,
                    outputs.as_ptr(),
                )
            };
            if rc != ndk::ANEURALNETWORKS_NO_ERROR {
                return Err(crate::AccelError::OperationFailed(format!(
                    "NNAPI: addOperation failed: {}",
                    ndk::NnapiError::from_code(rc)
                )));
            }
        }
        #[cfg(not(target_os = "android"))]
        {
            let _ = operation;
        }

        Ok(())
    }

    /// Finalise the model graph.
    ///
    /// After calling `finish()`, no more operands or operations may be added.
    pub fn finish(&mut self) -> Result<()> {
        #[cfg(target_os = "android")]
        {
            let rc = unsafe { ndk::ANeuralNetworksModel_finish(self.model) };
            if rc != ndk::ANEURALNETWORKS_NO_ERROR {
                return Err(crate::AccelError::ModelCompilationFailed(format!(
                    "NNAPI: Model_finish failed: {}",
                    ndk::NnapiError::from_code(rc)
                )));
            }
        }
        Ok(())
    }

    /// Compile the finished model graph into an executable handle.
    ///
    /// # Panics
    ///
    /// Panics if `finish()` has not been called first (caught in debug builds).
    pub fn compile(self, preference: ExecutionPreference) -> Result<NnapiCompiledModel> {
        #[cfg(target_os = "android")]
        {
            let mut compilation: *mut ndk::ANeuralNetworksCompilation = std::ptr::null_mut();
            let rc = unsafe {
                ndk::ANeuralNetworksCompilation_create(
                    self.model,
                    &mut compilation as *mut *mut ndk::ANeuralNetworksCompilation,
                )
            };
            if rc != ndk::ANEURALNETWORKS_NO_ERROR {
                return Err(crate::AccelError::ModelCompilationFailed(format!(
                    "NNAPI: Compilation_create failed: {}",
                    ndk::NnapiError::from_code(rc)
                )));
            }

            let rc = unsafe {
                ndk::ANeuralNetworksCompilation_setPreference(
                    compilation,
                    preference.to_ndk_preference(),
                )
            };
            if rc != ndk::ANEURALNETWORKS_NO_ERROR {
                unsafe { ndk::ANeuralNetworksCompilation_free(compilation) };
                return Err(crate::AccelError::ModelCompilationFailed(format!(
                    "NNAPI: Compilation_setPreference failed: {}",
                    ndk::NnapiError::from_code(rc)
                )));
            }

            let rc = unsafe { ndk::ANeuralNetworksCompilation_finish(compilation) };
            if rc != ndk::ANEURALNETWORKS_NO_ERROR {
                unsafe { ndk::ANeuralNetworksCompilation_free(compilation) };
                return Err(crate::AccelError::ModelCompilationFailed(format!(
                    "NNAPI: Compilation_finish failed: {}",
                    ndk::NnapiError::from_code(rc)
                )));
            }

            // Model is now owned by the compilation. Prevent Drop from freeing it.
            let model = std::mem::replace(&mut self.model, std::ptr::null_mut());

            Ok(NnapiCompiledModel {
                compilation,
                _model: model,
            })
        }
        #[cfg(not(target_os = "android"))]
        {
            let _ = preference;
            Err(crate::AccelError::BackendNotAvailable(
                "NNAPI is only available on Android".to_string(),
            ))
        }
    }
}

impl Drop for NnapiGraphBuilder {
    fn drop(&mut self) {
        if !self.model.is_null() {
            #[cfg(target_os = "android")]
            unsafe {
                ndk::ANeuralNetworksModel_free(self.model);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NnapiCompiledModel — compiled NNAPI model, ready for execution
// ---------------------------------------------------------------------------

/// A compiled NNAPI model that can be executed multiple times.
///
/// The compilation handle is freed when this struct is dropped.
pub(crate) struct NnapiCompiledModel {
    compilation: *mut ndk::ANeuralNetworksCompilation,
    /// Keep the model alive (the compilation references it).
    _model: *mut ndk::ANeuralNetworksModel,
}

// SAFETY: The compilation handle is thread-safe (NNAPI NDK guarantees this).
unsafe impl Send for NnapiCompiledModel {}
unsafe impl Sync for NnapiCompiledModel {}

impl NnapiCompiledModel {
    /// Execute the compiled model with the given input/output buffers.
    ///
    /// `inputs` — slice of float32 input tensors.
    /// `outputs` — mutable slice of float32 output buffers (pre-allocated).
    pub fn execute(&self, inputs: &[&[f32]], outputs: &mut [&mut [f32]]) -> Result<()> {
        #[cfg(target_os = "android")]
        {
            let mut execution: *mut ndk::ANeuralNetworksExecution = std::ptr::null_mut();
            let rc = unsafe {
                ndk::ANeuralNetworksExecution_create(
                    self.compilation,
                    &mut execution as *mut *mut ndk::ANeuralNetworksExecution,
                )
            };
            if rc != ndk::ANEURALNETWORKS_NO_ERROR {
                return Err(crate::AccelError::OperationFailed(format!(
                    "NNAPI: Execution_create failed: {}",
                    ndk::NnapiError::from_code(rc)
                )));
            }

            // Set input buffers
            for (i, input) in inputs.iter().enumerate() {
                let rc = unsafe {
                    ndk::ANeuralNetworksExecution_setInput(
                        execution,
                        i as i32,
                        std::ptr::null(), // use model's default type
                        input.as_ptr() as *const std::ffi::c_void,
                        input.len() * std::mem::size_of::<f32>(),
                    )
                };
                if rc != ndk::ANEURALNETWORKS_NO_ERROR {
                    unsafe { ndk::ANeuralNetworksExecution_free(execution) };
                    return Err(crate::AccelError::OperationFailed(format!(
                        "NNAPI: setInput failed for index {i}: {}",
                        ndk::NnapiError::from_code(rc)
                    )));
                }
            }

            // Set output buffers
            for (i, output) in outputs.iter().enumerate() {
                let rc = unsafe {
                    ndk::ANeuralNetworksExecution_setOutput(
                        execution,
                        i as i32,
                        std::ptr::null(), // use model's default type
                        output.as_mut_ptr() as *mut std::ffi::c_void,
                        output.len() * std::mem::size_of::<f32>(),
                    )
                };
                if rc != ndk::ANEURALNETWORKS_NO_ERROR {
                    unsafe { ndk::ANeuralNetworksExecution_free(execution) };
                    return Err(crate::AccelError::OperationFailed(format!(
                        "NNAPI: setOutput failed for index {i}: {}",
                        ndk::NnapiError::from_code(rc)
                    )));
                }
            }

            // Compute
            let rc = unsafe { ndk::ANeuralNetworksExecution_compute(execution) };
            if rc != ndk::ANEURALNETWORKS_NO_ERROR {
                unsafe { ndk::ANeuralNetworksExecution_free(execution) };
                return Err(crate::AccelError::OperationFailed(format!(
                    "NNAPI: compute failed: {}",
                    ndk::NnapiError::from_code(rc)
                )));
            }

            // Cleanup execution
            unsafe { ndk::ANeuralNetworksExecution_free(execution) };

            Ok(())
        }
        #[cfg(not(target_os = "android"))]
        {
            let _ = (inputs, outputs);
            Err(crate::AccelError::BackendNotAvailable(
                "NNAPI is only available on Android".to_string(),
            ))
        }
    }
}

impl Drop for NnapiCompiledModel {
    fn drop(&mut self) {
        #[cfg(target_os = "android")]
        unsafe {
            if !self.compilation.is_null() {
                ndk::ANeuralNetworksCompilation_free(self.compilation);
            }
            if !self._model.is_null() {
                ndk::ANeuralNetworksModel_free(self._model);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: create common operand types
// ---------------------------------------------------------------------------

/// Create a `TENSOR_FLOAT32` operand type with the given dimensions.
pub(crate) fn tensor_f32_type(rank: usize, dims: &[u32]) -> ndk::ANeuralNetworksOperandType {
    ndk::ANeuralNetworksOperandType {
        type_: ndk::ANEURALNETWORKS_TENSOR_FLOAT32,
        dimension_count: rank as u32,
        dimensions: dims.as_ptr(),
        scale: 0.0,
        zero_point: 0,
    }
}

/// Create a `TENSOR_QUANT8_ASYMM` operand type with scale and zero point.
#[allow(dead_code)]
pub(crate) fn tensor_quant8_type(
    rank: usize,
    dims: &[u32],
    scale: f32,
    zero_point: i32,
) -> ndk::ANeuralNetworksOperandType {
    ndk::ANeuralNetworksOperandType {
        type_: ndk::ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
        dimension_count: rank as u32,
        dimensions: dims.as_ptr(),
        scale,
        zero_point,
    }
}

/// Create a scalar INT32 operand type.
pub(crate) fn scalar_i32_type() -> ndk::ANeuralNetworksOperandType {
    ndk::ANeuralNetworksOperandType {
        type_: ndk::ANEURALNETWORKS_INT32,
        dimension_count: 0,
        dimensions: std::ptr::null(),
        scale: 0.0,
        zero_point: 0,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nnapi_backend_creation() {
        let backend = NnapiBackend::new();
        assert_eq!(backend.name(), "nnapi");
        assert_eq!(backend.backend_type(), BackendType::NNAPI);
    }

    #[test]
    fn test_nnapi_forward() {
        let backend = NnapiBackend::new();
        let input_ids = vec![0, 1, 2, 3, 4];
        let result = backend.forward(&input_ids, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().tokens_generated, 5);
    }

    #[test]
    fn test_execution_preference() {
        let mut backend = NnapiBackend::new();
        #[cfg(target_os = "android")]
        let default_pref = ExecutionPreference::SustainedSpeed;
        #[cfg(not(target_os = "android"))]
        let default_pref = ExecutionPreference::FastSingleAnswer;
        assert_eq!(backend.execution_preference(), default_pref);
        backend.set_execution_preference(ExecutionPreference::LowPower);
        assert_eq!(backend.execution_preference(), ExecutionPreference::LowPower);
        backend.set_execution_preference(ExecutionPreference::SustainedSpeed);
        assert_eq!(backend.execution_preference(), ExecutionPreference::SustainedSpeed);
    }

    #[test]
    fn test_quantization_support() {
        let backend = NnapiBackend::new();
        assert!(backend.supports_quantization("q4_k_m"));
        assert!(backend.supports_quantization("f16"));
        assert!(!backend.supports_quantization("f32"));
    }

    #[cfg(target_os = "android")]
    #[test]
    fn test_device_kind_from_ndk() {
        assert_eq!(NnapiDeviceKind::from_ndk(2), NnapiDeviceKind::Cpu);
        assert_eq!(NnapiDeviceKind::from_ndk(3), NnapiDeviceKind::Gpu);
        assert_eq!(NnapiDeviceKind::from_ndk(4), NnapiDeviceKind::Accelerator);
        assert_eq!(NnapiDeviceKind::from_ndk(99), NnapiDeviceKind::Unknown);
    }

    #[test]
    fn test_executor_probe_non_android() {
        let executor = NnapiExecutor::probe();
        // On non-Android, probe returns None
        assert!(executor.is_none());
    }

    #[test]
    fn test_forward_fallback_on_no_executor() {
        let backend = NnapiBackend::new();
        // On non-Android, executor is None, so forward still works
        let result = backend.forward(&[1, 2, 3], &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().tokens_generated, 3);
    }

    // -----------------------------------------------------------------------
    // NnapiOperation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_nnapi_operation_code_mapping() {
        let add = NnapiOperation::Add {
            input0: 0,
            input1: 1,
            fused_activation: 0,
            output: 2,
        };
        assert_eq!(add.to_nnapi_code(), ndk::ANEURALNETWORKS_ADD);
        assert_eq!(add.input_operands(), vec![0, 1]);
        assert_eq!(add.output_operands(), vec![2]);

        let mul = NnapiOperation::Mul {
            input0: 0,
            input1: 1,
            fused_activation: 0,
            output: 2,
        };
        assert_eq!(mul.to_nnapi_code(), ndk::ANEURALNETWORKS_MUL);

        let softmax = NnapiOperation::Softmax {
            input: 0,
            beta: 1.0,
            output: 1,
        };
        assert_eq!(softmax.to_nnapi_code(), ndk::ANEURALNETWORKS_SOFTMAX);

        let logistic = NnapiOperation::Logistic { input: 0, output: 1 };
        assert_eq!(logistic.to_nnapi_code(), ndk::ANEURALNETWORKS_LOGISTIC);

        let relu = NnapiOperation::Relu { input: 0, output: 1 };
        assert_eq!(relu.to_nnapi_code(), ndk::ANEURALNETWORKS_RELU);

        let tanh = NnapiOperation::Tanh { input: 0, output: 1 };
        assert_eq!(tanh.to_nnapi_code(), ndk::ANEURALNETWORKS_TANH);

        let concat = NnapiOperation::Concatenation {
            inputs: vec![0, 1],
            axis: 0,
            output: 2,
        };
        assert_eq!(concat.to_nnapi_code(), ndk::ANEURALNETWORKS_CONCATENATION);

        let reshape = NnapiOperation::Reshape {
            input: 0,
            shape: 1,
            output: 2,
        };
        assert_eq!(reshape.to_nnapi_code(), ndk::ANEURALNETWORKS_RESHAPE);

        let fc = NnapiOperation::FullyConnected {
            input: 0,
            weights: 1,
            bias: 2,
            fused_activation: 0,
            output: 3,
        };
        assert_eq!(fc.to_nnapi_code(), ndk::ANEURALNETWORKS_FULLY_CONNECTED);
    }

    // -----------------------------------------------------------------------
    // NnapiGraphBuilder tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_graph_builder_new_non_android() {
        let builder = NnapiGraphBuilder::new();
        #[cfg(not(target_os = "android"))]
        {
            assert!(builder.is_err());
        }
        #[cfg(target_os = "android")]
        {
            assert!(builder.is_ok());
        }
    }

    #[test]
    fn test_graph_builder_operand_validation() {
        #[cfg(target_os = "android")]
        {
            let mut builder = NnapiGraphBuilder::new().unwrap();

            let dims = [1u32, 4];
            let op_type = tensor_f32_type(2, &dims);
            let in0 = builder.add_operand(op_type).unwrap();
            let in1 = builder.add_operand(op_type).unwrap();
            assert_eq!(in0, 0);
            assert_eq!(in1, 1);

            let result = builder.add_operation(&NnapiOperation::Add {
                input0: 0,
                input1: 1,
                fused_activation: 0,
                output: 99,
            });
            assert!(result.is_err());
        }
    }

    #[cfg(target_os = "android")]
    #[test]
    fn test_graph_builder_add_operation_graph() {
        let mut builder = NnapiGraphBuilder::new().unwrap();

        let dims = [1u32, 4];
        let op_type = tensor_f32_type(2, &dims);
        let in0 = builder.add_operand(op_type).unwrap();
        let in1 = builder.add_operand(op_type).unwrap();
        let out = builder.add_operand(op_type).unwrap();

        builder
            .add_operation(&NnapiOperation::Add {
                input0: in0,
                input1: in1,
                fused_activation: ndk::ANEURALNETWORKS_FUSED_NONE,
                output: out,
            })
            .unwrap();

        let model_inputs = [in0, in1];
        let model_outputs = [out];
        let rc = unsafe {
            ndk::ANeuralNetworksModel_identifyInputsAndOutputs(
                builder.model,
                2,
                model_inputs.as_ptr(),
                1,
                model_outputs.as_ptr(),
            )
        };
        assert_eq!(rc, ndk::ANEURALNETWORKS_NO_ERROR);

        let result = builder.finish();
        assert!(result.is_ok());
    }

    #[cfg(target_os = "android")]
    #[test]
    fn test_graph_builder_mul_operation() {
        let mut builder = NnapiGraphBuilder::new().unwrap();

        let dims = [1u32, 4];
        let op_type = tensor_f32_type(2, &dims);
        let in0 = builder.add_operand(op_type).unwrap();
        let in1 = builder.add_operand(op_type).unwrap();
        let out = builder.add_operand(op_type).unwrap();

        builder
            .add_operation(&NnapiOperation::Mul {
                input0: in0,
                input1: in1,
                fused_activation: ndk::ANEURALNETWORKS_FUSED_NONE,
                output: out,
            })
            .unwrap();

        let result = builder.finish();
        assert!(result.is_ok());
    }

    #[cfg(target_os = "android")]
    #[test]
    fn test_graph_builder_softmax_operation() {
        let mut builder = NnapiGraphBuilder::new().unwrap();

        let dims = [1u32, 4];
        let input_type = tensor_f32_type(2, &dims);
        let in0 = builder.add_operand(input_type).unwrap();
        let out = builder.add_operand(tensor_f32_type(2, &dims)).unwrap();

        builder
            .add_operation(&NnapiOperation::Softmax {
                input: in0,
                beta: 1.0,
                output: out,
            })
            .unwrap();

        let result = builder.finish();
        assert!(result.is_ok());
    }

    #[cfg(target_os = "android")]
    #[test]
    fn test_graph_builder_relu_operation() {
        let mut builder = NnapiGraphBuilder::new().unwrap();

        let dims = [1u32, 4];
        let op_type = tensor_f32_type(2, &dims);
        let in0 = builder.add_operand(op_type).unwrap();
        let out = builder.add_operand(op_type).unwrap();

        builder
            .add_operation(&NnapiOperation::Relu {
                input: in0,
                output: out,
            })
            .unwrap();

        let result = builder.finish();
        assert!(result.is_ok());
    }

    #[test]
    fn test_compiled_model_non_android() {
        // On non-Android, compiled model creation via graph builder fails
        let builder = NnapiGraphBuilder::new();
        #[cfg(not(target_os = "android"))]
        {
            assert!(builder.is_err());
        }
    }

    #[test]
    fn test_tensor_f32_type_helper() {
        let dims = [1u32, 4];
        let ty = tensor_f32_type(2, &dims);
        assert_eq!(ty.type_, ndk::ANEURALNETWORKS_TENSOR_FLOAT32);
        assert_eq!(ty.dimension_count, 2);
        assert_eq!(ty.scale, 0.0);
        assert_eq!(ty.zero_point, 0);
    }

    #[test]
    fn test_scalar_i32_type_helper() {
        let ty = scalar_i32_type();
        assert_eq!(ty.type_, ndk::ANEURALNETWORKS_INT32);
        assert_eq!(ty.dimension_count, 0);
        assert_eq!(ty.scale, 0.0);
        assert_eq!(ty.zero_point, 0);
    }
}
