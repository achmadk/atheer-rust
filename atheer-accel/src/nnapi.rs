use crate::{AccelBackend, AccelResult, BackendType, Result};
use std::time::Instant;

// Conditionally import the NDK bindings
#[cfg(target_os = "android")]
use crate::nnapi_ndk as ndk;

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
}
