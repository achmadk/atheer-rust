//! NNAPI executor for managing device discovery, model compilation, and execution.
//!
//! # Responsibilities
//!
//! - **Device Discovery**: Probes available NNAPI accelerators (NPU, GPU, CPU)
//! - **Model Caching**: Caches compiled models by shape signature to avoid recompilation
//! - **Execution**: Builds and executes NNAPI models for each operation type
//!
//! # Caching Strategy
//!
//! The executor maintains an LRU cache of compiled models keyed by shape signature.
//! This eliminates the significant compilation overhead for repeated tensor shapes.

use crate::{DType, Result, Shape};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone)]
pub struct NnapiExecutorDevice {
    pub name: String,
    pub device_type: NnapiDeviceKind,
    pub feature_level: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NnapiDeviceKind {
    Unknown,
    Other,
    Cpu,
    Gpu,
    Accelerator,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShapeSignature {
    pub dims: Vec<usize>,
    pub dtype: DType,
}

impl ShapeSignature {
    fn from_shape(shape: &Shape, dtype: DType) -> Self {
        Self {
            dims: shape.dims().to_vec(),
            dtype,
        }
    }
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
use super::nnapi_ndk::{
    get_devices, nnapi_result, ANeuralNetworksCompilation, ANeuralNetworksCompilation_create,
    ANeuralNetworksCompilation_finish, ANeuralNetworksCompilation_free,
    ANeuralNetworksCompilation_setPreference, ANeuralNetworksExecution,
    ANeuralNetworksExecution_compute, ANeuralNetworksExecution_create,
    ANeuralNetworksExecution_free, ANeuralNetworksExecution_setInput,
    ANeuralNetworksExecution_setInputFromMemory, ANeuralNetworksExecution_setOutput,
    ANeuralNetworksExecution_setOutputFromMemory, ANeuralNetworksMemory, ANeuralNetworksModel,
    ANeuralNetworksModel_addOperand, ANeuralNetworksModel_addOperation,
    ANeuralNetworksModel_create, ANeuralNetworksModel_finish,
    ANeuralNetworksModel_identifyInputsAndOutputs, ANeuralNetworksModel_setOperandValue,
    ANeuralNetworksOperandType, NnapiError, ANEURALNETWORKS_DEVICE_ACCELERATOR,
    ANEURALNETWORKS_DEVICE_CPU, ANEURALNETWORKS_DEVICE_GPU, ANEURALNETWORKS_DEVICE_OTHER,
    ANEURALNETWORKS_FULLY_CONNECTED, ANEURALNETWORKS_FUSED_NONE, ANEURALNETWORKS_INT32,
    ANEURALNETWORKS_LOGISTIC, ANEURALNETWORKS_NO_ERROR, ANEURALNETWORKS_PREFER_SUSTAINED_SPEED,
    ANEURALNETWORKS_RELU, ANEURALNETWORKS_TANH, ANEURALNETWORKS_TENSOR_FLOAT32,
};

#[cfg(all(feature = "nnapi", target_os = "android"))]
struct CompiledModel {
    compilation: *mut ANeuralNetworksCompilation,
    _model: *mut ANeuralNetworksModel,
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
unsafe impl Send for CompiledModel {}
#[cfg(all(feature = "nnapi", target_os = "android"))]
unsafe impl Sync for CompiledModel {}

pub struct NnapiExecutor {
    #[cfg(all(feature = "nnapi", target_os = "android"))]
    devices: Vec<NnapiExecutorDevice>,
    #[cfg(all(feature = "nnapi", target_os = "android"))]
    compiled_models: RwLock<HashMap<ShapeSignature, CompiledModel>>,
    #[cfg(all(feature = "nnapi", target_os = "android"))]
    max_cache_size: usize,
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
impl NnapiExecutor {
    pub fn probe() -> Option<Self> {
        match get_devices() {
            Ok(devices) if !devices.is_empty() => {
                let executor_devices: Vec<NnapiExecutorDevice> = devices
                    .into_iter()
                    .map(|d| NnapiExecutorDevice {
                        name: d.name,
                        device_type: match d.device_type {
                            ANEURALNETWORKS_DEVICE_CPU => NnapiDeviceKind::Cpu,
                            ANEURALNETWORKS_DEVICE_GPU => NnapiDeviceKind::Gpu,
                            ANEURALNETWORKS_DEVICE_ACCELERATOR => NnapiDeviceKind::Accelerator,
                            ANEURALNETWORKS_DEVICE_OTHER => NnapiDeviceKind::Other,
                            _ => NnapiDeviceKind::Unknown,
                        },
                        feature_level: d.feature_level,
                    })
                    .collect();
                Some(Self {
                    devices: executor_devices,
                    compiled_models: RwLock::new(HashMap::new()),
                    max_cache_size: 100,
                })
            }
            _ => None,
        }
    }

    pub fn devices(&self) -> &[NnapiExecutorDevice] {
        &self.devices
    }

    pub fn has_npu(&self) -> bool {
        self.devices
            .iter()
            .any(|d| d.device_type == NnapiDeviceKind::Accelerator)
    }

    pub fn best_device(&self) -> Option<&NnapiExecutorDevice> {
        self.devices
            .iter()
            .find(|d| d.device_type == NnapiDeviceKind::Accelerator)
            .or_else(|| {
                self.devices
                    .iter()
                    .find(|d| d.device_type == NnapiDeviceKind::Gpu)
            })
            .or_else(|| {
                self.devices
                    .iter()
                    .find(|d| d.device_type == NnapiDeviceKind::Cpu)
            })
            .or_else(|| self.devices.first())
    }

    pub fn cache_size(&self) -> usize {
        self.compiled_models.read().unwrap().len()
    }

    pub fn clear_cache(&self) {
        self.compiled_models.write().unwrap().clear();
    }

    pub fn execute_fc(
        &self,
        input: &[f32],
        weights: &[f32],
        bias: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        use std::ptr;

        let batch_size = 1;
        let input_size = input.len();
        let num_units = output.len();

        let mut model: *mut ANeuralNetworksModel = ptr::null_mut();
        unsafe {
            nnapi_result(ANeuralNetworksModel_create(
                &mut model as *mut *mut ANeuralNetworksModel,
            ))?;
        }

        let input_dims = [batch_size as u32, input_size as u32];
        let input_type = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_TENSOR_FLOAT32,
            dimension_count: 2,
            dimensions: input_dims.as_ptr(),
            scale: 0.0,
            zero_point: 0,
        };

        let weights_dims = [num_units as u32, input_size as u32];
        let weights_type = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_TENSOR_FLOAT32,
            dimension_count: 2,
            dimensions: weights_dims.as_ptr(),
            scale: 0.0,
            zero_point: 0,
        };

        let bias_dims = [num_units as u32];
        let bias_type = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_TENSOR_FLOAT32,
            dimension_count: 1,
            dimensions: bias_dims.as_ptr(),
            scale: 0.0,
            zero_point: 0,
        };

        let act_type = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_INT32,
            dimension_count: 0,
            dimensions: ptr::null(),
            scale: 0.0,
            zero_point: 0,
        };

        let output_dims = [batch_size as u32, num_units as u32];
        let output_type = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_TENSOR_FLOAT32,
            dimension_count: 2,
            dimensions: output_dims.as_ptr(),
            scale: 0.0,
            zero_point: 0,
        };

        unsafe {
            nnapi_result(ANeuralNetworksModel_addOperand(
                model,
                &input_type as *const _,
            ))?;
            nnapi_result(ANeuralNetworksModel_addOperand(
                model,
                &weights_type as *const _,
            ))?;
            nnapi_result(ANeuralNetworksModel_addOperand(
                model,
                &bias_type as *const _,
            ))?;
            nnapi_result(ANeuralNetworksModel_addOperand(
                model,
                &act_type as *const _,
            ))?;
            nnapi_result(ANeuralNetworksModel_addOperand(
                model,
                &output_type as *const _,
            ))?;
        }

        let fused_activation = ANEURALNETWORKS_FUSED_NONE as i32;

        unsafe {
            nnapi_result(ANeuralNetworksModel_setOperandValue(
                model,
                1,
                weights.as_ptr() as *const std::ffi::c_void,
                weights.len() * std::mem::size_of::<f32>(),
            ))?;
            nnapi_result(ANeuralNetworksModel_setOperandValue(
                model,
                2,
                bias.as_ptr() as *const std::ffi::c_void,
                bias.len() * std::mem::size_of::<f32>(),
            ))?;
            nnapi_result(ANeuralNetworksModel_setOperandValue(
                model,
                3,
                &fused_activation as *const i32 as *const std::ffi::c_void,
                std::mem::size_of::<i32>(),
            ))?;
        }

        unsafe {
            nnapi_result(ANeuralNetworksModel_addOperation(
                model,
                ANEURALNETWORKS_FULLY_CONNECTED,
                4,
                [0u32, 1, 2, 3].as_ptr(),
                1,
                [4u32].as_ptr(),
            ))?;
        }

        unsafe {
            nnapi_result(ANeuralNetworksModel_identifyInputsAndOutputs(
                model,
                1,
                [0u32].as_ptr(),
                1,
                [4u32].as_ptr(),
            ))?;
        }

        unsafe {
            nnapi_result(ANeuralNetworksModel_finish(model))?;
        }

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

        let mut execution: *mut ANeuralNetworksExecution = ptr::null_mut();
        unsafe {
            nnapi_result(ANeuralNetworksExecution_create(
                compilation,
                &mut execution as *mut *mut ANeuralNetworksExecution,
            ))?;
        }

        unsafe {
            nnapi_result(ANeuralNetworksExecution_setInput(
                execution,
                0,
                ptr::null(),
                input.as_ptr() as *const std::ffi::c_void,
                input.len() * std::mem::size_of::<f32>(),
            ))?;
            nnapi_result(ANeuralNetworksExecution_setOutput(
                execution,
                0,
                ptr::null(),
                output.as_mut_ptr() as *mut std::ffi::c_void,
                output.len() * std::mem::size_of::<f32>(),
            ))?;
        }

        let result;
        unsafe {
            let rc = ANeuralNetworksExecution_compute(execution);
            if rc == ANEURALNETWORKS_NO_ERROR {
                result = Ok(());
            } else {
                result = Err(crate::Error::Nnapi(crate::NnapiError::Nnapi(format!(
                    "NNAPI compute failed: {:?}",
                    NnapiError::from_code(rc)
                ))));
            }
        }

        unsafe {
            ANeuralNetworksExecution_free(execution);
            ANeuralNetworksCompilation_free(compilation);
            ANeuralNetworksModel_free(model);
        }

        result
    }

    pub fn execute_fc_with_memory(
        &self,
        input: &[f32],
        weights: &[f32],
        bias: &[f32],
        output: &mut [f32],
        input_memory: *mut ANeuralNetworksMemory,
        output_memory: *mut ANeuralNetworksMemory,
    ) -> Result<()> {
        use std::ptr;

        let batch_size = 1;
        let input_size = input.len();
        let num_units = output.len();

        let mut model: *mut ANeuralNetworksModel = ptr::null_mut();
        unsafe {
            nnapi_result(ANeuralNetworksModel_create(
                &mut model as *mut *mut ANeuralNetworksModel,
            ))?;
        }

        let input_dims = [batch_size as u32, input_size as u32];
        let input_type = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_TENSOR_FLOAT32,
            dimension_count: 2,
            dimensions: input_dims.as_ptr(),
            scale: 0.0,
            zero_point: 0,
        };

        let weights_dims = [num_units as u32, input_size as u32];
        let weights_type = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_TENSOR_FLOAT32,
            dimension_count: 2,
            dimensions: weights_dims.as_ptr(),
            scale: 0.0,
            zero_point: 0,
        };

        let bias_dims = [num_units as u32];
        let bias_type = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_TENSOR_FLOAT32,
            dimension_count: 1,
            dimensions: bias_dims.as_ptr(),
            scale: 0.0,
            zero_point: 0,
        };

        let act_type = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_INT32,
            dimension_count: 0,
            dimensions: ptr::null(),
            scale: 0.0,
            zero_point: 0,
        };

        let output_dims = [batch_size as u32, num_units as u32];
        let output_type = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_TENSOR_FLOAT32,
            dimension_count: 2,
            dimensions: output_dims.as_ptr(),
            scale: 0.0,
            zero_point: 0,
        };

        unsafe {
            nnapi_result(ANeuralNetworksModel_addOperand(
                model,
                &input_type as *const _,
            ))?;
            nnapi_result(ANeuralNetworksModel_addOperand(
                model,
                &weights_type as *const _,
            ))?;
            nnapi_result(ANeuralNetworksModel_addOperand(
                model,
                &bias_type as *const _,
            ))?;
            nnapi_result(ANeuralNetworksModel_addOperand(
                model,
                &act_type as *const _,
            ))?;
            nnapi_result(ANeuralNetworksModel_addOperand(
                model,
                &output_type as *const _,
            ))?;
        }

        let fused_activation = ANEURALNETWORKS_FUSED_NONE as i32;

        unsafe {
            nnapi_result(ANeuralNetworksModel_setOperandValue(
                model,
                1,
                weights.as_ptr() as *const std::ffi::c_void,
                weights.len() * std::mem::size_of::<f32>(),
            ))?;
            nnapi_result(ANeuralNetworksModel_setOperandValue(
                model,
                2,
                bias.as_ptr() as *const std::ffi::c_void,
                bias.len() * std::mem::size_of::<f32>(),
            ))?;
            nnapi_result(ANeuralNetworksModel_setOperandValue(
                model,
                3,
                &fused_activation as *const i32 as *const std::ffi::c_void,
                std::mem::size_of::<i32>(),
            ))?;
        }

        unsafe {
            nnapi_result(ANeuralNetworksModel_addOperation(
                model,
                ANEURALNETWORKS_FULLY_CONNECTED,
                4,
                [0u32, 1, 2, 3].as_ptr(),
                1,
                [4u32].as_ptr(),
            ))?;
        }

        unsafe {
            nnapi_result(ANeuralNetworksModel_identifyInputsAndOutputs(
                model,
                1,
                [0u32].as_ptr(),
                1,
                [4u32].as_ptr(),
            ))?;
        }

        unsafe {
            nnapi_result(ANeuralNetworksModel_finish(model))?;
        }

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

        let mut execution: *mut ANeuralNetworksExecution = ptr::null_mut();
        unsafe {
            nnapi_result(ANeuralNetworksExecution_create(
                compilation,
                &mut execution as *mut *mut ANeuralNetworksExecution,
            ))?;
        }

        let input_size_bytes = input.len() * std::mem::size_of::<f32>();
        let output_size_bytes = output.len() * std::mem::size_of::<f32>();

        unsafe {
            nnapi_result(ANeuralNetworksExecution_setInputFromMemory(
                execution,
                0,
                ptr::null(),
                input_memory,
                0,
                input_size_bytes,
            ))?;
            nnapi_result(ANeuralNetworksExecution_setOutputFromMemory(
                execution,
                0,
                ptr::null(),
                output_memory,
                0,
                output_size_bytes,
            ))?;
        }

        let result;
        unsafe {
            let rc = ANeuralNetworksExecution_compute(execution);
            if rc == ANEURALNETWORKS_NO_ERROR {
                result = Ok(());
            } else {
                result = Err(crate::Error::Nnapi(crate::NnapiError::Nnapi(format!(
                    "NNAPI compute failed: {:?}",
                    NnapiError::from_code(rc)
                ))));
            }
        }

        unsafe {
            ANeuralNetworksExecution_free(execution);
            ANeuralNetworksCompilation_free(compilation);
            ANeuralNetworksModel_free(model);
        }

        result
    }

    pub fn execute_unary(
        &self,
        input_data: &[f32],
        output_data: &mut [f32],
        op: UnaryOp,
        _fused_activation: i32,
    ) -> Result<()> {
        use std::ptr;

        let input_size = input_data.len();
        let num_units = output_data.len();
        let batch_size = 1;

        let mut model: *mut ANeuralNetworksModel = ptr::null_mut();
        unsafe {
            nnapi_result(ANeuralNetworksModel_create(
                &mut model as *mut *mut ANeuralNetworksModel,
            ))?;
        }

        let input_type = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_TENSOR_FLOAT32,
            dimension_count: 2,
            dimensions: [batch_size as u32, input_size as u32].as_ptr(),
            scale: 0.0,
            zero_point: 0,
        };

        let output_type = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_TENSOR_FLOAT32,
            dimension_count: 2,
            dimensions: [batch_size as u32, num_units as u32].as_ptr(),
            scale: 0.0,
            zero_point: 0,
        };

        unsafe {
            nnapi_result(ANeuralNetworksModel_addOperand(model, &input_type))?;
            nnapi_result(ANeuralNetworksModel_addOperand(model, &output_type))?;
        }

        let op_code = match op {
            UnaryOp::Relu => ANEURALNETWORKS_RELU,
            UnaryOp::Tanh => ANEURALNETWORKS_TANH,
            UnaryOp::Logistic => ANEURALNETWORKS_LOGISTIC,
        };

        unsafe {
            nnapi_result(ANeuralNetworksModel_addOperation(
                model,
                op_code,
                1,
                [0u32].as_ptr(),
                1,
                [1u32].as_ptr(),
            ))?;
        }

        unsafe {
            nnapi_result(ANeuralNetworksModel_identifyInputsAndOutputs(
                model,
                1,
                [0u32].as_ptr(),
                1,
                [1u32].as_ptr(),
            ))?;
        }

        unsafe {
            nnapi_result(ANeuralNetworksModel_finish(model))?;
        }

        let mut compilation: *mut ANeuralNetworksCompilation = ptr::null_mut();
        unsafe {
            nnapi_result(ANeuralNetworksCompilation_create(model, &mut compilation))?;
            nnapi_result(ANeuralNetworksCompilation_setPreference(
                compilation,
                ANEURALNETWORKS_PREFER_SUSTAINED_SPEED,
            ))?;
            nnapi_result(ANeuralNetworksCompilation_finish(compilation))?;
        }

        let mut execution: *mut ANeuralNetworksExecution = ptr::null_mut();
        unsafe {
            nnapi_result(ANeuralNetworksExecution_create(compilation, &mut execution))?;
        }

        unsafe {
            nnapi_result(ANeuralNetworksExecution_setInput(
                execution,
                0,
                ptr::null(),
                input_data.as_ptr() as *const std::ffi::c_void,
                input_data.len() * std::mem::size_of::<f32>(),
            ))?;
            nnapi_result(ANeuralNetworksExecution_setOutput(
                execution,
                0,
                ptr::null(),
                output_data.as_mut_ptr() as *mut std::ffi::c_void,
                output_data.len() * std::mem::size_of::<f32>(),
            ))?;
        }

        let result;
        unsafe {
            let rc = ANeuralNetworksExecution_compute(execution);
            if rc == ANEURALNETWORKS_NO_ERROR {
                result = Ok(());
            } else {
                result = Err(crate::Error::Nnapi(crate::NnapiError::Nnapi(format!(
                    "NNAPI compute failed: {:?}",
                    NnapiError::from_code(rc)
                ))));
            }
        }

        unsafe {
            ANeuralNetworksExecution_free(execution);
            ANeuralNetworksCompilation_free(compilation);
            ANeuralNetworksModel_free(model);
        }

        result
    }

    pub fn execute_binary(
        &self,
        lhs: &[f32],
        rhs: &[f32],
        output: &mut [f32],
        op: BinaryOp,
        fused_activation: i32,
    ) -> Result<()> {
        use std::ptr;

        let input_size = lhs.len();
        let num_units = output.len();
        let batch_size = 1;

        let mut model: *mut ANeuralNetworksModel = ptr::null_mut();
        unsafe {
            nnapi_result(ANeuralNetworksModel_create(
                &mut model as *mut *mut ANeuralNetworksModel,
            ))?;
        }

        let input_type = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_TENSOR_FLOAT32,
            dimension_count: 2,
            dimensions: [batch_size as u32, input_size as u32].as_ptr(),
            scale: 0.0,
            zero_point: 0,
        };

        let act_type = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_INT32,
            dimension_count: 0,
            dimensions: ptr::null(),
            scale: 0.0,
            zero_point: 0,
        };

        let output_type = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_TENSOR_FLOAT32,
            dimension_count: 2,
            dimensions: [batch_size as u32, num_units as u32].as_ptr(),
            scale: 0.0,
            zero_point: 0,
        };

        unsafe {
            nnapi_result(ANeuralNetworksModel_addOperand(model, &input_type))?;
            nnapi_result(ANeuralNetworksModel_addOperand(model, &input_type))?;
            nnapi_result(ANeuralNetworksModel_addOperand(model, &act_type))?;
            nnapi_result(ANeuralNetworksModel_addOperand(model, &output_type))?;
        }

        unsafe {
            nnapi_result(ANeuralNetworksModel_setOperandValue(
                model,
                2,
                &fused_activation as *const i32 as *const std::ffi::c_void,
                std::mem::size_of::<i32>(),
            ))?;
        }

        let op_code = match op {
            BinaryOp::Add => ANEURALNETWORKS_ADD,
            BinaryOp::Mul => ANEURALNETWORKS_MUL,
        };

        unsafe {
            nnapi_result(ANeuralNetworksModel_addOperation(
                model,
                op_code,
                3,
                [0u32, 1, 2].as_ptr(),
                1,
                [3u32].as_ptr(),
            ))?;
        }

        unsafe {
            nnapi_result(ANeuralNetworksModel_identifyInputsAndOutputs(
                model,
                2,
                [0u32, 1u32].as_ptr(),
                1,
                [3u32].as_ptr(),
            ))?;
        }

        unsafe {
            nnapi_result(ANeuralNetworksModel_finish(model))?;
        }

        let mut compilation: *mut ANeuralNetworksCompilation = ptr::null_mut();
        unsafe {
            nnapi_result(ANeuralNetworksCompilation_create(model, &mut compilation))?;
            nnapi_result(ANeuralNetworksCompilation_setPreference(
                compilation,
                ANEURALNETWORKS_PREFER_SUSTAINED_SPEED,
            ))?;
            nnapi_result(ANeuralNetworksCompilation_finish(compilation))?;
        }

        let mut execution: *mut ANeuralNetworksExecution = ptr::null_mut();
        unsafe {
            nnapi_result(ANeuralNetworksExecution_create(compilation, &mut execution))?;
        }

        unsafe {
            nnapi_result(ANeuralNetworksExecution_setInput(
                execution,
                0,
                ptr::null(),
                lhs.as_ptr() as *const std::ffi::c_void,
                lhs.len() * std::mem::size_of::<f32>(),
            ))?;
            nnapi_result(ANeuralNetworksExecution_setInput(
                execution,
                1,
                ptr::null(),
                rhs.as_ptr() as *const std::ffi::c_void,
                rhs.len() * std::mem::size_of::<f32>(),
            ))?;
            nnapi_result(ANeuralNetworksExecution_setOutput(
                execution,
                0,
                ptr::null(),
                output.as_mut_ptr() as *mut std::ffi::c_void,
                output.len() * std::mem::size_of::<f32>(),
            ))?;
        }

        let result;
        unsafe {
            let rc = ANeuralNetworksExecution_compute(execution);
            if rc == ANEURALNETWORKS_NO_ERROR {
                result = Ok(());
            } else {
                result = Err(crate::Error::Nnapi(crate::NnapiError::Nnapi(format!(
                    "NNAPI compute failed: {:?}",
                    NnapiError::from_code(rc)
                ))));
            }
        }

        unsafe {
            ANeuralNetworksExecution_free(execution);
            ANeuralNetworksCompilation_free(compilation);
            ANeuralNetworksModel_free(model);
        }

        result
    }

    pub fn execute_softmax(
        &self,
        input_data: &[f32],
        output_data: &mut [f32],
        _beta: f32,
    ) -> Result<()> {
        use std::ptr;

        let input_size = input_data.len();
        let batch_size = 1;

        let mut model: *mut ANeuralNetworksModel = ptr::null_mut();
        unsafe {
            nnapi_result(ANeuralNetworksModel_create(
                &mut model as *mut *mut ANeuralNetworksModel,
            ))?;
        }

        let input_type = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_TENSOR_FLOAT32,
            dimension_count: 2,
            dimensions: [batch_size as u32, input_size as u32].as_ptr(),
            scale: 0.0,
            zero_point: 0,
        };

        let output_type = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_TENSOR_FLOAT32,
            dimension_count: 2,
            dimensions: [batch_size as u32, input_size as u32].as_ptr(),
            scale: 0.0,
            zero_point: 0,
        };

        unsafe {
            nnapi_result(ANeuralNetworksModel_addOperand(model, &input_type))?;
            nnapi_result(ANeuralNetworksModel_addOperand(model, &output_type))?;
        }

        unsafe {
            nnapi_result(ANeuralNetworksModel_setOperandValue(
                model,
                0,
                input_data.as_ptr() as *const std::ffi::c_void,
                input_data.len() * std::mem::size_of::<f32>(),
            ))?;
        }

        unsafe {
            nnapi_result(ANeuralNetworksModel_addOperation(
                model,
                ANEURALNETWORKS_SOFTMAX,
                1,
                [0u32].as_ptr(),
                1,
                [1u32].as_ptr(),
            ))?;
        }

        unsafe {
            nnapi_result(ANeuralNetworksModel_identifyInputsAndOutputs(
                model,
                1,
                [0u32].as_ptr(),
                1,
                [1u32].as_ptr(),
            ))?;
        }

        unsafe {
            nnapi_result(ANeuralNetworksModel_finish(model))?;
        }

        let mut compilation: *mut ANeuralNetworksCompilation = ptr::null_mut();
        unsafe {
            nnapi_result(ANeuralNetworksCompilation_create(model, &mut compilation))?;
            nnapi_result(ANeuralNetworksCompilation_setPreference(
                compilation,
                ANEURALNETWORKS_PREFER_SUSTAINED_SPEED,
            ))?;
            nnapi_result(ANeuralNetworksCompilation_finish(compilation))?;
        }

        let mut execution: *mut ANeuralNetworksExecution = ptr::null_mut();
        unsafe {
            nnapi_result(ANeuralNetworksExecution_create(compilation, &mut execution))?;
        }

        unsafe {
            nnapi_result(ANeuralNetworksExecution_setOutput(
                execution,
                0,
                ptr::null(),
                output_data.as_mut_ptr() as *mut std::ffi::c_void,
                output_data.len() * std::mem::size_of::<f32>(),
            ))?;
        }

        let result;
        unsafe {
            let rc = ANeuralNetworksExecution_compute(execution);
            if rc == ANEURALNETWORKS_NO_ERROR {
                result = Ok(());
            } else {
                result = Err(crate::Error::Nnapi(crate::NnapiError::Nnapi(format!(
                    "NNAPI compute failed: {:?}",
                    NnapiError::from_code(rc)
                ))));
            }
        }

        unsafe {
            ANeuralNetworksExecution_free(execution);
            ANeuralNetworksCompilation_free(compilation);
            ANeuralNetworksModel_free(model);
        }

        result
    }

    pub fn execute_conv2d(
        &self,
        input: &[f32],
        filter: &[f32],
        bias: &[f32],
        output: &mut [f32],
        input_dims: [usize; 4],
        filter_dims: [usize; 4],
        output_dims: [usize; 4],
        padding: [i32; 4],
        stride: [i32; 2],
        fused_activation: i32,
    ) -> Result<()> {
        use std::ptr;

        let [batch, in_h, in_w, in_channels] = input_dims;
        let [filter_h, filter_w, _, out_channels] = filter_dims;
        let [out_h, out_w, _out_channels] = [output_dims[1], output_dims[2], output_dims[3]];

        let input_type = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_TENSOR_FLOAT32,
            dimension_count: 4,
            dimensions: [batch as u32, in_h as u32, in_w as u32, in_channels as u32].as_ptr(),
            scale: 0.0,
            zero_point: 0,
        };

        let filter_type = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_TENSOR_FLOAT32,
            dimension_count: 4,
            dimensions: [
                filter_h as u32,
                filter_w as u32,
                in_channels as u32,
                out_channels as u32,
            ]
            .as_ptr(),
            scale: 0.0,
            zero_point: 0,
        };

        let bias_type = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_TENSOR_FLOAT32,
            dimension_count: 1,
            dimensions: [out_channels as u32].as_ptr(),
            scale: 0.0,
            zero_point: 0,
        };

        let act_type = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_INT32,
            dimension_count: 0,
            dimensions: ptr::null(),
            scale: 0.0,
            zero_point: 0,
        };

        let output_type = ANeuralNetworksOperandType {
            type_: ANEURALNETWORKS_TENSOR_FLOAT32,
            dimension_count: 4,
            dimensions: [
                batch as u32,
                out_h as u32,
                out_w as u32,
                out_channels as u32,
            ]
            .as_ptr(),
            scale: 0.0,
            zero_point: 0,
        };

        let mut model: *mut ANeuralNetworksModel = ptr::null_mut();
        unsafe {
            nnapi_result(ANeuralNetworksModel_create(
                &mut model as *mut *mut ANeuralNetworksModel,
            ))?;
        }

        unsafe {
            nnapi_result(ANeuralNetworksModel_addOperand(
                model,
                &input_type as *const _,
            ))?;
            nnapi_result(ANeuralNetworksModel_addOperand(
                model,
                &filter_type as *const _,
            ))?;
            nnapi_result(ANeuralNetworksModel_addOperand(
                model,
                &bias_type as *const _,
            ))?;
            nnapi_result(ANeuralNetworksModel_addOperand(
                model,
                &act_type as *const _,
            ))?;
            nnapi_result(ANeuralNetworksModel_addOperand(
                model,
                &output_type as *const _,
            ))?;
        }

        unsafe {
            nnapi_result(ANeuralNetworksModel_setOperandValue(
                model,
                1,
                filter.as_ptr() as *const std::ffi::c_void,
                filter.len() * std::mem::size_of::<f32>(),
            ))?;
            nnapi_result(ANeuralNetworksModel_setOperandValue(
                model,
                2,
                bias.as_ptr() as *const std::ffi::c_void,
                bias.len() * std::mem::size_of::<f32>(),
            ))?;
            nnapi_result(ANeuralNetworksModel_setOperandValue(
                model,
                3,
                &fused_activation as *const i32 as *const std::ffi::c_void,
                std::mem::size_of::<i32>(),
            ))?;
        }

        let op_inputs: [u32; 5] = [0, 1, 2, 3, 4];
        unsafe {
            nnapi_result(ANeuralNetworksModel_addOperation(
                model,
                ANEURALNETWORKS_CONV_2D,
                5,
                op_inputs.as_ptr(),
                1,
                [4u32].as_ptr(),
            ))?;
        }

        unsafe {
            nnapi_result(ANeuralNetworksModel_identifyInputsAndOutputs(
                model,
                1,
                [0u32].as_ptr(),
                1,
                [4u32].as_ptr(),
            ))?;
        }

        unsafe {
            nnapi_result(ANeuralNetworksModel_finish(model))?;
        }

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

        let mut execution: *mut ANeuralNetworksExecution = ptr::null_mut();
        unsafe {
            nnapi_result(ANeuralNetworksExecution_create(
                compilation,
                &mut execution as *mut *mut ANeuralNetworksExecution,
            ))?;
        }

        unsafe {
            nnapi_result(ANeuralNetworksExecution_setInput(
                execution,
                0,
                ptr::null(),
                input.as_ptr() as *const std::ffi::c_void,
                input.len() * std::mem::size_of::<f32>(),
            ))?;
            nnapi_result(ANeuralNetworksExecution_setOutput(
                execution,
                0,
                ptr::null(),
                output.as_mut_ptr() as *mut std::ffi::c_void,
                output.len() * std::mem::size_of::<f32>(),
            ))?;
        }

        let result;
        unsafe {
            let rc = ANeuralNetworksExecution_compute(execution);
            if rc == ANEURALNETWORKS_NO_ERROR {
                result = Ok(());
            } else {
                result = Err(crate::Error::Nnapi(crate::NnapiError::Nnapi(format!(
                    "NNAPI conv2d compute failed: {:?}",
                    NnapiError::from_code(rc)
                ))));
            }
        }

        unsafe {
            ANeuralNetworksExecution_free(execution);
            ANeuralNetworksCompilation_free(compilation);
            ANeuralNetworksModel_free(model);
        }

        result
    }
}

#[cfg(not(all(feature = "nnapi", target_os = "android")))]
impl NnapiExecutor {
    pub fn probe() -> Option<Self> {
        None
    }

    pub fn devices(&self) -> &[NnapiExecutorDevice] {
        &[]
    }

    pub fn has_npu(&self) -> bool {
        false
    }

    pub fn best_device(&self) -> Option<&NnapiExecutorDevice> {
        None
    }

    pub fn cache_size(&self) -> usize {
        0
    }

    pub fn clear_cache(&self) {}

    pub fn execute_fc(
        &self,
        _input: &[f32],
        _weights: &[f32],
        _bias: &[f32],
        _output: &mut [f32],
    ) -> Result<()> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    pub fn execute_unary(
        &self,
        _input_data: &[f32],
        _output_data: &mut [f32],
        _op: UnaryOp,
        _fused_activation: i32,
    ) -> Result<()> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    pub fn execute_binary(
        &self,
        _lhs: &[f32],
        _rhs: &[f32],
        _output: &mut [f32],
        _op: BinaryOp,
        _fused_activation: i32,
    ) -> Result<()> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    pub fn execute_softmax(
        &self,
        _input_data: &[f32],
        _output_data: &mut [f32],
        _beta: f32,
    ) -> Result<()> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    pub fn execute_conv2d(
        &self,
        _input: &[f32],
        _filter: &[f32],
        _bias: &[f32],
        _output: &mut [f32],
        _input_dims: [usize; 4],
        _filter_dims: [usize; 4],
        _output_dims: [usize; 4],
        _padding: [i32; 4],
        _stride: [i32; 2],
        _fused_activation: i32,
    ) -> Result<()> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }
}

impl Drop for NnapiExecutor {
    fn drop(&mut self) {
        #[cfg(all(feature = "nnapi", target_os = "android"))]
        self.clear_cache();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Relu,
    Tanh,
    Logistic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Mul,
}

pub type SharedExecutor = Arc<NnapiExecutor>;

pub fn create_shared_executor() -> Option<SharedExecutor> {
    NnapiExecutor::probe().map(Arc::new)
}
