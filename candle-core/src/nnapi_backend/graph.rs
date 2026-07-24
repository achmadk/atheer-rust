//! NNAPI Graph Builder for constructing model graphs.
//!
//! This module provides `NnapiGraphBuilder` for constructing NNAPI model graphs
//! by adding operands and operations, and `NnapiCompiledModel` for compiled
//! executable handles.
//!
//! # Architecture Note
//!
//! The current implementation in `executor.rs` builds and executes NNAPI models
//! inline for each operation type (FC, conv2d, binary, unary, softmax).
//! This module is intended for a higher-level graph-based approach where
//! operations are added to a builder and the model is compiled once,
//! then executed multiple times.
//!
//! # Example
//!
//! ```ignore
//! let mut builder = NnapiGraphBuilder::new()?;
//! let input = builder.add_operand(tensor_f32_type(2, &[1, 128]))?;
//! let weight = builder.add_operand(tensor_f32_type(2, &[128, 512]))?;
//! let bias = builder.add_operand(tensor_f32_type(1, &[512]))?;
//! let output = builder.add_operand(tensor_f32_type(2, &[1, 512]))?;
//!
//! builder.add_operation(NnapiOperation::FullyConnected {
//!     input,
//!     weights: weight,
//!     bias,
//!     fused_activation: 0,
//!     output,
//! })?;
//!
//! builder.finish()?;
//! let compiled = builder.compile(ExecutionPreference::SustainedSpeed)?;
//! ```

use crate::{DType, Layout, Result, Shape};

#[cfg(all(feature = "nnapi", target_os = "android"))]
use crate::nnapi_backend::nnapi_ndk as ndk;

#[derive(Debug, Clone)]
pub enum UnaryOp {
    Relu,
    Tanh,
    Logistic,
    Softmax,
}

#[derive(Debug, Clone)]
pub enum BinaryOp {
    Add,
    Mul,
}

#[derive(Debug, Clone)]
pub enum NnapiOperation {
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
    Conv2d {
        input: u32,
        filter: u32,
        bias: u32,
        padding_left: u32,
        padding_right: u32,
        padding_top: u32,
        padding_bottom: u32,
        stride_w: u32,
        stride_h: u32,
        fused_activation: i32,
        output: u32,
    },
}

impl NnapiOperation {
    #[cfg(all(feature = "nnapi", target_os = "android"))]
    pub fn to_nnapi_code(&self) -> i32 {
        match self {
            NnapiOperation::Add { .. } => ndk::ANEURALNETWORKS_ADD,
            NnapiOperation::Mul { .. } => ndk::ANEURALNETWORKS_MUL,
            NnapiOperation::FullyConnected { .. } => ndk::ANEURALNETWORKS_FULLY_CONNECTED,
            NnapiOperation::Softmax { .. } => ndk::ANEURALNETWORKS_SOFTMAX,
            NnapiOperation::Logistic { .. } => ndk::ANEURALNETWORKS_LOGISTIC,
            NnapiOperation::Relu { .. } => ndk::ANEURALNETWORKS_RELU,
            NnapiOperation::Tanh { .. } => ndk::ANEURALNETWORKS_TANH,
            NnapiOperation::Conv2d { .. } => ndk::ANEURALNETWORKS_CONV_2D,
        }
    }
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub struct NnapiGraphBuilder {
    model: *mut ndk::ANeuralNetworksModel,
    operand_count: u32,
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
impl NnapiGraphBuilder {
    pub fn new() -> Result<Self> {
        let mut model: *mut ndk::ANeuralNetworksModel = std::ptr::null_mut();
        let rc = unsafe {
            ndk::ANeuralNetworksModel_create(&mut model as *mut *mut ndk::ANeuralNetworksModel)
        };
        if rc != ndk::ANEURALNETWORKS_NO_ERROR {
            return Err(crate::Error::Nnapi(crate::NnapiError::Message(format!(
                "NNAPI: ANeuralNetworksModel_create failed: {:?}",
                ndk::NnapiError::from_code(rc)
            ))));
        }
        Ok(Self {
            model,
            operand_count: 0,
        })
    }

    pub fn add_operand(&mut self, operand_type: ndk::ANeuralNetworksOperandType) -> Result<u32> {
        let index = self.operand_count;
        #[cfg(target_os = "android")]
        {
            let rc = unsafe {
                ndk::ANeuralNetworksModel_addOperand(self.model, &operand_type as *const _)
            };
            if rc != ndk::ANEURALNETWORKS_NO_ERROR {
                return Err(crate::Error::Nnapi(crate::NnapiError::Message(format!(
                    "NNAPI: addOperand failed: {:?}",
                    ndk::NnapiError::from_code(rc)
                ))));
            }
        }
        self.operand_count += 1;
        Ok(index)
    }

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
                return Err(crate::Error::Nnapi(crate::NnapiError::Message(format!(
                    "NNAPI: setOperandValue failed for operand {}: {:?}",
                    index,
                    ndk::NnapiError::from_code(rc)
                ))));
            }
        }
        #[cfg(not(target_os = "android"))]
        {
            let _ = (index, data);
        }
        Ok(())
    }

    pub fn add_operation(&mut self, operation: &NnapiOperation) -> Result<()> {
        #[cfg(target_os = "android")]
        {
            let inputs = operation.input_operands();
            let outputs = operation.output_operands();

            let full_inputs = match operation {
                NnapiOperation::Add {
                    fused_activation, ..
                }
                | NnapiOperation::Mul {
                    fused_activation, ..
                }
                | NnapiOperation::FullyConnected {
                    fused_activation, ..
                } => {
                    let mut v = inputs.clone();
                    v.push(*fused_activation as u32);
                    v
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
                return Err(crate::Error::Nnapi(crate::NnapiError::Message(format!(
                    "NNAPI: addOperation failed: {:?}",
                    ndk::NnapiError::from_code(rc)
                ))));
            }
        }
        #[cfg(not(target_os = "android"))]
        {
            let _ = operation;
        }
        Ok(())
    }

    pub fn finish(&mut self) -> Result<()> {
        #[cfg(target_os = "android")]
        {
            let rc = unsafe { ndk::ANeuralNetworksModel_finish(self.model) };
            if rc != ndk::ANEURALNETWORKS_NO_ERROR {
                return Err(crate::Error::Nnapi(crate::NnapiError::Message(format!(
                    "NNAPI: Model_finish failed: {:?}",
                    ndk::NnapiError::from_code(rc)
                ))));
            }
        }
        Ok(())
    }

    pub fn compile(
        mut self,
        preference: crate::nnapi_backend::ExecutionPreference,
    ) -> Result<NnapiCompiledModel> {
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
                return Err(crate::Error::Nnapi(crate::NnapiError::Message(format!(
                    "NNAPI: Compilation_create failed: {:?}",
                    ndk::NnapiError::from_code(rc)
                ))));
            }

            let rc = unsafe {
                ndk::ANeuralNetworksCompilation_setPreference(
                    compilation,
                    preference.to_ndk_preference(),
                )
            };
            if rc != ndk::ANEURALNETWORKS_NO_ERROR {
                unsafe { ndk::ANeuralNetworksCompilation_free(compilation) };
                return Err(crate::Error::Nnapi(crate::NnapiError::Message(format!(
                    "NNAPI: Compilation_setPreference failed: {:?}",
                    ndk::NnapiError::from_code(rc)
                ))));
            }

            let rc = unsafe { ndk::ANeuralNetworksCompilation_finish(compilation) };
            if rc != ndk::ANEURALNETWORKS_NO_ERROR {
                unsafe { ndk::ANeuralNetworksCompilation_free(compilation) };
                return Err(crate::Error::Nnapi(crate::NnapiError::Message(format!(
                    "NNAPI: Compilation_finish failed: {:?}",
                    ndk::NnapiError::from_code(rc)
                ))));
            }

            let model = std::mem::replace(&mut self.model, std::ptr::null_mut());

            Ok(NnapiCompiledModel {
                compilation,
                _model: model,
            })
        }
        #[cfg(not(target_os = "android"))]
        {
            let _ = preference;
            Err(crate::Error::NotCompiledWithNnapiSupport)
        }
    }
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
impl Drop for NnapiGraphBuilder {
    fn drop(&mut self) {
        if !self.model.is_null() {
            unsafe {
                ndk::ANeuralNetworksModel_free(self.model);
            }
        }
    }
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub struct NnapiCompiledModel {
    compilation: *mut ndk::ANeuralNetworksCompilation,
    _model: *mut ndk::ANeuralNetworksModel,
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
unsafe impl Send for NnapiCompiledModel {}
#[cfg(all(feature = "nnapi", target_os = "android"))]
unsafe impl Sync for NnapiCompiledModel {}

#[cfg(all(feature = "nnapi", target_os = "android"))]
impl NnapiCompiledModel {
    pub fn execute(&self, inputs: &[&[f32]], outputs: &mut [&mut [f32]]) -> Result<()> {
        let mut execution: *mut ndk::ANeuralNetworksExecution = std::ptr::null_mut();
        let rc = unsafe {
            ndk::ANeuralNetworksExecution_create(
                self.compilation,
                &mut execution as *mut *mut ndk::ANeuralNetworksExecution,
            )
        };
        if rc != ndk::ANEURALNETWORKS_NO_ERROR {
            return Err(crate::Error::Nnapi(crate::NnapiError::Message(format!(
                "NNAPI: Execution_create failed: {:?}",
                ndk::NnapiError::from_code(rc)
            ))));
        }

        for (i, input) in inputs.iter().enumerate() {
            let rc = unsafe {
                ndk::ANeuralNetworksExecution_setInput(
                    execution,
                    i as i32,
                    std::ptr::null(),
                    input.as_ptr() as *const std::ffi::c_void,
                    input.len() * std::mem::size_of::<f32>(),
                )
            };
            if rc != ndk::ANEURALNETWORKS_NO_ERROR {
                unsafe { ndk::ANeuralNetworksExecution_free(execution) };
                return Err(crate::Error::Nnapi(crate::NnapiError::Message(format!(
                    "NNAPI: setInput failed for index {}: {:?}",
                    i,
                    ndk::NnapiError::from_code(rc)
                ))));
            }
        }

        for (i, output) in outputs.iter_mut().enumerate() {
            let rc = unsafe {
                ndk::ANeuralNetworksExecution_setOutput(
                    execution,
                    i as i32,
                    std::ptr::null(),
                    output.as_mut_ptr() as *mut std::ffi::c_void,
                    output.len() * std::mem::size_of::<f32>(),
                )
            };
            if rc != ndk::ANEURALNETWORKS_NO_ERROR {
                unsafe { ndk::ANeuralNetworksExecution_free(execution) };
                return Err(crate::Error::Nnapi(crate::NnapiError::Message(format!(
                    "NNAPI: setOutput failed for index {}: {:?}",
                    i,
                    ndk::NnapiError::from_code(rc)
                ))));
            }
        }

        let rc = unsafe { ndk::ANeuralNetworksExecution_compute(execution) };
        if rc != ndk::ANEURALNETWORKS_NO_ERROR {
            unsafe { ndk::ANeuralNetworksExecution_free(execution) };
            return Err(crate::Error::Nnapi(crate::NnapiError::Message(format!(
                "NNAPI: compute failed: {:?}",
                ndk::NnapiError::from_code(rc)
            ))));
        }

        unsafe { ndk::ANeuralNetworksExecution_free(execution) };

        Ok(())
    }
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
impl Drop for NnapiCompiledModel {
    fn drop(&mut self) {
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

#[cfg(not(all(feature = "nnapi", target_os = "android")))]
pub struct NnapiGraphBuilder;

#[cfg(not(all(feature = "nnapi", target_os = "android")))]
impl NnapiGraphBuilder {
    pub fn new() -> Result<Self> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }
}

#[cfg(not(all(feature = "nnapi", target_os = "android")))]
pub struct NnapiCompiledModel;

pub fn tensor_f32_type(rank: usize, dims: &[u32]) -> ndk::ANeuralNetworksOperandType {
    ndk::ANeuralNetworksOperandType {
        type_: ndk::ANEURALNETWORKS_TENSOR_FLOAT32,
        dimension_count: rank as u32,
        dimensions: dims.as_ptr(),
        scale: 0.0,
        zero_point: 0,
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ExecutionPreference {
    LowPower,
    FastSingleAnswer,
    SustainedSpeed,
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
impl ExecutionPreference {
    fn to_ndk_preference(&self) -> i32 {
        match self {
            ExecutionPreference::LowPower => ndk::ANEURALNETWORKS_PREFER_LOW_POWER,
            ExecutionPreference::FastSingleAnswer => ndk::ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER,
            ExecutionPreference::SustainedSpeed => ndk::ANEURALNETWORKS_PREFER_SUSTAINED_SPEED,
        }
    }
}
