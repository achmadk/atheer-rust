//! NNAPI storage implementation for candle-core.
//!
//! # Overview
//!
//! `NnapiStorage` wraps tensor data for NNAPI execution. It owns the raw byte buffer
//! and maintains a reference to the shared executor for operations.
//!
//! # Memory Model
//!
//! Currently uses direct byte buffers. Future versions will support zero-copy
//! via AHardwareBuffer for improved performance.
//!
//! # Thread Safety
//!
//! The executor is shared via `Arc<RwLock<>>` allowing concurrent access from
//! multiple threads.

use crate::backend::BackendStorage;
use crate::nnapi_backend::executor::{BinaryOp, SharedExecutor, UnaryOp};
use crate::nnapi_backend::{create_shared_executor, NnapiDevice, NnapiError};
use crate::{CpuStorage, DType, Layout, Result, Shape};
use std::os::unix::io::AsRawFd;
use std::sync::Arc;

#[cfg(all(feature = "nnapi", target_os = "android"))]
use crate::nnapi_backend::nnapi_ndk::{
    nnapi_result, AHardwareBuffer, AHardwareBuffer_Desc, AHardwareBuffer_allocate,
    AHardwareBuffer_release, ANeuralNetworksMemory, ANeuralNetworksMemory_createFromFd,
    ANeuralNetworksMemory_createFromHardwareBuffer, ANeuralNetworksMemory_free, NnapiError,
    AHARDWAREBUFFER_FORMAT_BLOB, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN,
    AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN,
};

#[cfg(all(feature = "nnapi", target_os = "android"))]
pub struct NnapiStorage {
    data: Vec<u8>,
    dtype: DType,
    device: NnapiDevice,
    executor: SharedExecutor,
    memory: Option<*mut ANeuralNetworksMemory>,
    hwbuffer: Option<*mut AHardwareBuffer>,
}

#[cfg(not(all(feature = "nnapi", target_os = "android")))]
pub struct NnapiStorage;

static SHARED_EXECUTOR: std::sync::OnceLock<SharedExecutor> = std::sync::OnceLock::new();

fn get_or_create_executor() -> Result<SharedExecutor> {
    SHARED_EXECUTOR
        .get_or_try_init(|| {
            create_shared_executor().ok_or_else(|| {
                crate::Error::Nnapi(NnapiError::Message(
                    "NNAPI executor not available".to_string(),
                ))
            })
        })
        .cloned()
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
impl NnapiStorage {
    pub fn new(data: Vec<u8>, dtype: DType, device: NnapiDevice) -> Result<Self> {
        let executor = get_or_create_executor()?;
        Ok(Self {
            data,
            dtype,
            device,
            executor,
            memory: None,
            hwbuffer: None,
        })
    }

    pub fn with_executor(
        data: Vec<u8>,
        dtype: DType,
        device: NnapiDevice,
        executor: SharedExecutor,
    ) -> Self {
        Self {
            data,
            dtype,
            device,
            executor,
            memory: None,
            hwbuffer: None,
        }
    }

    /// Allocates NNAPI-backed storage using AHardwareBuffer for zero-copy operations.
    ///
    /// This function attempts to allocate memory using AHardwareBuffer, which allows
    /// the NNAPI driver to access tensor data directly without copying. If AHardwareBuffer
    /// allocation fails (e.g., due to memory pressure), it falls back to using
    /// `ANeuralNetworksMemory_createFromFd` with a temporary file.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor to allocate
    /// * `dtype` - The data type of the tensor (F32, F16, BF16 supported)
    /// * `device` - The NNAPI device to associate this storage with
    ///
    /// # Returns
    ///
    /// Returns `NnapiStorage` wrapped in `Result`, or an error if allocation fails.
    ///
    /// # Memory Alignment
    ///
    /// Allocations are 16-byte aligned to meet NNAPI requirements. The actual
    /// allocated size may be larger than `shape.elem_count() * dtype.size_in_bytes()`
    /// if alignment padding is added.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use candle_core::{Device, Tensor, DType, Shape, Result};
    ///
    /// fn example() -> Result<()> {
    ///     let device = Device::new_nnapi(0)?;
    ///     let shape = Shape::from_dims(&[128, 512]);
    ///
    ///     // Allocate zero-copy storage for a 128x512 F32 tensor
    ///     let storage = candle_core::nnapi_backend::NnapiStorage::allocate(
    ///         &shape,
    ///         DType::F32,
    ///         &device,
    ///     )?;
    ///
    ///     // Check if zero-copy path is being used
    ///     assert!(storage.is_zero_copy());
    ///     assert!(storage.memory().is_some());
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn allocate(shape: &Shape, dtype: DType, device: &crate::NnapiDevice) -> Result<Self> {
        use std::fs;
        use std::io::Write;
        use std::os::unix::io::FromRawFd;
        use std::ptr;

        let executor = get_or_create_executor()?;
        let count = shape.elem_count();
        let size = count * dtype.size_in_bytes();
        let aligned_size = Self::align_size(size);

        let (memory, hwbuffer, data) = match Self::allocate_ahardware_buffer(aligned_size) {
            Ok((mem, buf)) => {
                let data = vec![0u8; aligned_size];
                (Some(mem), Some(buf), data)
            }
            Err(_) => {
                let (mem, fd) = Self::allocate_from_fd(aligned_size)?;
                let data = vec![0u8; aligned_size];
                (Some(mem), None, data)
            }
        };

        Ok(Self {
            data,
            dtype,
            device: device.clone(),
            executor,
            memory,
            hwbuffer,
        })
    }

    const NNAPI_MEMORY_ALIGNMENT: usize = 16;

    fn align_size(size: usize) -> usize {
        (size + Self::NNAPI_MEMORY_ALIGNMENT - 1) & !(Self::NNAPI_MEMORY_ALIGNMENT - 1)
    }

    fn allocate_ahardware_buffer(
        size: usize,
    ) -> Result<(*mut ANeuralNetworksMemory, *mut AHardwareBuffer)> {
        use std::ptr;

        let desc = AHardwareBuffer_Desc {
            width: size as u64,
            height: 1,
            layers: 1,
            format: AHARDWAREBUFFER_FORMAT_BLOB,
            usage: AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN,
            stride: 0,
        };

        let mut hw_buffer: *mut AHardwareBuffer = ptr::null_mut();
        let rc =
            unsafe { AHardwareBuffer_allocate(&desc, &mut hw_buffer as *mut *mut AHardwareBuffer) };
        nnapi_result(rc)?;

        let mut memory: *mut ANeuralNetworksMemory = ptr::null_mut();
        let rc = unsafe {
            ANeuralNetworksMemory_createFromHardwareBuffer(
                ptr::null(),
                hw_buffer,
                &mut memory as *mut *mut ANeuralNetworksMemory,
            )
        };
        if rc != 0 {
            unsafe {
                AHardwareBuffer_release(hw_buffer);
            }
            return Err(crate::Error::Nnapi(NnapiError::Nnapi(format!(
                "ANeuralNetworksMemory_createFromHardwareBuffer failed: {:?}",
                NnapiError::from_code(rc)
            ))));
        }

        Ok((memory, hw_buffer))
    }

    fn allocate_from_fd(size: usize) -> Result<(*mut ANeuralNetworksMemory, i32)> {
        use std::fs;
        use std::io::Write;
        use std::os::unix::io::FromRawFd;
        use std::ptr;

        let mut tmpfile = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open("/tmp/candle_nnapiXXXXXX")?;
        tmpfile.write_all(&vec![0u8; size])?;
        let fd = tmpfile.into_raw_fd();

        let mut memory: *mut ANeuralNetworksMemory = ptr::null_mut();
        let rc = unsafe {
            ANeuralNetworksMemory_createFromFd(
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                fd,
                0,
                &mut memory as *mut *mut ANeuralNetworksMemory,
            )
        };
        nnapi_result(rc)?;

        Ok((memory, fd))
    }

    pub fn memory(&self) -> Option<*mut ANeuralNetworksMemory> {
        self.memory
    }

    pub fn hwbuffer(&self) -> Option<*mut AHardwareBuffer> {
        self.hwbuffer
    }

    pub fn is_zero_copy(&self) -> bool {
        self.memory.is_some()
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> &NnapiDevice {
        &self.device
    }

    pub fn executor(&self) -> &SharedExecutor {
        &self.executor
    }

    pub fn count(&self) -> usize {
        self.data.len() / self.dtype.size_in_bytes()
    }

    pub fn zeros_impl(
        device: &crate::NnapiDevice,
        shape: &crate::Shape,
        dtype: DType,
    ) -> Result<Self> {
        let count = shape.elem_count();
        let size = count * dtype.size_in_bytes();
        let data = vec![0u8; size];
        Self::new(data, dtype, device.clone())
    }

    pub fn from_slice(data: &[u8], dtype: DType, device: &crate::NnapiDevice) -> Result<Self> {
        Self::new(data.to_vec(), dtype, device.clone())
    }

    pub fn from_slice_impl<D: crate::WithDType>(
        device: &crate::NnapiDevice,
        data: &[D],
    ) -> Result<Self> {
        let bytes: Vec<u8> = data.iter().flat_map(|&x| x.to_bytes()).collect();
        Self::new(bytes, D::DTYPE, device.clone())
    }

    pub unsafe fn alloc_uninit_impl(
        device: &crate::NnapiDevice,
        shape: &crate::Shape,
        dtype: DType,
    ) -> Result<Self> {
        let count = shape.elem_count();
        let size = count * dtype.size_in_bytes();
        let data = vec![0u8; size];
        Self::new(data, dtype, device.clone())
    }

    pub fn rand_uniform_impl(
        device: &crate::NnapiDevice,
        shape: &crate::Shape,
        dtype: DType,
        lo: f64,
        up: f64,
    ) -> Result<Self> {
        let count = shape.elem_count();
        let size = count * dtype.size_in_bytes();
        let mut data = vec![0u8; size];
        let count_f32 = count * (dtype.size_in_bytes() / 4);
        let ptr = data.as_mut_ptr() as *mut f32;
        for i in 0..count_f32 {
            unsafe {
                *ptr.add(i) = (lo + (up - lo) * rand::random::<f64>()) as f32;
            }
        }
        Self::new(data, dtype, device.clone())
    }

    pub fn rand_normal_impl(
        device: &crate::NnapiDevice,
        shape: &crate::Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Self> {
        let count = shape.elem_count();
        let size = count * dtype.size_in_bytes();
        let mut data = vec![0u8; size];
        let count_f32 = count * (dtype.size_in_bytes() / 4);
        let ptr = data.as_mut_ptr() as *mut f32;
        for i in 0..count_f32 {
            unsafe {
                *ptr.add(i) = (mean + std * rand::random::<f64>()) as f32;
            }
        }
        Self::new(data, dtype, device.clone())
    }

    pub fn to_cpu_storage_impl(&self) -> Result<crate::CpuStorage> {
        use crate::CpuStorage;
        match self.dtype {
            DType::F32 => {
                let ptr = self.data.as_ptr() as *const f32;
                let len = self.data.len() / 4;
                let values = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };
                Ok(CpuStorage::F32(values))
            }
            DType::F16 => {
                let ptr = self.data.as_ptr() as *const u16;
                let len = self.data.len() / 2;
                let values: Vec<half::f16> =
                    unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };
                Ok(CpuStorage::F16(values))
            }
            DType::BF16 => {
                let ptr = self.data.as_ptr() as *const u16;
                let len = self.data.len() / 2;
                let values: Vec<half::bf16> =
                    unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };
                Ok(CpuStorage::BF16(values))
            }
            _ => Err(crate::Error::Nnapi(NnapiError::Message(format!(
                "to_cpu_storage not supported for dtype {:?}",
                self.dtype
            )))),
        }
    }

    pub fn from_cpu_storage_impl(
        cpu: &crate::CpuStorage,
        device: &crate::NnapiDevice,
    ) -> Result<Self> {
        use crate::CpuStorage;
        match cpu {
            CpuStorage::F32(data) => {
                let bytes: Vec<u8> = data.iter().flat_map(|&x| x.to_le_bytes()).collect();
                Self::new(bytes, DType::F32, device.clone())
            }
            CpuStorage::F16(data) => {
                let bytes: Vec<u8> = data
                    .iter()
                    .flat_map(|&x| x.to_bits().to_le_bytes())
                    .collect();
                Self::new(bytes, DType::F16, device.clone())
            }
            CpuStorage::BF16(data) => {
                let bytes: Vec<u8> = data
                    .iter()
                    .flat_map(|&x| x.to_bits().to_le_bytes())
                    .collect();
                Self::new(bytes, DType::BF16, device.clone())
            }
            _ => Err(crate::Error::Nnapi(NnapiError::Message(
                "from_cpu_storage not supported for this dtype".to_string(),
            ))),
        }
    }

    fn as_f32_slice(&self) -> Result<&[f32]> {
        if self.dtype != DType::F32 {
            return Err(crate::Error::Nnapi(NnapiError::Message(format!(
                "Expected F32 dtype, got {:?}",
                self.dtype
            ))));
        }
        let ptr = self.data.as_ptr() as *const f32;
        let len = self.data.len() / 4;
        Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    fn as_f32_slice_mut(&mut self) -> Result<&mut [f32]> {
        if self.dtype != DType::F32 {
            return Err(crate::Error::Nnapi(NnapiError::Message(format!(
                "Expected F32 dtype, got {:?}",
                self.dtype
            ))));
        }
        let ptr = self.data.as_mut_ptr() as *mut f32;
        let len = self.data.len() / 4;
        Ok(unsafe { std::slice::from_raw_parts_mut(ptr, len) })
    }
}

#[cfg(not(all(feature = "nnapi", target_os = "android")))]
impl NnapiStorage {
    pub fn new(_data: Vec<u8>, _dtype: DType, _device: NnapiDevice) -> Result<Self> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    pub fn with_executor(
        _data: Vec<u8>,
        _dtype: DType,
        _device: NnapiDevice,
        _executor: SharedExecutor,
    ) -> Self {
        Self
    }

    pub fn data(&self) -> &[u8] {
        unreachable!()
    }

    pub fn dtype(&self) -> DType {
        unreachable!()
    }

    pub fn device(&self) -> &NnapiDevice {
        unreachable!()
    }

    pub fn executor(&self) -> &SharedExecutor {
        unreachable!()
    }

    pub fn count(&self) -> usize {
        unreachable!()
    }

    pub fn zeros_impl(
        _device: &crate::NnapiDevice,
        _shape: &crate::Shape,
        _dtype: DType,
    ) -> Result<Self> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    pub fn from_slice(_data: &[u8], _dtype: DType, _device: &crate::NnapiDevice) -> Result<Self> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    pub fn to_cpu_storage_impl(&self) -> Result<crate::CpuStorage> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    pub fn from_cpu_storage_impl(
        _cpu: &crate::CpuStorage,
        _device: &crate::NnapiDevice,
    ) -> Result<Self> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }
}

impl std::fmt::Debug for NnapiStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #[cfg(all(feature = "nnapi", target_os = "android"))]
        {
            f.debug_struct("NnapiStorage")
                .field("count", &self.count())
                .field("dtype", &self.dtype())
                .finish()
        }
        #[cfg(not(all(feature = "nnapi", target_os = "android")))]
        {
            write!(f, "NnapiStorage")
        }
    }
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
impl Drop for NnapiStorage {
    fn drop(&mut self) {
        if let Some(memory) = self.memory {
            unsafe {
                ANeuralNetworksMemory_free(memory);
            }
        }
        if let Some(hwbuffer) = self.hwbuffer {
            unsafe {
                AHardwareBuffer_release(hwbuffer);
            }
        }
    }
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
impl BackendStorage for NnapiStorage {
    type Device = NnapiDevice;

    fn try_clone(&self, _layout: &Layout) -> Result<Self> {
        Ok(NnapiStorage::with_executor(
            self.data.clone(),
            self.dtype,
            self.device.clone(),
            self.executor.clone(),
        ))
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        match self.dtype {
            DType::F32 => {
                let ptr = self.data.as_ptr() as *const f32;
                let len = self.data.len() / 4;
                let values = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };
                Ok(CpuStorage::F32(values))
            }
            DType::F16 => {
                let ptr = self.data.as_ptr() as *const u16;
                let len = self.data.len() / 2;
                let values: Vec<half::f16> =
                    unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };
                Ok(CpuStorage::F16(values))
            }
            DType::BF16 => {
                let ptr = self.data.as_ptr() as *const u16;
                let len = self.data.len() / 2;
                let values: Vec<half::bf16> =
                    unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };
                Ok(CpuStorage::BF16(values))
            }
            _ => Err(crate::Error::Nnapi(NnapiError::Message(format!(
                "to_cpu_storage not supported for dtype {:?}",
                self.dtype
            )))),
        }
    }

    fn affine(&self, _layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let input = self.as_f32_slice()?;
        let mut output = vec![0.0f32; input.len()];
        for (i, &val) in input.iter().enumerate() {
            output[i] = (val as f64 * mul + add) as f32;
        }
        let bytes: Vec<u8> = output.iter().flat_map(|&x| x.to_le_bytes()).collect();
        Ok(NnapiStorage::with_executor(
            bytes,
            DType::F32,
            self.device.clone(),
            self.executor.clone(),
        ))
    }

    fn powf(&self, _layout: &Layout, alpha: f64) -> Result<Self> {
        let input = self.as_f32_slice()?;
        let mut output = vec![0.0f32; input.len()];
        for (i, &val) in input.iter().enumerate() {
            output[i] = val.powf(alpha as f32);
        }
        let bytes: Vec<u8> = output.iter().flat_map(|&x| x.to_le_bytes()).collect();
        Ok(NnapiStorage::with_executor(
            bytes,
            DType::F32,
            self.device.clone(),
            self.executor.clone(),
        ))
    }

    fn elu(&self, _layout: &Layout, alpha: f64) -> Result<Self> {
        let input = self.as_f32_slice()?;
        let mut output = vec![0.0f32; input.len()];
        for (i, &val) in input.iter().enumerate() {
            output[i] = if val > 0.0 {
                val
            } else {
                alpha * (val.exp() - 1.0)
            };
        }
        let bytes: Vec<u8> = output.iter().flat_map(|&x| x.to_le_bytes()).collect();
        Ok(NnapiStorage::with_executor(
            bytes,
            DType::F32,
            self.device.clone(),
            self.executor.clone(),
        ))
    }

    fn reduce_op(
        &self,
        _op: crate::op::ReduceOp,
        _layout: &Layout,
        _dims: &[usize],
    ) -> Result<Self> {
        Err(crate::Error::Nnapi(NnapiError::Message(
            "reduce_op not implemented for nnapi".to_string(),
        )))
    }

    fn cmp(
        &self,
        _op: crate::op::CmpOp,
        _rhs: &Self,
        _lhs_l: &Layout,
        _rhs_l: &Layout,
    ) -> Result<Self> {
        Err(crate::Error::Nnapi(NnapiError::Message(
            "cmp not implemented for nnapi".to_string(),
        )))
    }

    fn to_dtype(&self, _layout: &Layout, _dtype: DType) -> Result<Self> {
        Err(crate::Error::Nnapi(NnapiError::Message(
            "to_dtype not implemented for nnapi".to_string(),
        )))
    }

    fn unary_impl<B: crate::op::UnaryOpT>(&self, _layout: &Layout) -> Result<Self> {
        let op = if B::NAME == "relu" {
            UnaryOp::Relu
        } else if B::NAME == "tanh" {
            UnaryOp::Tanh
        } else if B::NAME == "sigmoid" || B::NAME == "logistic" {
            UnaryOp::Logistic
        } else {
            return Err(crate::Error::Nnapi(NnapiError::Message(format!(
                "unary op {} not implemented for nnapi",
                B::NAME
            ))));
        };

        let input = self.as_f32_slice()?;
        let mut output = vec![0.0f32; input.len()];
        self.executor.execute_unary(input, &mut output, op, 0)?;

        let bytes: Vec<u8> = output.iter().flat_map(|&x| x.to_le_bytes()).collect();
        Ok(NnapiStorage::with_executor(
            bytes,
            DType::F32,
            self.device.clone(),
            self.executor.clone(),
        ))
    }

    fn binary_impl<B: crate::op::BinaryOpT>(
        &self,
        rhs: &Self,
        _lhs_l: &Layout,
        _rhs_l: &Layout,
    ) -> Result<Self> {
        let op = if B::NAME == "add" || B::NAME == "add_broadcast" {
            BinaryOp::Add
        } else if B::NAME == "mul" || B::NAME == "mul_broadcast" {
            BinaryOp::Mul
        } else {
            return Err(crate::Error::Nnapi(NnapiError::Message(format!(
                "binary op {} not implemented for nnapi",
                B::NAME
            ))));
        };

        let lhs = self.as_f32_slice()?;
        let rhs = rhs.as_f32_slice()?;
        let mut output = vec![0.0f32; lhs.len()];
        self.executor.execute_binary(lhs, rhs, &mut output, op, 0)?;

        let bytes: Vec<u8> = output.iter().flat_map(|&x| x.to_le_bytes()).collect();
        Ok(NnapiStorage::with_executor(
            bytes,
            DType::F32,
            self.device.clone(),
            self.executor.clone(),
        ))
    }

    fn where_cond(
        &self,
        _layout: &Layout,
        _t: &Self,
        _layout_t: &Layout,
        _f: &Self,
        _layout_f: &Layout,
    ) -> Result<Self> {
        Err(crate::Error::Nnapi(NnapiError::Message(
            "where_cond not implemented for nnapi".to_string(),
        )))
    }

    fn conv1d(
        &self,
        _layout: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        Err(crate::Error::Nnapi(NnapiError::Message(
            "conv1d not implemented for nnapi".to_string(),
        )))
    }

    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        Err(crate::Error::Nnapi(NnapiError::Message(
            "conv_transpose1d not implemented for nnapi".to_string(),
        )))
    }

    fn conv2d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        if self.dtype != DType::F32 || kernel.dtype() != DType::F32 {
            return Err(crate::Error::Nnapi(NnapiError::Message(
                "conv2d on NNAPI only supports F32 dtype".to_string(),
            )));
        }

        let input_data = self.as_f32_slice()?;
        let kernel_data = kernel.as_f32_slice()?;

        let batch = params.b_size;
        let in_h = params.i_h;
        let in_w = params.i_w;
        let k_h = params.k_h;
        let k_w = params.k_w;
        let out_channels = params.c_out;
        let in_channels = params.c_in;
        let padding = params.padding;
        let stride = params.stride;

        let out_h = params.out_h();
        let out_w = params.out_w();

        let input_dims = [batch, in_h, in_w, in_channels];
        let filter_dims = [k_h, k_w, in_channels, out_channels];
        let output_dims = [batch, out_h, out_w, out_channels];

        let padding_arr: [i32; 4] = [
            padding as i32,
            padding as i32,
            padding as i32,
            padding as i32,
        ];
        let stride_arr: [i32; 2] = [stride as i32, stride as i32];

        let bias = vec![0.0f32; out_channels];

        let mut output = vec![0.0f32; batch * out_channels * out_h * out_w];

        match self.executor.execute_conv2d(
            input_data,
            kernel_data,
            &bias,
            &mut output,
            input_dims,
            filter_dims,
            output_dims,
            padding_arr,
            stride_arr,
            ANEURALNETWORKS_FUSED_NONE as i32,
        ) {
            Ok(()) => {
                let bytes: Vec<u8> = output.iter().flat_map(|&x| x.to_le_bytes()).collect();
                Ok(NnapiStorage::with_executor(
                    bytes,
                    DType::F32,
                    self.device.clone(),
                    self.executor.clone(),
                ))
            }
            Err(e) => Err(e),
        }
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        Err(crate::Error::Nnapi(NnapiError::Message(
            "conv_transpose2d not implemented for nnapi".to_string(),
        )))
    }

    fn index_select(
        &self,
        _ids: &Self,
        _lhs_l: &Layout,
        _rhs_l: &Layout,
        _dim: usize,
    ) -> Result<Self> {
        Err(crate::Error::Nnapi(NnapiError::Message(
            "index_select not implemented for nnapi".to_string(),
        )))
    }

    fn gather(&self, _layout: &Layout, _rhs: &Self, _rhs_l: &Layout, _dim: usize) -> Result<Self> {
        Err(crate::Error::Nnapi(NnapiError::Message(
            "gather not implemented for nnapi".to_string(),
        )))
    }

    fn scatter_set(
        &mut self,
        _layout: &Layout,
        _indexes: &Self,
        _indexes_l: &Layout,
        _source: &Self,
        _source_l: &Layout,
        _dim: usize,
    ) -> Result<()> {
        Err(crate::Error::Nnapi(NnapiError::Message(
            "scatter_set not implemented for nnapi".to_string(),
        )))
    }

    fn scatter_add_set(
        &mut self,
        _layout: &Layout,
        _indexes: &Self,
        _indexes_l: &Layout,
        _source: &Self,
        _source_l: &Layout,
        _dim: usize,
    ) -> Result<()> {
        Err(crate::Error::Nnapi(NnapiError::Message(
            "scatter_add_set not implemented for nnapi".to_string(),
        )))
    }

    fn index_add(
        &self,
        _layout: &Layout,
        _indexes: &Self,
        _indexes_l: &Layout,
        _source: &Self,
        _source_l: &Layout,
        _dim: usize,
    ) -> Result<Self> {
        Err(crate::Error::Nnapi(NnapiError::Message(
            "index_add not implemented for nnapi".to_string(),
        )))
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        if self.dtype != DType::F32 || rhs.dtype != DType::F32 {
            return Err(crate::Error::Nnapi(NnapiError::Message(
                "matmul on NNAPI only supports F32 dtype".to_string(),
            )));
        }

        let (batched, m, n, k) = bmnk;

        if batched != 1 || lhs_l.stride().len() > 2 || rhs_l.stride().len() > 2 {
            return Err(crate::Error::Nnapi(NnapiError::Message(
                "matmul on NNAPI only supports non-batched 2D matmul with contiguous layouts"
                    .to_string(),
            )));
        }

        let lhs_data = self.as_f32_slice()?;
        let rhs_data = rhs.as_f32_slice()?;

        let lhs = unsafe { std::slice::from_raw_parts(lhs_data.as_ptr(), m * k) };
        let rhs = unsafe { std::slice::from_raw_parts(rhs_data.as_ptr(), k * n) };

        let mut output = vec![0.0f32; m * n];

        self.executor.execute_fc(lhs, rhs, &mut output)?;

        let bytes: Vec<u8> = output.iter().flat_map(|&x| x.to_le_bytes()).collect();
        Ok(NnapiStorage::with_executor(
            bytes,
            DType::F32,
            self.device.clone(),
            self.executor.clone(),
        ))
    }

    fn copy_strided_src(&self, _dst: &mut Self, _dst_offset: usize, _src_l: &Layout) -> Result<()> {
        Err(crate::Error::Nnapi(NnapiError::Message(
            "copy_strided_src not implemented for nnapi".to_string(),
        )))
    }

    fn copy2d(
        &self,
        _dst: &mut Self,
        _d1: usize,
        _d2: usize,
        _src_s: usize,
        _dst_s: usize,
        _src_o: usize,
        _dst_o: usize,
    ) -> Result<()> {
        Err(crate::Error::Nnapi(NnapiError::Message(
            "copy2d not implemented for nnapi".to_string(),
        )))
    }

    fn avg_pool2d(
        &self,
        _layout: &Layout,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
    ) -> Result<Self> {
        Err(crate::Error::Nnapi(NnapiError::Message(
            "avg_pool2d not implemented for nnapi".to_string(),
        )))
    }

    fn max_pool2d(
        &self,
        _layout: &Layout,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
    ) -> Result<Self> {
        Err(crate::Error::Nnapi(NnapiError::Message(
            "max_pool2d not implemented for nnapi".to_string(),
        )))
    }

    fn upsample_nearest1d(&self, _layout: &Layout, _sz: usize) -> Result<Self> {
        Err(crate::Error::Nnapi(NnapiError::Message(
            "upsample_nearest1d not implemented for nnapi".to_string(),
        )))
    }

    fn upsample_nearest2d(&self, _layout: &Layout, _h: usize, _w: usize) -> Result<Self> {
        Err(crate::Error::Nnapi(NnapiError::Message(
            "upsample_nearest2d not implemented for nnapi".to_string(),
        )))
    }

    fn upsample_bilinear2d(
        &self,
        _layout: &Layout,
        _h: usize,
        _w: usize,
        _align_corners: bool,
        _scale_h: Option<f64>,
        _scale_w: Option<f64>,
    ) -> Result<Self> {
        Err(crate::Error::Nnapi(NnapiError::Message(
            "upsample_bilinear2d not implemented for nnapi".to_string(),
        )))
    }

    fn const_set(&mut self, _scalar: crate::scalar::Scalar, _layout: &Layout) -> Result<()> {
        Err(crate::Error::Nnapi(NnapiError::Message(
            "const_set not implemented for nnapi".to_string(),
        )))
    }
}

#[cfg(not(all(feature = "nnapi", target_os = "android")))]
impl BackendStorage for NnapiStorage {
    type Device = NnapiDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    fn dtype(&self) -> DType {
        unreachable!()
    }

    fn device(&self) -> &Self::Device {
        unreachable!()
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    fn reduce_op(&self, _: crate::op::ReduceOp, _: &Layout, _: &[usize]) -> Result<Self> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    fn cmp(&self, _: crate::op::CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    fn unary_impl<B: crate::op::UnaryOpT>(&self, _: &Layout) -> Result<Self> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    fn binary_impl<B: crate::op::BinaryOpT>(
        &self,
        _: &Self,
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    fn conv1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    fn conv_transpose1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    fn conv2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    fn conv_transpose2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    fn scatter_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    fn scatter_add_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    fn index_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    fn copy2d(
        &self,
        _: &mut Self,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
    ) -> Result<()> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    fn upsample_bilinear2d(
        &self,
        _: &Layout,
        _: usize,
        _: usize,
        _: bool,
        _: Option<f64>,
        _: Option<f64>,
    ) -> Result<Self> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }

    fn const_set(&mut self, _: crate::scalar::Scalar, _: &Layout) -> Result<()> {
        Err(crate::Error::NotCompiledWithNnapiSupport)
    }
}
