#[cfg(all(feature = "vulkan", target_os = "android"))]
mod device;
#[cfg(all(feature = "vulkan", target_os = "android"))]
mod shaders;

#[cfg(all(feature = "vulkan", target_os = "android"))]
pub use device::VulkanDevice;
#[cfg(all(feature = "vulkan", target_os = "android"))]
pub use shaders::{ADD_SHADER, AFFINE_SHADER, BROADCAST_ADD_SHADER, GEMM_SHADER};

#[derive(thiserror::Error, Debug)]
pub enum VulkanError {
    #[error("{0}")]
    Message(String),
    #[error("wgpu error: {0}")]
    Wgpu(String),
    #[error("buffer mapping error: {0}")]
    Mapping(String),
}

impl From<String> for VulkanError {
    fn from(e: String) -> Self {
        VulkanError::Message(e)
    }
}

impl From<wgpu::Error> for VulkanError {
    fn from(e: wgpu::Error) -> Self {
        VulkanError::Wgpu(e.to_string())
    }
}

impl std::fmt::Debug for VulkanStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanStorage")
            .field("count", &self.count)
            .field("dtype", &self.dtype)
            .finish()
    }
}

use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};

#[cfg(all(feature = "vulkan", target_os = "android"))]
use shaders;

#[cfg(all(feature = "vulkan", target_os = "android"))]
pub struct VulkanStorage {
    buffer: wgpu::Buffer,
    count: usize,
    dtype: DType,
    device: VulkanDevice,
}

#[cfg(not(all(feature = "vulkan", target_os = "android")))]
pub struct VulkanStorage;

#[cfg(all(feature = "vulkan", target_os = "android"))]
impl VulkanStorage {
    pub fn new(buffer: wgpu::Buffer, count: usize, dtype: DType, device: VulkanDevice) -> Self {
        Self {
            buffer,
            count,
            dtype,
            device,
        }
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    fn alloc_result_storage(&self, elem_count: usize) -> Result<VulkanStorage> {
        let size = elem_count * self.dtype.size_in_bytes();
        let buffer = self
            .device
            .allocate_buffer(size as u64, wgpu::BufferUsages::STORAGE)?;
        Ok(VulkanStorage::new(
            buffer,
            elem_count,
            self.dtype,
            self.device.clone(),
        ))
    }

    fn bytes_to_f32(&self, bytes: &[u8], layout: &Layout) -> Result<Vec<f32>> {
        let shape = layout.shape();
        let elem_count = shape.elem_count();
        match self.dtype {
            DType::F32 => {
                let ptr = bytes.as_ptr() as *const f32;
                Ok(unsafe { std::slice::from_raw_parts(ptr, elem_count).to_vec() })
            }
            DType::F16 => {
                let ptr = bytes.as_ptr() as *const u16;
                let halfs: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, elem_count) }
                    .iter()
                    .map(|&h| half::f16::from_bits(h).to_f32())
                    .collect();
                Ok(halfs)
            }
            _ => Err(Error::Vulkan(VulkanError::Message(format!(
                "bytes_to_f32 not supported for dtype {:?}",
                self.dtype
            )))),
        }
    }

    fn bytes_to_u32(&self, bytes: &[u8], layout: &Layout) -> Result<Vec<u32>> {
        let shape = layout.shape();
        let elem_count = shape.elem_count();
        match self.dtype {
            DType::U32 => {
                let ptr = bytes.as_ptr() as *const u32;
                Ok(unsafe { std::slice::from_raw_parts(ptr, elem_count).to_vec() })
            }
            DType::I64 => {
                let ptr = bytes.as_ptr() as *const i64;
                Ok(unsafe { std::slice::from_raw_parts(ptr, elem_count) }
                    .iter()
                    .map(|&x| x as u32)
                    .collect())
            }
            _ => Err(Error::Vulkan(VulkanError::Message(format!(
                "bytes_to_u32 not supported for dtype {:?}",
                self.dtype
            )))),
        }
    }

    fn write_from_f32(&mut self, data: &[f32], offset: usize) -> Result<()> {
        let elem_size = self.dtype.size_in_bytes();
        let offset_bytes = offset * elem_size;
        if data.len() * elem_size + offset_bytes > self.count * elem_size {
            crate::bail!("data too large for wgpu storage");
        }

        match self.dtype {
            DType::F32 => {
                let bytes: Vec<u8> = data.iter().flat_map(|&x| x.to_le_bytes()).collect();
                self.device
                    .write_buffer(&self.buffer, offset_bytes as u64, &bytes)?;
            }
            DType::F16 => {
                let halfs: Vec<u16> = data
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();
                let bytes: Vec<u8> = halfs.iter().flat_map(|&h| h.to_le_bytes()).collect();
                self.device
                    .write_buffer(&self.buffer, offset_bytes as u64, &bytes)?;
            }
            _ => {
                return Err(Error::Vulkan(VulkanError::Message(format!(
                    "write_from_f32 not supported for dtype {:?}",
                    self.dtype
                ))))
            }
        }
        Ok(())
    }

    fn read_to_bytes(&self) -> Result<Vec<u8>> {
        self.device.sync()?;
        let size = self.count * self.dtype.size_in_bytes();
        let slice = self.buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        rx.recv().unwrap().map_err(|e| {
            Error::Vulkan(VulkanError::Mapping(format!("failed to map buffer: {}", e)))
        })?;
        let data = slice.get_mapped_range().to_vec();
        Ok(data)
    }
}

#[cfg(not(all(feature = "vulkan", target_os = "android")))]
impl VulkanStorage {
    pub fn new(_buffer: (), _count: usize, _dtype: DType, _device: VulkanDevice) -> Self {
        unreachable!()
    }
}

#[cfg(all(feature = "vulkan", target_os = "android"))]
impl BackendStorage for VulkanStorage {
    type Device = VulkanDevice;

    fn try_clone(&self, _layout: &Layout) -> Result<Self> {
        Err(Error::Vulkan(VulkanError::Message(
            "try_clone not implemented for wgpu".to_string(),
        )))
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        let size = self.count * self.dtype.size_in_bytes();
        let data = self.read_to_bytes()?;
        match self.dtype {
            DType::U8 => Ok(CpuStorage::U8(data)),
            DType::U32 => {
                let values: Vec<u32> = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u32, size / 4).to_vec()
                };
                Ok(CpuStorage::U32(values))
            }
            DType::I64 => {
                let values: Vec<i64> = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const i64, size / 8).to_vec()
                };
                Ok(CpuStorage::I64(values))
            }
            DType::F16 => {
                let values: Vec<half::f16> = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const half::f16, size / 2).to_vec()
                };
                Ok(CpuStorage::F16(values))
            }
            DType::BF16 => {
                let values: Vec<half::bf16> = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const half::bf16, size / 2)
                        .to_vec()
                };
                Ok(CpuStorage::BF16(values))
            }
            DType::F32 => {
                let values: Vec<f32> = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const f32, size / 4).to_vec()
                };
                Ok(CpuStorage::F32(values))
            }
            DType::F64 => {
                let values: Vec<f64> = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const f64, size / 8).to_vec()
                };
                Ok(CpuStorage::F64(values))
            }
            DType::I32 => {
                let values: Vec<i32> = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const i32, size / 4).to_vec()
                };
                Ok(CpuStorage::I32(values))
            }
            DType::I16 => {
                let values: Vec<i16> = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const i16, size / 2).to_vec()
                };
                Ok(CpuStorage::I16(values))
            }
            DType::F8E4M3 | DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                Err(Error::UnsupportedDTypeForOp(self.dtype, "to_cpu_storage"))
            }
        }
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let shape = layout.shape();
        let count = shape.elem_count();

        let mut output = self.device.allocate_buffer(
            (count * self.dtype.size_in_bytes()) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )?;

        self.device.dispatch_affine(
            shaders::AFFINE_SHADER,
            &self.buffer,
            &output,
            count as u32,
            mul as f32,
            add as f32,
        )?;

        Ok(VulkanStorage::new(
            output,
            count,
            self.dtype,
            self.device.clone(),
        ))
    }

    fn powf(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        let bytes = self.read_to_bytes()?;
        let shape = layout.shape();
        let values = self.bytes_to_f32(&bytes, layout)?;
        let result: Vec<f32> = values.iter().map(|&x| x.powf(alpha as f32)).collect();

        let mut storage = self.device.zeros_impl(&shape, self.dtype)?;
        let storage_mut = &mut storage;
        storage_mut.write_from_f32(&result, 0)?;
        Ok(storage)
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        let bytes = self.read_to_bytes()?;
        let shape = layout.shape();
        let values = self.bytes_to_f32(&bytes, layout)?;
        let result: Vec<f32> = values
            .iter()
            .map(|&x| {
                if x < 0.0 {
                    alpha as f32 * (x.exp() - 1.0)
                } else {
                    x
                }
            })
            .collect();

        let mut storage = self.device.zeros_impl(&shape, self.dtype)?;
        let storage_mut = &mut storage;
        storage_mut.write_from_f32(&result, 0)?;
        Ok(storage)
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, _dims: &[usize]) -> Result<Self> {
        let bytes = self.read_to_bytes()?;
        let shape = layout.shape();
        let values = self.bytes_to_f32(&bytes, layout)?;

        let _elems: usize = shape.dims().iter().product();
        let result = match op {
            ReduceOp::Sum => values.iter().sum::<f32>(),
            ReduceOp::Max => values.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            ReduceOp::Min => values.iter().cloned().fold(f32::INFINITY, f32::min),
            _ => {
                return Err(Error::Vulkan(VulkanError::Message(format!(
                    "reduce op {:?} not implemented for wgpu",
                    op
                ))))
            }
        };

        let mut storage = self.device.zeros_impl(&crate::Shape::from(1), self.dtype)?;
        let storage_mut = &mut storage;
        storage_mut.write_from_f32(&[result], 0)?;
        Ok(storage)
    }

    fn cmp(&self, _op: CmpOp, _rhs: &Self, _lhs_l: &Layout, _rhs_l: &Layout) -> Result<Self> {
        Err(Error::Vulkan(VulkanError::Message(
            "cmp not implemented for wgpu".to_string(),
        )))
    }

    fn to_dtype(&self, _layout: &Layout, _dtype: DType) -> Result<Self> {
        Err(Error::Vulkan(VulkanError::Message(
            "to_dtype not implemented for wgpu".to_string(),
        )))
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let bytes = self.read_to_bytes()?;
        let shape = layout.shape();
        let values = self.bytes_to_f32(&bytes, layout)?;

        let result: Vec<f32> = match B::KERNEL {
            "usilu" => values.iter().map(|&x| x / (1.0 + (-x).exp())).collect(),
            "uexp" => values.iter().map(|&x| x.exp()).collect(),
            "usqrt" => values.iter().map(|&x| x.sqrt()).collect(),
            "usqr" => values.iter().map(|&x| x * x).collect(),
            "urecip" => values.iter().map(|&x| 1.0 / x).collect(),
            "uneg" => values.iter().map(|&x| -x).collect(),
            "uabs" => values.iter().map(|&x| x.abs()).collect(),
            "urelu" => values.iter().map(|&x| x.max(0.0)).collect(),
            _ => {
                return Err(Error::Vulkan(VulkanError::Message(format!(
                    "unary op {} not implemented for wgpu",
                    B::KERNEL
                ))))
            }
        };

        let mut storage = self.device.zeros_impl(&shape, self.dtype)?;
        let storage_mut = &mut storage;
        storage_mut.write_from_f32(&result, 0)?;
        Ok(storage)
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let shape = lhs_l.shape();
        let count = shape.elem_count();

        match B::KERNEL {
            "add" => {
                let mut output = self.device.allocate_buffer(
                    (count * self.dtype.size_in_bytes()) as u64,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                )?;

                self.device.dispatch_binary_op(
                    shaders::ADD_SHADER,
                    &self.buffer,
                    &rhs.buffer,
                    &output,
                    count as u32,
                )?;

                return Ok(VulkanStorage::new(
                    output,
                    count,
                    self.dtype,
                    self.device.clone(),
                ));
            }
            _ => {}
        }

        let lhs_bytes = self.read_to_bytes()?;
        let rhs_bytes = rhs.read_to_bytes()?;
        let lhs_values = self.bytes_to_f32(&lhs_bytes, lhs_l)?;
        let rhs_values = self.bytes_to_f32(&rhs_bytes, rhs_l)?;

        let result: Vec<f32> = match B::KERNEL {
            "add" => lhs_values
                .iter()
                .zip(rhs_values.iter())
                .map(|(&a, &b)| a + b)
                .collect(),
            "mul" => lhs_values
                .iter()
                .zip(rhs_values.iter())
                .map(|(&a, &b)| a * b)
                .collect(),
            "div" => lhs_values
                .iter()
                .zip(rhs_values.iter())
                .map(|(&a, &b)| a / b)
                .collect(),
            _ => {
                return Err(Error::Vulkan(VulkanError::Message(format!(
                    "binary op {} not implemented for wgpu",
                    B::KERNEL
                ))))
            }
        };

        let mut storage = self.device.zeros_impl(&shape, self.dtype)?;
        let storage_mut = &mut storage;
        storage_mut.write_from_f32(&result, 0)?;
        Ok(storage)
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        layout_t: &Layout,
        f: &Self,
        layout_f: &Layout,
    ) -> Result<Self> {
        let cond_bytes = self.read_to_bytes()?;
        let t_bytes = t.read_to_bytes()?;
        let f_bytes = f.read_to_bytes()?;
        let cond_values = self.bytes_to_f32(&cond_bytes, layout)?;
        let t_values = t.bytes_to_f32(&t_bytes, layout_t)?;
        let f_values = f.bytes_to_f32(&f_bytes, layout_f)?;

        let result: Vec<f32> = cond_values
            .iter()
            .zip(t_values.iter().zip(f_values.iter()))
            .map(|(&c, (&tv, &fv))| if c != 0.0 { tv } else { fv })
            .collect();

        let shape = layout.shape();
        let mut storage = self.device.zeros_impl(&shape, self.dtype)?;
        let storage_mut = &mut storage;
        storage_mut.write_from_f32(&result, 0)?;
        Ok(storage)
    }

    fn conv1d(
        &self,
        _layout: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        Err(Error::Vulkan(VulkanError::Message(
            "conv1d not implemented for wgpu".to_string(),
        )))
    }

    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        Err(Error::Vulkan(VulkanError::Message(
            "conv_transpose1d not implemented for wgpu".to_string(),
        )))
    }

    fn conv2d(
        &self,
        _layout: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        Err(Error::Vulkan(VulkanError::Message(
            "conv2d not implemented for wgpu".to_string(),
        )))
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        Err(Error::Vulkan(VulkanError::Message(
            "conv_transpose2d not implemented for wgpu".to_string(),
        )))
    }

    fn index_select(&self, ids: &Self, lhs_l: &Layout, rhs_l: &Layout, dim: usize) -> Result<Self> {
        let self_bytes = self.read_to_bytes()?;
        let ids_bytes = ids.read_to_bytes()?;

        let self_f32 = self.bytes_to_f32(&self_bytes, lhs_l)?;
        let ids_u32 = ids.bytes_to_u32(&ids_bytes, rhs_l)?;

        let lhs_dims = lhs_l.shape().dims();
        let left_size: usize = lhs_dims[..dim].iter().product();
        let right_size: usize = lhs_dims[dim + 1..].iter().product();
        let ids_el = rhs_l.shape().elem_count();
        let dst_el = ids_el * left_size * right_size;

        let mut result = vec![0.0f32; dst_el];

        for (i, &id) in ids_u32.iter().enumerate() {
            let id = id as usize;
            for j in 0..left_size {
                for k in 0..right_size {
                    let src_idx = j + id * right_size + k;
                    let dst_idx = i * left_size * right_size + j * right_size + k;
                    result[dst_idx] = self_f32[src_idx];
                }
            }
        }

        let _dst_shape =
            Shape::from_dims(&[&lhs_dims[..dim], &[ids_el], &lhs_dims[dim + 1..]].concat());
        self.device.storage_from_slice(&result)
    }

    fn gather(&self, layout: &Layout, rhs: &Self, rhs_l: &Layout, dim: usize) -> Result<Self> {
        let self_bytes = self.read_to_bytes()?;
        let indices_bytes = rhs.read_to_bytes()?;

        let self_f32 = self.bytes_to_f32(&self_bytes, layout)?;
        let indices_u32 = rhs.bytes_to_u32(&indices_bytes, rhs_l)?;

        let self_dims = layout.shape().dims();
        let indices_count = rhs_l.shape().elem_count();

        let mut result = Vec::with_capacity(indices_count);
        for &idx in &indices_u32 {
            let idx = idx as usize;
            if idx >= self_dims[dim] {
                crate::bail!(
                    "gather: index {} out of bounds for dim {} with size {}",
                    idx,
                    dim,
                    self_dims[dim]
                );
            }
            let offset = idx * self_dims[dim + 1..].iter().product::<usize>();
            for i in 0..self_dims[dim + 1..].iter().product::<usize>() {
                result.push(self_f32[offset + i]);
            }
        }

        let _dst_shape = Shape::from_dims(
            &[&self_dims[..dim], &[indices_count], &self_dims[dim + 1..]].concat(),
        );
        self.device.storage_from_slice(&result)
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
        Err(Error::Vulkan(VulkanError::Message(
            "scatter_set not implemented for wgpu".to_string(),
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
        Err(Error::Vulkan(VulkanError::Message(
            "scatter_add_set not implemented for wgpu".to_string(),
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
        Err(Error::Vulkan(VulkanError::Message(
            "index_add not implemented for wgpu".to_string(),
        )))
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let (b, m, n, k) = bmnk;
        let lhs_bytes = self.read_to_bytes()?;
        let rhs_bytes = rhs.read_to_bytes()?;

        let lhs_f32 = self.bytes_to_f32(&lhs_bytes, lhs_l)?;
        let rhs_f32 = rhs.bytes_to_f32(&rhs_bytes, rhs_l)?;

        let mut result = vec![0.0f32; b * m * n];

        for bi in 0..b {
            for mi in 0..m {
                for ni in 0..n {
                    let mut sum = 0.0f32;
                    for ki in 0..k {
                        let lhs_idx = bi * m * k + mi * k + ki;
                        let rhs_idx = bi * k * n + ki * n + ni;
                        sum += lhs_f32[lhs_idx] * rhs_f32[rhs_idx];
                    }
                    result[bi * m * n + mi * n + ni] = sum;
                }
            }
        }

        self.device.storage_from_slice(&result)
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let elem_size = self.dtype.size_in_bytes();
        let src_bytes = self.read_to_bytes()?;
        let dst_count = dst.count;

        match src_l.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => {
                let to_copy = (dst_count - dst_offset).min(len);
                let src_start = start_offset * elem_size;
                let src_end = src_start + to_copy * elem_size;
                let _dst_start = dst_offset * elem_size;
                let data = &src_bytes[src_start..src_end];
                dst.device
                    .write_buffer(&dst.buffer, (dst_offset * elem_size) as u64, data)?;
            }
            crate::StridedBlocks::MultipleBlocks {
                block_start_index,
                block_len,
            } => {
                if block_len == 1 {
                    let mut dst_idx = dst_offset;
                    for src_idx in block_start_index {
                        if dst_idx >= dst_count {
                            break;
                        }
                        let src_byte_offset = src_idx * elem_size;
                        let src_end = src_byte_offset + elem_size;
                        dst.device.write_buffer(
                            &dst.buffer,
                            (dst_idx * elem_size) as u64,
                            &src_bytes[src_byte_offset..src_end],
                        )?;
                        dst_idx += 1;
                    }
                } else {
                    return Err(Error::Vulkan(VulkanError::Message(
                        "copy_strided_src with multiple blocks of len > 1 not implemented"
                            .to_string(),
                    )));
                }
            }
        }
        Ok(())
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()> {
        let elem_size = self.dtype.size_in_bytes();
        let src_bytes = self.read_to_bytes()?;
        let dst_count = dst.count;

        let total_elements = d1 * d2;
        if dst_o + total_elements > dst_count {
            crate::bail!("copy2d: insufficient space in destination");
        }

        let mut dst_idx = dst_o;
        let mut src_idx = src_o;
        for _ in 0..d1 {
            for _ in 0..d2 {
                let src_byte_offset = src_idx * elem_size;
                let src_end = src_byte_offset + elem_size;
                dst.device.write_buffer(
                    &dst.buffer,
                    (dst_idx * elem_size) as u64,
                    &src_bytes[src_byte_offset..src_end],
                )?;
                src_idx += 1;
                dst_idx += 1;
            }
            src_idx += src_s - d2;
            dst_idx += dst_s - d2;
        }
        Ok(())
    }

    fn avg_pool2d(
        &self,
        _layout: &Layout,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
    ) -> Result<Self> {
        Err(Error::Vulkan(VulkanError::Message(
            "avg_pool2d not implemented for wgpu".to_string(),
        )))
    }

    fn max_pool2d(
        &self,
        _layout: &Layout,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
    ) -> Result<Self> {
        Err(Error::Vulkan(VulkanError::Message(
            "max_pool2d not implemented for wgpu".to_string(),
        )))
    }

    fn upsample_nearest1d(&self, _layout: &Layout, _sz: usize) -> Result<Self> {
        Err(Error::Vulkan(VulkanError::Message(
            "upsample_nearest1d not implemented for wgpu".to_string(),
        )))
    }

    fn upsample_nearest2d(&self, _layout: &Layout, _h: usize, _w: usize) -> Result<Self> {
        Err(Error::Vulkan(VulkanError::Message(
            "upsample_nearest2d not implemented for wgpu".to_string(),
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
        Err(Error::Vulkan(VulkanError::Message(
            "upsample_bilinear2d not implemented for wgpu".to_string(),
        )))
    }

    fn const_set(&mut self, _scalar: crate::scalar::Scalar, _layout: &Layout) -> Result<()> {
        Err(Error::Vulkan(VulkanError::Message(
            "const_set not implemented for wgpu".to_string(),
        )))
    }
}

#[cfg(not(all(feature = "vulkan", target_os = "android")))]
impl BackendStorage for VulkanStorage {
    type Device = VulkanDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn dtype(&self) -> DType {
        Error::NotCompiledWithVulkanSupport
    }

    fn device(&self) -> &Self::Device {
        Error::NotCompiledWithVulkanSupport
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn reduce_op(&self, _: ReduceOp, _: &Layout, _: &[usize]) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn conv1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn conv_transpose1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn conv2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn conv_transpose2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
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
        Err(Error::NotCompiledWithVulkanSupport)
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
        Err(Error::NotCompiledWithVulkanSupport)
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
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()> {
        Err(Error::NotCompiledWithVulkanSupport)
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
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithVulkanSupport)
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
        Err(Error::NotCompiledWithVulkanSupport)
    }

    fn const_set(&mut self, _: crate::scalar::Scalar, _: &Layout) -> Result<()> {
        Err(Error::NotCompiledWithVulkanSupport)
    }
}
