mod device;

pub use device::VulkanDevice;

#[derive(thiserror::Error, Debug)]
pub enum VulkanError {
    #[error("{0}")]
    Message(String),
}

impl From<String> for VulkanError {
    fn from(e: String) -> Self {
        VulkanError::Message(e)
    }
}

use crate::backend::BackendStorage;
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape, StridedBlocks};

#[cfg(all(feature = "vulkan", target_os = "android"))]
use gpu_allocator::vulkan::Allocation;

#[cfg(all(feature = "vulkan", target_os = "android"))]
use ash::vk;

#[cfg(all(feature = "vulkan", target_os = "android"))]
pub struct VulkanStorage {
    buffer: vk::Buffer,
    allocation: Allocation,
    count: usize,
    dtype: DType,
    device: VulkanDevice,
}

#[cfg(not(all(feature = "vulkan", target_os = "android")))]
pub struct VulkanStorage;

impl std::fmt::Debug for VulkanStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanStorage").finish()
    }
}

#[cfg(all(feature = "vulkan", target_os = "android"))]
impl VulkanStorage {
    pub(crate) fn new(
        buffer: vk::Buffer,
        allocation: Allocation,
        count: usize,
        dtype: DType,
        device: VulkanDevice,
    ) -> Self {
        Self {
            buffer,
            allocation,
            count,
            dtype,
            device,
        }
    }

    pub(crate) fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub(crate) fn allocation(&self) -> &Allocation {
        &self.allocation
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

    fn alloc_result_storage(&self, elem_count: usize) -> Result<VulkanStorage> {
        let size = elem_count * self.dtype.size_in_bytes();
        let (buffer, allocation) = self
            .device
            .allocate_buffer(size as u64, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        Ok(VulkanStorage::new(
            buffer,
            allocation,
            elem_count,
            self.dtype,
            self.device.clone(),
        ))
    }

    fn copy_to_cpu_impl(&self) -> Result<CpuStorage> {
        let size = self.count * self.dtype.size_in_bytes();
        let mut data = vec![0u8; size];
        self.device
            .download_and_free(self.buffer, self.allocation, &mut data)?;
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
                Err(Error::UnsupportedDTypeForOp(self.dtype, "copy_to_cpu_impl"))
            }
        }
    }

    fn read_to_bytes(&self) -> Result<Vec<u8>> {
        let size = self.count * self.dtype.size_in_bytes();
        let mut data = vec![0u8; size];
        unsafe {
            let mapped = self.allocation.mapped_ptr().unwrap().as_ptr();
            std::ptr::copy_nonoverlapping(mapped, data.as_mut_ptr(), size);
        }
        Ok(data)
    }

    fn write_from_bytes(&mut self, data: &[u8], offset: usize) -> Result<()> {
        let offset_bytes = offset * self.dtype.size_in_bytes();
        if data.len() + offset_bytes > self.count * self.dtype.size_in_bytes() {
            crate::bail!("data too large for vulkan storage");
        }
        unsafe {
            let mapped = self.allocation.mapped_ptr().unwrap().as_ptr();
            std::ptr::copy_nonoverlapping(data.as_ptr(), mapped.add(offset_bytes), data.len());
        }
        Ok(())
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
                "matmul not supported for dtype {:?}",
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
            crate::bail!("data too large for vulkan storage");
        }

        match self.dtype {
            DType::F32 => {
                let ptr = data.as_ptr() as *const u8;
                unsafe {
                    let mapped = self.allocation.mapped_ptr().unwrap().as_ptr();
                    std::ptr::copy_nonoverlapping(ptr, mapped.add(offset_bytes), data.len() * 4);
                }
            }
            DType::F16 => {
                let halfs: Vec<u16> = data
                    .iter()
                    .map(|&x| half::f16::from_f32(x).to_bits())
                    .collect();
                let ptr = halfs.as_ptr() as *const u8;
                unsafe {
                    let mapped = self.allocation.mapped_ptr().unwrap().as_ptr();
                    std::ptr::copy_nonoverlapping(ptr, mapped.add(offset_bytes), halfs.len() * 2);
                }
            }
            _ => Err(Error::Vulkan(VulkanError::Message(format!(
                "write_from_f32 not supported for dtype {:?}",
                self.dtype
            )))),
        }
        Ok(())
    }
}

#[cfg(not(all(feature = "vulkan", target_os = "android")))]
impl VulkanStorage {
    pub fn new(
        _buffer: (),
        _allocation: (),
        _count: usize,
        _dtype: DType,
        _device: VulkanDevice,
    ) -> Self {
        unreachable!()
    }
}

#[cfg(all(feature = "vulkan", target_os = "android"))]
impl BackendStorage for VulkanStorage {
    type Device = VulkanDevice;

    fn try_clone(&self, _layout: &Layout) -> Result<Self> {
        Err(Error::Vulkan(VulkanError::Message(
            "try_clone not implemented for vulkan".to_string(),
        )))
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        self.copy_to_cpu_impl()
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let bytes = self.read_to_bytes()?;
        let shape = layout.shape();
        let elem_count = shape.elem_count();

        let values = self.bytes_to_f32(&bytes, layout)?;
        let result: Vec<f32> = values
            .iter()
            .map(|&x| x * mul as f32 + add as f32)
            .collect();

        let mut storage = self.device.zeros_impl(&shape, self.dtype)?;
        let storage_mut = &mut storage;
        storage_mut.write_from_f32(&result, 0)?;
        Ok(storage)
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
                    alpha as f32 * (exp(x) - 1.0)
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

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, dims: &[usize]) -> Result<Self> {
        let bytes = self.read_to_bytes()?;
        let shape = layout.shape();
        let values = self.bytes_to_f32(&bytes, layout)?;

        let elems: usize = shape.dims().iter().product();
        let mut result = match op {
            ReduceOp::Sum => values.iter().sum::<f32>(),
            ReduceOp::Max => values.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            ReduceOp::Min => values.iter().cloned().fold(f32::INFINITY, f32::min),
            _ => {
                return Err(Error::Vulkan(VulkanError::Message(format!(
                    "reduce op {:?} not implemented for vulkan",
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
            "cmp not implemented for vulkan".to_string(),
        )))
    }

    fn to_dtype(&self, _layout: &Layout, _dtype: DType) -> Result<Self> {
        Err(Error::Vulkan(VulkanError::Message(
            "to_dtype not implemented for vulkan".to_string(),
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
                    "unary op {} not implemented for vulkan",
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
        let lhs_bytes = self.read_to_bytes()?;
        let rhs_bytes = rhs.read_to_bytes()?;
        let lhs_values = self.bytes_to_f32(&lhs_bytes, lhs_l)?;
        let rhs_values = rhs.bytes_to_f32(&rhs_bytes, rhs_l)?;

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
                    "binary op {} not implemented for vulkan",
                    B::KERNEL
                ))))
            }
        };

        let shape = lhs_l.shape();
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
            "conv1d not implemented for vulkan".to_string(),
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
            "conv_transpose1d not implemented for vulkan".to_string(),
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
            "conv2d not implemented for vulkan".to_string(),
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
            "conv_transpose2d not implemented for vulkan".to_string(),
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
            for j in 0..left_size {
                for k in 0..right_size {
                    let src_idx = j + id * right_size + k;
                    let dst_idx = i * left_size * right_size + j * right_size + k;
                    result[dst_idx] = self_f32[src_idx];
                }
            }
        }

        let dst_shape =
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

        let dst_shape = Shape::from_dims(
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
            "scatter_set not implemented for vulkan".to_string(),
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
            "scatter_add_set not implemented for vulkan".to_string(),
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
            "index_add not implemented for vulkan".to_string(),
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
            StridedBlocks::SingleBlock { start_offset, len } => {
                let to_copy = (dst_count - dst_offset).min(len);
                let src_start = start_offset * elem_size;
                let src_end = src_start + to_copy * elem_size;
                let dst_start = dst_offset * elem_size;
                dst.write_from_bytes(&src_bytes[src_start..src_end], dst_offset)?;
            }
            StridedBlocks::MultipleBlocks {
                block_start_index,
                block_len,
            } => {
                if block_len == 1 {
                    let mut dst_idx = dst_offset;
                    for &src_idx in block_start_index {
                        if dst_idx >= dst_count {
                            break;
                        }
                        let src_byte_offset = src_idx * elem_size;
                        let dst_byte_offset = dst_idx * elem_size;
                        dst.write_from_bytes(
                            &src_bytes[src_byte_offset..src_byte_offset + elem_size],
                            dst_idx,
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
                dst.write_from_bytes(
                    &src_bytes[src_byte_offset..src_byte_offset + elem_size],
                    dst_idx,
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
            "avg_pool2d not implemented for vulkan".to_string(),
        )))
    }

    fn max_pool2d(
        &self,
        _layout: &Layout,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
    ) -> Result<Self> {
        Err(Error::Vulkan(VulkanError::Message(
            "max_pool2d not implemented for vulkan".to_string(),
        )))
    }

    fn upsample_nearest1d(&self, _layout: &Layout, _sz: usize) -> Result<Self> {
        Err(Error::Vulkan(VulkanError::Message(
            "upsample_nearest1d not implemented for vulkan".to_string(),
        )))
    }

    fn upsample_nearest2d(&self, _layout: &Layout, _h: usize, _w: usize) -> Result<Self> {
        Err(Error::Vulkan(VulkanError::Message(
            "upsample_nearest2d not implemented for vulkan".to_string(),
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
            "upsample_bilinear2d not implemented for vulkan".to_string(),
        )))
    }

    fn const_set(&mut self, _scalar: crate::scalar::Scalar, _layout: &Layout) -> Result<()> {
        Err(Error::Vulkan(VulkanError::Message(
            "const_set not implemented for vulkan".to_string(),
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

#[cfg(target_os = "android")]
mod tests {
    use super::*;

    #[test]
    fn test_vulkan_device_creation() {
        let device = VulkanDevice::new(0);
        assert!(
            device.is_ok(),
            "VulkanDevice::new should succeed on Android with Vulkan"
        );
    }

    #[test]
    fn test_vulkan_device_location() {
        let device = VulkanDevice::new(0).unwrap();
        let location = device.location();
        match location {
            crate::DeviceLocation::Vulkan { gpu_id } => {
                assert_eq!(gpu_id, 0, "Vulkan device location should have gpu_id 0");
            }
            _ => panic!("Expected Vulkan device location"),
        }
    }

    #[test]
    fn test_vulkan_device_same_device() {
        let device1 = VulkanDevice::new(0).unwrap();
        let device2 = VulkanDevice::new(0).unwrap();
        assert!(
            device1.same_device(&device2),
            "Same ordinal Vulkan devices should be same device"
        );
    }

    #[test]
    fn test_vulkan_storage_alloc_and_read() {
        let device = VulkanDevice::new(0).unwrap();
        let storage = device
            .zeros_impl(&crate::Shape::from((2, 3)), crate::DType::F32)
            .unwrap();
        assert_eq!(
            storage.count(),
            6,
            "Storage should have 6 elements for shape (2,3)"
        );
        assert_eq!(
            storage.dtype(),
            crate::DType::F32,
            "Storage dtype should be F32"
        );
    }

    #[test]
    fn test_vulkan_storage_from_slice() {
        let device = VulkanDevice::new(0).unwrap();
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let storage = device.storage_from_slice(&data).unwrap();
        assert_eq!(storage.count(), 4, "Storage should have 4 elements");
        assert_eq!(
            storage.dtype(),
            crate::DType::F32,
            "Storage dtype should be F32"
        );
    }

    #[test]
    fn test_vulkan_storage_roundtrip() {
        let device = VulkanDevice::new(0).unwrap();
        let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let storage = device.storage_from_slice(&original).unwrap();

        let cpu_storage = storage.to_cpu_storage().unwrap();
        match cpu_storage {
            crate::CpuStorage::F32(data) => {
                assert_eq!(data.len(), original.len());
                for (a, b) in data.iter().zip(original.iter()) {
                    assert!((a - b).abs() < 1e-6, "Roundtrip should preserve values");
                }
            }
            _ => panic!("Expected F32 cpu storage"),
        }
    }

    #[test]
    fn test_vulkan_matmul_f32_matches_cpu() {
        let device = VulkanDevice::new(0).unwrap();
        let lhs_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let rhs_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let lhs = device.storage_from_slice(&lhs_data).unwrap();
        let rhs = device.storage_from_slice(&rhs_data).unwrap();

        let lhs_l = Layout::contiguous((2, 2));
        let rhs_l = Layout::contiguous((2, 2));
        let result = lhs.matmul(&rhs, (1, 2, 2, 2), &lhs_l, &rhs_l).unwrap();
        let cpu = result.to_cpu_storage().unwrap();

        let expected = vec![7.0, 10.0, 15.0, 22.0];
        match cpu {
            crate::CpuStorage::F32(data) => {
                assert_eq!(data.len(), expected.len());
                for (a, &b) in data.iter().zip(&expected) {
                    assert!(
                        (a - b).abs() < 1e-3,
                        "Matmul mismatch: got {}, expected {}",
                        a,
                        b
                    );
                }
            }
            _ => panic!("Expected F32 cpu storage"),
        }
    }

    #[test]
    fn test_vulkan_affine_f32() {
        let device = VulkanDevice::new(0).unwrap();
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let storage = device.storage_from_slice(&data).unwrap();
        let layout = Layout::contiguous((1, 4));

        let result = storage.affine(&layout, 2.0, 1.0).unwrap();
        let cpu = result.to_cpu_storage().unwrap();

        let expected = vec![3.0, 5.0, 7.0, 9.0];
        match cpu {
            crate::CpuStorage::F32(out) => {
                assert_eq!(out.len(), expected.len());
                for (a, &b) in out.iter().zip(&expected) {
                    assert!(
                        (a - b).abs() < 1e-3,
                        "Affine mismatch: got {}, expected {}",
                        a,
                        b
                    );
                }
            }
            _ => panic!("Expected F32 cpu storage"),
        }
    }

    #[test]
    fn test_vulkan_binary_add_f32() {
        use crate::op::BinaryOp;
        let device = VulkanDevice::new(0).unwrap();
        let lhs_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let rhs_data = vec![0.5f32, 0.5, 0.5, 0.5];
        let lhs = device.storage_from_slice(&lhs_data).unwrap();
        let rhs = device.storage_from_slice(&rhs_data).unwrap();
        let layout = Layout::contiguous((1, 4));

        let result = lhs.binary_impl::<BinaryOp>(&rhs, &layout, &layout).unwrap();
        let cpu = result.to_cpu_storage().unwrap();

        let expected = vec![1.5, 2.5, 3.5, 4.5];
        match cpu {
            crate::CpuStorage::F32(out) => {
                assert_eq!(out.len(), expected.len());
                for (a, &b) in out.iter().zip(&expected) {
                    assert!(
                        (a - b).abs() < 1e-3,
                        "Add mismatch: got {}, expected {}",
                        a,
                        b
                    );
                }
            }
            _ => panic!("Expected F32 cpu storage"),
        }
    }

    #[test]
    fn test_vulkan_reduce_sum_f32() {
        use crate::op::ReduceOp;
        let device = VulkanDevice::new(0).unwrap();
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let storage = device.storage_from_slice(&data).unwrap();
        let layout = Layout::contiguous((2, 3));

        let result = storage.reduce_op(ReduceOp::Sum, &layout, &[1]).unwrap();
        let cpu = result.to_cpu_storage().unwrap();

        let expected = vec![6.0, 15.0];
        match cpu {
            crate::CpuStorage::F32(out) => {
                assert_eq!(out.len(), expected.len());
                for (a, &b) in out.iter().zip(&expected) {
                    assert!(
                        (a - b).abs() < 1e-3,
                        "Reduce sum mismatch: got {}, expected {}",
                        a,
                        b
                    );
                }
            }
            _ => panic!("Expected F32 cpu storage"),
        }
    }

    #[test]
    fn test_vulkan_reduce_max_f32() {
        use crate::op::ReduceOp;
        let device = VulkanDevice::new(0).unwrap();
        let data = vec![1.0f32, 5.0, 3.0, 4.0, 2.0, 6.0];
        let storage = device.storage_from_slice(&data).unwrap();
        let layout = Layout::contiguous((2, 3));

        let result = storage.reduce_op(ReduceOp::Max, &layout, &[1]).unwrap();
        let cpu = result.to_cpu_storage().unwrap();

        let expected = vec![5.0, 6.0];
        match cpu {
            crate::CpuStorage::F32(out) => {
                assert_eq!(out.len(), expected.len());
                for (a, &b) in out.iter().zip(&expected) {
                    assert!(
                        (a - b).abs() < 1e-3,
                        "Reduce max mismatch: got {}, expected {}",
                        a,
                        b
                    );
                }
            }
            _ => panic!("Expected F32 cpu storage"),
        }
    }

    #[test]
    fn test_vulkan_unary_silu_f32() {
        use crate::op::UnaryOp;
        let device = VulkanDevice::new(0).unwrap();
        let data = vec![0.0f32, 1.0, -1.0, 0.5];
        let storage = device.storage_from_slice(&data).unwrap();
        let layout = Layout::contiguous((1, 4));

        let result = storage.unary_impl::<UnaryOp>(&layout).unwrap();
        let cpu = result.to_cpu_storage().unwrap();

        fn silu(x: f32) -> f32 {
            x / (1.0 + (-x).exp())
        }
        let expected: Vec<f32> = data.iter().map(|&x| silu(x)).collect();
        match cpu {
            crate::CpuStorage::F32(out) => {
                assert_eq!(out.len(), expected.len());
                for (a, &b) in out.iter().zip(&expected) {
                    assert!(
                        (a - b).abs() < 1e-3,
                        "Silu mismatch: got {}, expected {}",
                        a,
                        b
                    );
                }
            }
            _ => panic!("Expected F32 cpu storage"),
        }
    }

    #[test]
    fn test_vulkan_unary_exp_f32() {
        use crate::op::UnaryOp;
        let device = VulkanDevice::new(0).unwrap();
        let data = vec![0.0f32, 1.0, 2.0, 0.5];
        let storage = device.storage_from_slice(&data).unwrap();
        let layout = Layout::contiguous((1, 4));

        let result = storage.unary_impl::<UnaryOp>(&layout).unwrap();
        let cpu = result.to_cpu_storage().unwrap();

        let expected: Vec<f32> = data.iter().map(|x| x.exp()).collect();
        match cpu {
            crate::CpuStorage::F32(out) => {
                assert_eq!(out.len(), expected.len());
                for (a, &b) in out.iter().zip(&expected) {
                    assert!(
                        (a - b).abs() < 1e-3,
                        "Exp mismatch: got {}, expected {}",
                        a,
                        b
                    );
                }
            }
            _ => panic!("Expected F32 cpu storage"),
        }
    }
}

#[cfg(target_os = "android")]
mod quantized_vulkan_tests {
    use crate::quantized::vulkan::QVulkanStorage;
    use crate::quantized::GgmlDType;
    use crate::VulkanDevice;

    #[test]
    fn test_qvulkan_storage_zeros() {
        let device = VulkanDevice::new(0).unwrap();
        let qstorage = QVulkanStorage::zeros(&device, 256, GgmlDType::Q4K).unwrap();
        assert_eq!(qstorage.dtype(), GgmlDType::Q4K);
    }

    #[test]
    fn test_qvulkan_dequant_f32() {
        let device = VulkanDevice::new(0).unwrap();
        let original: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let qstorage = QVulkanStorage {
            dtype: GgmlDType::F32,
            device: device.clone(),
            cpu_data: original.iter().flat_map(|&x| x.to_le_bytes()).collect(),
        };
        let dequantized = qstorage.dequantize(256).unwrap();
        let cpu = dequantized.to_cpu_storage().unwrap();
        match cpu {
            crate::CpuStorage::F32(data) => {
                assert_eq!(data.len(), 256);
                for (a, b) in data.iter().zip(original.iter()) {
                    assert!((a - b).abs() < 1e-3, "Dequant F32 should preserve values");
                }
            }
            _ => panic!("Expected F32 cpu storage"),
        }
    }

    #[test]
    fn test_qvulkan_dequant_q4k_zeros() {
        let device = VulkanDevice::new(0).unwrap();
        let qstorage = QVulkanStorage::zeros(&device, 256, GgmlDType::Q4K).unwrap();
        let dequantized = qstorage.dequantize(256).unwrap();
        let cpu = dequantized.to_cpu_storage().unwrap();
        match cpu {
            crate::CpuStorage::F32(data) => {
                assert_eq!(data.len(), 256);
                for &val in &data {
                    assert!(
                        val.abs() < 1e-6,
                        "Zero Q4K blocks should dequantize to zeros, got {}",
                        val
                    );
                }
            }
            _ => panic!("Expected F32 cpu storage"),
        }
    }

    #[test]
    fn test_qvulkan_dequant_q4k_matches_cpu_reference() {
        use crate::quantized::k_quants::{BlockQ4K, QK_K};
        use half::{bf16, f16};

        let device = VulkanDevice::new(0).unwrap();
        let elem_count = QK_K;
        let num_blocks = 1;

        let mut blocks = vec![BlockQ4K::zeros(); num_blocks];
        blocks[0].d = f16::from_f32(0.5);
        blocks[0].dmin = f16::from_f32(0.1);
        blocks[0].scales = [0u8; 12];
        blocks[0].qs = [0x88u8; 128];

        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(
                blocks.as_ptr() as *const u8,
                num_blocks * std::mem::size_of::<BlockQ4K>(),
            )
            .to_vec()
        };

        let qstorage = QVulkanStorage {
            dtype: GgmlDType::Q4K,
            device: device.clone(),
            cpu_data: bytes,
        };

        let vulkan_result = qstorage.dequantize(elem_count).unwrap();
        let vulkan_cpu = vulkan_result.to_cpu_storage().unwrap();

        let mut cpu_reference = vec![0.0f32; elem_count];
        BlockQ4K::to_float(&blocks, &mut cpu_reference);

        match vulkan_cpu {
            crate::CpuStorage::F32(data) => {
                assert_eq!(data.len(), cpu_reference.len());
                for (vulkan_val, cpu_val) in data.iter().zip(cpu_reference.iter()) {
                    let diff = (vulkan_val - cpu_val).abs();
                    assert!(
                        diff < 1e-3,
                        "Q4K dequant mismatch: vulkan={}, cpu={}, diff={}",
                        vulkan_val,
                        cpu_val,
                        diff
                    );
                }
            }
            _ => panic!("Expected F32 cpu storage"),
        }
    }

    #[test]
    fn test_qvulkan_weights_roundtrip() {
        let device = VulkanDevice::new(0).unwrap();
        let original: Vec<f32> = (0..256).map(|i| i as f32 / 256.0).collect();

        let storage = device.storage_from_slice(&original).unwrap();
        let roundtrip = storage.to_cpu_storage().unwrap();

        match roundtrip {
            crate::CpuStorage::F32(data) => {
                assert_eq!(data.len(), original.len());
                for (a, b) in data.iter().zip(original.iter()) {
                    assert!(
                        (a - b).abs() < 1e-5,
                        "Roundtrip should preserve values: got {}, expected {}",
                        a,
                        b
                    );
                }
            }
            _ => panic!("Expected F32 cpu storage"),
        }
    }
}
