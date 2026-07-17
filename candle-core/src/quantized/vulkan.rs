use super::{GgmlDType, QStorage};
use crate::backend::BackendDevice;
use crate::{Result, Shape, VulkanDevice, VulkanStorage};
use half::{bf16, f16};

pub struct QVulkanStorage {
    dtype: GgmlDType,
    device: VulkanDevice,
    cpu_data: Vec<u8>,
}

impl QVulkanStorage {
    pub fn zeros(device: &VulkanDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let block_size = dtype.block_size();
        let num_blocks = (elem_count + block_size - 1) / block_size;
        let size = num_blocks * dtype.type_size();
        Ok(Self {
            dtype,
            device: device.clone(),
            cpu_data: vec![0u8; size],
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.cpu_data.len()
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<VulkanStorage> {
        use crate::quantized::k_quants::GgmlType;

        let block_size = self.dtype.block_size();
        let num_blocks = elem_count / block_size;

        let out: Vec<f32> = match self.dtype {
            GgmlDType::F32 => {
                let data: &[f32] = unsafe {
                    std::slice::from_raw_parts(self.cpu_data.as_ptr() as *const f32, elem_count)
                };
                data.to_vec()
            }
            GgmlDType::F16 => {
                let data: &[f16] = unsafe {
                    std::slice::from_raw_parts(self.cpu_data.as_ptr() as *const f16, elem_count)
                };
                data.iter().map(|&x| x.to_f32()).collect()
            }
            GgmlDType::BF16 => {
                let data: &[bf16] = unsafe {
                    std::slice::from_raw_parts(self.cpu_data.as_ptr() as *const bf16, elem_count)
                };
                data.iter().map(|&x| x.to_f32()).collect()
            }
            GgmlDType::Q4K => {
                let blocks: &[crate::quantized::BlockQ4K] = unsafe {
                    std::slice::from_raw_parts(self.cpu_data.as_ptr() as *const _, num_blocks)
                };
                let mut out = vec![0.0f32; elem_count];
                crate::quantized::BlockQ4K::to_float(blocks, &mut out);
                out
            }
            GgmlDType::Q8K => {
                let blocks: &[crate::quantized::BlockQ8K] = unsafe {
                    std::slice::from_raw_parts(self.cpu_data.as_ptr() as *const _, num_blocks)
                };
                let mut out = vec![0.0f32; elem_count];
                crate::quantized::BlockQ8K::to_float(blocks, &mut out);
                out
            }
            _ => {
                crate::bail!("unsupported dtype for vulkan dequantize: {:?}", self.dtype)
            }
        };

        let vulkan_storage = self.device.storage_from_slice(&out)?;
        Ok(vulkan_storage)
    }

    pub fn quantize(&mut self, _src: &VulkanStorage) -> Result<()> {
        Err(crate::Error::Vulkan(crate::VulkanError::Message(
            "quantize not implemented for vulkan".to_string(),
        )))
    }

    pub fn quantize_imatrix(
        &mut self,
        _src: &VulkanStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        Err(crate::Error::Vulkan(crate::VulkanError::Message(
            "quantize_imatrix not implemented for vulkan".to_string(),
        )))
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        _src: &crate::CpuStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        Err(crate::Error::Vulkan(crate::VulkanError::Message(
            "quantize_imatrix_onto not implemented for vulkan".to_string(),
        )))
    }

    pub fn quantize_onto(&mut self, _src: &crate::CpuStorage) -> Result<()> {
        Err(crate::Error::Vulkan(crate::VulkanError::Message(
            "quantize_onto not implemented for vulkan".to_string(),
        )))
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        Ok(self.cpu_data.clone())
    }

    pub fn fwd(
        &self,
        _self_shape: &Shape,
        _storage: &VulkanStorage,
        _layout: &crate::Layout,
    ) -> Result<(VulkanStorage, Shape)> {
        Err(crate::Error::Vulkan(crate::VulkanError::Message(
            "fwd matmul not implemented for vulkan (use CANDLE_DEQUANTIZE_ALL_F16=1 for now)"
                .to_string(),
        )))
    }

    pub fn indexed_moe_forward(
        &self,
        _: &Shape,
        _: &VulkanStorage,
        _: &crate::Layout,
        _: &VulkanStorage,
        _: &crate::Layout,
    ) -> Result<(VulkanStorage, Shape)> {
        Err(crate::Error::Vulkan(crate::VulkanError::Message(
            "indexed_moe_forward not implemented for vulkan".to_string(),
        )))
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &VulkanDevice,
    data: &[T],
) -> Result<QStorage> {
    let bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<T>(),
        )
    }
    .to_vec();
    Ok(QStorage::Vulkan(QVulkanStorage {
        dtype: T::DTYPE,
        device: device.clone(),
        cpu_data: bytes,
    }))
}
