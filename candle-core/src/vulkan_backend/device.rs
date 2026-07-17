#![allow(dead_code)]

use crate::backend::BackendDevice;
#[cfg(not(all(feature = "vulkan", target_os = "android")))]
use crate::VulkanError;
use crate::{CpuStorage, DType, DeviceLocation, Result, Shape, VulkanStorage};
use std::ffi::CString;
use std::mem::ManuallyDrop;
use std::sync::Arc;

#[cfg(all(feature = "vulkan", target_os = "android"))]
use ash::vk;

#[cfg(all(feature = "vulkan", target_os = "android"))]
use gpu_allocator::vulkan::*;
#[cfg(all(feature = "vulkan", target_os = "android"))]
use gpu_allocator::MemoryLocation;

#[cfg(all(feature = "vulkan", target_os = "android"))]
use super::VulkanError;

#[cfg(all(feature = "vulkan", target_os = "android"))]
pub struct VulkanDevice {
    ordinal: usize,
    inner: Arc<VulkanDeviceInner>,
}

#[cfg(all(feature = "vulkan", target_os = "android"))]
struct VulkanDeviceInner {
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    queue: std::sync::Mutex<vk::Queue>,
    queue_family_index: u32,
    allocator: std::sync::Mutex<ManuallyDrop<Allocator>>,
    command_pool: std::sync::Mutex<vk::CommandPool>,
    location: DeviceLocation,
}

#[cfg(not(all(feature = "vulkan", target_os = "android")))]
pub struct VulkanDevice {
    ordinal: usize,
}

#[cfg(not(all(feature = "vulkan", target_os = "android")))]
impl VulkanDevice {
    pub fn new(ordinal: usize) -> Result<Self> {
        Err(crate::Error::NotCompiledWithVulkanSupport)
    }
}

#[cfg(all(feature = "vulkan", target_os = "android"))]
impl VulkanDevice {
    pub fn new(ordinal: usize) -> Result<Self> {
        if ordinal != 0 {
            crate::bail!("Vulkan only supports ordinal 0 on Android")
        }
        Self::new_internal(ordinal)
    }

    fn new_internal(ordinal: usize) -> Result<Self> {
        unsafe {
            let entry = ash::Entry::load().map_err(|e| {
                crate::Error::Vulkan(VulkanError::Message(format!(
                    "Failed to load Vulkan entry: {:?}",
                    e
                )))
            })?;

            let extension_names = [
                vk::KHR_SURFACE_NAME.as_ptr(),
                vk::KHR_ANDROID_SURFACE_NAME.as_ptr(),
                vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME.as_ptr(),
            ];

            let app_name = CString::new("candle").unwrap();
            let engine_name = CString::new("candle").unwrap();
            let app_info = vk::ApplicationInfo::default()
                .application_name(&app_name)
                .application_version(vk::make_api_version(0, 1, 0, 0))
                .engine_name(&engine_name)
                .engine_version(vk::make_api_version(0, 1, 0, 0))
                .api_version(vk::API_VERSION_1_1);

            let create_info = vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_extension_names(&extension_names);

            let instance = entry.create_instance(&create_info, None).map_err(|e| {
                crate::Error::Vulkan(VulkanError::Message(format!(
                    "Failed to create Vulkan instance: {:?}",
                    e
                )))
            })?;

            let physical_devices = instance.enumerate_physical_devices().map_err(|e| {
                crate::Error::Vulkan(VulkanError::Message(format!(
                    "Failed to enumerate physical devices: {:?}",
                    e
                )))
            })?;

            if physical_devices.is_empty() {
                return Err(crate::Error::Vulkan(VulkanError::Message(
                    "No Vulkan-capable devices found".to_string(),
                )));
            }

            let physical_device = physical_devices[ordinal.min(physical_devices.len() - 1)];
            let queue_family_properties =
                instance.get_physical_device_queue_family_properties(physical_device);

            let queue_family_index = queue_family_properties
                .iter()
                .enumerate()
                .find_map(|(i, props)| {
                    if props.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                        Some(i as u32)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| {
                    crate::Error::Vulkan(VulkanError::Message(
                        "No compute queue family found".to_string(),
                    ))
                })?;

            let queue_priorities = [1.0f32];
            let queue_create_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&queue_priorities);

            let device_extension_names = [vk::KHR_SHADER_NON_SEMANTIC_INFO_NAME.as_ptr()];
            let device_features = vk::PhysicalDeviceFeatures::default();

            let device_create_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&queue_create_info))
                .enabled_extension_names(&device_extension_names)
                .enabled_features(&device_features);

            let device = instance
                .create_device(physical_device, &device_create_info, None)
                .map_err(|e| {
                    crate::Error::Vulkan(VulkanError::Message(format!(
                        "Failed to create logical device: {:?}",
                        e
                    )))
                })?;

            let queue = device.get_device_queue(queue_family_index, 0);

            let allocator = Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device,
                debug_settings: Default::default(),
                buffer_device_address: false,
                allocation_sizes: Default::default(),
            })
            .map_err(|e| {
                crate::Error::Vulkan(VulkanError::Message(format!(
                    "Failed to create GPU allocator: {:?}",
                    e
                )))
            })?;

            let command_pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(queue_family_index)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(|e| {
                    crate::Error::Vulkan(VulkanError::Message(format!(
                        "Failed to create command pool: {:?}",
                        e
                    )))
                })?;

            let location = DeviceLocation::Vulkan { gpu_id: ordinal };

            Ok(Self {
                ordinal,
                inner: Arc::new(VulkanDeviceInner {
                    entry,
                    instance,
                    device,
                    physical_device,
                    queue: std::sync::Mutex::new(queue),
                    queue_family_index,
                    allocator: std::sync::Mutex::new(ManuallyDrop::new(allocator)),
                    command_pool: std::sync::Mutex::new(command_pool),
                    location,
                }),
            })
        }
    }

    pub(crate) fn allocate_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<(vk::Buffer, Allocation)> {
        unsafe {
            let buffer_info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage | vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let buffer = self
                .inner
                .device
                .create_buffer(&buffer_info, None)
                .map_err(|e| {
                    crate::Error::Vulkan(VulkanError::Message(format!(
                        "Failed to create buffer: {:?}",
                        e
                    )))
                })?;

            let requirements = self.inner.device.get_buffer_memory_requirements(buffer);

            let mut allocator = self.inner.allocator.lock().unwrap();
            let allocation = allocator
                .allocate(&AllocationCreateDesc {
                    name: "buffer",
                    requirements,
                    location: MemoryLocation::GpuToCpu,
                    linear: true,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                })
                .map_err(|e| {
                    crate::Error::Vulkan(VulkanError::Message(format!(
                        "Failed to allocate GPU memory: {:?}",
                        e
                    )))
                })?;

            self.inner
                .device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .map_err(|e| {
                    crate::Error::Vulkan(VulkanError::Message(format!(
                        "Failed to bind buffer memory: {:?}",
                        e
                    )))
                })?;

            Ok((buffer, allocation))
        }
    }

    pub(crate) fn device(&self) -> &ash::Device {
        &self.inner.device
    }

    pub(crate) fn queue(&self) -> vk::Queue {
        *self.inner.queue.lock().unwrap()
    }

    pub(crate) fn command_pool(&self) -> vk::CommandPool {
        *self.inner.command_pool.lock().unwrap()
    }

    pub(crate) fn physical_device(&self) -> vk::PhysicalDevice {
        self.inner.physical_device
    }

    pub(crate) fn instance(&self) -> &ash::Instance {
        &self.inner.instance
    }

    pub(crate) fn allocate_and_upload(
        &self,
        data: &[u8],
        usage: vk::BufferUsageFlags,
    ) -> Result<(vk::Buffer, Allocation)> {
        let (buffer, mut allocation) = self.allocate_buffer(data.len() as u64, usage)?;
        unsafe {
            let mapped = allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), mapped, data.len());
        }
        Ok((buffer, allocation))
    }

    pub(crate) fn download_and_free(
        &self,
        buffer: vk::Buffer,
        allocation: Allocation,
        data: &mut [u8],
    ) -> Result<()> {
        unsafe {
            let mapped = allocation.mapped_ptr().unwrap().as_ptr() as *const u8;
            std::ptr::copy_nonoverlapping(mapped, data.as_mut_ptr(), data.len());
        }
        let mut allocator = self.inner.allocator.lock().unwrap();
        allocator.free(allocation).map_err(|e| {
            crate::Error::Vulkan(VulkanError::Message(format!(
                "Failed to free GPU memory: {:?}",
                e
            )))
        })?;
        unsafe {
            self.inner.device.destroy_buffer(buffer, None);
        }
        Ok(())
    }
}

impl std::fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanDevice")
            .field("ordinal", &self.ordinal)
            .finish()
    }
}

impl Clone for VulkanDevice {
    fn clone(&self) -> Self {
        Self {
            ordinal: self.ordinal,
            #[cfg(all(feature = "vulkan", target_os = "android"))]
            inner: self.inner.clone(),
        }
    }
}

#[cfg(all(feature = "vulkan", target_os = "android"))]
impl BackendDevice for VulkanDevice {
    type Storage = VulkanStorage;

    fn new(ordinal: usize) -> Result<Self> {
        Self::new(ordinal)
    }

    fn location(&self) -> DeviceLocation {
        self.location
    }

    fn same_device(&self, other: &Self) -> bool {
        self.ordinal == other.ordinal
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let size = elem_count * dtype.size_in_bytes();
        let (buffer, allocation) =
            self.allocate_buffer(size as u64, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        unsafe {
            let mapped = allocation.mapped_ptr().unwrap().as_ptr();
            std::ptr::write_bytes(mapped, 0, size);
        }
        Ok(VulkanStorage::new(
            buffer,
            allocation,
            elem_count,
            dtype,
            self.clone(),
        ))
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let size = elem_count * dtype.size_in_bytes();
        let (buffer, allocation) =
            self.allocate_buffer(size as u64, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        Ok(VulkanStorage::new(
            buffer,
            allocation,
            elem_count,
            dtype,
            self.clone(),
        ))
    }

    fn storage_from_slice<T: crate::WithDType>(&self, data: &[T]) -> Result<Self::Storage> {
        let dtype = T::DTYPE;
        let elem_count = data.len();
        let size = elem_count * dtype.size_in_bytes();
        let (buffer, allocation) =
            self.allocate_buffer(size as u64, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        unsafe {
            let mapped = allocation.mapped_ptr().unwrap().as_ptr() as *mut T;
            std::ptr::copy_nonoverlapping(data.as_ptr(), mapped, elem_count);
        }
        Ok(VulkanStorage::new(
            buffer,
            allocation,
            elem_count,
            dtype,
            self.clone(),
        ))
    }

    fn storage_from_cpu_storage(&self, cpu_storage: &CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage_owned(cpu_storage.clone())
    }

    fn storage_from_cpu_storage_owned(&self, cpu_storage: CpuStorage) -> Result<Self::Storage> {
        let (data, dtype, elem_count) = match cpu_storage {
            CpuStorage::U8(v) => (v, DType::U8, v.len()),
            CpuStorage::U32(v) => {
                let bytes: Vec<u8> = v.iter().flat_map(|&x| x.to_le_bytes()).collect();
                (bytes, DType::U32, v.len())
            }
            CpuStorage::I64(v) => {
                let bytes: Vec<u8> = v.iter().flat_map(|&x| x.to_le_bytes()).collect();
                (bytes, DType::I64, v.len())
            }
            CpuStorage::F16(v) => {
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                (bytes, DType::F16, v.len())
            }
            CpuStorage::BF16(v) => {
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                (bytes, DType::BF16, v.len())
            }
            CpuStorage::F32(v) => {
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                (bytes, DType::F32, v.len())
            }
            CpuStorage::F64(v) => {
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                (bytes, DType::F64, v.len())
            }
            CpuStorage::I32(v) => {
                let bytes: Vec<u8> = v.iter().flat_map(|&x| x.to_le_bytes()).collect();
                (bytes, DType::I32, v.len())
            }
            CpuStorage::I16(v) => {
                let bytes: Vec<u8> = v.iter().flat_map(|&x| x.to_le_bytes()).collect();
                (bytes, DType::I16, v.len())
            }
            CpuStorage::F8E4M3(v) => {
                let bytes: Vec<u8> =
                    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) }
                        .to_vec();
                (bytes, DType::F8E4M3, v.len())
            }
            CpuStorage::F6E2M3(v) => {
                let bytes: Vec<u8> =
                    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) }
                        .to_vec();
                (bytes, DType::F6E2M3, v.len())
            }
            CpuStorage::F6E3M2(v) => {
                let bytes: Vec<u8> =
                    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) }
                        .to_vec();
                (bytes, DType::F6E3M2, v.len())
            }
            CpuStorage::F4(v) => {
                let bytes: Vec<u8> =
                    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) }
                        .to_vec();
                (bytes, DType::F4, v.len())
            }
            CpuStorage::F8E8M0(v) => {
                let bytes: Vec<u8> =
                    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) }
                        .to_vec();
                (bytes, DType::F8E8M0, v.len())
            }
        };

        let size = data.len();
        let (buffer, allocation) =
            self.allocate_buffer(size as u64, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        unsafe {
            let mapped = allocation.mapped_ptr().unwrap().as_ptr();
            std::ptr::copy_nonoverlapping(data.as_ptr(), mapped, size);
        }
        Ok(VulkanStorage::new(
            buffer,
            allocation,
            elem_count,
            dtype,
            self.clone(),
        ))
    }

    fn rand_uniform(
        &self,
        _shape: &Shape,
        _dtype: DType,
        _lo: f64,
        _up: f64,
    ) -> Result<Self::Storage> {
        Err(crate::Error::Vulkan(VulkanError::Message(
            "rand_uniform not implemented for vulkan".to_string(),
        )))
    }

    fn rand_normal(
        &self,
        _shape: &Shape,
        _dtype: DType,
        _mean: f64,
        _std: f64,
    ) -> Result<Self::Storage> {
        Err(crate::Error::Vulkan(VulkanError::Message(
            "rand_normal not implemented for vulkan".to_string(),
        )))
    }

    fn set_seed(&self, _seed: u64) -> Result<()> {
        Err(crate::Error::Vulkan(VulkanError::Message(
            "set_seed not implemented for vulkan".to_string(),
        )))
    }

    fn get_current_seed(&self) -> Result<u64> {
        Err(crate::Error::Vulkan(VulkanError::Message(
            "get_current_seed not implemented for vulkan".to_string(),
        )))
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(not(all(feature = "vulkan", target_os = "android")))]
impl BackendDevice for VulkanDevice {
    type Storage = VulkanStorage;

    fn new(_: usize) -> Result<Self> {
        Err(crate::Error::NotCompiledWithVulkanSupport)
    }

    fn location(&self) -> DeviceLocation {
        DeviceLocation::Vulkan {
            gpu_id: self.ordinal,
        }
    }

    fn same_device(&self, _: &Self) -> bool {
        true
    }

    fn zeros_impl(&self, _: &Shape, _: DType) -> Result<Self::Storage> {
        Err(crate::Error::NotCompiledWithVulkanSupport)
    }

    unsafe fn alloc_uninit(&self, _: &Shape, _: DType) -> Result<Self::Storage> {
        Err(crate::Error::NotCompiledWithVulkanSupport)
    }

    fn storage_from_slice<T: crate::WithDType>(&self, _: &[T]) -> Result<Self::Storage> {
        Err(crate::Error::NotCompiledWithVulkanSupport)
    }

    fn storage_from_cpu_storage(&self, _: &CpuStorage) -> Result<Self::Storage> {
        Err(crate::Error::NotCompiledWithVulkanSupport)
    }

    fn storage_from_cpu_storage_owned(&self, _: CpuStorage) -> Result<Self::Storage> {
        Err(crate::Error::NotCompiledWithVulkanSupport)
    }

    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        Err(crate::Error::NotCompiledWithVulkanSupport)
    }

    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        Err(crate::Error::NotCompiledWithVulkanSupport)
    }

    fn set_seed(&self, _: u64) -> Result<()> {
        Err(crate::Error::NotCompiledWithVulkanSupport)
    }

    fn get_current_seed(&self) -> Result<u64> {
        Err(crate::Error::NotCompiledWithVulkanSupport)
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(all(feature = "vulkan", target_os = "android"))]
impl Drop for VulkanDeviceInner {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            if let Ok(command_pool) = self.command_pool.lock() {
                self.device.destroy_command_pool(*command_pool, None);
            }
            if let Ok(mut allocator) = self.allocator.lock() {
                ManuallyDrop::drop(&mut *allocator);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
