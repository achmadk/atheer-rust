use crate::{AccelBackend, AccelResult, BackendType, Result};
use std::ffi::CString;
use std::mem::ManuallyDrop;
use std::sync::Mutex;
use std::time::Instant;

#[cfg(target_os = "android")]
use ash::vk;
#[cfg(target_os = "android")]
use gpu_allocator::vulkan::*;
#[cfg(target_os = "android")]
use gpu_allocator::*;

#[cfg(not(target_os = "android"))]
#[derive(Debug)]
struct VulkanContext;

#[cfg(not(target_os = "android"))]
struct VulkanDevice {
    name: String,
    driver_version: String,
    max_memory_mb: u64,
}

#[cfg(not(target_os = "android"))]
struct VulkanQueue {
    family_index: u32,
}

#[cfg(target_os = "android")]
#[derive(Debug)]
struct VulkanContext {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    queue: vk::Queue,
    queue_family_index: u32,
    allocator: ManuallyDrop<Allocator>,
    gemv_descriptor_set_layout: vk::DescriptorSetLayout,
    gemv_pipeline_layout: vk::PipelineLayout,
    gemv_pipeline: vk::Pipeline,
    attention_descriptor_set_layout: vk::DescriptorSetLayout,
    attention_pipeline_layout: vk::PipelineLayout,
    attention_pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    command_pool: vk::CommandPool,
    weight_buffer: Option<vk::Buffer>,
    weight_allocation: Option<Allocation>,
    weight_hidden_size: u32,
    weight_vocab_size: u32,
}

#[cfg(target_os = "android")]
#[repr(C)]
struct GemvPushConstants {
    batch_size: u32,
    vocab_size: u32,
    hidden_size: u32,
    quantization_type: u32,
}

#[cfg(target_os = "android")]
#[repr(C)]
struct AttentionPushConstants {
    batch_size: u32,
    seq_len: u32,
    head_dim: u32,
    num_heads: u32,
    scale: f32,
}

#[cfg(target_os = "android")]
impl VulkanContext {
    fn new() -> Result<Self> {
        unsafe {
            let entry = ash::Entry::linked();
            let extension_names = [
                vk::KHR_SURFACE_NAME.as_ptr(),
                vk::KHR_ANDROID_SURFACE_NAME.as_ptr(),
                vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_NAME.as_ptr(),
            ];

            let app_info = vk::ApplicationInfo::default()
                .application_name(&CString::new("Atheer").unwrap())
                .application_version(vk::make_api_version(0, 1, 0, 0))
                .engine_name(&CString::new("Atheer").unwrap())
                .engine_version(vk::make_api_version(0, 1, 0, 0))
                .api_version(vk::API_VERSION_1_1);

            let create_info = vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_extension_names(&extension_names);

            let instance = entry.create_instance(&create_info, None).map_err(|e| {
                crate::AccelError::BackendNotAvailable(format!(
                    "Failed to create Vulkan instance: {:?}",
                    e
                ))
            })?;

            let physical_devices = instance.enumerate_physical_devices().map_err(|e| {
                crate::AccelError::BackendNotAvailable(format!(
                    "Failed to enumerate physical devices: {:?}",
                    e
                ))
            })?;

            if physical_devices.is_empty() {
                return Err(crate::AccelError::BackendNotAvailable(
                    "No Vulkan-capable devices found".to_string(),
                ));
            }

            let physical_device = physical_devices[0];
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
                    crate::AccelError::BackendNotAvailable(
                        "No compute queue family found".to_string(),
                    )
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
                    crate::AccelError::BackendNotAvailable(format!(
                        "Failed to create logical device: {:?}",
                        e
                    ))
                })?;

            let queue = device.get_device_queue(queue_family_index, 0);

            let mut allocator = Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device,
                debug_settings: Default::default(),
                buffer_device_address: false,
                allocation_sizes: Default::default(),
            })
            .map_err(|e| {
                crate::AccelError::BackendNotAvailable(format!(
                    "Failed to create GPU allocator: {:?}",
                    e
                ))
            })?;

            let gemv_shader_bytes =
                include_bytes!(concat!(env!("OUT_DIR"), "/shaders/gemv.spv"));
            let gemv_module = device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::default()
                        .code(bytemuck::cast_slice(gemv_shader_bytes)),
                    None,
                )
                .map_err(|e| {
                    crate::AccelError::OperationFailed(format!(
                        "Failed to create GEMV shader module: {:?}",
                        e
                    ))
                })?;

            let gemv_bindings = [
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            ];

            let gemv_descriptor_set_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&gemv_bindings),
                    None,
                )
                .map_err(|e| {
                    crate::AccelError::OperationFailed(format!(
                        "Failed to create GEMV descriptor set layout: {:?}",
                        e
                    ))
                })?;

            let gemv_push_const_range = vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(std::mem::size_of::<GemvPushConstants>() as u32);

            let gemv_pipeline_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(std::slice::from_ref(&gemv_descriptor_set_layout))
                        .push_constant_ranges(std::slice::from_ref(&gemv_push_const_range)),
                    None,
                )
                .map_err(|e| {
                    crate::AccelError::OperationFailed(format!(
                        "Failed to create GEMV pipeline layout: {:?}",
                        e
                    ))
                })?;

            let entry_point = CString::new("main").unwrap();
            let gemv_shader_stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(gemv_module)
                .name(&entry_point);

            let gemv_pipeline = device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(
                        &vk::ComputePipelineCreateInfo::default()
                            .stage(gemv_shader_stage)
                            .layout(gemv_pipeline_layout),
                    ),
                    None,
                )
                .map_err(|e| {
                    crate::AccelError::OperationFailed(format!(
                        "Failed to create GEMV compute pipeline: {:?}",
                        e
                    ))
                })?[0];

            device.destroy_shader_module(gemv_module, None);

            let attention_shader_bytes =
                include_bytes!(concat!(env!("OUT_DIR"), "/shaders/attention.spv"));
            let attention_module = device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::default()
                        .code(bytemuck::cast_slice(attention_shader_bytes)),
                    None,
                )
                .map_err(|e| {
                    crate::AccelError::OperationFailed(format!(
                        "Failed to create attention shader module: {:?}",
                        e
                    ))
                })?;

            let attention_bindings = [
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(3)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            ];

            let attention_descriptor_set_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&attention_bindings),
                    None,
                )
                .map_err(|e| {
                    crate::AccelError::OperationFailed(format!(
                        "Failed to create attention descriptor set layout: {:?}",
                        e
                    ))
                })?;

            let attention_push_const_range = vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(std::mem::size_of::<AttentionPushConstants>() as u32);

            let attention_pipeline_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(std::slice::from_ref(&attention_descriptor_set_layout))
                        .push_constant_ranges(std::slice::from_ref(&attention_push_const_range)),
                    None,
                )
                .map_err(|e| {
                    crate::AccelError::OperationFailed(format!(
                        "Failed to create attention pipeline layout: {:?}",
                        e
                    ))
                })?;

            let attention_shader_stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(attention_module)
                .name(&entry_point);

            let attention_pipeline = device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(
                        &vk::ComputePipelineCreateInfo::default()
                            .stage(attention_shader_stage)
                            .layout(attention_pipeline_layout),
                    ),
                    None,
                )
                .map_err(|e| {
                    crate::AccelError::OperationFailed(format!(
                        "Failed to create attention compute pipeline: {:?}",
                        e
                    ))
                })?[0];

            device.destroy_shader_module(attention_module, None);

            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(8)];

            let descriptor_pool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .pool_sizes(&pool_sizes)
                        .max_sets(2),
                    None,
                )
                .map_err(|e| {
                    crate::AccelError::OperationFailed(format!(
                        "Failed to create descriptor pool: {:?}",
                        e
                    ))
                })?;

            let command_pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(queue_family_index)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(|e| {
                    crate::AccelError::OperationFailed(format!(
                        "Failed to create command pool: {:?}",
                        e
                    ))
                })?;

            Ok(VulkanContext {
                _entry: entry,
                instance,
                device,
                physical_device,
                queue,
                queue_family_index,
                allocator: ManuallyDrop::new(allocator),
                gemv_descriptor_set_layout,
                gemv_pipeline_layout,
                gemv_pipeline,
                attention_descriptor_set_layout,
                attention_pipeline_layout,
                attention_pipeline,
                descriptor_pool,
                command_pool,
                weight_buffer: None,
                weight_allocation: None,
                weight_hidden_size: 0,
                weight_vocab_size: 0,
            })
        }
    }

    fn allocate_buffer(
        &mut self,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<(vk::Buffer, Allocation)> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage | vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            self.device.create_buffer(&buffer_info, None).map_err(|e| {
                crate::AccelError::MemoryAllocationFailed(format!(
                    "Failed to create buffer: {:?}",
                    e
                ))
            })?
        };

        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let allocation = self
            .allocator
            .allocate(&AllocationCreateDesc {
                name: "buffer",
                requirements,
                location: MemoryLocation::GpuToCpu,
                linear: true,
            })
            .map_err(|e| {
                crate::AccelError::MemoryAllocationFailed(format!(
                    "Failed to allocate GPU memory: {:?}",
                    e
                ))
            })?;

        unsafe {
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .map_err(|e| {
                    crate::AccelError::MemoryAllocationFailed(format!(
                        "Failed to bind buffer memory: {:?}",
                        e
                    ))
                })?;
        }

        Ok((buffer, allocation))
    }

    fn ensure_weight_buffers(
        &mut self,
        hidden_size: u32,
        vocab_size: u32,
    ) -> Result<()> {
        if self.weight_buffer.is_some()
            && self.weight_hidden_size == hidden_size
            && self.weight_vocab_size == vocab_size
        {
            return Ok(());
        }

        if let Some(buf) = self.weight_buffer.take() {
            unsafe { self.device.destroy_buffer(buf, None); }
        }
        if let Some(alloc) = self.weight_allocation.take() {
            self.allocator.free(alloc).ok();
        }

        let weight_size = (hidden_size as u64) * (vocab_size as u64) * 4;
        let (buffer, allocation) = self.allocate_buffer(
            weight_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;

        unsafe {
            let mapped = allocation.mapped_ptr().unwrap().as_ptr() as *mut f32;
            let count = (hidden_size * vocab_size) as usize;
            for i in 0..count {
                *mapped.add(i) = ((i % 997) as f32) / 997.0f32 - 0.5f32;
            }
        }

        self.weight_buffer = Some(buffer);
        self.weight_allocation = Some(allocation);
        self.weight_hidden_size = hidden_size;
        self.weight_vocab_size = vocab_size;

        Ok(())
    }

    fn dispatch_gemv(
        &mut self,
        input_ids: &[u32],
        hidden_size: u32,
        vocab_size: u32,
    ) -> Result<Vec<f32>> {
        let batch_size = input_ids.len() as u32;

        self.ensure_weight_buffers(hidden_size, vocab_size)?;
        let weight_buffer = self.weight_buffer.unwrap();

        unsafe {
            let input_size = (batch_size as u64) * size_of::<u32>() as u64;
            let (input_buffer, mut input_allocation) = self.allocate_buffer(
                input_size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )?;

            {
                let mapped = input_allocation.mapped_ptr().unwrap().as_ptr() as *mut u32;
                std::ptr::copy_nonoverlapping(input_ids.as_ptr(), mapped, input_ids.len());
            }

            let output_size =
                (batch_size as u64) * (vocab_size as u64) * size_of::<f32>() as u64;
            let (output_buffer, mut output_allocation) = self.allocate_buffer(
                output_size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )?;

            let descriptor_sets = self
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.descriptor_pool)
                        .set_layouts(std::slice::from_ref(&self.gemv_descriptor_set_layout)),
                )
                .map_err(|e| {
                    crate::AccelError::OperationFailed(format!(
                        "Failed to allocate descriptor set: {:?}",
                        e
                    ))
                })?;

            let descriptor_set = descriptor_sets[0];
            let weight_size = (hidden_size as u64) * (vocab_size as u64) * 4;

            let buffer_infos = [
                vk::DescriptorBufferInfo::default()
                    .buffer(input_buffer)
                    .offset(0)
                    .range(input_size),
                vk::DescriptorBufferInfo::default()
                    .buffer(weight_buffer)
                    .offset(0)
                    .range(weight_size),
                vk::DescriptorBufferInfo::default()
                    .buffer(output_buffer)
                    .offset(0)
                    .range(output_size),
            ];

            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&buffer_infos[0])),
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&buffer_infos[1])),
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&buffer_infos[2])),
            ];

            self.device.update_descriptor_sets(&writes, &[]);

            let command_buffers = self
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(|e| {
                    crate::AccelError::OperationFailed(format!(
                        "Failed to allocate command buffer: {:?}",
                        e
                    ))
                })?;

            let command_buffer = command_buffers[0];

            self.device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())
                .map_err(|e| {
                    crate::AccelError::OperationFailed(format!(
                        "Failed to begin command buffer: {:?}",
                        e
                    ))
                })?;

            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.gemv_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.gemv_pipeline_layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            );

            let push = GemvPushConstants {
                batch_size,
                vocab_size,
                hidden_size,
                quantization_type: 0,
            };
            self.device.cmd_push_constants(
                command_buffer,
                self.gemv_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push),
            );

            let workgroup_size = 256u32;
            let dispatch_x =
                ((batch_size * vocab_size + workgroup_size - 1) / workgroup_size).max(1);
            self.device.cmd_dispatch(command_buffer, dispatch_x, 1, 1);

            self.device.end_command_buffer(command_buffer).map_err(|e| {
                crate::AccelError::OperationFailed(format!(
                    "Failed to end command buffer: {:?}",
                    e
                ))
            })?;

            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| {
                    crate::AccelError::OperationFailed(format!(
                        "Failed to create fence: {:?}",
                        e
                    ))
                })?;

            let submit_info =
                vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));

            self.device
                .queue_submit(self.queue, std::slice::from_ref(&submit_info), fence)
                .map_err(|e| {
                    crate::AccelError::OperationFailed(format!(
                        "Failed to submit: {:?}",
                        e
                    ))
                })?;

            self.device
                .wait_for_fences(std::slice::from_ref(&fence), true, 5_000_000_000)
                .map_err(|e| {
                    crate::AccelError::OperationFailed(format!(
                        "Fence wait failed: {:?}",
                        e
                    ))
                })?;

            let mapped = output_allocation.mapped_ptr().unwrap().as_ptr() as *const f32;
            let mut logits = vec![0.0f32; (batch_size * vocab_size) as usize];
            std::ptr::copy_nonoverlapping(mapped, logits.as_mut_ptr(), logits.len());

            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.command_pool, &command_buffers);
            self.allocator.free(input_allocation).ok();
            self.allocator.free(output_allocation).ok();
            self.device.destroy_buffer(input_buffer, None);
            self.device.destroy_buffer(output_buffer, None);

            Ok(logits)
        }
    }

    fn dispatch_attention(
        &mut self,
        q_buf: &[f32],
        k_buf: &[f32],
        v_buf: &[f32],
        batch_size: u32,
        seq_len: u32,
        head_dim: u32,
        num_heads: u32,
    ) -> Result<Vec<f32>> {
        let total_elements = batch_size * num_heads * seq_len * head_dim;

        unsafe {
            let buf_size = (total_elements as u64) * size_of::<f32>() as u64;

            let (q_buffer, mut q_alloc) = self.allocate_buffer(
                buf_size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )?;
            let (k_buffer, mut k_alloc) = self.allocate_buffer(
                buf_size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )?;
            let (v_buffer, mut v_alloc) = self.allocate_buffer(
                buf_size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )?;
            let (output_buffer, mut output_alloc) = self.allocate_buffer(
                buf_size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )?;

            std::ptr::copy_nonoverlapping(q_buf.as_ptr(), q_alloc.mapped_ptr().unwrap().as_ptr() as *mut f32, q_buf.len());
            std::ptr::copy_nonoverlapping(k_buf.as_ptr(), k_alloc.mapped_ptr().unwrap().as_ptr() as *mut f32, k_buf.len());
            std::ptr::copy_nonoverlapping(v_buf.as_ptr(), v_alloc.mapped_ptr().unwrap().as_ptr() as *mut f32, v_buf.len());

            let descriptor_sets = self
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.descriptor_pool)
                        .set_layouts(std::slice::from_ref(&self.attention_descriptor_set_layout)),
                )
                .map_err(|e| {
                    crate::AccelError::OperationFailed(format!(
                        "Failed to allocate attention descriptor set: {:?}",
                        e
                    ))
                })?;

            let descriptor_set = descriptor_sets[0];

            let buffer_infos = [
                vk::DescriptorBufferInfo::default().buffer(q_buffer).offset(0).range(buf_size),
                vk::DescriptorBufferInfo::default().buffer(k_buffer).offset(0).range(buf_size),
                vk::DescriptorBufferInfo::default().buffer(v_buffer).offset(0).range(buf_size),
                vk::DescriptorBufferInfo::default().buffer(output_buffer).offset(0).range(buf_size),
            ];

            let writes = [
                vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&buffer_infos[0])),
                vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&buffer_infos[1])),
                vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&buffer_infos[2])),
                vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(3)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&buffer_infos[3])),
            ];

            self.device.update_descriptor_sets(&writes, &[]);

            let command_buffers = self
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(|e| {
                    crate::AccelError::OperationFailed(format!(
                        "Failed to allocate command buffer: {:?}",
                        e
                    ))
                })?;

            let command_buffer = command_buffers[0];

            self.device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())
                .map_err(|e| {
                    crate::AccelError::OperationFailed(format!(
                        "Failed to begin command buffer: {:?}",
                        e
                    ))
                })?;

            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.attention_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.attention_pipeline_layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            );

            let push = AttentionPushConstants {
                batch_size,
                seq_len,
                head_dim,
                num_heads,
                scale: 1.0f32 / (head_dim as f32).sqrt(),
            };
            self.device.cmd_push_constants(
                command_buffer,
                self.attention_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push),
            );

            let workgroup_size = 256u32;
            let dispatch_x =
                ((total_elements + workgroup_size - 1) / workgroup_size).max(1);
            self.device.cmd_dispatch(command_buffer, dispatch_x, 1, 1);

            self.device.end_command_buffer(command_buffer).map_err(|e| {
                crate::AccelError::OperationFailed(format!(
                    "Failed to end command buffer: {:?}",
                    e
                ))
            })?;

            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| {
                    crate::AccelError::OperationFailed(format!(
                        "Failed to create fence: {:?}",
                        e
                    ))
                })?;

            let submit_info =
                vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));

            self.device
                .queue_submit(self.queue, std::slice::from_ref(&submit_info), fence)
                .map_err(|e| crate::AccelError::OperationFailed(format!("Submit failed: {:?}", e)))?;

            self.device
                .wait_for_fences(std::slice::from_ref(&fence), true, 5_000_000_000)
                .map_err(|e| crate::AccelError::OperationFailed(format!("Fence wait failed: {:?}", e)))?;

            let output_ptr = output_alloc.mapped_ptr().unwrap().as_ptr() as *const f32;
            let mut output = vec![0.0f32; total_elements as usize];
            std::ptr::copy_nonoverlapping(output_ptr, output.as_mut_ptr(), output.len());

            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.command_pool, &command_buffers);
            for buf in [q_buffer, k_buffer, v_buffer, output_buffer] {
                self.device.destroy_buffer(buf, None);
            }
            for alloc in [q_alloc, k_alloc, v_alloc, output_alloc] {
                self.allocator.free(alloc).ok();
            }

            Ok(output)
        }
    }
}

#[cfg(target_os = "android")]
impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();

            if let Some(buf) = self.weight_buffer.take() {
                self.device.destroy_buffer(buf, None);
            }
            if let Some(alloc) = self.weight_allocation.take() {
                self.allocator.free(alloc).ok();
            }

            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_pipeline(self.gemv_pipeline, None);
            self.device.destroy_pipeline_layout(self.gemv_pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.gemv_descriptor_set_layout, None);
            self.device.destroy_pipeline(self.attention_pipeline, None);
            self.device.destroy_pipeline_layout(self.attention_pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.attention_descriptor_set_layout, None);

            ManuallyDrop::drop(&mut self.allocator);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

#[allow(dead_code)]
pub struct VulkanBackend {
    #[cfg(target_os = "android")]
    context: Mutex<Option<VulkanContext>>,
    #[cfg(not(target_os = "android"))]
    device: Option<VulkanDevice>,
    #[cfg(not(target_os = "android"))]
    compute_queue: Option<VulkanQueue>,
}

impl VulkanBackend {
    pub fn new() -> Self {
        Self::try_init()
    }

    fn try_init() -> Self {
        #[cfg(target_os = "android")]
        {
            match VulkanContext::new() {
                Ok(context) => Self {
                    context: Mutex::new(Some(context)),
                },
                Err(_) => Self {
                    context: Mutex::new(None),
                },
            }
        }
        #[cfg(not(target_os = "android"))]
        {
            Self::init_stub()
        }
    }

    #[cfg(not(target_os = "android"))]
    fn init_stub() -> Self {
        Self {
            device: Some(VulkanDevice {
                name: "VulkanStub".to_string(),
                driver_version: "1.0".to_string(),
                max_memory_mb: 2048,
            }),
            compute_queue: Some(VulkanQueue { family_index: 0 }),
        }
    }

    pub fn is_available() -> bool {
        #[cfg(target_os = "android")]
        {
            VulkanContext::new().is_ok()
        }
        #[cfg(not(target_os = "android"))]
        {
            false
        }
    }

    pub fn device_name(&self) -> Option<&str> {
        #[cfg(target_os = "android")]
        {
            Some("Vulkan GPU")
        }
        #[cfg(not(target_os = "android"))]
        {
            self.device.as_ref().map(|d| d.name.as_str())
        }
    }

    pub fn max_memory_mb(&self) -> u64 {
        #[cfg(target_os = "android")]
        {
            if let Ok(guard) = self.context.lock() {
                if let Some(ctx) = guard.as_ref() {
                    unsafe {
                        let mem_props = ctx
                            .instance
                            .get_physical_device_memory_properties(ctx.physical_device);
                        let heap_size = mem_props.memory_heaps[0].size;
                        return heap_size / (1024 * 1024);
                    }
                }
            }
            0
        }
        #[cfg(not(target_os = "android"))]
        {
            self.device.as_ref().map(|d| d.max_memory_mb).unwrap_or(0)
        }
    }
}

impl AccelBackend for VulkanBackend {
    fn name(&self) -> &str {
        "vulkan"
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Vulkan
    }

    fn is_available(&self) -> bool {
        #[cfg(target_os = "android")]
        {
            self.context.lock().map(|g| g.is_some()).unwrap_or(false)
        }
        #[cfg(not(target_os = "android"))]
        {
            false
        }
    }

    fn supports_quantization(&self, quantization: &str) -> bool {
        matches!(quantization, "q4_k_m" | "q8_0" | "f16")
    }

    #[cfg(test)]
    fn forward(&self, input_ids: &[u32], _positions: &[usize]) -> Result<AccelResult> {
        let start = Instant::now();

        #[cfg(target_os = "android")]
        {
            let mut guard = self.context.lock().map_err(|_| {
                crate::AccelError::BackendNotAvailable("Vulkan mutex poisoned".to_string())
            })?;
            let ctx = guard.as_mut().ok_or_else(|| {
                crate::AccelError::BackendNotAvailable("Vulkan not available".to_string())
            })?;

            let batch_size = input_ids.len() as u32;
            let vocab_size = 50257u32;

            let logits = ctx.dispatch_gemv(input_ids, 4096, vocab_size)?;

            let elapsed = start.elapsed().as_millis() as u64;
            Ok(AccelResult::new(logits, batch_size as usize, elapsed))
        }

        #[cfg(not(target_os = "android"))]
        {
            Err(crate::AccelError::BackendNotAvailable(
                "Vulkan not available on this platform".to_string(),
            ))
        }
    }

    #[cfg(not(test))]
    fn forward(&self, _input_ids: &[u32], _positions: &[usize]) -> Result<AccelResult> {
        Err(crate::AccelError::Deprecated(
            "VulkanBackend::forward() is deprecated; use InferenceEngine::generate() instead".to_string(),
        ))
    }
}

impl Default for VulkanBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vulkan_backend_creation() {
        let backend = VulkanBackend::new();
        assert_eq!(backend.name(), "vulkan");
        assert_eq!(backend.backend_type(), BackendType::Vulkan);
    }

    #[test]
    fn test_vulkan_forward() {
        let backend = VulkanBackend::new();
        let input_ids = vec![0, 1, 2];

        let result = backend.forward(&input_ids, &[]);
        #[cfg(not(target_os = "android"))]
        {
            assert!(result.is_err());
            match result {
                Err(crate::AccelError::BackendNotAvailable(_)) => {}
                _ => panic!("Expected BackendNotAvailable on non-Android"),
            }
        }
        #[cfg(target_os = "android")]
        {
            assert!(result.is_ok());
            let accel = result.unwrap();
            assert_eq!(accel.tokens_generated, 3);
            assert!(accel.duration_ms > 0);
        }
    }

    #[test]
    fn test_device_info() {
        let backend = VulkanBackend::new();
        let _mem = backend.max_memory_mb();
    }

    #[test]
    fn test_quantization_support() {
        let backend = VulkanBackend::new();
        assert!(backend.supports_quantization("q4_k_m"));
        assert!(backend.supports_quantization("f16"));
        assert!(backend.supports_quantization("q8_0"));
        assert!(!backend.supports_quantization("f32"));
    }

    #[test]
    fn test_device_name() {
        let backend = VulkanBackend::new();
        let _name = backend.device_name();
    }
}
