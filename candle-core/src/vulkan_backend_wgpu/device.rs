#![allow(dead_code)]

use crate::backend::BackendDevice;
use crate::{CpuStorage, DType, DeviceLocation, Result, Shape, VulkanError, VulkanStorage};
use std::sync::Arc;

#[cfg(all(feature = "vulkan", target_os = "android"))]
pub struct VulkanDevice {
    ordinal: usize,
    inner: Arc<VulkanDeviceInner>,
}

#[cfg(all(feature = "vulkan", target_os = "android"))]
struct VulkanDeviceInner {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    limits: wgpu::Limits,
    location: DeviceLocation,
}

#[cfg(not(all(feature = "vulkan", target_os = "android")))]
pub struct VulkanDevice {
    ordinal: usize,
}

#[cfg(not(all(feature = "vulkan", target_os = "android")))]
impl VulkanDevice {
    pub fn new(_ordinal: usize) -> Result<Self> {
        Err(crate::Error::NotCompiledWithVulkanSupport)
    }
}

#[cfg(all(feature = "vulkan", target_os = "android"))]
impl VulkanDevice {
    pub fn new(ordinal: usize) -> Result<Self> {
        if ordinal != 0 {
            crate::bail!("Wgpu only supports ordinal 0 on Android")
        }
        Self::new_internal(ordinal)
    }

    fn new_internal(ordinal: usize) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .map_err(|e| {
            crate::Error::Vulkan(VulkanError::Message(format!(
                "Failed to request Vulkan adapter: {}",
                e
            )))
        })?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("candle-wgpu-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            experimental_features: Default::default(),
            memory_hints: Default::default(),
            trace: Default::default(),
        }))
        .map_err(|e| {
            crate::Error::Vulkan(VulkanError::Message(format!(
                "Failed to request device: {}",
                e
            )))
        })?;

        let limits = device.limits();

        let location = DeviceLocation::Vulkan { gpu_id: ordinal };

        Ok(Self {
            ordinal,
            inner: Arc::new(VulkanDeviceInner {
                instance,
                adapter,
                device,
                queue,
                limits,
                location,
            }),
        })
    }

    pub(crate) fn allocate_buffer(
        &self,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> Result<wgpu::Buffer> {
        let buffer = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-buffer"),
            size,
            usage,
            mapped_at_creation: false,
        });
        Ok(buffer)
    }

    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.inner.device
    }

    pub(crate) fn queue(&self) -> &wgpu::Queue {
        &self.inner.queue
    }

    pub(crate) fn limits(&self) -> &wgpu::Limits {
        &self.inner.limits
    }

    pub(crate) fn write_buffer(
        &self,
        buffer: &wgpu::Buffer,
        offset: u64,
        data: &[u8],
    ) -> Result<()> {
        self.inner.queue.write_buffer(buffer, offset, data);
        Ok(())
    }

    pub fn new_buffer_with_data(&self, data: &[u8]) -> Result<wgpu::Buffer> {
        let buffer = self.allocate_buffer(
            data.len() as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )?;
        self.queue().write_buffer(&buffer, 0, data);
        Ok(buffer)
    }

    pub(crate) fn sync(&self) -> Result<()> {
        self.inner.queue.on_submitted_work_done(|| {});
        Ok(())
    }

    pub(crate) fn max_buffer_size(&self) -> u64 {
        self.inner.limits.max_buffer_size
    }

    pub(crate) fn max_storage_buffer_binding_size(&self) -> u64 {
        self.inner.limits.max_storage_buffer_binding_size as u64
    }

    pub(crate) fn compile_shader(&self, wgsl: &str) -> Result<wgpu::ShaderModule> {
        let shader = self
            .inner
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("wgpu-compute-shader"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
        Ok(shader)
    }

    pub(crate) fn dispatch_binary_op(
        &self,
        shader_src: &str,
        lhs: &wgpu::Buffer,
        rhs: &wgpu::Buffer,
        output: &wgpu::Buffer,
        count: u32,
    ) -> Result<()> {
        let shader = self.compile_shader(shader_src)?;

        let bind_group_layout =
            self.inner
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("binary-op-bind-group"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout =
            self.inner
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("binary-op-pipeline"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        let pipeline =
            self.inner
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("binary-op"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        let bind_group = self
            .inner
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("binary-op-bind-group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: lhs,
                            offset: 0,
                            size: None,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: rhs,
                            offset: 0,
                            size: None,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: output,
                            offset: 0,
                            size: None,
                        }),
                    },
                ],
            });

        let mut encoder =
            self.inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("binary-op-encoder"),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("binary-op-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((count + 255) / 256, 1, 1);
        }

        self.inner.queue.submit(Some(encoder.finish()));
        self.sync()?;

        Ok(())
    }

    pub(crate) fn dispatch_affine(
        &self,
        shader_src: &str,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        count: u32,
        mul: f32,
        add: f32,
    ) -> Result<()> {
        let shader = self.compile_shader(shader_src)?;

        let bind_group_layout =
            self.inner
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("affine-bind-group"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let uniform_buffer = self.allocate_buffer(
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        )?;
        let mut uniform_data = vec![0u8; 16];
        uniform_data[0..4].copy_from_slice(&mul.to_le_bytes());
        uniform_data[4..8].copy_from_slice(&add.to_le_bytes());
        self.queue().write_buffer(&uniform_buffer, 0, &uniform_data);

        let pipeline_layout =
            self.inner
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("affine-pipeline"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        let pipeline =
            self.inner
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("affine"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        let bind_group = self
            .inner
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("affine-bind-group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: input,
                            offset: 0,
                            size: None,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: output,
                            offset: 0,
                            size: None,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &uniform_buffer,
                            offset: 0,
                            size: None,
                        }),
                    },
                ],
            });

        let mut encoder =
            self.inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("affine-encoder"),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("affine-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((count + 255) / 256, 1, 1);
        }

        self.inner.queue.submit(Some(encoder.finish()));
        self.sync()?;

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_gemm(
        &self,
        shader_src: &str,
        lhs: &wgpu::Buffer,
        rhs: &wgpu::Buffer,
        output: &wgpu::Buffer,
        meta: GemmMeta,
    ) -> Result<()> {
        let shader = self.compile_shader(shader_src)?;

        let bind_group_layout =
            self.inner
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("gemm-bind-group"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        // Uniform buffer: 15 u32 fields = 60 bytes; WGSL uniform structs require
        // a 16-byte-aligned size, so allocate 64 bytes (one word of padding).
        let meta_words = meta.to_words();
        let mut meta_bytes = vec![0u8; 64];
        for (i, w) in meta_words.iter().enumerate() {
            meta_bytes[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        let uniform_buffer = self.allocate_buffer(
            meta_bytes.len() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        )?;
        self.queue().write_buffer(&uniform_buffer, 0, &meta_bytes);

        let pipeline_layout =
            self.inner
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("gemm-pipeline"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        let pipeline =
            self.inner
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("gemm"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        let bind_group = self
            .inner
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gemm-bind-group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: lhs,
                            offset: 0,
                            size: None,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: rhs,
                            offset: 0,
                            size: None,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: output,
                            offset: 0,
                            size: None,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &uniform_buffer,
                            offset: 0,
                            size: None,
                        }),
                    },
                ],
            });

        let mut encoder =
            self.inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("gemm-encoder"),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gemm-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // Workgroup is 8x8 over (m, n); z axis is batch.
            let wg_m = (meta.m + 7) / 8;
            let wg_n = (meta.n + 7) / 8;
            pass.dispatch_workgroups(wg_m, wg_n, meta.batch);
        }

        self.inner.queue.submit(Some(encoder.finish()));
        self.sync()?;

        Ok(())
    }
}

/// Metadata passed to the GEMM compute shader as a uniform buffer.
///
/// Field order MUST match the `Metadata` struct in `shaders::GEMM_SHADER`.
/// Stride/offset semantics mirror candle's CPU `MatMul` (batch skip strides plus
/// row/col strides per operand, contiguous destination), so transposed and
/// offset operand layouts compute correctly.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GemmMeta {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub batch: u32,
    pub lhs_offset: u32,
    pub rhs_offset: u32,
    pub dst_offset: u32,
    pub lhs_batch_stride: u32,
    pub rhs_batch_stride: u32,
    pub lhs_stride_m: u32,
    pub lhs_stride_k: u32,
    pub rhs_stride_k: u32,
    pub rhs_stride_n: u32,
    pub dst_stride_m: u32,
    pub dst_stride_n: u32,
}

impl GemmMeta {
    fn to_words(self) -> [u32; 15] {
        [
            self.m,
            self.n,
            self.k,
            self.batch,
            self.lhs_offset,
            self.rhs_offset,
            self.dst_offset,
            self.lhs_batch_stride,
            self.rhs_batch_stride,
            self.lhs_stride_m,
            self.lhs_stride_k,
            self.rhs_stride_k,
            self.rhs_stride_n,
            self.dst_stride_m,
            self.dst_stride_n,
        ]
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
        self.inner.location
    }

    fn same_device(&self, other: &Self) -> bool {
        self.ordinal == other.ordinal
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let size = elem_count * dtype.size_in_bytes();
        let buffer = self.allocate_buffer(
            size as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )?;

        let zero_bytes = vec![0u8; size as usize];
        self.queue().write_buffer(&buffer, 0, &zero_bytes);

        Ok(VulkanStorage::new(buffer, elem_count, dtype, self.clone()))
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let size = elem_count * dtype.size_in_bytes();
        let buffer = self.allocate_buffer(
            size as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )?;
        Ok(VulkanStorage::new(buffer, elem_count, dtype, self.clone()))
    }

    fn storage_from_slice<T: crate::WithDType>(&self, data: &[T]) -> Result<Self::Storage> {
        let dtype = T::DTYPE;
        let elem_count = data.len();
        let size = elem_count * dtype.size_in_bytes();
        let buffer = self.allocate_buffer(
            size as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )?;

        let bytes: Vec<u8> =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, size).to_vec() };

        self.queue().write_buffer(&buffer, 0, &bytes);

        Ok(VulkanStorage::new(buffer, elem_count, dtype, self.clone()))
    }

    fn storage_from_cpu_storage(&self, cpu_storage: &CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage_owned(cpu_storage.clone())
    }

    fn storage_from_cpu_storage_owned(&self, cpu_storage: CpuStorage) -> Result<Self::Storage> {
        let (data, dtype, elem_count) = match cpu_storage {
            CpuStorage::U8(v) => {
                let len = v.len();
                (v, DType::U8, len)
            }
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
        let buffer = self.allocate_buffer(
            size as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )?;
        self.queue().write_buffer(&buffer, 0, &data);

        Ok(VulkanStorage::new(buffer, elem_count, dtype, self.clone()))
    }

    fn rand_uniform(
        &self,
        _shape: &Shape,
        _dtype: DType,
        _lo: f64,
        _up: f64,
    ) -> Result<Self::Storage> {
        Err(crate::Error::Vulkan(VulkanError::Message(
            "rand_uniform not implemented for wgpu".to_string(),
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
            "rand_normal not implemented for wgpu".to_string(),
        )))
    }

    fn set_seed(&self, _seed: u64) -> Result<()> {
        Err(crate::Error::Vulkan(VulkanError::Message(
            "set_seed not implemented for wgpu".to_string(),
        )))
    }

    fn get_current_seed(&self) -> Result<u64> {
        Err(crate::Error::Vulkan(VulkanError::Message(
            "get_current_seed not implemented for wgpu".to_string(),
        )))
    }

    fn synchronize(&self) -> Result<()> {
        self.queue().on_submitted_work_done(|| {});
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
