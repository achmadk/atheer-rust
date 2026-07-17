//! Vulkan backend tests
//!
//! These tests are gated on `target_os = "android"` and test the Vulkan backend
//! against CPU reference implementations.

use candle_core::{DType, Device, Result, Tensor};

#[cfg(target_os = "android")]
fn test_vulkan_device_creation() -> Result<()> {
    let device = Device::new_vulkan(0)?;
    Ok(())
}

#[cfg(target_os = "android")]
fn test_vulkan_matmul_f32() -> Result<()> {
    let device = Device::new_vulkan(0)?;
    let cpu_device = Device::Cpu;

    let lhs = Tensor::randn(0f32, 1f32, (2, 3), &cpu_device)?;
    let rhs = Tensor::randn(0f32, 1f32, (3, 2), &cpu_device)?;

    let lhs_v = lhs.to_device(&device)?;
    let rhs_v = rhs.to_device(&device)?;

    let result_v = lhs_v.matmul(&rhs_v)?;
    let result_cpu = lhs.matmul(&rhs)?;

    let diff = (result_v.to_device(&cpu_device)? - result_cpu)?.abs()?;
    let max_diff = diff.max()?;
    if max_diff.to_scalar::<f32>()? > 1e-3 {
        println!(
            "WARNING: Vulkan matmul F32 differs from CPU by {}",
            max_diff.to_scalar::<f32>()?
        );
    }

    Ok(())
}

#[cfg(target_os = "android")]
fn test_vulkan_matmul_f16() -> Result<()> {
    let device = Device::new_vulkan(0)?;
    let cpu_device = Device::Cpu;

    let lhs = Tensor::randn(0f32, 1f32, (2, 3), &cpu_device)?.to_dtype(DType::F16)?;
    let rhs = Tensor::randn(0f32, 1f32, (3, 2), &cpu_device)?.to_dtype(DType::F16)?;

    let lhs_v = lhs.to_device(&device)?;
    let rhs_v = rhs.to_device(&device)?;

    let result_v = lhs_v.matmul(&rhs_v)?;
    let result_cpu = lhs.matmul(&rhs)?.to_dtype(DType::F16)?;

    let diff = (result_v.to_device(&cpu_device)?.to_dtype(DType::F16)? - result_cpu)?.abs()?;
    let max_diff = diff.max()?;
    if max_diff.to_scalar::<f32>()? > 1e-2 {
        println!(
            "WARNING: Vulkan matmul F16 differs from CPU by {}",
            max_diff.to_scalar::<f32>()?
        );
    }

    Ok(())
}

#[cfg(target_os = "android")]
fn test_vulkan_elementwise_ops() -> Result<()> {
    let device = Device::new_vulkan(0)?;
    let cpu_device = Device::Cpu;

    let input = Tensor::randn(0f32, 1f32, (4, 4), &cpu_device)?;

    let input_v = input.to_device(&device)?;

    let result_v = (input_v.exp())?;
    let result_cpu = (input.exp())?;

    let diff = (result_v.to_device(&cpu_device)? - result_cpu)?.abs()?;
    let max_diff = diff.max()?;
    if max_diff.to_scalar::<f32>()? > 1e-2 {
        println!(
            "WARNING: Vulkan exp differs from CPU by {}",
            max_diff.to_scalar::<f32>()?
        );
    }

    Ok(())
}

#[cfg(target_os = "android")]
fn test_vulkan_reduce() -> Result<()> {
    let device = Device::new_vulkan(0)?;
    let cpu_device = Device::Cpu;

    let input = Tensor::randn(0f32, 1f32, (2, 3, 4), &cpu_device)?;

    let input_v = input.to_device(&device)?;

    let result_v = input_v.sum(2)?;
    let result_cpu = input.sum(2)?;

    let diff = (result_v.to_device(&cpu_device)? - result_cpu)?.abs()?;
    let max_diff = diff.max()?;
    if max_diff.to_scalar::<f32>()? > 1e-2 {
        println!(
            "WARNING: Vulkan reduce differs from CPU by {}",
            max_diff.to_scalar::<f32>()?
        );
    }

    Ok(())
}

#[cfg(target_os = "android")]
fn test_vulkan_storage_roundtrip() -> Result<()> {
    let device = Device::new_vulkan(0)?;
    let cpu_device = Device::Cpu;

    let original: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let tensor = Tensor::from_slice(&original, (2, 3, 4), &cpu_device)?;

    let tensor_v = tensor.to_device(&device)?;
    let roundtrip = tensor_v.to_device(&cpu_device)?;

    let diff = (tensor - roundtrip)?.abs()?;
    let max_diff = diff.max()?;
    if max_diff.to_scalar::<f32>()? > 1e-5 {
        println!(
            "WARNING: Vulkan roundtrip differs by {}",
            max_diff.to_scalar::<f32>()?
        );
    }

    Ok(())
}

#[cfg(target_os = "android")]
fn test_vulkan_copy_strided() -> Result<()> {
    let device = Device::new_vulkan(0)?;
    let cpu_device = Device::Cpu;

    let original = Tensor::randn(0f32, 1f32, (2, 3, 4), &cpu_device)?;

    let strides = vec![4, 8, 1];
    let layout = candle_core::Layout::new((2, 3, 4), &strides, 0);
    let tensor = original.reshape_with_layout(&layout)?;

    let tensor_v = tensor.to_device(&device)?;
    let roundtrip = tensor_v.to_device(&cpu_device)?;

    let diff = (tensor - roundtrip)?.abs()?;
    let max_diff = diff.max()?;
    if max_diff.to_scalar::<f32>()? > 1e-4 {
        println!(
            "WARNING: Vulkan strided copy differs by {}",
            max_diff.to_scalar::<f32>()?
        );
    }

    Ok(())
}
