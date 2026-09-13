//! Vulkan backend tests
//!
//! These tests are gated on `target_os = "android"` and verify the wgpu Vulkan
//! backend against CPU reference implementations. GEMM tests assert parity
//! (including the transposed-RHS case that guards stride correctness). Full GPU
//! execution requires real Android hardware; on emulators without a real Vulkan
//! device, `Device::new_vulkan(0)` may fail and the test returns early.

#![cfg(target_os = "android")]
#![allow(unused_imports)]

use candle_core::{DType, Device, Result, Tensor};

/// Try to construct the Vulkan device; if unavailable (e.g. no real GPU on an
/// emulator), return `None` so the test can skip rather than falsely fail.
fn try_vulkan() -> Option<Device> {
    Device::new_vulkan(0).ok()
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let diff = (a - b)?.abs()?.flatten_all()?.max(0)?;
    diff.to_scalar::<f32>()
}

#[test]
fn test_vulkan_device_creation() -> Result<()> {
    // Only asserts that construction does not panic; absence of a device is ok.
    let _ = try_vulkan();
    Ok(())
}

#[test]
fn test_vulkan_matmul_f32_contiguous() -> Result<()> {
    let Some(device) = try_vulkan() else {
        return Ok(());
    };
    let cpu = Device::Cpu;

    let lhs = Tensor::randn(0f32, 1f32, (4, 5), &cpu)?;
    let rhs = Tensor::randn(0f32, 1f32, (5, 3), &cpu)?;

    let result_v = lhs.to_device(&device)?.matmul(&rhs.to_device(&device)?)?;
    let result_cpu = lhs.matmul(&rhs)?;

    let max_diff = max_abs_diff(&result_v.to_device(&cpu)?, &result_cpu)?;
    assert!(
        max_diff <= 1e-3,
        "Vulkan F32 matmul differs from CPU by {max_diff}"
    );
    Ok(())
}

#[test]
fn test_vulkan_matmul_f32_transposed_rhs() -> Result<()> {
    // Regression guard: linear layers do `x.matmul(&w.t())`, producing a
    // transposed (non-contiguous) RHS layout. The GEMM shader must honor rhs
    // strides/offset — the old scalar loop and naive shader got this wrong.
    let Some(device) = try_vulkan() else {
        return Ok(());
    };
    let cpu = Device::Cpu;

    let x = Tensor::randn(0f32, 1f32, (4, 6), &cpu)?;
    // Weight stored [out=3, in=6]; matmul as x @ w.t() -> [4, 3].
    let w = Tensor::randn(0f32, 1f32, (3, 6), &cpu)?;

    let result_v = x.to_device(&device)?.matmul(&w.to_device(&device)?.t()?)?;
    let result_cpu = x.matmul(&w.t()?)?;

    let max_diff = max_abs_diff(&result_v.to_device(&cpu)?, &result_cpu)?;
    assert!(
        max_diff <= 1e-3,
        "Vulkan F32 transposed-RHS matmul differs from CPU by {max_diff}"
    );
    Ok(())
}

#[test]
fn test_vulkan_matmul_f32_batched() -> Result<()> {
    let Some(device) = try_vulkan() else {
        return Ok(());
    };
    let cpu = Device::Cpu;

    let lhs = Tensor::randn(0f32, 1f32, (3, 4, 5), &cpu)?;
    let rhs = Tensor::randn(0f32, 1f32, (3, 5, 2), &cpu)?;

    let result_v = lhs.to_device(&device)?.matmul(&rhs.to_device(&device)?)?;
    let result_cpu = lhs.matmul(&rhs)?;

    let max_diff = max_abs_diff(&result_v.to_device(&cpu)?, &result_cpu)?;
    assert!(
        max_diff <= 1e-3,
        "Vulkan F32 batched matmul differs from CPU by {max_diff}"
    );
    Ok(())
}

#[test]
fn test_vulkan_matmul_f16() -> Result<()> {
    let Some(device) = try_vulkan() else {
        return Ok(());
    };
    let cpu = Device::Cpu;

    let lhs = Tensor::randn(0f32, 1f32, (4, 5), &cpu)?.to_dtype(DType::F16)?;
    let rhs = Tensor::randn(0f32, 1f32, (5, 3), &cpu)?.to_dtype(DType::F16)?;

    let result_v = lhs.to_device(&device)?.matmul(&rhs.to_device(&device)?)?;
    let result_cpu = lhs.matmul(&rhs)?;

    let diff_t =
        (result_v.to_device(&cpu)?.to_dtype(DType::F32)? - result_cpu.to_dtype(DType::F32)?)?;
    let max_diff = diff_t.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    assert!(
        max_diff <= 5e-2,
        "Vulkan F16 matmul differs from CPU by {max_diff}"
    );
    Ok(())
}

#[test]
fn test_vulkan_matmul_unsupported_dtype_errors() -> Result<()> {
    // Integer matmul is not supported by the GPU GEMM path; it must error rather
    // than silently fall back to a CPU computation.
    let Some(device) = try_vulkan() else {
        return Ok(());
    };
    let cpu = Device::Cpu;

    let lhs = Tensor::ones((4, 5), DType::U32, &cpu)?;
    let rhs = Tensor::ones((5, 3), DType::U32, &cpu)?;

    let res = lhs
        .to_device(&device)
        .and_then(|l| rhs.to_device(&device).map(|r| (l, r)))
        .and_then(|(l, r)| l.matmul(&r));
    assert!(
        res.is_err(),
        "Vulkan matmul on U32 should error, not fall back to CPU"
    );
    Ok(())
}

#[test]
fn test_vulkan_elementwise_ops() -> Result<()> {
    let Some(device) = try_vulkan() else {
        return Ok(());
    };
    let cpu = Device::Cpu;

    let input = Tensor::randn(0f32, 1f32, (4, 4), &cpu)?;
    let result_v = input.to_device(&device)?.exp()?;
    let result_cpu = input.exp()?;

    let max_diff = max_abs_diff(&result_v.to_device(&cpu)?, &result_cpu)?;
    assert!(
        max_diff <= 1e-2,
        "Vulkan exp differs from CPU by {max_diff}"
    );
    Ok(())
}

#[test]
fn test_vulkan_storage_roundtrip() -> Result<()> {
    let Some(device) = try_vulkan() else {
        return Ok(());
    };
    let cpu = Device::Cpu;

    let original: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let tensor = Tensor::from_slice(&original, (2, 3, 4), &cpu)?;

    let roundtrip = tensor.to_device(&device)?.to_device(&cpu)?;
    let max_diff = max_abs_diff(&tensor, &roundtrip)?;
    assert!(max_diff <= 1e-5, "Vulkan roundtrip differs by {max_diff}");
    Ok(())
}
