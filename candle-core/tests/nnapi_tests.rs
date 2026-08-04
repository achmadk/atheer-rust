//! NNAPI backend tests
//!
//! These tests are gated on `all(feature = "nnapi", target_os = "android")`
//! and test the NNAPI backend against CPU reference implementations.
//!
//! Each test requires a physical Android device with an NNAPI driver, so all are
//! marked `#[ignore]`. Run them on-device with `cargo test -- --ignored`.

#[cfg(all(feature = "nnapi", target_os = "android"))]
use candle_core::{Device, Result, Tensor};

#[cfg(all(feature = "nnapi", target_os = "android"))]
#[test]
#[ignore = "requires an Android device with an NNAPI driver"]
fn test_nnapi_device_creation() -> Result<()> {
    let device = Device::new_nnapi(0)?;
    assert!(device.is_nnapi());
    Ok(())
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
#[test]
#[ignore = "requires an Android device with an NNAPI driver"]
fn test_nnapi_add_f32() -> Result<()> {
    let device = Device::new_nnapi(0)?;
    let cpu_device = Device::Cpu;

    let lhs = Tensor::randn(0f32, 1.0, (2, 3), &cpu_device)?;
    let rhs = Tensor::randn(0f32, 1.0, (2, 3), &cpu_device)?;

    let lhs_n = lhs.to_device(&device)?;
    let rhs_n = rhs.to_device(&device)?;

    let result_n = (&lhs_n + &rhs_n)?;

    let result_cpu = &lhs + &rhs;

    let diff = (result_n.to_device(&cpu_device)? - result_cpu)?.abs()?;
    let max_diff = diff.max_all()?.to_scalar::<f32>()?;
    assert!(
        max_diff < 1e-5,
        "NNAPI add differs from CPU by {}",
        max_diff
    );

    Ok(())
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
#[test]
#[ignore = "requires an Android device with an NNAPI driver"]
fn test_nnapi_mul_f32() -> Result<()> {
    let device = Device::new_nnapi(0)?;
    let cpu_device = Device::Cpu;

    let lhs = Tensor::randn(0f32, 1.0, (2, 3), &cpu_device)?;
    let rhs = Tensor::randn(0f32, 1.0, (2, 3), &cpu_device)?;

    let lhs_n = lhs.to_device(&device)?;
    let rhs_n = rhs.to_device(&device)?;

    let result_n = (&lhs_n * &rhs_n)?;

    let result_cpu = &lhs * &rhs;

    let diff = (result_n.to_device(&cpu_device)? - result_cpu)?.abs()?;
    let max_diff = diff.max_all()?.to_scalar::<f32>()?;
    assert!(
        max_diff < 1e-5,
        "NNAPI mul differs from CPU by {}",
        max_diff
    );

    Ok(())
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
#[test]
#[ignore = "requires an Android device with an NNAPI driver"]
fn test_nnapi_matmul_f32() -> Result<()> {
    let device = Device::new_nnapi(0)?;
    let cpu_device = Device::Cpu;

    let lhs = Tensor::randn(0f32, 1.0, (2, 3), &cpu_device)?;
    let rhs = Tensor::randn(0f32, 1.0, (3, 2), &cpu_device)?;

    let lhs_n = lhs.to_device(&device)?;
    let rhs_n = rhs.to_device(&device)?;

    let result_n = lhs_n.matmul(&rhs_n)?;

    let result_cpu = lhs.matmul(&rhs)?;

    let diff = (result_n.to_device(&cpu_device)? - result_cpu)?.abs()?;
    let max_diff = diff.max_all()?.to_scalar::<f32>()?;
    assert!(
        max_diff < 1e-5,
        "NNAPI matmul differs from CPU by {}",
        max_diff
    );

    Ok(())
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
#[test]
#[ignore = "requires an Android device with an NNAPI driver"]
fn test_nnapi_relu() -> Result<()> {
    let device = Device::new_nnapi(0)?;
    let cpu_device = Device::Cpu;

    let lhs = Tensor::randn(-1f32, 1.0, (2, 3), &cpu_device)?;

    let lhs_n = lhs.to_device(&device)?;

    let result_n = lhs_n.relu()?;

    let result_cpu = lhs.relu()?;

    let diff = (result_n.to_device(&cpu_device)? - result_cpu)?.abs()?;
    let max_diff = diff.max_all()?.to_scalar::<f32>()?;
    assert!(
        max_diff < 1e-5,
        "NNAPI relu differs from CPU by {}",
        max_diff
    );

    Ok(())
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
#[test]
#[ignore = "requires an Android device with an NNAPI driver"]
fn test_nnapi_conv2d() -> Result<()> {
    let device = Device::new_nnapi(0)?;
    let cpu_device = Device::Cpu;

    let input = Tensor::randn(0f32, 1.0, (1, 3, 8, 8), &cpu_device)?;
    let kernel = Tensor::randn(0f32, 1.0, (16, 3, 3, 3), &cpu_device)?;

    let input_n = input.to_device(&device)?;
    let kernel_n = kernel.to_device(&device)?;

    let result_n = input_n.conv2d(&kernel_n, 1, 1, 1, 1)?;

    let result_cpu = input.conv2d(&kernel, 1, 1, 1, 1)?;

    let diff = (result_n.to_device(&cpu_device)? - result_cpu)?.abs()?;
    let max_diff = diff.max_all()?.to_scalar::<f32>()?;
    assert!(
        max_diff < 1e-4,
        "NNAPI conv2d differs from CPU by {}",
        max_diff
    );

    Ok(())
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
#[test]
#[ignore = "requires an Android device with an NNAPI driver"]
fn test_nnapi_to_vec() -> Result<()> {
    let device = Device::new_nnapi(0)?;
    let cpu_device = Device::Cpu;

    let tensor = Tensor::randn(0f32, 1.0, (2, 3), &cpu_device)?;
    let tensor_n = tensor.to_device(&device)?;

    let vec_n: Vec<Vec<f32>> = tensor_n.to_vec2()?;
    let vec_cpu: Vec<Vec<f32>> = tensor.to_vec2()?;

    assert_eq!(vec_n.len(), vec_cpu.len());
    assert_eq!(vec_n[0].len(), vec_cpu[0].len());

    for (row_n, row_cpu) in vec_n.iter().zip(vec_cpu.iter()) {
        for (&v_n, &v_cpu) in row_n.iter().zip(row_cpu.iter()) {
            assert!((v_n - v_cpu).abs() < 1e-5);
        }
    }

    Ok(())
}
