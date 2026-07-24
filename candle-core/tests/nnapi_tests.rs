//! NNAPI backend tests
//!
//! These tests are gated on `all(feature = "nnapi", target_os = "android")`
//! and test the NNAPI backend against CPU reference implementations.

#[cfg(all(feature = "nnapi", target_os = "android"))]
use candle_core::{Device, Result, Tensor};

#[cfg(all(feature = "nnapi", target_os = "android"))]
fn test_nnapi_device_creation() -> Result<()> {
    let device = Device::new_nnapi(0)?;
    assert!(device.is_nnapi());
    Ok(())
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
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
fn test_nnapi_relu() -> Result<()> {
    let device = Device::new_nnapi(0)?;
    let cpu_device = Device::Cpu;

    let lhs = Tensor::randn(-1f32, 1.0, (2, 3), &cpu_device)?;

    let lhs_n = lhs.to_device(&device)?;

    let result_n = candle_core::ops::relu(&lhs_n)?;

    let result_cpu = candle_core::ops::relu(&lhs)?;

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
fn test_nnapi_softmax() -> Result<()> {
    let device = Device::new_nnapi(0)?;
    let cpu_device = Device::Cpu;

    let lhs = Tensor::randn(0f32, 1.0, (1, 4), &cpu_device)?;

    let lhs_n = lhs.to_device(&device)?;

    let result_n = candle_core::ops::softmax(&lhs_n, 1)?;
    let result_cpu = candle_core::ops::softmax(&lhs, 1)?;

    let diff = (result_n.to_device(&cpu_device)? - result_cpu)?.abs()?;
    let max_diff = diff.max_all()?.to_scalar::<f32>()?;
    assert!(
        max_diff < 1e-4,
        "NNAPI softmax differs from CPU by {}",
        max_diff
    );

    Ok(())
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
fn test_nnapi_conv2d() -> Result<()> {
    let device = Device::new_nnapi(0)?;
    let cpu_device = Device::Cpu;

    let input = Tensor::randn(0f32, 1.0, (1, 3, 8, 8), &cpu_device)?;
    let kernel = Tensor::randn(0f32, 1.0, (16, 3, 3, 3), &cpu_device)?;

    let input_n = input.to_device(&device)?;
    let kernel_n = kernel.to_device(&device)?;

    let result_n = input_n.conv2d(&kernel_n, 1, 1)?;

    let result_cpu = input.conv2d(&kernel, 1, 1)?;

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
fn test_nnapi_to_vec() -> Result<()> {
    let device = Device::new_nnapi(0)?;
    let cpu_device = Device::Cpu;

    let tensor = Tensor::randn(0f32, 1.0, (2, 3), &cpu_device)?;
    let tensor_n = tensor.to_device(&device)?;

    let vec_n: Vec<Vec<f32>> = tensor_n.to_vec2()?;
    let vec_cpu = tensor.to_vec2()?;

    assert_eq!(vec_n.len(), vec_cpu.len());
    assert_eq!(vec_n[0].len(), vec_cpu[0].len());

    for (row_n, row_cpu) in vec_n.iter().zip(vec_cpu.iter()) {
        for (&v_n, &v_cpu) in row_n.iter().zip(row_cpu.iter()) {
            assert!((v_n - v_cpu).abs() < 1e-5);
        }
    }

    Ok(())
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
fn test_nnapi_storage_allocate() -> Result<()> {
    use candle_core::nnapi_backend::NnapiStorage;
    use candle_core::Shape;

    let device = Device::new_nnapi(0)?;
    let shape = Shape::from_dims(&[2, 3]);

    let storage = NnapiStorage::allocate(&shape, candle_core::DType::F32, &device)?;

    assert!(
        storage.is_zero_copy(),
        "NnapiStorage::allocate should create zero-copy storage"
    );
    assert!(
        storage.memory().is_some(),
        "NnapiStorage should have ANeuralNetworksMemory handle"
    );
    assert!(
        storage.hwbuffer().is_some(),
        "NnapiStorage should have AHardwareBuffer handle"
    );

    let data = storage.data();
    assert_eq!(
        data.len(),
        2 * 3 * 4,
        "Storage data size should match shape * dtype size"
    );

    Ok(())
}

#[cfg(all(feature = "nnapi", target_os = "android"))]
fn test_nnapi_zero_copy_alignment() -> Result<()> {
    use candle_core::nnapi_backend::NnapiStorage;

    let device = Device::new_nnapi(0)?;

    for (name, shape) in [
        ("1x1", Shape::from_dims(&[1, 1])),
        ("3x3", Shape::from_dims(&[3, 3])),
        ("7x7", Shape::from_dims(&[7, 7])),
        ("17x17", Shape::from_dims(&[17, 17])),
    ] {
        let storage = NnapiStorage::allocate(&shape, candle_core::DType::F32, &device)?;
        assert!(
            storage.is_zero_copy(),
            "Alignment test {}: should be zero-copy",
            name
        );
        let data = storage.data();
        assert_eq!(
            data.len() % 16,
            0,
            "Alignment test {}: data size should be 16-byte aligned",
            name
        );
    }

    Ok(())
}
