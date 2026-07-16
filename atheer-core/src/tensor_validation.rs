use crate::error::{AtheerCoreError, Result};

/// Result of validating tensor parameters before GPU/NPU submission.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorValidation {
    pub dims: Vec<usize>,
    pub num_elements: usize,
}

/// Validate tensor dimensions before GPU/NPU dispatch.
///
/// Checks:
/// - Dimension count is between 1 and 4 (inclusive)
/// - No dimension is zero
/// - Total element count does not overflow `usize`
///
/// Returns a `TensorValidation` on success containing computed metadata.
pub fn validate_tensor_dims(dims: &[usize]) -> Result<TensorValidation> {
    if dims.is_empty() {
        return Err(AtheerCoreError::InvalidParameters(
            "Tensor must have at least 1 dimension".to_string(),
        ));
    }
    if dims.len() > 4 {
        return Err(AtheerCoreError::InvalidParameters(format!(
            "Tensor dimension count {} exceeds maximum of 4",
            dims.len()
        )));
    }

    let mut num_elements = 1usize;
    for (i, &dim) in dims.iter().enumerate() {
        if dim == 0 {
            return Err(AtheerCoreError::InvalidParameters(format!(
                "Tensor dimension {} has size 0",
                i
            )));
        }
        num_elements = num_elements.checked_mul(dim).ok_or_else(|| {
            AtheerCoreError::InvalidParameters("Tensor total element count overflow".to_string())
        })?;
    }

    Ok(TensorValidation {
        dims: dims.to_vec(),
        num_elements,
    })
}

/// Validate batch input tensors for inference.
///
/// Checks:
/// - Each token_id is within expected vocabulary range
/// - Token and position arrays have matching length
/// - Total batch size does not exceed maximum
pub fn validate_batch_input(
    token_ids: &[u32],
    positions: &[usize],
    vocab_size: usize,
    max_batch_size: usize,
) -> Result<()> {
    if token_ids.is_empty() {
        return Err(AtheerCoreError::InvalidParameters(
            "Batch token_ids must not be empty".to_string(),
        ));
    }

    if token_ids.len() != positions.len() {
        return Err(AtheerCoreError::InvalidParameters(format!(
            "token_ids length ({}) does not match positions length ({})",
            token_ids.len(),
            positions.len(),
        )));
    }

    if token_ids.len() > max_batch_size {
        return Err(AtheerCoreError::InvalidParameters(format!(
            "Batch size {} exceeds maximum {}",
            token_ids.len(),
            max_batch_size,
        )));
    }

    for (i, &tid) in token_ids.iter().enumerate() {
        if tid as usize >= vocab_size {
            return Err(AtheerCoreError::InvalidParameters(format!(
                "token_id {} at index {} exceeds vocab size {}",
                tid, i, vocab_size,
            )));
        }
    }

    Ok(())
}

/// Validate tensor strides (for strided tensor access).
///
/// Checks:
/// - Strides length matches dims length
/// - No stride is zero
/// - Total extent (stride * dim for each axis) does not overflow
pub fn validate_strides(dims: &[usize], strides: &[usize]) -> Result<()> {
    if dims.len() != strides.len() {
        return Err(AtheerCoreError::InvalidParameters(format!(
            "dims length ({}) does not match strides length ({})",
            dims.len(),
            strides.len(),
        )));
    }

    for (i, (&dim, &stride)) in dims.iter().zip(strides.iter()).enumerate() {
        if stride == 0 {
            return Err(AtheerCoreError::InvalidParameters(format!(
                "Stride for dimension {} is 0",
                i
            )));
        }
        // Check that stride * (dim - 1) doesn't overflow
        if dim > 1 {
            let _ = (dim - 1).checked_mul(stride).ok_or_else(|| {
                AtheerCoreError::InvalidParameters(format!(
                    "Stride*extent overflow for dimension {}: stride={}, dim={}",
                    i, stride, dim,
                ))
            })?;
        }
    }

    Ok(())
}

/// Validate a memory offset (used for tensor data in GPU memory).
pub fn validate_offset(offset: u64, total_size: u64, buffer_size: u64) -> Result<()> {
    if offset + total_size > buffer_size {
        return Err(AtheerCoreError::InvalidParameters(format!(
            "Tensor access out of bounds: offset={}, size={}, buffer_size={}",
            offset, total_size, buffer_size,
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_tensor_dims_valid_1d() {
        let result = validate_tensor_dims(&[256]);
        assert!(result.is_ok());
        let v = result.unwrap();
        assert_eq!(v.num_elements, 256);
    }

    #[test]
    fn test_validate_tensor_dims_valid_2d() {
        let result = validate_tensor_dims(&[1, 50257]);
        assert!(result.is_ok());
        let v = result.unwrap();
        assert_eq!(v.num_elements, 50257);
    }

    #[test]
    fn test_validate_tensor_dims_valid_4d() {
        let result = validate_tensor_dims(&[1, 32, 128, 128]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().num_elements, 1 * 32 * 128 * 128);
    }

    #[test]
    fn test_validate_tensor_dims_empty() {
        let result = validate_tensor_dims(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_tensor_dims_zero_dim() {
        let result = validate_tensor_dims(&[128, 0, 64]);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_tensor_dims_overflow() {
        let result = validate_tensor_dims(&[usize::MAX, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_tensor_dims_too_many_dims() {
        let result = validate_tensor_dims(&[1, 2, 3, 4, 5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_batch_input_empty() {
        let result = validate_batch_input(&[], &[], 50257, 512);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_batch_input_mismatched_lengths() {
        let result = validate_batch_input(&[0, 1], &[0], 50257, 512);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_batch_input_token_exceeds_vocab() {
        let result = validate_batch_input(&[999999], &[0], 50257, 512);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_batch_input_exceeds_max() {
        let result = validate_batch_input(&[0, 1, 2], &[0, 1, 2], 50257, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_batch_input_valid() {
        let result = validate_batch_input(&[10, 20, 30], &[0, 1, 2], 50257, 512);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_strides_mismatched_length() {
        let result = validate_strides(&[128, 256], &[256]);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_strides_zero_stride() {
        let result = validate_strides(&[128], &[0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_strides_valid() {
        let result = validate_strides(&[128, 256], &[256, 1]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_offset_out_of_bounds() {
        let result = validate_offset(100, 50, 120);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_offset_valid() {
        let result = validate_offset(100, 50, 200);
        assert!(result.is_ok());
    }

    // ── T1.8 — Integration test: full validation chain ───────────

    /// Simulates the validation flow that would occur before GPU dispatch:
    /// validate tensor dims → batch input → strides → offset.
    #[test]
    fn test_full_validation_chain_before_gpu_dispatch() {
        // Step 1: Validate tensor dimensions (simulates model weight tensor)
        let dims = validate_tensor_dims(&[1, 32, 128, 128]).unwrap();
        assert_eq!(dims.num_elements, 1 * 32 * 128 * 128);

        // Step 2: Validate batch input (simulates inference request)
        let token_ids = vec![10, 20, 30];
        let positions = vec![0, 1, 2];
        validate_batch_input(&token_ids, &positions, 50257, 512).unwrap();

        // Step 3: Validate strides for tensor access
        validate_strides(&[128, 256], &[256, 1]).unwrap();

        // Step 4: Validate memory offset for GPU buffer access
        let buffer_size = (dims.num_elements * 4) as u64; // f32 = 4 bytes
        validate_offset(0, buffer_size, buffer_size * 2).unwrap();
    }

    /// Simulates the validation failure path on invalid input.
    #[test]
    fn test_full_validation_chain_rejects_invalid_input() {
        // Invalid: empty token_ids
        assert!(validate_batch_input(&[], &[], 50257, 512).is_err());

        // Invalid: zero dimension
        assert!(validate_tensor_dims(&[128, 0, 64]).is_err());

        // Invalid: stride mismatch
        assert!(validate_strides(&[128, 256], &[256]).is_err());

        // Invalid: out-of-bounds buffer access
        assert!(validate_offset(100, 50, 120).is_err());
    }

    /// Simulates the validation rejection chain: fails early on first bad input.
    #[test]
    fn test_full_validation_chain_fails_fast_on_first_error() {
        // Empty dims → fail before batch validation
        assert!(validate_tensor_dims(&[]).is_err());

        // Token exceeds vocab size → fail before strides check
        assert!(validate_batch_input(&[999999], &[0], 50257, 512).is_err());

        // Zero stride → fail before offset check
        assert!(validate_strides(&[128], &[0]).is_err());
    }
}
