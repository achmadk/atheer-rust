use crate::safe_content::{SafeLoadLimits, MAX_ALIGNMENT, MIN_ALIGNMENT};
use crate::AtheerCoreError;
use candle_core::quantized::gguf_file::Content;
use std::collections::HashSet;

pub struct GgufValidator {
    max_tensors: usize,
    max_metadata_kv: usize,
    max_string_bytes: usize,
    max_tensor_name_bytes: usize,
    max_dimensions: usize,
    max_alignment: u64,
    max_tensor_bytes: u64,
}

impl GgufValidator {
    pub fn new(file_size: u64) -> Self {
        Self::with_limits(file_size, SafeLoadLimits::default())
    }

    pub fn with_limits(_file_size: u64, safe_load_limits: SafeLoadLimits) -> Self {
        Self {
            max_tensors: safe_load_limits.max_tensor_count as usize,
            max_metadata_kv: safe_load_limits.max_metadata_kv_count as usize,
            max_string_bytes: 10 * 1024 * 1024,
            max_tensor_name_bytes: 1024 * 1024,
            max_dimensions: 16,
            max_alignment: safe_load_limits.max_alignment,
            max_tensor_bytes: safe_load_limits.max_total_tensor_bytes,
        }
    }

    /// Backwards-compatible entry point. Prefer [`Self::validate_full`] which
    /// accepts the file size at the call site.
    pub fn validate(&self, content: &Content) -> crate::Result<()> {
        self.validate_full(content, u64::MAX)
    }

    /// Deep structural validation of a fully-parsed GGUF [`Content`].
    ///
    /// This is the S5/S6 second-stage validator. It runs **after** candle's
    /// `Content::read` has allocated metadata buffers but **before**
    /// `WeightsVariant::from_gguf` allocates per-tensor data. The
    /// pre-allocation header gate is `safe_content::parse_header`.
    pub fn validate_full(&self, content: &Content, file_size: u64) -> crate::Result<()> {
        self.validate_counts(content)?;
        self.validate_metadata_strings(content)?;
        self.validate_alignment(content)?;
        self.validate_tensor_data_offset(content, file_size)?;
        self.validate_tensors(content, file_size)?;
        self.validate_unique_tensor_names(content)?;
        self.validate_required_metadata(content)?;
        Ok(())
    }

    fn validate_counts(&self, content: &Content) -> crate::Result<()> {
        let tensor_count = content.tensor_infos.len();
        if tensor_count > self.max_tensors {
            return Err(AtheerCoreError::InvalidCounts {
                tensor_count: tensor_count as u64,
                metadata_kv_count: content.metadata.len() as u64,
                max_tensor_bytes: self.max_tensor_bytes,
                requested_tensor_bytes: 0,
            });
        }

        let metadata_kv_count = content.metadata.len();
        if metadata_kv_count > self.max_metadata_kv {
            return Err(AtheerCoreError::InvalidCounts {
                tensor_count: tensor_count as u64,
                metadata_kv_count: metadata_kv_count as u64,
                max_tensor_bytes: self.max_tensor_bytes,
                requested_tensor_bytes: 0,
            });
        }

        Ok(())
    }

    fn validate_metadata_strings(&self, content: &Content) -> crate::Result<()> {
        use candle_core::quantized::gguf_file::Value;
        for (key, value) in &content.metadata {
            if key.len() > self.max_tensor_name_bytes {
                return Err(AtheerCoreError::ModelLoadFailed(format!(
                    "GGUF validation: metadata key length ({}) exceeds maximum ({})",
                    key.len(),
                    self.max_tensor_name_bytes
                )));
            }

            if let Value::String(s) = value {
                if s.len() > self.max_string_bytes {
                    return Err(AtheerCoreError::ModelLoadFailed(format!(
                        "GGUF validation: metadata string value length for key '{key}' ({}) exceeds maximum ({})",
                        s.len(),
                        self.max_string_bytes
                    )));
                }
            }
        }
        Ok(())
    }

    fn validate_alignment(&self, content: &Content) -> crate::Result<()> {
        use candle_core::quantized::gguf_file::Value;
        let alignment = match content.metadata.get("general.alignment") {
            Some(Value::U8(v)) => *v as i64,
            Some(Value::U16(v)) => *v as i64,
            Some(Value::U32(v)) => *v as i64,
            Some(Value::I8(v)) => *v as i64,
            Some(Value::I16(v)) => *v as i64,
            Some(Value::I32(v)) => *v as i64,
            Some(Value::U64(v)) if *v <= i64::MAX as u64 => *v as i64,
            Some(Value::I64(v)) => *v,
            _ => return Ok(()),
        };

        if alignment < MIN_ALIGNMENT as i64 || alignment > MAX_ALIGNMENT as i64 {
            return Err(AtheerCoreError::InvalidAlignment { value: alignment });
        }
        let alignment_u = alignment as u64;
        if !alignment_u.is_power_of_two() {
            return Err(AtheerCoreError::InvalidAlignment { value: alignment });
        }
        if alignment_u > self.max_alignment {
            return Err(AtheerCoreError::InvalidAlignment { value: alignment });
        }
        Ok(())
    }

    fn validate_tensor_data_offset(&self, content: &Content, file_size: u64) -> crate::Result<()> {
        if content.tensor_data_offset > file_size {
            return Err(AtheerCoreError::InvalidTensorBounds {
                tensor_name: "<tensor_data_offset>".into(),
                offset: content.tensor_data_offset,
                size: 0,
                file_size,
            });
        }
        Ok(())
    }

    fn validate_tensors(&self, content: &Content, file_size: u64) -> crate::Result<()> {
        let tensor_data_offset = content.tensor_data_offset;

        for (name, info) in &content.tensor_infos {
            if name.len() > self.max_tensor_name_bytes {
                return Err(AtheerCoreError::ModelLoadFailed(format!(
                    "GGUF validation: tensor name length ({}) exceeds maximum ({})",
                    name.len(),
                    self.max_tensor_name_bytes
                )));
            }

            let dims = info.shape.dims();
            if dims.is_empty() {
                return Err(AtheerCoreError::ModelLoadFailed(format!(
                    "GGUF validation: tensor '{name}' has zero dimensions",
                )));
            }
            if dims.len() > self.max_dimensions {
                return Err(AtheerCoreError::ModelLoadFailed(format!(
                    "GGUF validation: tensor '{name}' has {} dimensions, maximum is {}",
                    dims.len(),
                    self.max_dimensions
                )));
            }
            for (i, &dim) in dims.iter().enumerate() {
                if dim == 0 {
                    return Err(AtheerCoreError::ModelLoadFailed(format!(
                        "GGUF validation: tensor '{name}' has zero in dimension {i}",
                    )));
                }
            }

            let file_offset = tensor_data_offset.checked_add(info.offset).ok_or_else(|| {
                AtheerCoreError::InvalidTensorBounds {
                    tensor_name: name.clone(),
                    offset: info.offset,
                    size: 0,
                    file_size,
                }
            })?;

            let tensor_elems = info.shape.elem_count();
            let block_size = info.ggml_dtype.block_size();
            let tensor_bytes: u64 = if tensor_elems > 0 && block_size > 0 {
                let blocks = tensor_elems.div_ceil(block_size);
                (blocks as u64) * (info.ggml_dtype.type_size() as u64)
            } else {
                0
            };

            if tensor_bytes > self.max_tensor_bytes {
                return Err(AtheerCoreError::ModelLoadFailed(format!(
                    "GGUF validation: tensor '{name}' size ({tensor_bytes}) exceeds maximum ({})",
                    self.max_tensor_bytes
                )));
            }

            if file_offset > file_size {
                return Err(AtheerCoreError::InvalidTensorBounds {
                    tensor_name: name.clone(),
                    offset: file_offset,
                    size: tensor_bytes,
                    file_size,
                });
            }

            let end_offset = file_offset.checked_add(tensor_bytes).ok_or_else(|| {
                AtheerCoreError::InvalidTensorBounds {
                    tensor_name: name.clone(),
                    offset: file_offset,
                    size: tensor_bytes,
                    file_size,
                }
            })?;

            if end_offset > file_size {
                return Err(AtheerCoreError::InvalidTensorBounds {
                    tensor_name: name.clone(),
                    offset: file_offset,
                    size: tensor_bytes,
                    file_size,
                });
            }
        }

        Ok(())
    }

    fn validate_unique_tensor_names(&self, content: &Content) -> crate::Result<()> {
        let mut seen: HashSet<&str> = HashSet::with_capacity(content.tensor_infos.len());
        for name in content.tensor_infos.keys() {
            if !seen.insert(name.as_str()) {
                return Err(AtheerCoreError::DuplicateTensorName { name: name.clone() });
            }
        }
        Ok(())
    }

    fn validate_required_metadata(&self, content: &Content) -> crate::Result<()> {
        use candle_core::quantized::gguf_file::Value;
        let arch = match content.metadata.get("general.architecture") {
            Some(Value::String(s)) => s.as_str(),
            _ => return Ok(()),
        };
        let required: &[&str] = match arch {
            "llama" | "lfm2" => &["llama.block_count"],
            _ => return Ok(()),
        };
        for key in required {
            if !content.metadata.contains_key(*key) {
                return Err(AtheerCoreError::ModelLoadFailed(format!(
                    "GGUF validation: missing required metadata '{key}' for architecture '{arch}'"
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::gguf_file::{Content, TensorInfo, Value};
    use candle_core::quantized::GgmlDType;
    use candle_core::Shape;
    use std::collections::HashMap;

    fn make_content(tensor_count: usize, metadata_kv_count: usize) -> Content {
        let mut metadata = HashMap::new();
        for i in 0..metadata_kv_count {
            metadata.insert(format!("key_{i}"), Value::String(format!("value_{i}")));
        }

        let mut tensor_infos = HashMap::new();
        for i in 0..tensor_count {
            tensor_infos.insert(
                format!("tensor_{i}"),
                TensorInfo {
                    shape: Shape::from(vec![1, 2]),
                    offset: 0,
                    ggml_dtype: GgmlDType::Q4K,
                },
            );
        }

        Content {
            magic: candle_core::quantized::gguf_file::VersionedMagic::GgufV3,
            metadata,
            tensor_infos,
            tensor_data_offset: 32,
        }
    }

    #[test]
    fn test_valid_content_passes() {
        let content = make_content(10, 10);
        let validator = GgufValidator::new(1024 * 1024);
        assert!(validator.validate_full(&content, 1024 * 1024).is_ok());
    }

    #[test]
    fn test_tensor_count_exceeds_max() {
        let content = make_content(20_000, 10);
        let validator = GgufValidator::new(1024 * 1024);
        let result = validator.validate_full(&content, 1024 * 1024);
        assert!(matches!(result, Err(AtheerCoreError::InvalidCounts { .. })));
    }

    #[test]
    fn test_metadata_kv_count_exceeds_max() {
        let content = make_content(10, 200_000);
        let validator = GgufValidator::new(1024 * 1024);
        let result = validator.validate_full(&content, 1024 * 1024);
        assert!(matches!(result, Err(AtheerCoreError::InvalidCounts { .. })));
    }

    #[test]
    fn test_zero_dimensions_rejected() {
        let mut content = make_content(1, 1);
        content.tensor_infos.insert(
            "bad_tensor".to_string(),
            TensorInfo {
                shape: Shape::from(vec![]),
                offset: 0,
                ggml_dtype: GgmlDType::Q4K,
            },
        );
        let validator = GgufValidator::new(1024 * 1024);
        let result = validator.validate_full(&content, 1024 * 1024);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("zero dimensions"));
    }

    #[test]
    fn test_too_many_dimensions_rejected() {
        let mut content = make_content(1, 1);
        content.tensor_infos.insert(
            "bad_tensor".to_string(),
            TensorInfo {
                shape: Shape::from(vec![1; 20]),
                offset: 0,
                ggml_dtype: GgmlDType::Q4K,
            },
        );
        let validator = GgufValidator::new(1024 * 1024);
        let result = validator.validate_full(&content, 1024 * 1024);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("dimensions"));
    }

    #[test]
    fn test_zero_in_dimension_rejected() {
        let mut content = make_content(1, 1);
        content.tensor_infos.insert(
            "bad_tensor".to_string(),
            TensorInfo {
                shape: Shape::from(vec![1, 0, 3]),
                offset: 0,
                ggml_dtype: GgmlDType::Q4K,
            },
        );
        let validator = GgufValidator::new(1024 * 1024);
        let result = validator.validate_full(&content, 1024 * 1024);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("zero in dimension"));
    }

    #[test]
    fn test_alignment_not_power_of_two_rejected() {
        let mut content = make_content(1, 1);
        content
            .metadata
            .insert("general.alignment".to_string(), Value::U32(100));
        let validator = GgufValidator::new(1024 * 1024);
        let result = validator.validate_full(&content, 1024 * 1024);
        assert!(matches!(
            result,
            Err(AtheerCoreError::InvalidAlignment { .. })
        ));
    }

    #[test]
    fn test_alignment_exceeds_max_rejected() {
        let mut content = make_content(1, 1);
        content
            .metadata
            .insert("general.alignment".to_string(), Value::U32(8192));
        let validator = GgufValidator::new(1024 * 1024);
        let result = validator.validate_full(&content, 1024 * 1024);
        assert!(matches!(
            result,
            Err(AtheerCoreError::InvalidAlignment { .. })
        ));
    }

    #[test]
    fn test_offset_overflow_rejected() {
        let mut content = make_content(1, 1);
        content.tensor_infos.insert(
            "bad_tensor".to_string(),
            TensorInfo {
                shape: Shape::from(vec![1]),
                offset: u64::MAX,
                ggml_dtype: GgmlDType::Q4K,
            },
        );
        let validator = GgufValidator::new(u64::MAX);
        let result = validator.validate_full(&content, u64::MAX);
        assert!(matches!(
            result,
            Err(AtheerCoreError::InvalidTensorBounds { .. })
        ));
    }

    #[test]
    fn test_offset_exceeds_file_size_rejected() {
        let mut content = make_content(1, 1);
        content.tensor_data_offset = 1000;
        content.tensor_infos.insert(
            "bad_tensor".to_string(),
            TensorInfo {
                shape: Shape::from(vec![1]),
                offset: 100,
                ggml_dtype: GgmlDType::Q4K,
            },
        );
        let validator = GgufValidator::new(1000);
        let result = validator.validate_full(&content, 1000);
        assert!(matches!(
            result,
            Err(AtheerCoreError::InvalidTensorBounds { .. })
        ));
    }

    #[test]
    fn test_tensor_name_too_long_rejected() {
        let mut content = make_content(1, 1);
        content.tensor_infos.insert(
            "x".repeat(2 * 1024 * 1024),
            TensorInfo {
                shape: Shape::from(vec![1]),
                offset: 0,
                ggml_dtype: GgmlDType::Q4K,
            },
        );
        let validator = GgufValidator::new(1024 * 1024);
        let result = validator.validate_full(&content, 1024 * 1024);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("tensor name length"));
    }

    #[test]
    fn test_tensor_data_offset_past_file_size_rejected() {
        let mut content = make_content(1, 1);
        content.tensor_data_offset = 10_000;
        let validator = GgufValidator::new(1024);
        let result = validator.validate_full(&content, 1024);
        assert!(matches!(
            result,
            Err(AtheerCoreError::InvalidTensorBounds {
                ref tensor_name,
                ..
            }) if tensor_name == "<tensor_data_offset>"
        ));
    }

    #[test]
    fn test_unique_tensor_names_passes() {
        let content = make_content(5, 0);
        let validator = GgufValidator::new(1024 * 1024);
        assert!(validator.validate_full(&content, 1024 * 1024).is_ok());
    }

    #[test]
    fn test_missing_required_metadata_for_llama_rejected() {
        let mut content = make_content(1, 1);
        content.metadata.insert(
            "general.architecture".to_string(),
            Value::String("llama".to_string()),
        );
        let validator = GgufValidator::new(1024 * 1024);
        let result = validator.validate_full(&content, 1024 * 1024);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("llama.block_count"),
            "expected missing metadata error, got: {err}"
        );
    }

    #[test]
    fn test_required_metadata_present_passes() {
        let mut content = make_content(1, 2);
        content.metadata.insert(
            "general.architecture".to_string(),
            Value::String("llama".to_string()),
        );
        content
            .metadata
            .insert("llama.block_count".to_string(), Value::U32(32));
        let validator = GgufValidator::new(1024 * 1024);
        assert!(validator.validate_full(&content, 1024 * 1024).is_ok());
    }
}
