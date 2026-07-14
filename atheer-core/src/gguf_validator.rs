use crate::AtheerCoreError;
use candle_core::quantized::gguf_file::Content;

pub struct GgufValidator {
    file_size: u64,
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
        Self {
            file_size,
            max_tensors: 10_000,
            max_metadata_kv: 100_000,
            max_string_bytes: 10 * 1024 * 1024,
            max_tensor_name_bytes: 1024 * 1024,
            max_dimensions: 16,
            max_alignment: 4096,
            max_tensor_bytes: 500 * 1024 * 1024 * 1024,
        }
    }

    pub fn validate(&self, content: &Content) -> Result<(), AtheerCoreError> {
        self.validate_counts(content)?;
        self.validate_metadata_strings(content)?;
        self.validate_alignment(content)?;
        self.validate_tensors(content)?;
        Ok(())
    }

    fn validate_counts(&self, content: &Content) -> Result<(), AtheerCoreError> {
        let tensor_count = content.tensor_infos.len();
        if tensor_count > self.max_tensors {
            return Err(AtheerCoreError::ModelLoadFailed(format!(
                "GGUF validation: tensor_count ({tensor_count}) exceeds maximum ({})",
                self.max_tensors
            )));
        }

        let metadata_kv_count = content.metadata.len();
        if metadata_kv_count > self.max_metadata_kv {
            return Err(AtheerCoreError::ModelLoadFailed(format!(
                "GGUF validation: metadata_kv_count ({metadata_kv_count}) exceeds maximum ({})",
                self.max_metadata_kv
            )));
        }

        Ok(())
    }

    fn validate_metadata_strings(&self, content: &Content) -> Result<(), AtheerCoreError> {
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
                        "GGUF validation: metadata string value length for key '{}' ({}) exceeds maximum ({})",
                        key,
                        s.len(),
                        self.max_string_bytes
                    )));
                }
            }
        }
        Ok(())
    }

    fn validate_alignment(&self, content: &Content) -> Result<(), AtheerCoreError> {
        use candle_core::quantized::gguf_file::Value;
        let alignment = match content.metadata.get("general.alignment") {
            Some(Value::U8(v)) => *v as u64,
            Some(Value::U16(v)) => *v as u64,
            Some(Value::U32(v)) => *v as u64,
            Some(Value::I8(v)) if *v >= 0 => *v as u64,
            Some(Value::I16(v)) if *v >= 0 => *v as u64,
            Some(Value::I32(v)) if *v >= 0 => *v as u64,
            _ => 32,
        };

        if alignment > self.max_alignment {
            return Err(AtheerCoreError::ModelLoadFailed(format!(
                "GGUF validation: alignment ({alignment}) exceeds maximum ({})",
                self.max_alignment
            )));
        }

        if alignment.count_ones() != 1 {
            return Err(AtheerCoreError::ModelLoadFailed(format!(
                "GGUF validation: alignment ({alignment}) is not a power of 2",
            )));
        }

        Ok(())
    }

    fn validate_tensors(&self, content: &Content) -> Result<(), AtheerCoreError> {
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
                    "GGUF validation: tensor '{}' has zero dimensions",
                    name
                )));
            }
            if dims.len() > self.max_dimensions {
                return Err(AtheerCoreError::ModelLoadFailed(format!(
                    "GGUF validation: tensor '{}' has {} dimensions, maximum is {}",
                    name,
                    dims.len(),
                    self.max_dimensions
                )));
            }
            for (i, &dim) in dims.iter().enumerate() {
                if dim == 0 {
                    return Err(AtheerCoreError::ModelLoadFailed(format!(
                        "GGUF validation: tensor '{}' has zero in dimension {i}",
                        name
                    )));
                }
            }

            let file_offset = tensor_data_offset.checked_add(info.offset).ok_or_else(|| {
                AtheerCoreError::ModelLoadFailed(format!(
                    "GGUF validation: tensor '{}' offset overflow (tensor_data_offset + offset)",
                    name
                ))
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
                    "GGUF validation: tensor '{}' size ({tensor_bytes}) exceeds maximum ({})",
                    name, self.max_tensor_bytes
                )));
            }

            if file_offset > self.file_size {
                return Err(AtheerCoreError::ModelLoadFailed(format!(
                    "GGUF validation: tensor '{}' offset ({file_offset}) exceeds file size ({})",
                    name, self.file_size
                )));
            }

            let end_offset = file_offset.checked_add(tensor_bytes).ok_or_else(|| {
                AtheerCoreError::ModelLoadFailed(format!(
                    "GGUF validation: tensor '{}' end offset overflow",
                    name
                ))
            })?;

            if end_offset > self.file_size {
                return Err(AtheerCoreError::ModelLoadFailed(format!(
                    "GGUF validation: tensor '{}' end offset ({end_offset}) exceeds file size ({})",
                    name, self.file_size
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
        assert!(validator.validate(&content).is_ok());
    }

    #[test]
    fn test_tensor_count_exceeds_max() {
        let content = make_content(20_000, 10);
        let validator = GgufValidator::new(1024 * 1024);
        let result = validator.validate(&content);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("tensor_count"));
    }

    #[test]
    fn test_metadata_kv_count_exceeds_max() {
        let content = make_content(10, 200_000);
        let validator = GgufValidator::new(1024 * 1024);
        let result = validator.validate(&content);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("metadata_kv_count"));
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
        let result = validator.validate(&content);
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
        let result = validator.validate(&content);
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
        let result = validator.validate(&content);
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
        let result = validator.validate(&content);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not a power of 2"));
    }

    #[test]
    fn test_alignment_exceeds_max_rejected() {
        let mut content = make_content(1, 1);
        content
            .metadata
            .insert("general.alignment".to_string(), Value::U32(8192));
        let validator = GgufValidator::new(1024 * 1024);
        let result = validator.validate(&content);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("alignment"));
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
        let result = validator.validate(&content);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("offset overflow"));
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
        let result = validator.validate(&content);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("exceeds file size"));
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
        let result = validator.validate(&content);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("tensor name length"));
    }
}
