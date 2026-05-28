use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductionConfig {
    pub model_path: String,
    pub draft_model_path: Option<String>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub quantization: String,
    pub memory_limit_mb: u32,
    pub enable_crash_reporting: bool,
    pub crash_log_path: Option<String>,
    pub enable_security_audit: bool,
    pub allowed_model_paths: Vec<String>,
    pub log_level: String,
}

impl ProductionConfig {
    pub fn validate(&self) -> Vec<ConfigError> {
        let mut errors = Vec::new();

        if self.model_path.is_empty() {
            errors.push(ConfigError::MissingField("model_path".to_string()));
        }

        if self.max_tokens == 0 {
            errors.push(ConfigError::InvalidValue(
                "max_tokens".to_string(),
                "must be > 0".to_string(),
            ));
        }

        if self.max_tokens > 32_768 {
            errors.push(ConfigError::InvalidValue(
                "max_tokens".to_string(),
                "exceeds maximum".to_string(),
            ));
        }

        if self.temperature < 0.0 || self.temperature > 2.0 {
            errors.push(ConfigError::InvalidValue(
                "temperature".to_string(),
                "must be 0-2".to_string(),
            ));
        }

        if self.memory_limit_mb < 256 {
            errors.push(ConfigError::InvalidValue(
                "memory_limit_mb".to_string(),
                "minimum 256MB".to_string(),
            ));
        }

        if !["q4_k_m", "q8_0", "f16"].contains(&self.quantization.as_str()) {
            errors.push(ConfigError::InvalidValue(
                "quantization".to_string(),
                "unknown type".to_string(),
            ));
        }

        if !["trace", "debug", "info", "warn", "error"].contains(&self.log_level.as_str()) {
            errors.push(ConfigError::InvalidValue(
                "log_level".to_string(),
                "unknown level".to_string(),
            ));
        }

        errors
    }

    pub fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }
}

impl Default for ProductionConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            draft_model_path: None,
            max_tokens: 512,
            temperature: 0.7,
            quantization: "q4_k_m".to_string(),
            memory_limit_mb: 2048,
            enable_crash_reporting: true,
            crash_log_path: None,
            enable_security_audit: true,
            allowed_model_paths: vec![],
            log_level: "info".to_string(),
        }
    }
}

#[derive(Debug)]
pub enum ConfigError {
    MissingField(String),
    InvalidValue(String, String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_production_config_defaults() {
        let config = ProductionConfig::default();
        assert_eq!(config.max_tokens, 512);
        assert_eq!(config.temperature, 0.7);
    }

    #[test]
    fn test_config_validation_valid() {
        let config = ProductionConfig {
            model_path: "/models/test.gguf".to_string(),
            ..Default::default()
        };
        assert!(config.is_valid());
    }

    #[test]
    fn test_config_validation_missing_path() {
        let config = ProductionConfig::default();
        let errors = config.validate();
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_config_validation_invalid_temperature() {
        let mut config = ProductionConfig::default();
        config.model_path = "/models/test.gguf".to_string();
        config.temperature = 5.0;

        let errors = config.validate();
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_config_validation_invalid_max_tokens() {
        let mut config = ProductionConfig::default();
        config.model_path = "/models/test.gguf".to_string();
        config.max_tokens = 100_000;

        let errors = config.validate();
        assert!(!errors.is_empty());
    }
}
