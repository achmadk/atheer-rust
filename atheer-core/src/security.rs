use std::collections::HashSet;

pub struct SecurityAudit {
    allowed_paths: HashSet<String>,
    max_model_size_mb: usize,
    enable_signature_verify: bool,
}

impl SecurityAudit {
    pub fn new() -> Self {
        Self {
            allowed_paths: HashSet::new(),
            max_model_size_mb: 10_000,
            enable_signature_verify: false,
        }
    }

    pub fn with_allowed_paths(mut self, paths: Vec<String>) -> Self {
        self.allowed_paths = paths.into_iter().collect();
        self
    }

    pub fn with_max_model_size_mb(mut self, size_mb: usize) -> Self {
        self.max_model_size_mb = size_mb;
        self
    }

    pub fn with_signature_verification(mut self, enabled: bool) -> Self {
        self.enable_signature_verify = enabled;
        self
    }

    pub fn validate_model_path(&self, path: &str) -> Result<(), SecurityError> {
        if self.allowed_paths.is_empty() {
            return Ok(());
        }

        if !self.allowed_paths.contains(path) {
            return Err(SecurityError::PathNotAllowed(path.to_string()));
        }

        Ok(())
    }

    pub fn validate_model_size(&self, size_mb: usize) -> Result<(), SecurityError> {
        if size_mb > self.max_model_size_mb {
            return Err(SecurityError::ModelTooLarge {
                size_mb,
                max_mb: self.max_model_size_mb,
            });
        }
        Ok(())
    }

    pub fn sanitize_prompt(&self, prompt: &str) -> String {
        let max_len = 32_000;
        if prompt.len() > max_len {
            return prompt[..max_len].to_string();
        }
        prompt.to_string()
    }
}

impl Default for SecurityAudit {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub enum SecurityError {
    PathNotAllowed(String),
    ModelTooLarge { size_mb: usize, max_mb: usize },
    SignatureInvalid,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_audit_creation() {
        let audit = SecurityAudit::new();
        assert_eq!(audit.max_model_size_mb, 10_000);
    }

    #[test]
    fn test_allowed_paths() {
        let audit = SecurityAudit::new().with_allowed_paths(vec!["/models".to_string()]);

        let result = audit.validate_model_path("/models");
        assert!(result.is_ok());

        let result = audit.validate_model_path("/other");
        assert!(result.is_err());
    }

    #[test]
    fn test_model_size_validation() {
        let audit = SecurityAudit::new().with_max_model_size_mb(1000);

        let result = audit.validate_model_size(500);
        assert!(result.is_ok());

        let result = audit.validate_model_size(2000);
        assert!(result.is_err());
    }

    #[test]
    fn test_prompt_sanitization() {
        let audit = SecurityAudit::new();
        let long_prompt = "a".repeat(50_000);
        let sanitized = audit.sanitize_prompt(&long_prompt);
        assert_eq!(sanitized.len(), 32_000);
    }
}
