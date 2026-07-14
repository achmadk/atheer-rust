pub mod analyzer;
pub mod detector;
pub mod normalizer;
pub mod output_check;
pub mod patterns;
pub mod verdict;

use serde::{Deserialize, Serialize};

pub use detector::GuardrailDetector;
pub use verdict::GuardrailVerdict;

/// Guardrail severity level controlling which detection layers run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GuardrailLevel {
    /// All guardrail checks disabled. Testing/dev use only.
    None = 0,
    /// L1 fast heuristics only. Catches obvious injections.
    Basic = 1,
    /// L1 + L2 token analysis. Catches encoding/leetspeak attacks.
    Balanced = 2,
    /// L1 + L2 + L3 output guard. Compliance/high-security deployments.
    Strict = 3,
}

/// Configuration for the guardrail detection pipeline.
pub struct GuardrailConfig {
    /// The detection level to use.
    pub level: GuardrailLevel,
    /// Optional path to a sidecar JSON pattern file.
    pub patterns_path: Option<String>,
    /// Additional custom patterns appended to the merged set.
    pub custom_patterns: Vec<String>,
}

impl GuardrailConfig {
    pub fn new(
        level: GuardrailLevel,
        patterns_path: Option<String>,
        custom_patterns: Vec<String>,
    ) -> Self {
        Self {
            level,
            patterns_path,
            custom_patterns,
        }
    }

    /// Build a [`GuardrailDetector`] from this configuration.
    ///
    /// Loads patterns from builtin → sidecar → custom, then constructs
    /// the detector with the appropriate thresholds.
    pub fn build_detector(self) -> Result<GuardrailDetector, String> {
        let mut patterns = if let Some(ref path) = self.patterns_path {
            patterns::PatternDatabase::load_sidecar(path)
        } else {
            patterns::PatternDatabase::load_builtin()
        };

        // Append custom patterns to the merged set
        patterns.append_custom(self.custom_patterns.clone());

        let detector = GuardrailDetector::new(
            self.level,
            patterns,
            self.patterns_path.clone(),
            self.custom_patterns.clone(),
        );

        Ok(detector)
    }

    /// Get the block threshold for the configured level.
    pub fn block_threshold(level: GuardrailLevel) -> f64 {
        match level {
            GuardrailLevel::None => 1.0,
            GuardrailLevel::Basic => 0.90,
            GuardrailLevel::Balanced => 0.80,
            GuardrailLevel::Strict => 0.70,
        }
    }

    /// Get the flag threshold for the configured level.
    pub fn flag_threshold(level: GuardrailLevel) -> f64 {
        match level {
            GuardrailLevel::None => 1.0,
            GuardrailLevel::Basic => 0.70,
            GuardrailLevel::Balanced => 0.60,
            GuardrailLevel::Strict => 0.50,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guardrail_level_ordering() {
        assert!(GuardrailLevel::None < GuardrailLevel::Basic);
        assert!(GuardrailLevel::Basic < GuardrailLevel::Balanced);
        assert!(GuardrailLevel::Balanced < GuardrailLevel::Strict);
    }

    #[test]
    fn test_block_thresholds() {
        assert_eq!(GuardrailConfig::block_threshold(GuardrailLevel::None), 1.0);
        assert_eq!(
            GuardrailConfig::block_threshold(GuardrailLevel::Basic),
            0.90
        );
        assert_eq!(
            GuardrailConfig::block_threshold(GuardrailLevel::Balanced),
            0.80
        );
        assert_eq!(
            GuardrailConfig::block_threshold(GuardrailLevel::Strict),
            0.70
        );
    }

    #[test]
    fn test_flag_thresholds() {
        assert_eq!(GuardrailConfig::flag_threshold(GuardrailLevel::None), 1.0);
        assert_eq!(GuardrailConfig::flag_threshold(GuardrailLevel::Basic), 0.70);
        assert_eq!(
            GuardrailConfig::flag_threshold(GuardrailLevel::Balanced),
            0.60
        );
        assert_eq!(
            GuardrailConfig::flag_threshold(GuardrailLevel::Strict),
            0.50
        );
    }

    #[test]
    fn test_build_detector_basic() {
        let config = GuardrailConfig::new(GuardrailLevel::Basic, None, Vec::new());
        let detector = config.build_detector().unwrap();
        assert_eq!(detector.level(), GuardrailLevel::Basic);
    }
}

#[cfg(test)]
mod test_suite;
