use atheer_core::AtheerCoreError;
use atheer_ffi::AtheerConfig;
use atheer_orchestrator::{InferenceMode, Orchestrator, OrchestratorConfig};

#[test]
fn test_inference_modes_exist() {
    assert_eq!(InferenceMode::Turbo.as_str(), "turbo");
    assert_eq!(InferenceMode::Balanced.as_str(), "balanced");
    assert_eq!(InferenceMode::Eco.as_str(), "eco");
}

#[test]
fn test_inference_modes_speculation_depth() {
    assert_eq!(InferenceMode::Turbo.speculation_depth(), 4);
    assert_eq!(InferenceMode::Balanced.speculation_depth(), 2);
    assert_eq!(InferenceMode::Eco.speculation_depth(), 0);
}

#[test]
fn test_orchestrator_config_default_adaptive() {
    let config = OrchestratorConfig::default();
    assert!(config.adaptive, "Default config should be adaptive");
}

#[test]
fn test_orchestrator_new_adaptive_mode() {
    let config = OrchestratorConfig::default();
    let orchestrator = Orchestrator::new(config);
    assert_eq!(orchestrator.current_mode(), InferenceMode::Eco);
}

#[test]
fn test_atheer_config_default() {
    let config = AtheerConfig::default();
    assert_eq!(config.max_tokens, 512);
    assert_eq!(config.temperature, 0.7);
}

#[test]
fn test_atheer_config_new() {
    let config = AtheerConfig::new("/path/to/model".to_string());
    assert_eq!(config.model_path, Some("/path/to/model".to_string()));
}

#[test]
fn test_error_display() {
    let err = AtheerCoreError::ModelLoadFailed("test".to_string());
    let display = format!("{}", err);
    assert!(display.contains("Failed to load model"));
}
