#[allow(unused_imports)]
use atheer_orchestrator::{
    BalancedMode, EcoMode, InferenceMode, NGramCache, Orchestrator, OrchestratorConfig,
    SpeculativeDecoder, TurboMode,
};

#[test]
fn test_orchestrator_default_mode() {
    let config = OrchestratorConfig::default();
    let orchestrator = Orchestrator::new(config);

    assert_eq!(orchestrator.current_mode(), InferenceMode::Eco);
}

#[test]
fn test_orchestrator_mode_switching() {
    let config = OrchestratorConfig::default();
    let mut orchestrator = Orchestrator::new(config);

    orchestrator.set_mode(InferenceMode::Turbo);
    assert_eq!(orchestrator.current_mode(), InferenceMode::Turbo);
    assert_eq!(orchestrator.previous_mode(), InferenceMode::Eco);

    orchestrator.set_mode(InferenceMode::Balanced);
    assert_eq!(orchestrator.current_mode(), InferenceMode::Balanced);
    assert_eq!(orchestrator.previous_mode(), InferenceMode::Turbo);
}

#[test]
fn test_turbo_mode_creation() {
    let _mode = TurboMode::new();
}

#[test]
fn test_balanced_mode_creation() {
    let _mode = BalancedMode::new();
}

#[test]
fn test_eco_mode_creation() {
    let _mode = EcoMode::new();
}

#[test]
fn test_speculative_decoder() {
    let mut decoder = SpeculativeDecoder::new(1, 4);
    assert_eq!(decoder.start_draft(), 4);

    decoder.propose(vec![1, 2, 3], vec![-0.1, -0.2, -0.3]);
    let verify = decoder.verify(&[1, 2, 3]);
    assert_eq!(verify.accepted, vec![1, 2, 3]);
}

#[test]
fn test_ngram_cache_basic() {
    let mut cache = NGramCache::new(3, 100);

    cache.insert(&[1, 2, 3], &[4, 5]);
    let result = cache.get(&[1, 2, 3]);

    assert!(result.is_some());
    assert_eq!(result.unwrap(), &[4, 5]);
}

#[test]
fn test_ngram_cache_lookup() {
    let mut cache = NGramCache::new(3, 100);
    cache.insert(&[1, 2, 3], &[4, 5, 6]);

    assert_eq!(cache.lookup(&[1, 2, 3]), Some(4));
}

#[test]
fn test_ngram_cache_eviction() {
    let mut cache = NGramCache::new(3, 2);

    cache.insert(&[1, 2, 3], &[4]);
    cache.insert(&[2, 3, 4], &[5]);
    cache.insert(&[3, 4, 5], &[6]);

    assert!(cache.get(&[1, 2, 3]).is_none());
}

#[test]
fn test_eco_mode_ngram() {
    let mut mode = EcoMode::new();
    assert!(mode.ngram_enabled());
    assert_eq!(mode.ngram_order(), 3);

    mode.train_on_sequence(&[1, 2, 3, 4, 5]);

    let prediction = mode.predict(&[2, 3, 4]);
    assert_eq!(prediction, Some(5));
}

#[test]
fn test_orchestrator_responds_to_thermal_change() {
    let mut config = OrchestratorConfig::default();
    config.adaptive = true;
    config.hysteresis_cooldown_ms = 0;
    config.thermal_threshold_c = 40.0;
    config.thermal_margin_c = 5.0;
    let mut orchestrator = Orchestrator::new(config);

    // Start with good conditions -> should select Turbo
    let mode = orchestrator.select_mode(Some(35.0), 8192, Some(80), false);
    assert_eq!(mode, InferenceMode::Turbo);

    // Rising temperature -> predictive scheduler pre-downgrades to Balanced
    let mode = orchestrator.select_mode(Some(45.0), 8192, Some(80), false);
    assert_eq!(mode, InferenceMode::Balanced);

    // Sustained heat -> reactive logic downgrades to Eco
    let mode = orchestrator.select_mode(Some(45.0), 8192, Some(80), false);
    assert_eq!(mode, InferenceMode::Eco);
}

#[test]
fn test_orchestrator_responds_to_memory_pressure() {
    let mut config = OrchestratorConfig::default();
    config.adaptive = true;
    config.hysteresis_cooldown_ms = 0;
    config.memory_threshold_mb = 1000;
    config.memory_critical_mb = 500;
    let mut orchestrator = Orchestrator::new(config);

    // Plenty of memory -> Turbo
    let mode = orchestrator.select_mode(Some(35.0), 4096, Some(80), false);
    assert_eq!(mode, InferenceMode::Turbo);

    // Low memory (below critical) -> Eco
    let mode = orchestrator.select_mode(Some(35.0), 300, Some(80), false);
    assert_eq!(mode, InferenceMode::Eco);
}

#[test]
fn test_orchestrator_responds_to_battery_drain() {
    let mut config = OrchestratorConfig::default();
    config.adaptive = true;
    config.hysteresis_cooldown_ms = 0;
    config.battery_threshold_percent = 20;
    let mut orchestrator = Orchestrator::new(config);

    // High battery, not on battery -> Turbo
    let mode = orchestrator.select_mode(Some(35.0), 4096, Some(80), false);
    assert_eq!(mode, InferenceMode::Turbo);

    // Low battery, on battery -> Eco
    let mode = orchestrator.select_mode(Some(35.0), 4096, Some(15), true);
    assert_eq!(mode, InferenceMode::Eco);
}

#[test]
fn test_orchestrator_all_modes_accessible() {
    let mut orchestrator = Orchestrator::new(OrchestratorConfig::default());

    for mode in [
        InferenceMode::Turbo,
        InferenceMode::Balanced,
        InferenceMode::Eco,
    ] {
        orchestrator.set_mode(mode);
        assert_eq!(orchestrator.current_mode(), mode);
    }
}
