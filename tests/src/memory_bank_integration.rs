#[allow(unused_imports)]
use atheer_memory_bank::{HandoffPhase, L1ActiveCache, L2WarmCache, MemoryBank};

#[test]
fn test_memory_bank_single_tier() {
    let bank = MemoryBank::new(512, None);
    bank.load_l1("model-1").expect("Should load L1");

    assert_eq!(bank.l1_active(), Some("model-1".to_string()));
    assert_eq!(bank.l2_warm(), None);
}

#[test]
fn test_memory_bank_multi_tier() {
    let bank = MemoryBank::new(2048, None);

    bank.load_l1("model-1").expect("Should load L1");
    bank.load_l2("model-1").expect("Should load L2");

    assert_eq!(bank.l1_active(), Some("model-1".to_string()));
    assert_eq!(bank.l2_warm(), Some("model-1".to_string()));
}

#[test]
fn test_l1_cache_basic() {
    let cache = L1ActiveCache::new("test-model".to_string());
    assert_eq!(cache.model_id, "test-model");
}

#[test]
fn test_l2_cache_basic() {
    let cache = L2WarmCache::new("test-model".to_string());
    assert_eq!(cache.model_id, "test-model");
    assert_eq!(cache.alignment_score(), 0.0);
}

#[test]
fn test_memory_bank_alignment_score() {
    let bank = MemoryBank::new(1024, None);
    bank.load_l1("model-1").expect("Should load");
    bank.load_l2("model-1").expect("Should load");

    let score = bank.alignment_score();
    assert!(score >= 0.0 && score <= 1.0);
}

#[test]
fn test_handoff_protocol_idle() {
    let bank = MemoryBank::new(1024, None);
    assert!(matches!(bank.handoff_phase(), HandoffPhase::Idle));
}
