#[cfg(test)]
mod property_tests {
    use atheer_memory_bank::{KvCache, L2WarmCache, MemoryBank};
    use atheer_orchestrator::{NGramCache, SpeculativeDecoder};
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_kv_cache_insert_get_roundtrip(
            num_layers in 1..=32usize,
            num_heads in 1..=16usize,
            head_dim in 1..=128usize,
            max_seq_len in 1..=1024usize,
            layer in 0..32usize,
            position in 0..256usize,
            token_id: u32
        ) {
            let mut cache = KvCache::new(num_layers, num_heads, head_dim, max_seq_len);
            let keys = vec![1.0f32; num_heads * head_dim];
            let values = vec![2.0f32; num_heads * head_dim];

            cache.insert(layer % num_layers, position % max_seq_len, token_id, keys.clone(), values.clone());

            prop_assert!(cache.get(layer % num_layers, position % max_seq_len).is_some());
        }

        #[test]
        fn test_kv_cache_memory_estimate_consistency(
            num_layers in 1..=64usize,
            num_heads in 1..=32usize,
            head_dim in 1..=256usize,
            max_seq_len in 1..=4096usize
        ) {
            let cache = KvCache::new(num_layers, num_heads, head_dim, max_seq_len);
            let bytes = cache.max_memory_bytes();

            let expected = num_layers * max_seq_len * num_heads * head_dim * 2 * 4;
            prop_assert_eq!(bytes, expected);
        }

        #[test]
        fn test_kv_cache_truncate_preserves_count(
            initial_positions in 10..=100usize,
            truncate_at in 5..=50usize
        ) {
            let mut cache = KvCache::new(1, 1, 64, 128);

            for pos in 0..initial_positions {
                cache.insert(0, pos, pos as u32, vec![1.0; 64], vec![1.0; 64]);
            }

            let count_before = cache.token_count();
            cache.truncate(truncate_at);
            let count_after = cache.token_count();

            prop_assert!(count_after < count_before || truncate_at >= initial_positions);
        }

        #[test]
        fn test_l2_alignment_score_bounds(score: f32) {
            let mut cache = L2WarmCache::new("test".to_string());
            cache.update_alignment(score);

            let actual = cache.alignment_score();
            prop_assert!(actual >= 0.0 && actual <= 1.0);
        }

        #[test]
        fn test_ngram_cache_insert_within_order(
            max_order in 2..=5usize,
            max_entries in 10..=100usize
        ) {
            let mut cache = NGramCache::new(max_order, max_entries);
            let prefix_len = max_order;

            let prefix: Vec<u32> = (0..prefix_len).map(|i| i as u32).collect();
            let continuation: Vec<u32> = (100..110).map(|i| i as u32).collect();

            cache.insert(&prefix, &continuation);

            prop_assert!(cache.get(&prefix).is_some());
            prop_assert_eq!(cache.lookup(&prefix), continuation.first().copied());
        }

        #[test]
        fn test_ngram_cache_eviction_order(
            max_entries in 1..=10usize
        ) {
            let mut cache = NGramCache::new(3, max_entries);

            for i in 0..(max_entries * 2) {
                let prefix = vec![i as u32];
                cache.insert(&prefix, &[i as u32 + 100]);
            }

            prop_assert!(cache.size() <= max_entries);
        }

        #[test]
        fn test_speculative_decoder_verify_bounds(
            draft_len in 1..=20usize
        ) {
            let mut decoder = SpeculativeDecoder::new(1, 8);

            let draft: Vec<u32> = (0..draft_len).map(|i| i as u32).collect();
            let target: Vec<u32> = (0..draft_len).map(|i| i as u32).collect();

            decoder.propose(draft.clone(), vec![-0.1; draft_len]);
            let result = decoder.verify(&target);

            prop_assert!(result.rejected.is_empty());
            prop_assert_eq!(result.accepted.len(), draft_len);
        }

        #[test]
        fn test_memory_bank_creation(max_size_mb in 1..=4096usize) {
            let bank = MemoryBank::new(max_size_mb);

            prop_assert!(bank.l1_active().is_none());
            prop_assert!(bank.l2_warm().is_none());
        }

        #[test]
        fn test_memory_bank_tier_loading(
            max_size in 128..=2048usize,
            model_id in "[a-z]{1,20}"
        ) {
            let bank = MemoryBank::new(max_size);

            bank.load_l1(&model_id).unwrap();

            prop_assert_eq!(bank.l1_active(), Some(model_id.to_string()));
        }
    }
}
