use atheer_ffi::AtheerConfig;
use atheer_memory_bank::KvCache;

fn fuzz_config_parse(data: &[u8]) {
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = serde_json::from_str::<AtheerConfig>(s);
    }
}

fn fuzz_kv_cache_operations(data: &[u8]) {
    let mut cache = KvCache::new(32, 8, 128, 2048);

    let mut hash: u64 = 0;
    for (i, &byte) in data.iter().enumerate() {
        hash = hash.wrapping_mul(31).wrapping_add(byte as u64);

        let layer = (hash % 32) as usize;
        let position = ((hash >> 8) % 2048) as usize;
        let token_id = (hash % 100000) as u32;

        let keys = vec![(hash % 1000) as f32; 1024];
        let values = vec![((hash >> 16) % 1000) as f32; 1024];

        cache.insert(layer, position, token_id, keys, values);

        if i > 100 {
            break;
        }
    }
}

fn fuzz_token_validation(data: &[u8]) {
    let mut hash: u64 = 0;
    for &byte in data.iter().take(100) {
        hash = hash.wrapping_mul(31).wrapping_add(byte as u64);

        let token_count = (hash % 500) as usize;
        let _ = format!("batch_size_{}", token_count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuzz_config_parse_empty() {
        fuzz_config_parse(b"{}");
    }

    #[test]
    fn test_fuzz_kv_cache_empty() {
        fuzz_kv_cache_operations(b"");
    }

    #[test]
    fn test_fuzz_token_empty() {
        fuzz_token_validation(b"");
    }
}

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    fuzz_config_parse(data);
    fuzz_kv_cache_operations(data);
    fuzz_token_validation(data);
});
