use atheer_memory_bank::KvCache;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn kv_cache_insert(c: &mut Criterion) {
    let mut cache = KvCache::new(32, 8, 128, 2048);
    let keys = vec![1.0f32; 1024];
    let values = vec![2.0f32; 1024];

    c.bench_function("kv_cache_insert_32x8x128", |b| {
        b.iter(|| {
            for layer in 0..32 {
                for pos in 0..128 {
                    cache.insert(
                        black_box(layer),
                        black_box(pos),
                        black_box(42u32),
                        black_box(keys.clone()),
                        black_box(values.clone()),
                    );
                }
            }
        })
    });
}

fn kv_cache_get(c: &mut Criterion) {
    let mut cache = KvCache::new(32, 8, 128, 2048);
    let keys = vec![1.0f32; 1024];
    let values = vec![2.0f32; 1024];

    for layer in 0..32 {
        for pos in 0..128 {
            cache.insert(layer, pos, 42, keys.clone(), values.clone());
        }
    }

    c.bench_function("kv_cache_get_32x8x128", |b| {
        b.iter(|| {
            for layer in 0..32 {
                for pos in 0..128 {
                    black_box(cache.get(layer, pos));
                }
            }
        })
    });
}

fn kv_cache_truncate(c: &mut Criterion) {
    let mut cache = KvCache::new(32, 8, 128, 2048);
    let keys = vec![1.0f32; 1024];
    let values = vec![2.0f32; 1024];

    for layer in 0..32 {
        for pos in 0..1024 {
            cache.insert(layer, pos, 42, keys.clone(), values.clone());
        }
    }

    c.bench_function("kv_cache_truncate_32x1024", |b| {
        b.iter(|| {
            cache.truncate(black_box(512));
        })
    });
}

fn kv_cache_memory_estimate(c: &mut Criterion) {
    let cache = KvCache::new(32, 8, 128, 2048);

    c.bench_function("kv_cache_memory_estimate", |b| {
        b.iter(|| {
            black_box(cache.max_memory_bytes());
        })
    });
}

criterion_group!(
    benches,
    kv_cache_insert,
    kv_cache_get,
    kv_cache_truncate,
    kv_cache_memory_estimate
);
criterion_main!(benches);
