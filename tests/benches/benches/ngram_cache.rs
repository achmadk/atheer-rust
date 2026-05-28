use atheer_orchestrator::NGramCache;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn ngram_cache_insert(c: &mut Criterion) {
    let mut cache = NGramCache::new(5, 10000);

    c.bench_function("ngram_cache_insert_5x10000", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let prefix = vec![i as u32, i as u32 + 1, i as u32 + 2];
                let continuation = vec![i as u32 + 3, i as u32 + 4];
                cache.insert(black_box(&prefix), black_box(&continuation));
            }
        })
    });
}

fn ngram_cache_lookup(c: &mut Criterion) {
    let mut cache = NGramCache::new(5, 10000);

    for i in 0..1000 {
        let prefix = vec![i as u32, i as u32 + 1, i as u32 + 2];
        let continuation = vec![i as u32 + 3, i as u32 + 4];
        cache.insert(&prefix, &continuation);
    }

    c.bench_function("ngram_cache_lookup_1000_entries", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let prefix = vec![i as u32, i as u32 + 1, i as u32 + 2];
                black_box(cache.lookup(&prefix));
            }
        })
    });
}

fn ngram_cache_eviction(c: &mut Criterion) {
    let mut cache = NGramCache::new(3, 100);

    c.bench_function("ngram_cache_eviction_3x100", |b| {
        b.iter(|| {
            for i in 0..200 {
                let prefix = vec![i as u32];
                cache.insert(black_box(&prefix), black_box(&[i as u32 + 100]));
            }
        })
    });
}

fn ngram_cache_large_scale(c: &mut Criterion) {
    let mut cache = NGramCache::new(5, 50000);

    c.bench_function("ngram_cache_large_scale_5x50000", |b| {
        b.iter(|| {
            for i in 0..10000 {
                let prefix: Vec<u32> = (i..i + 5).map(|x| x as u32).collect();
                let continuation: Vec<u32> = (i + 5..i + 10).map(|x| x as u32).collect();
                cache.insert(black_box(&prefix), black_box(&continuation));
            }
        })
    });
}

criterion_group!(
    benches,
    ngram_cache_insert,
    ngram_cache_lookup,
    ngram_cache_eviction,
    ngram_cache_large_scale
);
criterion_main!(benches);
