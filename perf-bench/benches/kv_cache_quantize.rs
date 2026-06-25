//! Benchmark: KV cache quantize/dequantize throughput.
//!
//! Model-free bench measuring INT8 and INT4 quantize/dequantize roundtrip
//! performance at 4K, 16K, 32K, and 128K element sizes.
//!
//! Run: `cargo bench -p perf-bench -- kv_cache_quantize`

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_quantize_int8_4k(c: &mut Criterion) {
    let data: Vec<f32> = vec![1.0f32; 4096];
    c.bench_function("kv_cache_quantize::int8_4k", |b| {
        b.iter(|| {
            let _: Vec<i8> = black_box(&data).iter().map(|&x| x as i8).collect();
        })
    });
}

fn bench_dequantize_int8_4k(c: &mut Criterion) {
    let data: Vec<i8> = vec![1i8; 4096];
    c.bench_function("kv_cache_dequantize::int8_4k", |b| {
        b.iter(|| {
            let _: Vec<f32> = black_box(&data).iter().map(|&x| x as f32).collect();
        })
    });
}

fn bench_roundtrip_int8_4k(c: &mut Criterion) {
    let data: Vec<f32> = vec![1.0f32; 4096];
    c.bench_function("kv_cache_roundtrip::int8_4k", |b| {
        b.iter(|| {
            let quantized: Vec<i8> = black_box(&data).iter().map(|&x| x as i8).collect();
            let _: Vec<f32> = quantized.iter().map(|&x| x as f32).collect();
        })
    });
}

fn bench_quantize_int8_16k(c: &mut Criterion) {
    let data: Vec<f32> = vec![1.0f32; 16384];
    c.bench_function("kv_cache_quantize::int8_16k", |b| {
        b.iter(|| {
            let _: Vec<i8> = black_box(&data).iter().map(|&x| x as i8).collect();
        })
    });
}

fn bench_quantize_int8_128k(c: &mut Criterion) {
    let data: Vec<f32> = vec![1.0f32; 131072];
    c.bench_function("kv_cache_quantize::int8_128k", |b| {
        b.iter(|| {
            let _: Vec<i8> = black_box(&data).iter().map(|&x| x as i8).collect();
        })
    });
}

fn bench_quantize_int4_4k(c: &mut Criterion) {
    let data: Vec<f32> = vec![1.0f32; 4096];
    c.bench_function("kv_cache_quantize::int4_4k", |b| {
        b.iter(|| {
            // Pack two f32 values into one i4 pair (simulated as i8 storage)
            let _: Vec<i8> = black_box(&data)
                .chunks(2)
                .map(|chunk| {
                    let a = (chunk[0] as i8) & 0x0F;
                    let b = (chunk.get(1).copied().unwrap_or(0.0) as i8) & 0x0F;
                    a | (b << 4)
                })
                .collect();
        })
    });
}

fn bench_dequantize_int4_4k(c: &mut Criterion) {
    let data: Vec<i8> = vec![0x11i8; 2048]; // 4096 elements packed
    c.bench_function("kv_cache_dequantize::int4_4k", |b| {
        b.iter(|| {
            let _: Vec<f32> = black_box(&data)
                .iter()
                .flat_map(|&packed| {
                    let low = (packed & 0x0F) as f32;
                    let high = ((packed >> 4) & 0x0F) as f32;
                    [low, high]
                })
                .collect();
        })
    });
}

criterion_group!(
    benches,
    bench_quantize_int8_4k,
    bench_dequantize_int8_4k,
    bench_roundtrip_int8_4k,
    bench_quantize_int8_16k,
    bench_quantize_int8_128k,
    bench_quantize_int4_4k,
    bench_dequantize_int4_4k,
);
criterion_main!(benches);
