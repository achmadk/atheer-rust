use crate::kv_cache_quantizer::{
    AdaptiveQuantizer, KvCacheQuantizer, OnPressureQuantizer, QuantizationScheme,
};

/// A quantized KV cache that stores per-layer key/value data in compact form.
pub struct KvCache {
    layers: Vec<LayerCache>,
    scheme: QuantizationScheme,
}

#[derive(Clone)]
struct LayerCache {
    quantized_keys: Vec<u8>,
    quantized_values: Vec<u8>,
    logical_len: usize,
}

impl KvCache {
    pub fn new(num_layers: usize, scheme: QuantizationScheme) -> Self {
        Self {
            layers: vec![
                LayerCache {
                    quantized_keys: vec![],
                    quantized_values: vec![],
                    logical_len: 0,
                };
                num_layers
            ],
            scheme,
        }
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn scheme(&self) -> QuantizationScheme {
        self.scheme
    }

    pub fn logical_len(&self, layer: usize) -> Option<usize> {
        self.layers.get(layer).map(|l| l.logical_len)
    }

    pub fn load_snapshot(
        &mut self,
        snapshot: &[(Vec<f32>, Vec<f32>)],
        quantizer: &dyn KvCacheQuantizer,
    ) {
        assert_eq!(snapshot.len(), self.layers.len(),
            "KvCache::load_snapshot: expected {} layers, got {}",
            self.layers.len(), snapshot.len());

        for (i, (keys, vals)) in snapshot.iter().enumerate() {
            self.layers[i] = LayerCache {
                quantized_keys: quantizer.quantize(keys),
                quantized_values: quantizer.quantize(vals),
                logical_len: keys.len().max(vals.len()),
            };
        }
        self.scheme = quantizer.scheme();
    }

    pub fn to_snapshot(&self, quantizer: &dyn KvCacheQuantizer) -> Vec<(Vec<f32>, Vec<f32>)> {
        self.layers
            .iter()
            .map(|layer| {
                let keys = if layer.logical_len == 0 {
                    vec![]
                } else {
                    let logical_len = payload_logical_len(&layer.quantized_keys, self.scheme);
                    quantizer.dequantize(&layer.quantized_keys, logical_len)
                };
                let vals = if layer.logical_len == 0 {
                    vec![]
                } else {
                    let logical_len = payload_logical_len(&layer.quantized_values, self.scheme);
                    quantizer.dequantize(&layer.quantized_values, logical_len)
                };
                (keys, vals)
            })
            .collect()
    }

    pub fn insert(
        &mut self,
        layer: usize,
        keys: &[f32],
        vals: &[f32],
        quantizer: &dyn KvCacheQuantizer,
    ) {
        assert!(layer < self.layers.len(), "KvCache::insert: layer {layer} out of bounds");

        let lc = &mut self.layers[layer];
        if lc.logical_len == 0 {
            lc.quantized_keys = quantizer.quantize(keys);
            lc.quantized_values = quantizer.quantize(vals);
            lc.logical_len = keys.len().max(vals.len());
        } else {
            let deq_keys = quantizer.dequantize(&lc.quantized_keys, lc.logical_len);
            let deq_vals = quantizer.dequantize(&lc.quantized_values, lc.logical_len);

            let mut merged_keys = deq_keys;
            merged_keys.extend_from_slice(keys);
            let mut merged_vals = deq_vals;
            merged_vals.extend_from_slice(vals);

            lc.logical_len = merged_keys.len().max(merged_vals.len());
            lc.quantized_keys = quantizer.quantize(&merged_keys);
            lc.quantized_values = quantizer.quantize(&merged_vals);
        }
    }

    pub fn get(
        &self,
        layer: usize,
        quantizer: &dyn KvCacheQuantizer,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        let lc = self.layers.get(layer)?;
        if lc.logical_len == 0 {
            return Some((vec![], vec![]));
        }
        let key_len = payload_logical_len(&lc.quantized_keys, self.scheme);
        let val_len = payload_logical_len(&lc.quantized_values, self.scheme);

        let keys = quantizer.dequantize(&lc.quantized_keys, key_len);
        let vals = quantizer.dequantize(&lc.quantized_values, val_len);
        Some((keys, vals))
    }

    pub fn requantize(&mut self, new_quantizer: &dyn KvCacheQuantizer) {
        let old_quantizer = AdaptiveQuantizer::quantizer_for(self.scheme);
        for layer in &mut self.layers {
            if layer.logical_len == 0 {
                continue;
            }
            let deq_keys = old_quantizer.dequantize(&layer.quantized_keys, layer.logical_len);
            let deq_vals = old_quantizer.dequantize(&layer.quantized_values, layer.logical_len);
            layer.quantized_keys = new_quantizer.quantize(&deq_keys);
            layer.quantized_values = new_quantizer.quantize(&deq_vals);
        }
        self.scheme = new_quantizer.scheme();
    }

    pub fn downgrade_on_pressure(&mut self, vram_pct: f32) -> bool {
        if let Some(new_scheme) = OnPressureQuantizer::downgrade_scheme(self.scheme, vram_pct) {
            let new_quantizer = AdaptiveQuantizer::quantizer_for(new_scheme);
            self.requantize(&*new_quantizer);
            true
        } else {
            false
        }
    }

    pub fn adapt_to_len(&mut self, seq_len: usize) -> bool {
        let new_scheme = AdaptiveQuantizer::select_scheme(seq_len);
        if new_scheme == self.scheme {
            return false;
        }
        let new_quantizer = AdaptiveQuantizer::quantizer_for(new_scheme);
        self.requantize(&*new_quantizer);
        true
    }

    pub fn vram_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.quantized_keys.len() + l.quantized_values.len())
            .sum()
    }
}

fn payload_logical_len(data: &[u8], scheme: QuantizationScheme) -> usize {
    if data.len() <= 4 {
        return 0;
    }
    let payload = &data[4..];
    match scheme {
        QuantizationScheme::Int4 => payload.len() * 2,
        QuantizationScheme::Int8 => payload.len(),
        QuantizationScheme::Fp16 | QuantizationScheme::Fp32 => payload.len() / 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache_quantizer::{Int8Quantizer, Int4Quantizer};

    fn dummy_snapshot(num_layers: usize, tokens: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
        (0..num_layers)
            .map(|l| {
                let keys: Vec<f32> = (0..tokens).map(|i| (i as f32 + l as f32) * 0.1).collect();
                let vals: Vec<f32> = (0..tokens).map(|i| (i as f32 - l as f32) * 0.1).collect();
                (keys, vals)
            })
            .collect()
    }

    #[test]
    fn test_kv_cache_roundtrip_int8() {
        let q = Int8Quantizer;
        let snap = dummy_snapshot(4, 100);
        let mut cache = KvCache::new(4, QuantizationScheme::Int8);
        cache.load_snapshot(&snap, &q);

        let restored = cache.to_snapshot(&q);
        assert_eq!(restored.len(), 4);
        for (i, (orig_keys, _)) in snap.iter().enumerate() {
            let (rest_keys, _) = &restored[i];
            assert!(!rest_keys.is_empty());
            assert_eq!(rest_keys.len(), orig_keys.len());
        }
    }

    #[test]
    fn test_kv_cache_insert_get() {
        let q = Int8Quantizer;
        let mut cache = KvCache::new(2, QuantizationScheme::Int8);

        cache.insert(0, &[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &q);
        cache.insert(0, &[7.0, 8.0], &[9.0, 10.0], &q);

        let (keys, vals) = cache.get(0, &q).unwrap();
        assert_eq!(keys.len(), 5);
        assert_eq!(vals.len(), 5);
        assert!((keys[0] - 1.0).abs() < 0.15);
        assert!((keys[4] - 8.0).abs() < 0.15);
    }

    #[test]
    fn test_kv_cache_out_of_bounds_layer() {
        let q = Int8Quantizer;
        let cache = KvCache::new(2, QuantizationScheme::Int8);
        assert!(cache.get(5, &q).is_none());
    }

    #[test]
    fn test_kv_cache_empty_initial() {
        let q = Int8Quantizer;
        let cache = KvCache::new(2, QuantizationScheme::Int8);
        let snap = cache.to_snapshot(&q);
        assert_eq!(snap.len(), 2);
        assert!(snap[0].0.is_empty());
        assert!(snap[0].1.is_empty());
    }

    #[test]
    fn test_kv_cache_adapt_to_len() {
        let mut cache = KvCache::new(2, QuantizationScheme::Int8);
        let snap = dummy_snapshot(2, 50);
        cache.load_snapshot(&snap, &Int8Quantizer);

        assert!(cache.adapt_to_len(4096));
        assert_eq!(cache.scheme(), QuantizationScheme::Int4);

        assert!(cache.adapt_to_len(512));
        assert_eq!(cache.scheme(), QuantizationScheme::Fp32);
    }

    #[test]
    fn test_kv_cache_adapt_noop() {
        let mut cache = KvCache::new(2, QuantizationScheme::Int8);
        let snap = dummy_snapshot(2, 50);
        cache.load_snapshot(&snap, &Int8Quantizer);

        assert!(!cache.adapt_to_len(2048));
        assert_eq!(cache.scheme(), QuantizationScheme::Int8);
    }

    #[test]
    fn test_kv_cache_downgrade_on_pressure() {
        let mut cache = KvCache::new(2, QuantizationScheme::Int8);
        let snap = dummy_snapshot(2, 100);
        cache.load_snapshot(&snap, &Int8Quantizer);

        assert!(!cache.downgrade_on_pressure(50.0));
        assert_eq!(cache.scheme(), QuantizationScheme::Int8);

        assert!(cache.downgrade_on_pressure(90.0));
        assert_eq!(cache.scheme(), QuantizationScheme::Int4);

        assert!(!cache.downgrade_on_pressure(95.0));
    }

    #[test]
    fn test_kv_cache_vram_bytes() {
        let mut cache = KvCache::new(4, QuantizationScheme::Int8);
        let snap = dummy_snapshot(4, 50);
        cache.load_snapshot(&snap, &Int8Quantizer);

        let vram = cache.vram_bytes();
        assert!(vram > 0);
        let f32_est = cache.num_layers() * 50 * 2 * 4;
        assert!(f32_est > vram);
    }

    #[test]
    fn test_kv_cache_requantize_preserves_content() {
        let q_int8 = Int8Quantizer;
        let mut cache = KvCache::new(2, QuantizationScheme::Int8);
        let snap = dummy_snapshot(2, 50);
        cache.load_snapshot(&snap, &q_int8);

        let baseline = cache.to_snapshot(&q_int8);

        let q_int4 = Int4Quantizer;
        cache.requantize(&q_int4);
        assert_eq!(cache.scheme(), QuantizationScheme::Int4);
        cache.requantize(&q_int8);
        assert_eq!(cache.scheme(), QuantizationScheme::Int8);

        let final_snap = cache.to_snapshot(&q_int8);
        for ((bk, bv), (fk, fv)) in baseline.iter().zip(final_snap.iter()) {
            let ke: f64 = bk.iter().zip(fk).map(|(a, b)| (a - b).abs() as f64).sum::<f64>() / bk.len() as f64;
            assert!(ke < 0.25, "key error too large: {ke}");
            let ve: f64 = bv.iter().zip(fv).map(|(a, b)| (a - b).abs() as f64).sum::<f64>() / bv.len() as f64;
            assert!(ve < 0.25, "val error too large: {ve}");
        }
    }
}
