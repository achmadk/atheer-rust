// KV cache quantization: trait, INT8/INT4 quantizers, adaptive depth, and
// on-pressure downgrade logic.
//
// Each quantized buffer stores the per-tensor scale as a little-endian f32 in
// the first 4 bytes, followed by the packed quantized payload.

/// The numeric format a quantized KV cache buffer is stored in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationScheme {
    /// 8-bit symmetric integer, 1 byte per element.
    Int8,
    /// 4-bit symmetric integer, 2 elements packed per byte.
    Int4,
    /// 16-bit half-precision float, 2 bytes per element.
    Fp16,
    /// Full 32-bit float, 4 bytes per element (no compression).
    Fp32,
}

impl QuantizationScheme {
    pub fn bytes_per_element(&self) -> usize {
        match self {
            Self::Int8 => 1,
            Self::Int4 => 1,
            Self::Fp16 => 2,
            Self::Fp32 => 4,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Int8 => "INT8",
            Self::Int4 => "INT4",
            Self::Fp16 => "FP16",
            Self::Fp32 => "FP32",
        }
    }
}

/// Stateless quantizer that converts between `Vec<f32>` and a compact `Vec<u8>`
/// representation.
///
/// Every quantized buffer starts with a 4-byte little-endian f32 scale followed
/// by the packed quantized payload.  `dequantize` reads the scale, then
/// reconstructs `len` float values from the payload.
pub trait KvCacheQuantizer {
    fn quantize(&self, data: &[f32]) -> Vec<u8>;
    fn dequantize(&self, data: &[u8], len: usize) -> Vec<f32>;
    fn scheme(&self) -> QuantizationScheme;
}

fn read_scale(buf: &[u8; 4]) -> f32 {
    f32::from_le_bytes(*buf)
}

fn symmetric_scale(data: &[f32], max_q: f32) -> f32 {
    let max_abs = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    if max_abs == 0.0 {
        1.0
    } else {
        max_abs / max_q
    }
}

/// Symmetric INT8 quantizer using per-tensor scaling.
///
/// Quantized layout (N elements):
/// ```text
/// [0..4)   scale (f32 LE)
/// [4..4+N) i8 bytes
/// ```
pub struct Int8Quantizer;

impl KvCacheQuantizer for Int8Quantizer {
    fn quantize(&self, data: &[f32]) -> Vec<u8> {
        if data.is_empty() {
            return vec![0u8; 4];
        }
        let scale = symmetric_scale(data, 127.0);
        let inv_scale = scale.recip();

        let mut out = Vec::with_capacity(4 + data.len());
        out.extend_from_slice(&[0u8; 4]);
        for &x in data {
            let q = (x * inv_scale).round().clamp(-128.0, 127.0) as i8;
            out.push(q as u8);
        }
        out[..4].copy_from_slice(&scale.to_le_bytes());
        out
    }

    fn dequantize(&self, data: &[u8], len: usize) -> Vec<f32> {
        if len == 0 {
            return vec![];
        }
        let scale = read_scale(&data[..4].try_into().expect("scale prefix"));
        let payload = &data[4..];
        assert!(
            payload.len() >= len,
            "Int8 dequantize: expected {} bytes, got {}",
            len,
            payload.len()
        );

        let mut out = Vec::with_capacity(len);
        for &b in &payload[..len] {
            out.push((b as i8) as f32 * scale);
        }
        out
    }

    fn scheme(&self) -> QuantizationScheme {
        QuantizationScheme::Int8
    }
}

/// Symmetric INT4 quantizer using per-tensor scaling, packed 2-per-byte.
///
/// Quantized layout (N elements):
/// ```text
/// [0..4)       scale (f32 LE)
/// [4..4+M)     packed nibbles, M = (N + 1) / 2
/// ```
///
/// Packing order per byte: low nibble → first element, high nibble → second.
pub struct Int4Quantizer;

fn clamp_i4(x: f32) -> u8 {
    let v = x.round().clamp(-8.0, 7.0) as i8;
    (v as u8) & 0x0F
}

fn sign_extend_nibble(nibble: u8) -> i8 {
    if nibble & 0x08 != 0 {
        (nibble | 0xF0) as i8
    } else {
        nibble as i8
    }
}

impl KvCacheQuantizer for Int4Quantizer {
    fn quantize(&self, data: &[f32]) -> Vec<u8> {
        if data.is_empty() {
            return vec![0u8; 4];
        }
        let scale = symmetric_scale(data, 7.0);
        let inv_scale = scale.recip();
        let packed_len = data.len().div_ceil(2);

        let mut out = Vec::with_capacity(4 + packed_len);
        out.extend_from_slice(&[0u8; 4]);

        for chunk in data.chunks(2) {
            let lo = clamp_i4(chunk[0] * inv_scale);
            let hi = if chunk.len() > 1 {
                clamp_i4(chunk[1] * inv_scale)
            } else {
                0
            };
            out.push(lo | (hi << 4));
        }

        out[..4].copy_from_slice(&scale.to_le_bytes());
        out
    }

    fn dequantize(&self, data: &[u8], len: usize) -> Vec<f32> {
        if len == 0 {
            return vec![];
        }
        let scale = read_scale(&data[..4].try_into().expect("scale prefix"));
        let payload = &data[4..];
        let needed = len.div_ceil(2);
        assert!(
            payload.len() >= needed,
            "Int4 dequantize: expected {} bytes, got {}",
            needed,
            payload.len()
        );

        let mut out = Vec::with_capacity(len);
        for &byte in payload.iter() {
            let lo = sign_extend_nibble(byte & 0x0F);
            out.push((lo as f32) * scale);
            if out.len() >= len {
                break;
            }
            let hi = sign_extend_nibble((byte >> 4) & 0x0F);
            out.push((hi as f32) * scale);
            if out.len() >= len {
                break;
            }
        }
        out
    }

    fn scheme(&self) -> QuantizationScheme {
        QuantizationScheme::Int4
    }
}

/// Adapts quantization depth based on KV cache sequence length:
///
/// | seq_len    | scheme |
/// |------------|--------|
/// | 4096+      | Int4   |
/// | 2048+      | Int8   |
/// | 1024+      | Fp16   |
/// | 0..1024    | Fp32   |
pub struct AdaptiveQuantizer {
    pub seq_len: usize,
}

impl AdaptiveQuantizer {
    pub fn select_scheme(seq_len: usize) -> QuantizationScheme {
        if seq_len >= 4096 {
            QuantizationScheme::Int4
        } else if seq_len >= 2048 {
            QuantizationScheme::Int8
        } else if seq_len >= 1024 {
            QuantizationScheme::Fp16
        } else {
            QuantizationScheme::Fp32
        }
    }

    pub fn quantizer_for(scheme: QuantizationScheme) -> Box<dyn KvCacheQuantizer> {
        match scheme {
            QuantizationScheme::Int8 => Box::new(Int8Quantizer),
            QuantizationScheme::Int4 => Box::new(Int4Quantizer),
            QuantizationScheme::Fp16 | QuantizationScheme::Fp32 => Box::new(IdentityQuantizer),
        }
    }

    pub fn current_quantizer(&self) -> Box<dyn KvCacheQuantizer> {
        Self::quantizer_for(Self::select_scheme(self.seq_len))
    }
}

/// Helpers for deciding when to downgrade the quantization scheme under VRAM
/// pressure (> 80%).
pub struct OnPressureQuantizer;

impl OnPressureQuantizer {
    pub fn downgrade_scheme(
        current: QuantizationScheme,
        vram_pct: f32,
    ) -> Option<QuantizationScheme> {
        if vram_pct <= 80.0 {
            return None;
        }
        match current {
            QuantizationScheme::Fp32 => Some(QuantizationScheme::Fp16),
            QuantizationScheme::Fp16 => Some(QuantizationScheme::Int8),
            QuantizationScheme::Int8 => Some(QuantizationScheme::Int4),
            QuantizationScheme::Int4 => None,
        }
    }

    pub fn downgrade_quantizer(
        current: QuantizationScheme,
        vram_pct: f32,
    ) -> Option<Box<dyn KvCacheQuantizer>> {
        Self::downgrade_scheme(current, vram_pct).map(AdaptiveQuantizer::quantizer_for)
    }

    pub fn savings_factor(current: QuantizationScheme, vram_pct: f32) -> f32 {
        match Self::downgrade_scheme(current, vram_pct) {
            Some(QuantizationScheme::Fp16) => 2.0,
            Some(QuantizationScheme::Int8) => 2.0,
            Some(QuantizationScheme::Int4) => 2.0,
            Some(QuantizationScheme::Fp32) => 1.0,
            None => 1.0,
        }
    }
}

/// Passthrough quantizer that stores f32 data verbatim.
struct IdentityQuantizer;

impl KvCacheQuantizer for IdentityQuantizer {
    fn quantize(&self, data: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + data.len() * 4);
        out.extend_from_slice(&1.0f32.to_le_bytes());
        for &x in data {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }

    fn dequantize(&self, data: &[u8], len: usize) -> Vec<f32> {
        if len == 0 {
            return vec![];
        }
        let _scale = read_scale(&data[..4].try_into().expect("scale prefix"));
        let payload = &data[4..];
        assert!(
            payload.len() >= len * 4,
            "Identity dequantize: expected {} bytes, got {}",
            len * 4,
            payload.len()
        );
        let mut out = Vec::with_capacity(len);
        for &chunk in payload.as_chunks::<4>().0 {
            if out.len() >= len {
                break;
            }
            out.push(f32::from_le_bytes(chunk));
        }
        out
    }

    fn scheme(&self) -> QuantizationScheme {
        QuantizationScheme::Fp32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int8_roundtrip_errors() {
        let q = Int8Quantizer;
        let data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.1).collect();
        let encoded = q.quantize(&data);
        let decoded = q.dequantize(&encoded, data.len());
        assert_eq!(decoded.len(), data.len());
        assert!(normalized_error(&data, &decoded) < 0.01);
    }

    #[test]
    fn test_int8_roundtrip_extremes() {
        let q = Int8Quantizer;
        let data = vec![-1000.0, 0.0, 1000.0];
        let encoded = q.quantize(&data);
        let decoded = q.dequantize(&encoded, data.len());
        assert_eq!(decoded.len(), 3);
        assert!(decoded[0] < 0.0);
        assert!(decoded[2] > 0.0);
    }

    #[test]
    fn test_int8_empty() {
        let q = Int8Quantizer;
        let encoded = q.quantize(&[]);
        assert_eq!(encoded.len(), 4);
        assert!(q.dequantize(&encoded, 0).is_empty());
    }

    #[test]
    fn test_int4_roundtrip_errors() {
        let q = Int4Quantizer;
        let data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.1).collect();
        let encoded = q.quantize(&data);
        let decoded = q.dequantize(&encoded, data.len());
        assert_eq!(decoded.len(), data.len());
        assert!(normalized_error(&data, &decoded) < 0.04);
    }

    #[test]
    fn test_int4_odd_count() {
        let q = Int4Quantizer;
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let encoded = q.quantize(&data);
        assert_eq!(encoded.len(), 4 + 2);
        let decoded = q.dequantize(&encoded, data.len());
        assert_eq!(decoded.len(), 3);
    }

    #[test]
    fn test_int4_empty() {
        let q = Int4Quantizer;
        assert_eq!(q.quantize(&[]).len(), 4);
        assert!(q.dequantize(&q.quantize(&[]), 0).is_empty());
    }

    #[test]
    fn test_int4_single_value() {
        let q = Int4Quantizer;
        let data = vec![3.5];
        let decoded = q.dequantize(&q.quantize(&data), data.len());
        assert_eq!(decoded.len(), 1);
        assert!(normalized_error(&data, &decoded) < 0.04);
    }

    #[test]
    fn test_adaptive_select_scheme() {
        assert_eq!(
            AdaptiveQuantizer::select_scheme(0),
            QuantizationScheme::Fp32
        );
        assert_eq!(
            AdaptiveQuantizer::select_scheme(512),
            QuantizationScheme::Fp32
        );
        assert_eq!(
            AdaptiveQuantizer::select_scheme(1024),
            QuantizationScheme::Fp16
        );
        assert_eq!(
            AdaptiveQuantizer::select_scheme(2048),
            QuantizationScheme::Int8
        );
        assert_eq!(
            AdaptiveQuantizer::select_scheme(4096),
            QuantizationScheme::Int4
        );
        assert_eq!(
            AdaptiveQuantizer::select_scheme(8192),
            QuantizationScheme::Int4
        );
    }

    #[test]
    fn test_downgrade_below_threshold() {
        assert_eq!(
            OnPressureQuantizer::downgrade_scheme(QuantizationScheme::Int8, 50.0),
            None
        );
    }

    #[test]
    fn test_downgrade_at_threshold() {
        assert_eq!(
            OnPressureQuantizer::downgrade_scheme(QuantizationScheme::Fp32, 85.0),
            Some(QuantizationScheme::Fp16)
        );
        assert_eq!(
            OnPressureQuantizer::downgrade_scheme(QuantizationScheme::Int8, 90.0),
            Some(QuantizationScheme::Int4)
        );
    }

    #[test]
    fn test_downgrade_at_lowest() {
        assert_eq!(
            OnPressureQuantizer::downgrade_scheme(QuantizationScheme::Int4, 95.0),
            None
        );
    }

    #[test]
    fn test_int8_roundtrip_random_like() {
        let q = Int8Quantizer;
        let data: Vec<f32> = (0..256).map(|i| ((i as f32) / 128.0) - 1.0).collect();
        let decoded = q.dequantize(&q.quantize(&data), data.len());
        assert!(normalized_error(&data, &decoded) < 0.01);
    }

    #[test]
    fn test_int4_roundtrip_random_like() {
        let q = Int4Quantizer;
        let data: Vec<f32> = (0..256).map(|i| ((i as f32) / 128.0) - 1.0).collect();
        let decoded = q.dequantize(&q.quantize(&data), data.len());
        assert!(normalized_error(&data, &decoded) < 0.04);
    }

    #[test]
    fn test_identity_roundtrip() {
        let q = IdentityQuantizer;
        let data: Vec<f32> = (0..50).map(|i| i as f32 * 0.5).collect();
        assert_eq!(q.dequantize(&q.quantize(&data), data.len()), data);
    }

    fn normalized_error(original: &[f32], decoded: &[f32]) -> f64 {
        let peak = original
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max)
            .max(1e-8);
        let mean_abs_err: f64 = original
            .iter()
            .zip(decoded)
            .map(|(a, b)| (a - b).abs() as f64)
            .sum::<f64>()
            / original.len() as f64;
        mean_abs_err / peak as f64
    }
}
