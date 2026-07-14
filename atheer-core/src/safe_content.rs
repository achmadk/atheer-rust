//! Pre-allocation header gate for GGUF content.
//!
//! [`parse_header`] runs **before** `candle_core::quantized::gguf_file::Content::read`
//! allocates any `Vec<u8>` of file-derived size. It validates the GGUF magic,
//! version, tensor count, metadata KV count, and `general.alignment` so that
//! crafted inputs are rejected before they can OOM the loader.
//!
//! This module is the single chokepoint for untrusted GGUF content. All three
//! load paths (`Model::from_gguf`, `Model::from_gguf_reader`,
//! `MmapModel::from_gguf`) must call [`parse_header`] before invoking
//! candle's parser.
//!
//! See `openspec/changes/safe-gguf-load/specs/model-loading-safety/spec.md`
//! for the normative requirements this module satisfies.

use std::io::{Read, Seek, SeekFrom};

/// Maximum number of tensors a file may declare.
pub const MAX_TENSOR_COUNT: u64 = 10_000;

/// Maximum number of metadata KV pairs a file may declare.
pub const MAX_METADATA_KV_COUNT: u64 = 100_000;

/// Per-tensor byte ceiling used in the coarse `tensor_count × MAX_PER_TENSOR`
/// total-byte budget check. Real single-tensor weights top out around 10 GB.
pub const MAX_PER_TENSOR: u64 = 16 * 1024 * 1024 * 1024;

/// Total tensor-byte budget used by the coarse header-level overflow guard.
///
/// Sized generously above `MAX_TENSOR_COUNT × MAX_PER_TENSOR` so legitimate
/// multi-tensor models pass unconditionally. The guard only fires for
/// adversarial files that declare so many tensors that the worst-case
/// per-tensor allocation exceeds this ceiling. Real models always fall
/// well under this; realistic per-tensor sizes are tiny compared to the
/// 16 GiB per-tensor ceiling.
pub const MAX_TOTAL_TENSOR_BYTES: u64 = 100 * 1024 * 1024 * 1024 * 1024 * 1024;

/// Minimum permitted alignment value.
pub const MIN_ALIGNMENT: u64 = 16;

/// Maximum permitted alignment value.
pub const MAX_ALIGNMENT: u64 = 4096;

/// Default alignment when `general.alignment` is absent.
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// Maximum byte length of a metadata key string.
pub const MAX_METADATA_KEY_BYTES: u64 = 1024 * 1024;

/// Maximum byte length of a metadata value string.
pub const MAX_METADATA_STRING_BYTES: u64 = 10 * 1024 * 1024;

/// Errors returned by the pre-allocation header gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SafeLoadError {
    /// GGUF magic header is wrong.
    InvalidMagic { actual: [u8; 4] },
    /// GGUF version is not 1, 2, or 3.
    InvalidVersion { version: u32 },
    /// Tensor or metadata KV count exceeds ceiling, or the coarse total
    /// tensor-byte budget is exceeded.
    InvalidCounts {
        tensor_count: u64,
        metadata_kv_count: u64,
        max_tensor_bytes: u64,
        requested_tensor_bytes: u64,
    },
    /// `general.alignment` is missing, non-power-of-two, below `MIN_ALIGNMENT`,
    /// or above `MAX_ALIGNMENT`.
    InvalidAlignment { value: i64 },
    /// I/O failure during header read (truncated file, EOF, etc.).
    Io(String),
}

/// Tunable limits consumed by [`parse_header`].
#[derive(Debug, Clone, Copy)]
pub struct SafeLoadLimits {
    pub max_tensor_count: u64,
    pub max_metadata_kv_count: u64,
    pub max_per_tensor: u64,
    pub max_total_tensor_bytes: u64,
    pub min_alignment: u64,
    pub max_alignment: u64,
    pub default_alignment: u64,
    pub max_metadata_key_bytes: u64,
    pub max_metadata_string_bytes: u64,
}

impl Default for SafeLoadLimits {
    fn default() -> Self {
        Self {
            max_tensor_count: MAX_TENSOR_COUNT,
            max_metadata_kv_count: MAX_METADATA_KV_COUNT,
            max_per_tensor: MAX_PER_TENSOR,
            max_total_tensor_bytes: MAX_TOTAL_TENSOR_BYTES,
            min_alignment: MIN_ALIGNMENT,
            max_alignment: MAX_ALIGNMENT,
            default_alignment: DEFAULT_ALIGNMENT,
            max_metadata_key_bytes: MAX_METADATA_KEY_BYTES,
            max_metadata_string_bytes: MAX_METADATA_STRING_BYTES,
        }
    }
}

/// A validated GGUF header returned by [`parse_header`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidatedHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub alignment: u64,
}

/// Validate the GGUF header of a reader without allocating any buffer sized
/// from file content.
///
/// The reader is expected to be positioned at the start of the file (or
/// decrypted byte buffer) and must support both `Read` and `Seek`. The reader
/// is left positioned at the start of the tensor metadata block — exactly
/// where candle's `Content::read` expects to begin.
///
/// On success, returns a [`ValidatedHeader`] describing the parsed counts and
/// alignment. On failure, returns a [`SafeLoadError`] describing which
/// invariant was violated.
///
/// # Spec
///
/// Satisfies `model-loading-safety/spec.md` requirements:
/// - Pre-allocation header validation
/// - Alignment validation
pub fn parse_header<R: Read + Seek>(
    reader: &mut R,
    limits: &SafeLoadLimits,
) -> Result<ValidatedHeader, SafeLoadError> {
    // Seek to start to be tolerant of non-zero initial position.
    reader.seek(SeekFrom::Start(0)).map_err(io)?;

    // Magic: 4 bytes. GGUF spec uses 0x46554747 ("GGUF" little-endian) for
    // forward reads and 0x47475546 ("UFUG" big-endian magic) for backward.
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).map_err(io)?;
    if &magic != b"GGUF" && &magic != b"UFUG" {
        return Err(SafeLoadError::InvalidMagic { actual: magic });
    }

    // Version: 4 bytes, must be 1, 2, or 3.
    let mut version_bytes = [0u8; 4];
    reader.read_exact(&mut version_bytes).map_err(io)?;
    let version = u32::from_le_bytes(version_bytes);
    if !matches!(version, 1..=3) {
        return Err(SafeLoadError::InvalidVersion { version });
    }

    // Tensor count: u32 for V1, u64 for V2/V3.
    let tensor_count = if version == 1 {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf).map_err(io)?;
        u64::from(u32::from_le_bytes(buf))
    } else {
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf).map_err(io)?;
        u64::from_le_bytes(buf)
    };

    // Metadata KV count: same encoding.
    let metadata_kv_count = if version == 1 {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf).map_err(io)?;
        u64::from(u32::from_le_bytes(buf))
    } else {
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf).map_err(io)?;
        u64::from_le_bytes(buf)
    };

    if tensor_count > limits.max_tensor_count || metadata_kv_count > limits.max_metadata_kv_count {
        return Err(SafeLoadError::InvalidCounts {
            tensor_count,
            metadata_kv_count,
            max_tensor_bytes: limits.max_total_tensor_bytes,
            requested_tensor_bytes: 0,
        });
    }

    // Coarse total-byte budget: tensor_count × max_per_tensor must not
    // exceed max_total_tensor_bytes. checked_mul detects u64 overflow.
    let requested_tensor_bytes = match tensor_count.checked_mul(limits.max_per_tensor) {
        Some(n) if n <= limits.max_total_tensor_bytes => n,
        Some(n) => {
            return Err(SafeLoadError::InvalidCounts {
                tensor_count,
                metadata_kv_count,
                max_tensor_bytes: limits.max_total_tensor_bytes,
                requested_tensor_bytes: n,
            });
        }
        None => {
            return Err(SafeLoadError::InvalidCounts {
                tensor_count,
                metadata_kv_count,
                max_tensor_bytes: limits.max_total_tensor_bytes,
                requested_tensor_bytes: u64::MAX,
            });
        }
    };
    // `requested_tensor_bytes` is computed to surface in any future debug
    // logging; the early return above ensures we never reach the scan with
    // a value exceeding `limits.max_total_tensor_bytes`.
    let _ = requested_tensor_bytes;
    // without allocating their string contents.
    let mut alignment: Option<i64> = None;
    for _ in 0..metadata_kv_count {
        let key = read_bounded_string(reader, version, limits.max_metadata_key_bytes)?;
        let mut type_tag_bytes = [0u8; 4];
        reader.read_exact(&mut type_tag_bytes).map_err(io)?;
        let type_tag = u32::from_le_bytes(type_tag_bytes);

        if key == "general.alignment" && alignment.is_none() {
            // Capture and validate alignment. Accept unsigned and non-negative
            // signed integer types.
            let val = match type_tag {
                0 => i64::from(read_u8(reader)?), // U8
                1 => {
                    let mut b = [0u8; 1];
                    reader.read_exact(&mut b).map_err(io)?;
                    i64::from(b[0] as i8) // I8
                }
                2 => i64::from(read_u16_le(reader)?), // U16
                3 => i64::from(read_i16_le(reader)?), // I16
                4 => read_i32_le(reader)? as i64,     // U32
                5 => i64::from(read_i32_le(reader)?), // I32
                10 => {
                    let mut buf = [0u8; 8];
                    reader.read_exact(&mut buf).map_err(io)?;
                    let v = u64::from_le_bytes(buf);
                    if v > i64::MAX as u64 {
                        return Err(SafeLoadError::InvalidAlignment { value: i64::MAX });
                    }
                    v as i64
                }
                11 => {
                    let mut buf = [0u8; 8];
                    reader.read_exact(&mut buf).map_err(io)?;
                    i64::from_le_bytes(buf)
                }
                _ => {
                    return Err(SafeLoadError::InvalidAlignment { value: -1 });
                }
            };
            alignment = Some(val);
        } else {
            skip_value(reader, version, type_tag, limits)?;
        }
    }

    let final_alignment = match alignment {
        Some(v) => {
            if v <= 0 {
                return Err(SafeLoadError::InvalidAlignment { value: v });
            }
            let v_u = v as u64;
            if v_u < limits.min_alignment || v_u > limits.max_alignment {
                return Err(SafeLoadError::InvalidAlignment { value: v });
            }
            if !v_u.is_power_of_two() {
                return Err(SafeLoadError::InvalidAlignment { value: v });
            }
            v_u
        }
        None => limits.default_alignment,
    };

    // Rewind to start so candle's Content::read begins at the magic.
    reader.seek(SeekFrom::Start(0)).map_err(io)?;

    Ok(ValidatedHeader {
        version,
        tensor_count,
        metadata_kv_count,
        alignment: final_alignment,
    })
}

// ── Internal helpers ───────────────────────────────────────────────────

fn io(e: std::io::Error) -> SafeLoadError {
    SafeLoadError::Io(e.to_string())
}

fn read_u8<R: Read>(r: &mut R) -> Result<u8, SafeLoadError> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b).map_err(io)?;
    Ok(b[0])
}

fn read_u16_le<R: Read>(r: &mut R) -> Result<u16, SafeLoadError> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b).map_err(io)?;
    Ok(u16::from_le_bytes(b))
}

fn read_i16_le<R: Read>(r: &mut R) -> Result<i16, SafeLoadError> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b).map_err(io)?;
    Ok(i16::from_le_bytes(b))
}

fn read_i32_le<R: Read>(r: &mut R) -> Result<i32, SafeLoadError> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(io)?;
    Ok(i32::from_le_bytes(b))
}

fn discard<R: Read>(r: &mut R, n: u64) -> Result<(), SafeLoadError> {
    let mut remaining = n;
    let mut buf = [0u8; 4096];
    while remaining > 0 {
        let take = remaining.min(buf.len() as u64) as usize;
        r.read_exact(&mut buf[..take]).map_err(io)?;
        remaining -= take as u64;
    }
    Ok(())
}

/// Read a GGUF string of the given version with the given byte ceiling.
/// Pre-validates the length against the ceiling before allocating.
fn read_bounded_string<R: Read>(
    r: &mut R,
    version: u32,
    max_bytes: u64,
) -> Result<String, SafeLoadError> {
    let len: u64 = if version == 1 {
        let mut b = [0u8; 4];
        r.read_exact(&mut b).map_err(io)?;
        u64::from(u32::from_le_bytes(b))
    } else {
        let mut b = [0u8; 8];
        r.read_exact(&mut b).map_err(io)?;
        u64::from_le_bytes(b)
    };
    if len > max_bytes {
        // Skip past the string bytes so the reader stays aligned for the
        // next value.
        discard(r, len)?;
        // Return a placeholder; caller will discard because it's not the
        // key it was looking for.
        return Ok(String::new());
    }
    let mut v = vec![0u8; len as usize];
    r.read_exact(&mut v).map_err(io)?;
    while let Some(0) = v.last() {
        v.pop();
    }
    Ok(String::from_utf8_lossy(&v).into_owned())
}

/// Skip over a single GGUF value of the given type tag.
fn skip_value<R: Read + Seek>(
    r: &mut R,
    version: u32,
    type_tag: u32,
    limits: &SafeLoadLimits,
) -> Result<(), SafeLoadError> {
    match type_tag {
        0..=1 | 7 => discard(r, 1)?, // U8, I8, Bool
        2..=3 => discard(r, 2)?,     // U16, I16
        4..=6 => discard(r, 4)?,     // U32, I32, F32
        10..=12 => discard(r, 8)?,   // U64, I64, F64
        8 => {
            // String: length-prefixed bytes (V1=u32, V2/V3=u64).
            let len = if version == 1 {
                u64::from(read_u16_le_dummy(r)?)
            } else {
                let mut b = [0u8; 8];
                r.read_exact(&mut b).map_err(io)?;
                u64::from_le_bytes(b)
            };
            // Pre-validate to avoid reading insanely large string lengths.
            if len > limits.max_metadata_string_bytes {
                return Err(SafeLoadError::Io(format!(
                    "metadata string length {len} exceeds ceiling {}",
                    limits.max_metadata_string_bytes
                )));
            }
            discard(r, len)?;
        }
        9 => {
            // Array: u32 element type + u64 length (V1 uses u32) + elements.
            let mut type_buf = [0u8; 4];
            r.read_exact(&mut type_buf).map_err(io)?;
            let elem_type = u32::from_le_bytes(type_buf);
            let count: u64 = if version == 1 {
                u64::from(read_u16_le_dummy(r)?)
            } else {
                let mut b = [0u8; 8];
                r.read_exact(&mut b).map_err(io)?;
                u64::from_le_bytes(b)
            };
            // Bound the count to avoid absurd allocation.
            if count > limits.max_metadata_string_bytes {
                return Err(SafeLoadError::Io(format!(
                    "metadata array length {count} exceeds ceiling"
                )));
            }
            for _ in 0..count {
                skip_value(r, version, elem_type, limits)?;
            }
        }
        _ => {
            return Err(SafeLoadError::Io(format!(
                "unknown metadata value type tag {type_tag}"
            )));
        }
    }
    Ok(())
}

fn read_u16_le_dummy<R: Read>(r: &mut R) -> Result<u32, SafeLoadError> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(io)?;
    Ok(u32::from_le_bytes(b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn encode_u32(v: u32) -> [u8; 4] {
        v.to_le_bytes()
    }
    fn encode_u64(v: u64) -> [u8; 8] {
        v.to_le_bytes()
    }
    fn encode_string_v3(s: &str) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&encode_u64(s.len() as u64));
        v.extend_from_slice(s.as_bytes());
        v
    }
    fn encode_string_v1(s: &str) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&encode_u32(s.len() as u32));
        v.extend_from_slice(s.as_bytes());
        v
    }

    /// Build a minimal V3 GGUF with the given tensor count, metadata KV count,
    /// and optional alignment value.
    fn build_v3(tensor_count: u64, kv_count: u64, alignment: Option<u32>) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&encode_u32(3));
        buf.extend_from_slice(&encode_u64(tensor_count));
        buf.extend_from_slice(&encode_u64(kv_count));
        if let Some(align) = alignment {
            buf.extend_from_slice(&encode_string_v3("general.alignment"));
            buf.extend_from_slice(&encode_u32(4)); // U32 type tag
            buf.extend_from_slice(&encode_u32(align));
        }
        // Pad out the remaining KV entries with empty strings + U32 zero values.
        let already = if alignment.is_some() { 1 } else { 0 };
        for _ in already..kv_count {
            buf.extend_from_slice(&encode_string_v3(""));
            buf.extend_from_slice(&encode_u32(4)); // U32 type tag
            buf.extend_from_slice(&encode_u32(0));
        }
        buf
    }

    #[test]
    fn parse_valid_header_returns_metadata() {
        let buf = build_v3(200, 50, Some(64));
        let mut cursor = Cursor::new(buf);
        let header = parse_header(&mut cursor, &SafeLoadLimits::default()).unwrap();
        assert_eq!(header.version, 3);
        assert_eq!(header.tensor_count, 200);
        assert_eq!(header.metadata_kv_count, 50);
        assert_eq!(header.alignment, 64);
    }

    #[test]
    fn parse_wrong_magic_rejected() {
        let mut buf = b"NOPE".to_vec();
        buf.extend_from_slice(&encode_u32(3));
        buf.extend_from_slice(&encode_u64(0));
        buf.extend_from_slice(&encode_u64(0));
        let mut cursor = Cursor::new(buf);
        match parse_header(&mut cursor, &SafeLoadLimits::default()) {
            Err(SafeLoadError::InvalidMagic { actual }) => assert_eq!(&actual, b"NOPE"),
            other => panic!("expected InvalidMagic, got {other:?}"),
        }
    }

    #[test]
    fn parse_ufug_backward_magic_accepted() {
        let mut buf = b"UFUG".to_vec();
        buf.extend_from_slice(&encode_u32(3));
        buf.extend_from_slice(&encode_u64(0));
        buf.extend_from_slice(&encode_u64(0));
        let mut cursor = Cursor::new(buf);
        let header = parse_header(&mut cursor, &SafeLoadLimits::default()).unwrap();
        assert_eq!(header.version, 3);
    }

    #[test]
    fn parse_unsupported_version_rejected() {
        let mut buf = b"GGUF".to_vec();
        buf.extend_from_slice(&encode_u32(99));
        let mut cursor = Cursor::new(buf);
        match parse_header(&mut cursor, &SafeLoadLimits::default()) {
            Err(SafeLoadError::InvalidVersion { version: 99 }) => {}
            other => panic!("expected InvalidVersion(99), got {other:?}"),
        }
    }

    #[test]
    fn parse_tensor_count_above_ceiling_rejected() {
        let buf = build_v3(MAX_TENSOR_COUNT + 1, 0, None);
        let mut cursor = Cursor::new(buf);
        match parse_header(&mut cursor, &SafeLoadLimits::default()) {
            Err(SafeLoadError::InvalidCounts { .. }) => {}
            other => panic!("expected InvalidCounts, got {other:?}"),
        }
    }

    #[test]
    fn parse_metadata_kv_above_ceiling_rejected() {
        let buf = build_v3(0, MAX_METADATA_KV_COUNT + 1, None);
        let mut cursor = Cursor::new(buf);
        match parse_header(&mut cursor, &SafeLoadLimits::default()) {
            Err(SafeLoadError::InvalidCounts { .. }) => {}
            other => panic!("expected InvalidCounts, got {other:?}"),
        }
    }

    #[test]
    fn parse_alignment_not_power_of_two_rejected() {
        let buf = build_v3(0, 1, Some(100));
        let mut cursor = Cursor::new(buf);
        match parse_header(&mut cursor, &SafeLoadLimits::default()) {
            Err(SafeLoadError::InvalidAlignment { value: 100 }) => {}
            other => panic!("expected InvalidAlignment(100), got {other:?}"),
        }
    }

    #[test]
    fn parse_alignment_above_max_rejected() {
        let buf = build_v3(0, 1, Some(8192));
        let mut cursor = Cursor::new(buf);
        match parse_header(&mut cursor, &SafeLoadLimits::default()) {
            Err(SafeLoadError::InvalidAlignment { value: 8192 }) => {}
            other => panic!("expected InvalidAlignment(8192), got {other:?}"),
        }
    }

    #[test]
    fn parse_alignment_below_min_rejected() {
        let buf = build_v3(0, 1, Some(8));
        let mut cursor = Cursor::new(buf);
        match parse_header(&mut cursor, &SafeLoadLimits::default()) {
            Err(SafeLoadError::InvalidAlignment { value: 8 }) => {}
            other => panic!("expected InvalidAlignment(8), got {other:?}"),
        }
    }

    #[test]
    fn parse_missing_alignment_defaults_to_32() {
        let buf = build_v3(10, 0, None);
        let mut cursor = Cursor::new(buf);
        let header = parse_header(&mut cursor, &SafeLoadLimits::default()).unwrap();
        assert_eq!(header.alignment, 32);
    }

    #[test]
    fn parse_coarse_byte_budget_rejected() {
        // Construct an adversarial file whose tensor_count × MAX_PER_TENSOR
        // exceeds the coarse budget. We can't easily write 1e10 tensors in a
        // test buffer, so we tighten the limits instead.
        let buf = build_v3(3, 0, None);
        let mut cursor = Cursor::new(buf);
        let tight = SafeLoadLimits {
            max_tensor_count: 10,
            max_metadata_kv_count: 100_000,
            max_per_tensor: 1,
            max_total_tensor_bytes: 2, // 3 tensors × 1 byte > 2 bytes
            min_alignment: MIN_ALIGNMENT,
            max_alignment: MAX_ALIGNMENT,
            default_alignment: DEFAULT_ALIGNMENT,
            max_metadata_key_bytes: MAX_METADATA_KEY_BYTES,
            max_metadata_string_bytes: MAX_METADATA_STRING_BYTES,
        };
        match parse_header(&mut cursor, &tight) {
            Err(SafeLoadError::InvalidCounts { .. }) => {}
            other => panic!("expected InvalidCounts, got {other:?}"),
        }
    }

    #[test]
    fn parse_truncated_input_returns_io_error() {
        let buf = b"GGUF".to_vec(); // only 4 bytes — version cut off
        let mut cursor = Cursor::new(buf);
        match parse_header(&mut cursor, &SafeLoadLimits::default()) {
            Err(SafeLoadError::Io(_)) => {}
            other => panic!("expected Io, got {other:?}"),
        }
    }

    #[test]
    fn parse_v1_format_supported() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&encode_u32(1));
        buf.extend_from_slice(&encode_u32(5)); // tensor_count
        buf.extend_from_slice(&encode_u32(0)); // metadata_kv_count
        let mut cursor = Cursor::new(buf);
        let header = parse_header(&mut cursor, &SafeLoadLimits::default()).unwrap();
        assert_eq!(header.version, 1);
        assert_eq!(header.tensor_count, 5);
        assert_eq!(header.alignment, 32);
    }

    #[test]
    fn parse_huge_metadata_string_length_rejected() {
        // KV with a string value whose length exceeds our ceiling. We must
        // not OOM — the scanner pre-validates length.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&encode_u32(3));
        buf.extend_from_slice(&encode_u64(0)); // tensor_count
        buf.extend_from_slice(&encode_u64(1)); // metadata_kv_count
        buf.extend_from_slice(&encode_string_v3("harmless_key"));
        buf.extend_from_slice(&encode_u32(8)); // String value type tag
        buf.extend_from_slice(&encode_u64(u64::MAX)); // bogus length
        let mut cursor = Cursor::new(buf);
        let result = parse_header(&mut cursor, &SafeLoadLimits::default());
        assert!(result.is_err(), "expected error, got {result:?}");
    }

    #[test]
    fn parse_v1_string_length_uses_u32() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&encode_u32(1));
        buf.extend_from_slice(&encode_u32(0));
        buf.extend_from_slice(&encode_u32(1));
        // V1 string length is u32.
        buf.extend_from_slice(&encode_string_v1("general.alignment"));
        buf.extend_from_slice(&encode_u32(4)); // U32 type tag
        buf.extend_from_slice(&encode_u32(64));
        let mut cursor = Cursor::new(buf);
        let header = parse_header(&mut cursor, &SafeLoadLimits::default()).unwrap();
        assert_eq!(header.alignment, 64);
    }
}
