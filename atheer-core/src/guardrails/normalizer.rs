use std::collections::HashMap;

use unicode_normalization::UnicodeNormalization;

/// The type of encoding detected in the input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingType {
    Base64,
    Hex,
    Rot13,
}

// ── Zero-width character set ────────────────────────────────────────────────

const ZERO_WIDTH_CHARS: &[char] = &[
    '\u{200B}', // ZERO WIDTH SPACE
    '\u{200C}', // ZERO WIDTH NON-JOINER
    '\u{200D}', // ZERO WIDTH JOINER
    '\u{FEFF}', // BOM / ZERO WIDTH NO-BREAK SPACE
    '\u{2060}', // WORD JOINER
    '\u{00AD}', // SOFT HYPHEN
    '\u{200E}', // LEFT-TO-RIGHT MARK
    '\u{200F}', // RIGHT-TO-LEFT MARK
    '\u{202A}', // LEFT-TO-RIGHT EMBEDDING
    '\u{202B}', // RIGHT-TO-LEFT EMBEDDING
    '\u{202C}', // POP DIRECTIONAL FORMATTING
    '\u{202D}', // LEFT-TO-RIGHT OVERRIDE
    '\u{202E}', // RIGHT-TO-LEFT OVERRIDE
    '\u{2066}', // LEFT-TO-RIGHT ISOLATE
    '\u{2067}', // RIGHT-TO-LEFT ISOLATE
    '\u{2068}', // FIRST STRONG ISOLATE
    '\u{2069}', // POP DIRECTIONAL ISOLATE
    '\u{061C}', // ARABIC LETTER MARK
    '\u{180E}', // MONGOLIAN VOWEL SEPARATOR
    '\u{FFA0}', // HALFWIDTH HANGUL FILLER
];

// ── Homoglyph mapping ──────────────────────────────────────────────────────

fn build_homoglyph_map() -> HashMap<char, &'static str> {
    let mut m: HashMap<char, &'static str> = HashMap::new();

    // Cyrillic → Latin
    let cyrillic_to_latin: &[(char, &str)] = &[
        ('а', "a"),
        ('А', "a"),
        ('е', "e"),
        ('Е', "e"),
        ('о', "o"),
        ('О', "o"),
        ('р', "p"),
        ('Р', "p"),
        ('с', "c"),
        ('С', "c"),
        ('і', "i"),
        ('І', "i"),
        ('ї', "i"),
        ('Ї', "i"),
        ('є', "e"),
        ('Є', "e"),
        ('ʙ', "b"),
        ('ɢ', "g"),
        ('ʜ', "h"),
        ('ᴀ', "a"),
        ('ᴄ', "c"),
        ('ᴅ', "d"),
        ('ᴇ', "e"),
        ('ꜰ', "f"),
        ('ᴊ', "j"),
        ('ᴋ', "k"),
        ('ʟ', "l"),
        ('ᴍ', "m"),
        ('ɴ', "n"),
        ('ᴘ', "p"),
        ('ǫ', "q"),
        ('ʀ', "r"),
        ('ᴛ', "t"),
        ('ᴜ', "u"),
        ('ᴡ', "w"),
        ('ʏ', "y"),
        ('ᴢ', "z"),
    ];
    for &(c, r) in cyrillic_to_latin {
        m.insert(c, r);
    }

    // Roman numerals (multi-char expansions)
    m.insert('Ⅰ', "i");
    m.insert('Ⅱ', "ii");
    m.insert('Ⅲ', "iii");
    m.insert('Ⅳ', "iv");
    m.insert('Ⅴ', "v");
    m.insert('Ⅵ', "vi");
    m.insert('Ⅶ', "vii");
    m.insert('Ⅷ', "viii");
    m.insert('Ⅸ', "ix");
    m.insert('Ⅹ', "x");
    m.insert('Ⅺ', "xi");
    m.insert('Ⅻ', "xii");
    m.insert('ⅰ', "i");
    m.insert('ⅱ', "ii");
    m.insert('ⅲ', "iii");
    m.insert('ⅳ', "iv");
    m.insert('ⅴ', "v");
    m.insert('ⅵ', "vi");
    m.insert('ⅶ', "vii");
    m.insert('ⅷ', "viii");
    m.insert('ⅸ', "ix");
    m.insert('ⅹ', "x");

    // Fullwidth Latin
    for i in 0..26u8 {
        let fullwidth = char::from_u32(0xFF21 + i as u32).unwrap();
        let latin = (b'A' + i) as char;
        let lower: &'static str =
            Box::leak(latin.to_lowercase().collect::<String>().into_boxed_str());
        m.insert(fullwidth, lower);
    }

    // Mathematical symbols → Latin
    for i in 0..26u8 {
        let math = char::from_u32(0x1D400 + i as u32).unwrap();
        let latin = (b'A' + i) as char;
        let lower: &'static str =
            Box::leak(latin.to_lowercase().collect::<String>().into_boxed_str());
        m.insert(math, lower);
    }

    // Additional confusables
    m.insert('¡', "i");
    m.insert('!', "!");
    m.insert('`', "'");
    m.insert('‚', ",");
    m.insert('‛', "'");
    m.insert('‟', "\"");
    m.insert('״', "\"");
    m.insert('＇', "'");
    m.insert('꩜', "o");

    m
}

// ── Leetspeak mapping ──────────────────────────────────────────────────────

fn build_leet_map() -> HashMap<char, char> {
    let mut m = HashMap::new();
    m.insert('1', 'i');
    m.insert('3', 'e');
    m.insert('4', 'a');
    m.insert('@', 'a');
    m.insert('5', 's');
    m.insert('0', 'o');
    m.insert('8', 'b');
    m.insert('$', 's');
    m.insert('7', 't');
    m.insert('!', 'i');
    m.insert('+', 't');
    m.insert('|', 'l');
    m.insert('2', 'z');
    m.insert('6', 'g');
    m.insert('9', 'g');
    m
}

// ── Normalize ──────────────────────────────────────────────────────────────

/// Normalize a prompt for pattern matching:
/// 1. NFKC Unicode normalization
/// 2. Strip zero-width characters
/// 3. Apply homoglyph mapping
/// 4. Apply leetspeak substitution
/// 5. Case fold to lowercase
pub fn normalize_text(input: &str) -> String {
    let homoglyphs = build_homoglyph_map();
    let leet = build_leet_map();

    let mut result = String::with_capacity(input.len());

    // Step 1: NFKC normalization
    for ch in input.nfkc() {
        // Step 2: Strip zero-width characters
        if ZERO_WIDTH_CHARS.contains(&ch) {
            continue;
        }
        // Step 3: Homoglyph mapping
        if let Some(mapped) = homoglyphs.get(&ch) {
            result.push_str(mapped);
            continue;
        }
        // Step 4: Leetspeak substitution
        if let Some(&mapped) = leet.get(&ch) {
            result.push(mapped);
            continue;
        }
        // Step 5: Case fold (lowercase)
        for lower in ch.to_lowercase() {
            result.push(lower);
        }
    }

    result
}

// ── Encoding detection ─────────────────────────────────────────────────────

/// Detect if the input string appears to be encoded.
pub fn detect_encoding(input: &str) -> Option<EncodingType> {
    let trimmed = input.trim();

    // Base64 detection: length divisible by 4, valid chars, min 16 chars
    if trimmed.len() >= 16 && is_base64_chars(trimmed) {
        let valid_len = trimmed.trim_end_matches('=').len();
        if valid_len.is_multiple_of(4) || valid_len.is_multiple_of(2) || valid_len % 4 == 3 {
            // Additional heuristic: base64 of English text has high entropy
            if let Some(decoded) = try_base64_decode(trimmed) {
                if decoded.iter().any(|&b| b.is_ascii_alphabetic()) {
                    return Some(EncodingType::Base64);
                }
            }
        }
    }

    // Hex detection: even length, all hex chars, min 16 chars
    if trimmed.len() >= 16
        && trimmed.len().is_multiple_of(2)
        && trimmed.chars().all(|c| c.is_ascii_hexdigit())
    {
        return Some(EncodingType::Hex);
    }

    // ROT13 detection: heuristic — check if decoded text has common English bigrams
    if trimmed.len() >= 20
        && trimmed
            .chars()
            .all(|c| c.is_ascii_alphabetic() || c.is_ascii_whitespace())
    {
        let decoded = decode_rot13(trimmed);
        if has_english_bigrams(&decoded) && !has_english_bigrams(trimmed) {
            return Some(EncodingType::Rot13);
        }
    }

    None
}

fn is_base64_chars(s: &str) -> bool {
    s.chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '=')
}

fn try_base64_decode(s: &str) -> Option<Vec<u8>> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.decode(s).ok()
}

/// Decode base64-encoded string.
pub fn decode_base64(input: &str) -> Option<String> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(input)
        .ok()?;
    String::from_utf8(bytes).ok()
}

/// Decode hex-encoded string.
pub fn decode_hex(input: &str) -> Option<String> {
    let bytes = hex::decode(input).ok()?;
    String::from_utf8(bytes).ok()
}

/// ROT13 decode.
pub fn decode_rot13(input: &str) -> String {
    input
        .chars()
        .map(|c| {
            if c.is_ascii_lowercase() {
                ((c as u8 - b'a' + 13) % 26 + b'a') as char
            } else if c.is_ascii_uppercase() {
                ((c as u8 - b'A' + 13) % 26 + b'A') as char
            } else {
                c
            }
        })
        .collect()
}

/// Check if text has common English bigrams (heuristic for ROT13 detection).
fn has_english_bigrams(text: &str) -> bool {
    let lower = text.to_lowercase();
    let common_bigrams = [
        "th", "he", "in", "er", "an", "re", "on", "at", "en", "nd", "ti", "es", "or", "te", "of",
        "ed", "is", "it", "al", "ar",
    ];
    let count = common_bigrams
        .iter()
        .filter(|&&bg| lower.contains(bg))
        .count();
    // At least 3 common bigrams suggests English text
    count >= 3
}
