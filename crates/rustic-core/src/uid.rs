//! Compact binary form of document ids (ES `Uid.encodeId` / `decodeId`).
//!
//! An id that is a positive decimal number is stored as packed decimal
//! digits, a canonical URL-safe base64 string without padding (every auto
//! id: 20 characters for 15 bytes) as the bytes it encodes, and any other
//! string as its UTF-8 bytes behind a marker byte. The mapping is lossless
//! for every string: [`decode`] of [`encode`] gives the id back, so stored
//! fields and term dictionaries can hold the binary form (5 bytes fewer an
//! auto id, and bytes LZ4 and front coding compress better than base64).

/// First byte of a UTF-8 id.
const UTF8: u8 = 0xff;
/// First byte of a numeric id.
const NUMERIC: u8 = 0xfe;
/// First byte of a base64 id whose decoded bytes start with a marker byte.
const BASE64_ESCAPE: u8 = 0xfd;

fn is_url_base64_without_padding(id: &[u8]) -> bool {
    let n = id.len();
    // 4k + 1 characters encode no whole byte.
    if n == 0 || n % 4 == 1 {
        return false;
    }
    let value = |c: u8| -> Option<u8> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'-' => 62,
            b'_' => 63,
            _ => return None,
        })
    };
    if id.iter().any(|&c| value(c).is_none()) {
        return false;
    }
    // Canonical: the unused low bits of the last character are zero, so
    // decoding and re-encoding gives the same string.
    let last = value(id[n - 1]).unwrap_or(0);
    match n % 4 {
        2 => last & 0x0f == 0,
        3 => last & 0x03 == 0,
        _ => true,
    }
}

fn is_positive_numeric(id: &[u8]) -> bool {
    !id.is_empty() && id.iter().all(u8::is_ascii_digit)
}

/// The binary form of `id`.
pub fn encode(id: &str) -> Vec<u8> {
    let b = id.as_bytes();
    if is_positive_numeric(b) {
        // Two digits a byte, high nibble first; an odd count ends with 0xf.
        let mut out = Vec::with_capacity(1 + b.len().div_ceil(2));
        out.push(NUMERIC);
        for pair in b.chunks(2) {
            let hi = pair[0] - b'0';
            let lo = pair.get(1).map_or(0x0f, |c| c - b'0');
            out.push((hi << 4) | lo);
        }
        out
    } else if is_url_base64_without_padding(b) {
        use base64::Engine;
        let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(b).expect("checked URL-safe base64");
        if decoded.first().is_some_and(|&x| x >= BASE64_ESCAPE) {
            let mut out = Vec::with_capacity(decoded.len() + 1);
            out.push(BASE64_ESCAPE);
            out.extend_from_slice(&decoded);
            out
        } else {
            decoded
        }
    } else {
        let mut out = Vec::with_capacity(b.len() + 1);
        out.push(UTF8);
        out.extend_from_slice(b);
        out
    }
}

/// The id whose binary form is `bytes` (`None`: not an [`encode`] output).
pub fn decode(bytes: &[u8]) -> Option<String> {
    use base64::Engine;
    match bytes.first()? {
        &UTF8 => String::from_utf8(bytes[1..].to_vec()).ok(),
        &NUMERIC => {
            let mut out = String::with_capacity((bytes.len() - 1) * 2);
            for (i, &x) in bytes[1..].iter().enumerate() {
                let (hi, lo) = (x >> 4, x & 0x0f);
                if hi > 9 {
                    return None;
                }
                out.push((b'0' + hi) as char);
                match lo {
                    0..=9 => out.push((b'0' + lo) as char),
                    // Only the last byte may end an odd digit count.
                    0x0f if i + 2 == bytes.len() => {}
                    _ => return None,
                }
            }
            (!out.is_empty()).then_some(out)
        }
        &BASE64_ESCAPE => Some(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&bytes[1..])),
        _ => Some(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_every_kind() {
        let ids = [
            "AaGbCX0F9q8YlnEvGqAa", "0", "7", "42", "0012", "123456789012345678901234567890", "abc", "ab", "a", "abcd",
            "AB", "AQ", "AR", "_-", "a b", "über", "id with \"quotes\"", "-", "--", "///", "1a", "a1", "____", "////",
            "\u{1}", "日本", "99999999999999999999", "+1", "1.5", "/w", "_w", "__w8", "AAAA", "AA", "AAA",
        ];
        for id in ids {
            let e = encode(id);
            assert_eq!(decode(&e).as_deref(), Some(id), "{id:?} -> {e:?}");
        }
        // Every 1-3 character string over a mixed alphabet.
        let alphabet: Vec<char> = "Az09-_+/ é\u{0}".chars().collect();
        let mut all = vec![String::new()];
        for _ in 0..3 {
            all = all.iter().flat_map(|p| alphabet.iter().map(move |c| format!("{p}{c}"))).collect();
            for id in &all {
                assert_eq!(decode(&encode(id)).as_deref(), Some(id.as_str()), "{id:?}");
            }
        }
    }

    #[test]
    fn auto_ids_take_their_bytes() {
        assert_eq!(encode("AaGbCX0F9q8YlnEvGqAa").len(), 15);
        assert_eq!(encode("12345").len(), 4);
        assert_eq!(encode("my-doc 1").len(), 9);
        // Distinct ids never collide.
        let mut seen = std::collections::HashSet::new();
        for id in ["AA", "AQ", "0", "00", "/w", "_w", "\u{ff}", "abc", "abcd"] {
            assert!(seen.insert(encode(id)), "{id}");
        }
    }
}
