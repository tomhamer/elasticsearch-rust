//! Password / secret hashing compatible with ES `Hasher`: bcrypt (`$2a$`),
//! `{PBKDF2}`, `{PBKDF2_STRETCH}`, `{SSHA256}` and clear text (noop).

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use sha2::{Digest, Sha256, Sha512};

pub fn random_bytes(n: usize) -> Vec<u8> {
    let mut b = vec![0u8; n];
    if getrandom::getrandom(&mut b).is_err() {
        // Fallback: uuid v4 randomness.
        for chunk in b.chunks_mut(16) {
            let u = uuid::Uuid::new_v4();
            chunk.copy_from_slice(&u.as_bytes()[..chunk.len()]);
        }
    }
    b
}

/// ES-style random secret / id: url-safe base64 (no padding) of `n` random bytes.
pub fn random_b64url(n: usize) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(random_bytes(n))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Hasher {
    Bcrypt(u32),
    Pbkdf2(u32),
    Pbkdf2Stretch(u32),
    Ssha256,
    Noop,
}

/// Algorithms accepted for stored password hashes (ES `getAvailableAlgoStoredPasswordHash`).
pub const STORED_PASSWORD_ALGOS: &[&str] = &[
    "bcrypt", "bcrypt4", "bcrypt5", "bcrypt6", "bcrypt7", "bcrypt8", "bcrypt9", "bcrypt10", "bcrypt11", "bcrypt12", "bcrypt13", "bcrypt14",
    "pbkdf2", "pbkdf2_1000", "pbkdf2_10000", "pbkdf2_50000", "pbkdf2_100000", "pbkdf2_500000", "pbkdf2_1000000",
    "pbkdf2_stretch", "pbkdf2_stretch_1000", "pbkdf2_stretch_10000", "pbkdf2_stretch_50000", "pbkdf2_stretch_100000",
    "pbkdf2_stretch_500000", "pbkdf2_stretch_1000000",
];

/// The list as ES prints it (enum order, deduplicated aliases).
pub fn stored_algos_display() -> String {
    let names = [
        "bcrypt", "bcrypt4", "bcrypt5", "bcrypt6", "bcrypt7", "bcrypt8", "bcrypt9", "bcrypt11", "bcrypt12", "bcrypt13", "bcrypt14",
        "pbkdf2", "pbkdf2_1000", "pbkdf2_10000", "pbkdf2_50000", "pbkdf2_100000", "pbkdf2_500000", "pbkdf2_1000000",
        "pbkdf2_stretch", "pbkdf2_stretch_1000", "pbkdf2_stretch_10000", "pbkdf2_stretch_50000", "pbkdf2_stretch_100000",
        "pbkdf2_stretch_500000", "pbkdf2_stretch_1000000",
    ];
    format!("[{}]", names.join(", "))
}

impl Hasher {
    pub fn resolve(name: &str) -> Result<Hasher, String> {
        let n = name.to_ascii_lowercase();
        let h = match n.as_str() {
            "bcrypt" => Hasher::Bcrypt(10),
            "pbkdf2" => Hasher::Pbkdf2(10000),
            "pbkdf2_stretch" => Hasher::Pbkdf2Stretch(10000),
            "ssha256" => Hasher::Ssha256,
            "noop" | "clear_text" => Hasher::Noop,
            _ => {
                if let Some(c) = n.strip_prefix("bcrypt").and_then(|c| c.parse::<u32>().ok()).filter(|c| (4..=14).contains(c)) {
                    Hasher::Bcrypt(c)
                } else if let Some(c) = n.strip_prefix("pbkdf2_stretch_").and_then(|c| c.parse::<u32>().ok()).filter(|c| [1000, 10000, 50000, 100000, 500000, 1000000].contains(c)) {
                    Hasher::Pbkdf2Stretch(c)
                } else if let Some(c) = n.strip_prefix("pbkdf2_").and_then(|c| c.parse::<u32>().ok()).filter(|c| [1000, 10000, 50000, 100000, 500000, 1000000].contains(c)) {
                    Hasher::Pbkdf2(c)
                } else {
                    return Err(format!("unknown hash function [{name}]"));
                }
            }
        };
        Ok(h)
    }

    /// Algorithm implied by a stored hash (ES `resolveFromHash`).
    pub fn from_hash(hash: &str) -> Hasher {
        if is_bcrypt(hash) {
            let cost = hash.get(4..6).and_then(|c| c.parse().ok()).unwrap_or(10);
            Hasher::Bcrypt(cost)
        } else if let Some(r) = hash.strip_prefix("{PBKDF2_STRETCH}") {
            Hasher::Pbkdf2Stretch(r.split('$').next().and_then(|c| c.parse().ok()).unwrap_or(10000))
        } else if let Some(r) = hash.strip_prefix("{PBKDF2}") {
            Hasher::Pbkdf2(r.split('$').next().and_then(|c| c.parse().ok()).unwrap_or(10000))
        } else if hash.starts_with("{SSHA256}") {
            Hasher::Ssha256
        } else {
            Hasher::Noop
        }
    }

    pub fn is_stored_password_algo(&self) -> bool {
        matches!(self, Hasher::Bcrypt(_) | Hasher::Pbkdf2(_) | Hasher::Pbkdf2Stretch(_))
    }

    pub fn hash(&self, pw: &str) -> String {
        match *self {
            Hasher::Bcrypt(cost) => {
                let mut salt = [0u8; 16];
                salt.copy_from_slice(&random_bytes(16));
                match bcrypt::hash_with_salt(pw.as_bytes(), cost, salt) {
                    Ok(h) => h.format_for_version(bcrypt::Version::TwoA),
                    Err(_) => String::new(),
                }
            }
            Hasher::Pbkdf2(c) => pbkdf2_hash(pw.as_bytes(), c, "{PBKDF2}"),
            Hasher::Pbkdf2Stretch(c) => pbkdf2_hash(sha512_hex(pw).as_bytes(), c, "{PBKDF2_STRETCH}"),
            Hasher::Ssha256 => {
                let salt = random_bytes(8);
                let mut h = Sha256::new();
                h.update(pw.as_bytes());
                h.update(&salt);
                format!("{{SSHA256}}{}{}", B64.encode(&salt), B64.encode(h.finalize()))
            }
            Hasher::Noop => pw.to_string(),
        }
    }
}

fn is_bcrypt(h: &str) -> bool {
    (h.starts_with("$2a$") || h.starts_with("$2b$") || h.starts_with("$2y$")) && h.len() == 60
}

fn sha512_hex(pw: &str) -> String {
    let d = Sha512::digest(pw.as_bytes());
    d.iter().map(|b| format!("{b:02x}")).collect()
}

fn pbkdf2_raw(pw: &[u8], salt: &[u8], iters: u32, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    pbkdf2::pbkdf2_hmac::<Sha512>(pw, salt, iters, &mut out);
    out
}

fn pbkdf2_hash(pw: &[u8], iters: u32, prefix: &str) -> String {
    let salt = random_bytes(32);
    let key = pbkdf2_raw(pw, &salt, iters, 32);
    format!("{prefix}{iters}${}${}", B64.encode(&salt), B64.encode(key))
}

fn pbkdf2_verify(pw: &[u8], hash: &str, prefix: &str) -> bool {
    let Some(rest) = hash.strip_prefix(prefix) else { return false };
    let parts: Vec<&str> = rest.split('$').collect();
    if parts.len() != 3 {
        return false;
    }
    let (Ok(iters), Ok(salt), Ok(expected)) = (parts[0].parse::<u32>(), B64.decode(parts[1]), B64.decode(parts[2])) else { return false };
    if salt.len() < 8 || expected.is_empty() {
        return false;
    }
    ct_eq(&pbkdf2_raw(pw, &salt, iters, expected.len()), &expected)
}

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Verify `pw` against any supported stored hash format.
pub fn verify(pw: &str, hash: &str) -> bool {
    if is_bcrypt(hash) {
        return bcrypt::verify(pw.as_bytes(), hash).unwrap_or(false);
    }
    if hash.starts_with("{PBKDF2_STRETCH}") {
        return pbkdf2_verify(sha512_hex(pw).as_bytes(), hash, "{PBKDF2_STRETCH}");
    }
    if hash.starts_with("{PBKDF2}") {
        return pbkdf2_verify(pw.as_bytes(), hash, "{PBKDF2}");
    }
    if let Some(rest) = hash.strip_prefix("{SSHA256}") {
        if rest.len() < 12 {
            return false;
        }
        let Ok(salt) = B64.decode(&rest[..12]) else { return false };
        let mut h = Sha256::new();
        h.update(pw.as_bytes());
        h.update(&salt);
        return ct_eq(B64.encode(h.finalize()).as_bytes(), rest[12..].as_bytes());
    }
    let plain = hash.strip_prefix("{plain}").unwrap_or(hash);
    ct_eq(pw.as_bytes(), plain.as_bytes())
}

/// Fast cache key for verified credentials.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    Sha256::digest(data).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        for h in [Hasher::Bcrypt(4), Hasher::Pbkdf2(1000), Hasher::Pbkdf2Stretch(1000), Hasher::Ssha256, Hasher::Noop] {
            let s = h.hash("s3krit-password");
            assert!(verify("s3krit-password", &s), "{h:?} {s}");
            assert!(!verify("nope-nope", &s));
        }
    }

    #[test]
    fn es_vectors() {
        // From ES test resources (service_tokens): the clear text secret is not known here,
        // but parsing must resolve the algorithm.
        assert_eq!(Hasher::from_hash("{PBKDF2}10000$0N2h5/AsDS5uO0/A+B6y8AnTCJ3Tqo8nygbzu1gkgpo=$5aTcCtteHf2g2ye7Y3p6jSZBoGhNJ7l6F3tmUhPTwRo="), Hasher::Pbkdf2(10000));
        assert_eq!(Hasher::from_hash("$2a$10$ML0BUUxdzs8ApPNf1ayAwuh61ZhfqlzN/1DgZWZn6vNiUhpu1GKTe"), Hasher::Bcrypt(10));
    }
}
