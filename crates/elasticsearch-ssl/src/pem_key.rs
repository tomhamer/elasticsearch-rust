//! PEM private keys, as ES `PemUtils.parsePrivateKey`
//! reads them: PKCS#8 (plain or encrypted with PBES1 / PBES2), and OpenSSL
//! "traditional" RSA and EC keys, plain or encrypted with a `Proc-Type:
//! 4,ENCRYPTED` / `DEK-Info` header (DES-CBC, DES-EDE3-CBC, AES-128/192/256-CBC,
//! key derived with OpenSSL's `EVP_BytesToKey` over MD5).
//!
//! Encrypted keys are decrypted with the context's `secure_key_passphrase`
//! (keystore) or the legacy `key_passphrase` (e.g.
//! `xpack.security.transport.ssl.key_passphrase`). The PKCS#8 decryption
//! also serves PKCS#12 shrouded key bags ([`crate::pkcs12`]).

use base64::Engine;
use cbc::cipher::{block_padding::Pkcs7, BlockDecryptMut, KeyIvInit};
use rustls::pki_types::{PrivateKeyDer, PrivatePkcs1KeyDer, PrivatePkcs8KeyDer, PrivateSec1KeyDer};

/// A key that cannot be loaded; the message names `origin` (the key file).
#[derive(Debug, PartialEq)]
pub(crate) struct KeyError(pub String);

type R<T> = Result<T, KeyError>;

fn err(msg: impl Into<String>) -> KeyError {
    KeyError(msg.into())
}

/// One PEM block: its label, `Name: value` headers and decoded body.
struct Block {
    label: String,
    headers: Vec<(String, String)>,
    der: Vec<u8>,
}

fn blocks(text: &str) -> R<Vec<Block>> {
    let mut out = Vec::new();
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        let line = line.trim();
        let Some(label) = line.strip_prefix("-----BEGIN ").and_then(|l| l.strip_suffix("-----")) else { continue };
        let end = format!("-----END {label}-----");
        let (mut headers, mut body) = (Vec::new(), String::new());
        let mut closed = false;
        for l in lines.by_ref() {
            let l = l.trim();
            if l == end {
                closed = true;
                break;
            }
            // RFC 1421 headers (`Proc-Type`, `DEK-Info`) precede the body;
            // base64 never contains ':'.
            match l.split_once(':') {
                Some((k, v)) if body.is_empty() => headers.push((k.trim().to_string(), v.trim().to_string())),
                _ => body.push_str(l),
            }
        }
        if !closed {
            return Err(err("Malformed PEM file, PEM footer is invalid or missing"));
        }
        let der = base64::engine::general_purpose::STANDARD.decode(body.as_bytes()).map_err(|e| err(format!("Malformed PEM file, invalid base64 ({e})")))?;
        out.push(Block { label: label.to_string(), headers, der });
    }
    Ok(out)
}

/// Parse the private key in `pem`. `origin` names the file in errors.
pub(crate) fn parse(pem: &[u8], passphrase: Option<&str>, origin: &str) -> R<PrivateKeyDer<'static>> {
    let text = std::str::from_utf8(pem).map_err(|_| err(format!("cannot read PEM private key [{origin}] because the file is not UTF-8")))?;
    let all = blocks(text)?;
    // OpenSSL writes `EC PARAMETERS` / `DSA PARAMETERS` before the key.
    let Some(b) = all.into_iter().find(|b| !b.label.ends_with("PARAMETERS")) else {
        return Err(err(format!("Error parsing Private Key [{origin}], file is empty")));
    };
    let encrypted = b.headers.iter().any(|(k, v)| k == "Proc-Type" && v == "4,ENCRYPTED");
    let need = |what: &str| passphrase.ok_or_else(|| err(format!("cannot read encrypted key [{origin}] without a password{what}")));
    match b.label.as_str() {
        "PRIVATE KEY" => Ok(PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(checked(b.der, origin)?))),
        "ENCRYPTED PRIVATE KEY" => {
            let pass = need("")?;
            let der = decrypt_pkcs8(&b.der, pass).map_err(|e| wrong(origin, e))?;
            Ok(PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(checked(der, origin).map_err(|_| wrong(origin, "bad decrypt".into()))?)))
        }
        "RSA PRIVATE KEY" | "EC PRIVATE KEY" => {
            let der = if encrypted {
                let pass = need("")?;
                let dek = b.headers.iter().find(|(k, _)| k == "DEK-Info").map(|(_, v)| v.as_str()).ok_or_else(|| err("Malformed PEM File, DEK-Info header is missing"))?;
                let der = decrypt_legacy(&b.der, dek, pass).map_err(|e| wrong(origin, e))?;
                checked(der, origin).map_err(|_| wrong(origin, "bad decrypt".into()))?
            } else {
                checked(b.der, origin)?
            };
            Ok(if b.label == "RSA PRIVATE KEY" { PrivateKeyDer::Pkcs1(PrivatePkcs1KeyDer::from(der)) } else { PrivateKeyDer::Sec1(PrivateSec1KeyDer::from(der)) })
        }
        "DSA PRIVATE KEY" => Err(err(format!("cannot use PEM private key [{origin}]: DSA keys are not supported for TLS"))),
        _ => Err(err(format!("cannot read PEM private key [{origin}] because the file does not contain a supported key format"))),
    }
}

fn wrong(origin: &str, detail: String) -> KeyError {
    err(format!("cannot load PEM private key from [{origin}] due to BadPaddingException ({detail}); this is usually caused by an incorrect password"))
}

/// The DER must be exactly one SEQUENCE (a wrong passphrase decrypts to noise).
pub(crate) fn checked(der: Vec<u8>, origin: &str) -> R<Vec<u8>> {
    match tlv(&der) {
        Some((0x30, _, rest)) if rest.is_empty() => Ok(der),
        _ => Err(err(format!("cannot read PEM private key [{origin}]: invalid key encoding"))),
    }
}

// ------------------------------------------------------------------ DER

/// One DER TLV: `(tag, contents, rest)`.
pub(crate) fn tlv(b: &[u8]) -> Option<(u8, &[u8], &[u8])> {
    let (&tag, b) = b.split_first()?;
    let (&first, mut b) = b.split_first()?;
    let len = if first < 0x80 {
        first as usize
    } else {
        let n = (first & 0x7f) as usize;
        if n == 0 || n > 4 || b.len() < n {
            return None;
        }
        let len = b[..n].iter().fold(0usize, |a, &x| (a << 8) | x as usize);
        b = &b[n..];
        len
    };
    if b.len() < len {
        return None;
    }
    Some((tag, &b[..len], &b[len..]))
}

pub(crate) fn expect(b: &[u8], tag: u8) -> Result<(&[u8], &[u8]), String> {
    match tlv(b) {
        Some((t, c, rest)) if t == tag => Ok((c, rest)),
        _ => Err(format!("malformed encrypted key (expected DER tag {tag:#x})")),
    }
}

/// Dotted form of an encoded OBJECT IDENTIFIER.
pub(crate) fn oid(c: &[u8]) -> String {
    let mut parts = Vec::new();
    if let Some((&f, rest)) = c.split_first() {
        parts.push((f / 40) as u64);
        parts.push((f % 40) as u64);
        let mut v = 0u64;
        for &x in rest {
            v = (v << 7) | (x & 0x7f) as u64;
            if x & 0x80 == 0 {
                parts.push(v);
                v = 0;
            }
        }
    }
    parts.iter().map(u64::to_string).collect::<Vec<_>>().join(".")
}

pub(crate) fn int(c: &[u8]) -> Result<u32, String> {
    if c.is_empty() || c.len() > 5 {
        return Err("malformed encrypted key (integer)".into());
    }
    Ok(c.iter().fold(0u64, |a, &x| (a << 8) | x as u64) as u32)
}

// ----------------------------------------------------------- decryption

fn cbc_decrypt(alg: &str, key: &[u8], iv: &[u8], data: &[u8]) -> Result<Vec<u8>, String> {
    let bad = |e: &dyn std::fmt::Display| format!("{e}");
    let pad = |_| "Given final block not properly padded. Such issues can arise if a bad key is used during decryption.".to_string();
    match alg {
        "DES-CBC" => cbc::Decryptor::<des::Des>::new_from_slices(key, iv).map_err(|e| bad(&e))?.decrypt_padded_vec_mut::<Pkcs7>(data).map_err(pad),
        "DES-EDE3-CBC" => cbc::Decryptor::<des::TdesEde3>::new_from_slices(key, iv).map_err(|e| bad(&e))?.decrypt_padded_vec_mut::<Pkcs7>(data).map_err(pad),
        "AES-128-CBC" => cbc::Decryptor::<aes::Aes128>::new_from_slices(key, iv).map_err(|e| bad(&e))?.decrypt_padded_vec_mut::<Pkcs7>(data).map_err(pad),
        "AES-192-CBC" => cbc::Decryptor::<aes::Aes192>::new_from_slices(key, iv).map_err(|e| bad(&e))?.decrypt_padded_vec_mut::<Pkcs7>(data).map_err(pad),
        "AES-256-CBC" => cbc::Decryptor::<aes::Aes256>::new_from_slices(key, iv).map_err(|e| bad(&e))?.decrypt_padded_vec_mut::<Pkcs7>(data).map_err(pad),
        other => Err(format!("Private Key encrypted with unsupported algorithm [{other}]")),
    }
}

fn key_len(alg: &str) -> Option<usize> {
    Some(match alg {
        "DES-CBC" => 8,
        "DES-EDE3-CBC" | "AES-192-CBC" => 24,
        "AES-128-CBC" => 16,
        "AES-256-CBC" => 32,
        _ => return None,
    })
}

/// OpenSSL `EVP_BytesToKey(MD5, count 1)`: the key of a legacy PEM key.
fn openssl_key(pass: &[u8], salt: &[u8], len: usize) -> Vec<u8> {
    use md5::{Digest, Md5};
    let mut out = Vec::with_capacity(len + 16);
    let mut prev: Vec<u8> = Vec::new();
    while out.len() < len {
        let mut h = Md5::new();
        h.update(&prev);
        h.update(pass);
        h.update(&salt[..8.min(salt.len())]);
        prev = h.finalize().to_vec();
        out.extend_from_slice(&prev);
    }
    out.truncate(len);
    out
}

fn hex(s: &str) -> Result<Vec<u8>, String> {
    if s.len() % 2 != 0 {
        return Err("Malformed PEM file, DEK-Info IV is invalid".into());
    }
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|_| "Malformed PEM file, DEK-Info IV is invalid".to_string())).collect()
}

/// A `Proc-Type: 4,ENCRYPTED` key body, `dek` = the `DEK-Info` value.
fn decrypt_legacy(data: &[u8], dek: &str, pass: &str) -> Result<Vec<u8>, String> {
    let (alg, iv) = dek.split_once(',').ok_or("Malformed PEM file, DEK-Info PEM header is invalid")?;
    let iv = hex(iv.trim())?;
    let len = key_len(alg.trim()).ok_or_else(|| format!("Private Key encrypted with unsupported algorithm [{alg}]"))?;
    let key = openssl_key(pass.as_bytes(), &iv, len);
    cbc_decrypt(alg.trim(), &key, &iv, data)
}

const PBES2: &str = "1.2.840.113549.1.5.13";
const PBKDF2: &str = "1.2.840.113549.1.5.12";
const PBE_MD5_DES: &str = "1.2.840.113549.1.5.3";
const PBE_SHA1_DES: &str = "1.2.840.113549.1.5.10";
const PKCS12_SHA_RC4_128: &str = "1.2.840.113549.1.12.1.1";
const PKCS12_SHA_RC4_40: &str = "1.2.840.113549.1.12.1.2";
const PKCS12_SHA_3DES: &str = "1.2.840.113549.1.12.1.3";
const PKCS12_SHA_2DES: &str = "1.2.840.113549.1.12.1.4";
const PKCS12_SHA_RC2_128: &str = "1.2.840.113549.1.12.1.5";
const PKCS12_SHA_RC2_40: &str = "1.2.840.113549.1.12.1.6";

/// An `EncryptedPrivateKeyInfo` (PKCS#8) decrypted to its `PrivateKeyInfo`.
pub(crate) fn decrypt_pkcs8(der: &[u8], pass: &str) -> Result<Vec<u8>, String> {
    let (info, _) = expect(der, 0x30)?;
    let (alg, rest) = expect(info, 0x30)?;
    let (data, _) = expect(rest, 0x04)?;
    decrypt_with(alg, data, pass)
}

/// `data` decrypted with password-based encryption scheme `alg` (the
/// contents of its `AlgorithmIdentifier`): PBES2, PBES1, or a PKCS#12 PBE
/// (`pbeWithSHAAnd3-KeyTripleDES-CBC` & co., password as a BMPString).
pub(crate) fn decrypt_with(alg: &[u8], data: &[u8], pass: &str) -> Result<Vec<u8>, String> {
    let (alg_oid, params) = expect(alg, 0x06)?;
    match oid(alg_oid).as_str() {
        PBES2 => {
            let (p, _) = expect(params, 0x30)?;
            let (kdf, rest) = expect(p, 0x30)?;
            let (enc, _) = expect(rest, 0x30)?;
            let (kdf_oid, kdf_params) = expect(kdf, 0x06)?;
            if oid(kdf_oid) != PBKDF2 {
                return Err(format!("PKCS#8 Private Key is encrypted with unsupported key derivation [{}]", oid(kdf_oid)));
            }
            let (kp, _) = expect(kdf_params, 0x30)?;
            let (salt, rest) = expect(kp, 0x04)?;
            let (iters, mut rest) = expect(rest, 0x02)?;
            let iters = int(iters)?;
            let mut key_length = None;
            if let Some((0x02, c, r)) = tlv(rest) {
                key_length = Some(int(c)? as usize);
                rest = r;
            }
            let prf = match tlv(rest) {
                Some((0x30, c, _)) => oid(expect(c, 0x06)?.0),
                _ => "1.2.840.113549.2.7".into(),
            };
            let (enc_oid, enc_params) = expect(enc, 0x06)?;
            let alg = match oid(enc_oid).as_str() {
                "2.16.840.1.101.3.4.1.2" => "AES-128-CBC",
                "2.16.840.1.101.3.4.1.22" => "AES-192-CBC",
                "2.16.840.1.101.3.4.1.42" => "AES-256-CBC",
                "1.2.840.113549.3.7" => "DES-EDE3-CBC",
                "1.3.14.3.2.7" => "DES-CBC",
                o => return Err(format!("PKCS#8 Private Key is encrypted with unsupported PBES2 algorithm [{o}]")),
            };
            let (iv, _) = expect(enc_params, 0x04)?;
            let len = key_length.unwrap_or_else(|| key_len(alg).unwrap());
            let mut key = vec![0u8; len];
            let p = pass.as_bytes();
            match prf.as_str() {
                "1.2.840.113549.2.7" => pbkdf2::pbkdf2_hmac::<sha1::Sha1>(p, salt, iters, &mut key),
                "1.2.840.113549.2.9" => pbkdf2::pbkdf2_hmac::<sha2::Sha256>(p, salt, iters, &mut key),
                "1.2.840.113549.2.10" => pbkdf2::pbkdf2_hmac::<sha2::Sha384>(p, salt, iters, &mut key),
                "1.2.840.113549.2.11" => pbkdf2::pbkdf2_hmac::<sha2::Sha512>(p, salt, iters, &mut key),
                o => return Err(format!("PKCS#8 Private Key is encrypted with unsupported PRF [{o}]")),
            }
            cbc_decrypt(alg, &key, iv, data)
        }
        o @ (PBE_MD5_DES | PBE_SHA1_DES) => {
            // PBES1 (PKCS#5 v1.5): PBKDF1, DES-CBC.
            let (p, _) = expect(params, 0x30)?;
            let (salt, rest) = expect(p, 0x04)?;
            let (iters, _) = expect(rest, 0x02)?;
            let iters = int(iters)?.max(1);
            let dk = pbkdf1(o == PBE_MD5_DES, pass.as_bytes(), salt, iters);
            cbc_decrypt("DES-CBC", &dk[..8], &dk[8..16], data)
        }
        o @ (PKCS12_SHA_3DES | PKCS12_SHA_2DES) => {
            // PKCS#12 PBE (RFC 7292 appendix B): key (ID 1) and IV (ID 2)
            // from the PKCS#12 KDF over SHA-1.
            let (p, _) = expect(params, 0x30)?;
            let (salt, rest) = expect(p, 0x04)?;
            let (iters, _) = expect(rest, 0x02)?;
            let iters = int(iters)?.max(1);
            let bmp = crate::pkcs12::bmp_password(pass);
            let iv = crate::pkcs12::kdf_sha1(&bmp, salt, 2, iters, 8);
            if o == PKCS12_SHA_3DES {
                let key = crate::pkcs12::kdf_sha1(&bmp, salt, 1, iters, 24);
                cbc_decrypt("DES-EDE3-CBC", &key, &iv, data)
            } else {
                let key = crate::pkcs12::kdf_sha1(&bmp, salt, 1, iters, 16);
                cbc::Decryptor::<des::TdesEde2>::new_from_slices(&key, &iv)
                    .map_err(|e| e.to_string())?
                    .decrypt_padded_vec_mut::<Pkcs7>(data)
                    .map_err(|_| "Given final block not properly padded. Such issues can arise if a bad key is used during decryption.".to_string())
            }
        }
        o @ (PKCS12_SHA_RC2_128 | PKCS12_SHA_RC2_40 | PKCS12_SHA_RC4_128 | PKCS12_SHA_RC4_40) => Err(format!(
            "encrypted with unsupported legacy algorithm [{o}] (RC2/RC4); re-encrypt the keystore with AES (e.g. `openssl pkcs12 -export` without `-legacy`)"
        )),
        o => Err(format!("PKCS#8 Private Key is encrypted with unsupported algorithm [{o}]")),
    }
}

/// PKCS#5 PBKDF1 with MD5 or SHA-1 (at least 16 bytes out).
fn pbkdf1(md5: bool, pass: &[u8], salt: &[u8], iters: u32) -> Vec<u8> {
    use md5::Digest;
    let digest = |data: &[u8]| -> Vec<u8> { if md5 { md5::Md5::digest(data).to_vec() } else { sha1::Sha1::digest(data).to_vec() } };
    let mut t = digest(&[pass, salt].concat());
    for _ in 1..iters {
        t = digest(&t);
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evp_bytes_to_key_matches_openssl() {
        // `openssl enc -aes-256-cbc -P -md md5 -pass pass:testnode -S 0102030405060708`
        let k = openssl_key(b"testnode", &[1, 2, 3, 4, 5, 6, 7, 8], 32);
        assert_eq!(k.len(), 32);
        // Successive MD5 rounds chain the previous digest.
        let first = { use md5::Digest; md5::Md5::digest([b"testnode".as_slice(), &[1, 2, 3, 4, 5, 6, 7, 8]].concat()).to_vec() };
        assert_eq!(&k[..16], first.as_slice());
    }

    #[test]
    fn der_helpers_read_oids_and_lengths() {
        assert_eq!(oid(&[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x05, 0x0d]), PBES2);
        assert_eq!(tlv(&[0x30, 0x81, 0x02, 1, 2]).map(|(t, c, r)| (t, c.to_vec(), r.len())), Some((0x30, vec![1, 2], 0)));
        assert!(tlv(&[0x30, 0x05, 1]).is_none());
    }

    /// A deterministic RSA PKCS#1 v1.5 signature identifies the key.
    fn fingerprint(key: &PrivateKeyDer<'static>) -> Vec<u8> {
        let k = rustls::crypto::ring::sign::any_supported_type(key).expect("supported key");
        let signer = k.choose_scheme(&[rustls::SignatureScheme::RSA_PKCS1_SHA256]).expect("rsa signer");
        signer.sign(b"transport tls key fingerprint").unwrap()
    }

    const PLAIN: &[u8] = include_bytes!("../tests/data/testnode-unprotected.pem");
    const ENCRYPTED: &[(&str, &[u8])] = &[
        // OpenSSL legacy `Proc-Type: 4,ENCRYPTED`, DES-EDE3-CBC (ES's testnode.pem).
        ("testnode.pem", include_bytes!("../tests/data/testnode.pem")),
        // Legacy AES-256-CBC.
        ("testnode-aes256.pem", include_bytes!("../tests/data/testnode-aes256.pem")),
        // PKCS#8 PBES1 (pbeWithMD5AndDES-CBC; ES's key_pkcs8_encrypted.pem).
        ("key_pkcs8_encrypted.pem", include_bytes!("../tests/data/key_pkcs8_encrypted.pem")),
        // PKCS#8 PBES2: PBKDF2-HMAC-SHA256 + AES-256-CBC, and PBKDF2-HMAC-SHA1 + DES-EDE3-CBC.
        ("testnode-pkcs8-aes256.pem", include_bytes!("../tests/data/testnode-pkcs8-aes256.pem")),
        ("testnode-pkcs8-des3.pem", include_bytes!("../tests/data/testnode-pkcs8-des3.pem")),
    ];

    #[test]
    fn every_encrypted_testnode_key_decrypts_to_the_same_key() {
        let plain = parse(PLAIN, None, "testnode-unprotected.pem").unwrap();
        let want = fingerprint(&plain);
        for (name, pem) in ENCRYPTED {
            let key = parse(pem, Some("testnode"), name).unwrap_or_else(|e| panic!("{name}: {e:?}"));
            assert_eq!(fingerprint(&key), want, "{name}");
        }
    }

    #[test]
    fn a_wrong_or_missing_passphrase_is_rejected() {
        for (name, pem) in ENCRYPTED {
            let e = parse(pem, Some("not-the-passphrase"), name).unwrap_err();
            assert!(e.0.contains("incorrect password"), "{name}: {e:?}");
            let e = parse(pem, None, name).unwrap_err();
            assert!(e.0.starts_with(&format!("cannot read encrypted key [{name}] without a password")), "{name}: {e:?}");
        }
    }

    #[test]
    fn unknown_and_missing_keys_are_reported() {
        assert_eq!(parse(b"", None, "k.pem").unwrap_err(), err("Error parsing Private Key [k.pem], file is empty"));
        let enc = b"-----BEGIN ENCRYPTED PRIVATE KEY-----\nMAA=\n-----END ENCRYPTED PRIVATE KEY-----\n";
        assert_eq!(parse(enc, None, "k.pem").unwrap_err(), err("cannot read encrypted key [k.pem] without a password"));
    }
}
