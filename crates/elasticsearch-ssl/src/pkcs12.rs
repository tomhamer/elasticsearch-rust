//! PKCS#12 keystores and truststores (`keystore.path` / `truststore.path`
//! with type `PKCS12`, ES's default store format; what
//! `elasticsearch-certutil` and `keytool -storetype PKCS12` write).
//!
//! Supported: password-integrity (MAC over SHA-1 / SHA-256 / SHA-384 /
//! SHA-512), `data` and `encryptedData` safe contents, key bags, shrouded key
//! bags and X.509 certificate bags, encrypted with PBES2 (PBKDF2 + AES /
//! 3DES: OpenSSL 3 and recent JDK defaults), PBES1, or the PKCS#12
//! `pbeWithSHAAnd3-KeyTripleDES-CBC` / `2-KeyTripleDES` schemes. RC2 / RC4
//! (`openssl pkcs12 -legacy`, very old JDKs) are reported as unsupported.

use crate::pem_key::{checked, decrypt_pkcs8, decrypt_with, expect, int, oid, tlv};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};

const DATA: &str = "1.2.840.113549.1.7.1";
const ENCRYPTED_DATA: &str = "1.2.840.113549.1.7.6";
const KEY_BAG: &str = "1.2.840.113549.1.12.10.1.1";
const SHROUDED_KEY_BAG: &str = "1.2.840.113549.1.12.10.1.2";
const CERT_BAG: &str = "1.2.840.113549.1.12.10.1.3";
const X509_CERTIFICATE: &str = "1.2.840.113549.1.9.22.1";
const LOCAL_KEY_ID: &str = "1.2.840.113549.1.9.21";

/// Why a store could not be read.
#[derive(Debug, PartialEq)]
pub(crate) enum StoreError {
    /// The integrity MAC or a key's decryption failed: almost always an
    /// incorrect password.
    Password,
    Invalid(String),
}

impl From<String> for StoreError {
    fn from(s: String) -> Self {
        StoreError::Invalid(s)
    }
}

/// The entries of a PKCS#12 file, in file order. `local_key_id` pairs a key
/// with its certificate.
#[derive(Default)]
pub(crate) struct Store {
    pub keys: Vec<(Option<Vec<u8>>, PrivateKeyDer<'static>)>,
    pub certs: Vec<(Option<Vec<u8>>, CertificateDer<'static>)>,
}

impl Store {
    /// Every private key with the certificate sharing its `localKeyId` (else
    /// the certificate at the key's position), in file order: ES's
    /// `StoreKeyConfig.getKeys()`.
    pub(crate) fn key_entries(&self) -> Vec<(PrivateKeyDer<'static>, CertificateDer<'static>)> {
        let mut out = Vec::new();
        for (i, (id, key)) in self.keys.iter().enumerate() {
            let cert = match id {
                Some(id) => self.certs.iter().find(|(cid, _)| cid.as_ref() == Some(id)),
                None => self.certs.get(i),
            };
            if let Some((_, cert)) = cert {
                out.push((key.clone_key(), cert.clone()));
            }
        }
        out
    }

    /// The first private key and its chain: the certificate sharing its
    /// `localKeyId` (else the first certificate), then its issuers found in
    /// the store.
    pub(crate) fn key_and_chain(&self) -> Option<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>)> {
        let (id, key) = self.keys.first()?;
        let leaf = match id {
            Some(id) => self.certs.iter().find(|(cid, _)| cid.as_ref() == Some(id)).or_else(|| self.certs.first()),
            None => self.certs.first(),
        }?;
        let mut chain = vec![leaf.1.clone()];
        while chain.len() < 10 {
            let Some((issuer, subject)) = names(chain.last().unwrap()) else { break };
            if issuer == subject {
                break;
            }
            let next = self.certs.iter().find(|(_, c)| names(c).is_some_and(|(_, s)| s == issuer) && !chain.contains(c));
            match next {
                Some((_, c)) => chain.push(c.clone()),
                None => break,
            }
        }
        Some((chain, key.clone_key()))
    }
}

/// `(issuer, subject)`: the DER `Name`s of an X.509 certificate.
pub(crate) fn names(cert: &[u8]) -> Option<(&[u8], &[u8])> {
    let (c, _) = expect(cert, 0x30).ok()?;
    let (tbs, _) = expect(c, 0x30).ok()?;
    let mut r = tbs;
    if let Some((0xA0, _, rest)) = tlv(r) {
        r = rest; // [0] version
    }
    let (_, _, r) = tlv(r)?; // serialNumber
    let (_, _, r) = tlv(r)?; // signature
    let (_, _, after_issuer) = tlv(r)?;
    let issuer = &r[..r.len() - after_issuer.len()];
    let (_, _, r) = tlv(after_issuer)?; // validity
    let (_, _, after_subject) = tlv(r)?;
    let subject = &r[..r.len() - after_subject.len()];
    Some((issuer, subject))
}

/// The password as PKCS#12 wants it for its KDF: a NUL-terminated
/// big-endian BMPString.
pub(crate) fn bmp_password(pass: &str) -> Vec<u8> {
    let mut out: Vec<u8> = pass.encode_utf16().flat_map(|u| u.to_be_bytes()).collect();
    out.extend_from_slice(&[0, 0]);
    out
}

/// RFC 7292 appendix B.2 key derivation: `n` bytes of purpose `id`
/// (1 = key, 2 = IV, 3 = MAC key) with digest `D` of block size `v`.
fn kdf<D: sha2::Digest>(v: usize, pass: &[u8], salt: &[u8], id: u8, iters: u32, n: usize) -> Vec<u8> {
    let u = <D as sha2::Digest>::output_size();
    let fill = |src: &[u8]| -> Vec<u8> {
        if src.is_empty() {
            return Vec::new();
        }
        let len = v * src.len().div_ceil(v);
        (0..len).map(|i| src[i % src.len()]).collect()
    };
    let d = vec![id; v];
    let mut i_buf = fill(salt);
    i_buf.extend(fill(pass));
    let mut out = Vec::with_capacity(n + u);
    while out.len() < n {
        let mut a = D::digest([d.as_slice(), i_buf.as_slice()].concat()).to_vec();
        for _ in 1..iters.max(1) {
            a = D::digest(&a).to_vec();
        }
        out.extend_from_slice(&a);
        if out.len() >= n {
            break;
        }
        let b: Vec<u8> = (0..v).map(|k| a[k % a.len()]).collect();
        for chunk in i_buf.chunks_mut(v) {
            let mut carry = 1u16;
            for k in (0..chunk.len()).rev() {
                let s = chunk[k] as u16 + b[k] as u16 + carry;
                chunk[k] = s as u8;
                carry = s >> 8;
            }
        }
    }
    out.truncate(n);
    out
}

/// The PKCS#12 KDF over SHA-1 (what the PKCS#12 PBE schemes use).
pub(crate) fn kdf_sha1(pass_bmp: &[u8], salt: &[u8], id: u8, iters: u32, n: usize) -> Vec<u8> {
    kdf::<sha1::Sha1>(64, pass_bmp, salt, id, iters, n)
}

fn mac_ok(alg: &str, pass: &str, salt: &[u8], iters: u32, data: &[u8], expected: &[u8]) -> Result<bool, String> {
    use hmac::{Hmac, Mac};
    let bmp = bmp_password(pass);
    macro_rules! check {
        ($d:ty, $v:expr) => {{
            let key = kdf::<$d>($v, &bmp, salt, 3, iters, <$d as sha2::Digest>::output_size());
            let mut m = <Hmac<$d> as Mac>::new_from_slice(&key).map_err(|e| e.to_string())?;
            m.update(data);
            Ok(m.verify_slice(expected).is_ok())
        }};
    }
    match alg {
        "1.3.14.3.2.26" => check!(sha1::Sha1, 64),
        "2.16.840.1.101.3.4.2.1" => check!(sha2::Sha256, 64),
        "2.16.840.1.101.3.4.2.2" => check!(sha2::Sha384, 128),
        "2.16.840.1.101.3.4.2.3" => check!(sha2::Sha512, 128),
        o => Err(format!("PKCS#12 integrity MAC uses unsupported digest [{o}]")),
    }
}

/// An OCTET STRING's bytes, primitive or (BER) constructed.
fn octets(b: &[u8]) -> Result<(Vec<u8>, &[u8]), String> {
    match tlv(b) {
        Some((0x04, c, rest)) => Ok((c.to_vec(), rest)),
        Some((0x24, mut c, rest)) => {
            let mut out = Vec::new();
            while !c.is_empty() {
                let (part, r) = octets(c)?;
                out.extend(part);
                c = r;
            }
            Ok((out, rest))
        }
        _ => Err("malformed PKCS#12 file (expected an OCTET STRING)".into()),
    }
}

/// Parse `der`, checking its MAC with `store_password` and decrypting
/// shrouded keys with `key_password`.
pub(crate) fn parse(der: &[u8], store_password: &str, key_password: &str) -> Result<Store, StoreError> {
    let (pfx, _) = expect(der, 0x30)?;
    let (version, rest) = expect(pfx, 0x02)?;
    if int(version)? != 3 {
        return Err(StoreError::Invalid("unsupported PKCS#12 version".into()));
    }
    let (auth_safe, mac) = expect(rest, 0x30)?;
    let (ct, r) = expect(auth_safe, 0x06)?;
    if oid(ct) != DATA {
        return Err(StoreError::Invalid(format!("unsupported PKCS#12 integrity mode [{}] (only password integrity is supported)", oid(ct))));
    }
    let (explicit, _) = expect(r, 0xA0)?;
    let (content, _) = octets(explicit)?;
    if !mac.is_empty() {
        let (mac_data, _) = expect(mac, 0x30)?;
        let (digest_info, r) = expect(mac_data, 0x30)?;
        let (alg, r2) = expect(digest_info, 0x30)?;
        let (alg_oid, _) = expect(alg, 0x06)?;
        let (digest, _) = expect(r2, 0x04)?;
        let (salt, r) = expect(r, 0x04)?;
        let iters = match tlv(r) {
            Some((0x02, c, _)) => int(c)?,
            _ => 1,
        };
        if !mac_ok(&oid(alg_oid), store_password, salt, iters, &content, digest)? {
            return Err(StoreError::Password);
        }
    }
    let mut store = Store::default();
    let (mut items, _) = expect(&content, 0x30)?;
    while !items.is_empty() {
        let (ci, next) = expect(items, 0x30)?;
        items = next;
        let (ct, r) = expect(ci, 0x06)?;
        let (explicit, _) = expect(r, 0xA0)?;
        match oid(ct).as_str() {
            DATA => {
                let (safe, _) = octets(explicit)?;
                bags(&safe, key_password, &mut store)?;
            }
            ENCRYPTED_DATA => {
                let (ed, _) = expect(explicit, 0x30)?;
                let (_version, r) = expect(ed, 0x02)?;
                let (eci, _) = expect(r, 0x30)?;
                let (_ct, r) = expect(eci, 0x06)?;
                let (alg, r) = expect(r, 0x30)?;
                let encrypted = match tlv(r) {
                    Some((0x80, c, _)) => c.to_vec(),
                    Some((0xA0, mut c, _)) => {
                        let mut out = Vec::new();
                        while !c.is_empty() {
                            let (part, rest) = octets(c)?;
                            out.extend(part);
                            c = rest;
                        }
                        out
                    }
                    _ => continue,
                };
                let plain = decrypt_with(alg, &encrypted, store_password).map_err(|e| if e.contains("padded") { StoreError::Password } else { StoreError::Invalid(e) })?;
                bags(&plain, key_password, &mut store)?;
            }
            _ => {}
        }
    }
    Ok(store)
}

fn bags(safe_contents: &[u8], key_password: &str, store: &mut Store) -> Result<(), StoreError> {
    let (mut items, _) = expect(safe_contents, 0x30)?;
    while !items.is_empty() {
        let (bag, next) = expect(items, 0x30)?;
        items = next;
        let (id, r) = expect(bag, 0x06)?;
        let (value, r) = expect(r, 0xA0)?;
        let local_key_id = match tlv(r) {
            Some((0x31, attrs, _)) => local_key_id(attrs),
            _ => None,
        };
        match oid(id).as_str() {
            KEY_BAG => {
                let der = checked(value.to_vec(), "PKCS#12 key bag").map_err(|e| StoreError::Invalid(e.0))?;
                store.keys.push((local_key_id, PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(der))));
            }
            SHROUDED_KEY_BAG => {
                let der = decrypt_pkcs8(value, key_password).map_err(|e| if e.contains("padded") { StoreError::Password } else { StoreError::Invalid(e) })?;
                let der = checked(der, "PKCS#12 key bag").map_err(|_| StoreError::Password)?;
                store.keys.push((local_key_id, PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(der))));
            }
            CERT_BAG => {
                let (cb, _) = expect(value, 0x30)?;
                let (cid, r) = expect(cb, 0x06)?;
                if oid(cid) == X509_CERTIFICATE {
                    let (explicit, _) = expect(r, 0xA0)?;
                    let (cert, _) = octets(explicit)?;
                    store.certs.push((local_key_id, CertificateDer::from(cert)));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// The `localKeyId` attribute of a bag's attribute SET.
fn local_key_id(mut attrs: &[u8]) -> Option<Vec<u8>> {
    while let Some((0x30, attr, rest)) = tlv(attrs) {
        attrs = rest;
        let Ok((id, values)) = expect(attr, 0x06) else { continue };
        if oid(id) == LOCAL_KEY_ID {
            if let Some((0x31, set, _)) = tlv(values) {
                if let Some((0x04, v, _)) = tlv(set) {
                    return Some(v.to_vec());
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kdf_matches_rfc7292_vectors() {
        // OpenSSL's PKCS12_key_gen test vector: "smeg", salt 0A58CF64530D823F,
        // ID 1, 1 iteration, 24 bytes -> 8AAAE6297B6CB04642AB5B077851284EB7128F1A2A7FBCA3.
        let k = kdf_sha1(&bmp_password("smeg"), &[0x0A, 0x58, 0xCF, 0x64, 0x53, 0x0D, 0x82, 0x3F], 1, 1, 24);
        assert_eq!(k, [0x8A, 0xAA, 0xE6, 0x29, 0x7B, 0x6C, 0xB0, 0x46, 0x42, 0xAB, 0x5B, 0x07, 0x78, 0x51, 0x28, 0x4E, 0xB7, 0x12, 0x8F, 0x1A, 0x2A, 0x7F, 0xBC, 0xA3]);
        // Same password and salt, ID 2, 8 bytes -> 79993DFE048D3B76.
        let iv = kdf_sha1(&bmp_password("smeg"), &[0x0A, 0x58, 0xCF, 0x64, 0x53, 0x0D, 0x82, 0x3F], 2, 1, 8);
        assert_eq!(iv, [0x79, 0x99, 0x3D, 0xFE, 0x04, 0x8D, 0x3B, 0x76]);
    }

    #[test]
    fn bmp_password_is_nul_terminated_utf16be() {
        assert_eq!(bmp_password("ab"), [0, b'a', 0, b'b', 0, 0]);
        assert_eq!(bmp_password(""), [0, 0]);
    }
}
