//! The Elasticsearch keystore (`$ES_PATH_CONF/elasticsearch.keystore`), in
//! ES's own on-disk format (`KeyStoreWrapper`), so a keystore made by ES's
//! `elasticsearch-keystore` opens here and ours opens in ES.
//!
//! File (format version 7, ES 9; 3 to 6 are read too):
//!
//! ```text
//! Lucene CodecUtil header : BE i32 0x3fd76c17, vint-prefixed "elasticsearch.keystore", BE i32 version
//! has password            : 1 byte (0 / 1)
//! data length             : i32 (little endian from version 5, big endian before)
//! data                    : i32 len + salt (64 bytes), i32 len + iv (12 bytes), i32 len + ciphertext
//! Lucene CodecUtil footer : BE i32 0xc02893e8, BE i32 0, BE i64 CRC-32 of every preceding byte
//! ```
//!
//! The ciphertext is AES-GCM (128-bit tag, the salt as additional data)
//! under a PBKDF2-HMAC-SHA512 key of the UTF-8 password (empty when the
//! keystore has none): 210000 iterations from version 6 (10000 before), a
//! 256-bit key from version 7 (128 before). The plaintext is Java
//! `DataOutputStream` data: BE i32 entry count, then per entry
//! `writeUTF(name)`, (version 3 only: `writeUTF(type)`), BE i32 length,
//! bytes. String and file entries are stored alike (UTF-8 for strings).
//!
//! At node start ([`load_into_settings`]) the keystore is loaded (created
//! when missing, upgraded when old), its entries become secure node
//! settings (registered with `elasticsearch_engine::settings_filter`, so no
//! API renders them) and secure-only settings found in `elasticsearch.yml`
//! are rejected. `POST _nodes/reload_secure_settings` ([`reload`]) re-reads
//! it; reloadable consumers (S3 clients) read [`overlay_secure`].

pub mod cli;
pub mod reload;

use crate::config::{ConfigError, Settings};
use ring::rand::SecureRandom;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

/// The keystore file in the config directory.
pub const KEYSTORE_FILENAME: &str = "elasticsearch.keystore";
/// `KeyStoreWrapper.PROMPT`.
pub const PROMPT: &str = "Enter password for the elasticsearch keystore : ";
/// The random per-keystore seed every keystore carries.
pub const SEED_SETTING: &str = "keystore.seed";

const CODEC_MAGIC: i32 = 0x3fd7_6c17;
const FOOTER_MAGIC: i32 = !CODEC_MAGIC;
const FOOTER_LENGTH: usize = 16;
const MIN_FORMAT_VERSION: i32 = 3;
pub const V3_VERSION: i32 = 3;
pub const V4_VERSION: i32 = 4;
/// From here the Lucene directory API (and so the data block) is little endian.
pub const LE_VERSION: i32 = 5;
pub const HIGHER_KDF_ITERATION_COUNT_VERSION: i32 = 6;
pub const CIPHER_KEY_BITS_256_VERSION: i32 = 7;
pub const CURRENT_VERSION: i32 = CIPHER_KEY_BITS_256_VERSION;
const KDF_ITERS: u32 = 210_000;
const LEGACY_KDF_ITERS: u32 = 10_000;
const SALT_LEN: usize = 64;
const IV_LEN: usize = 12;
const SEED_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789~!@#$%^&*-_=+?";
const CORRUPTED: &str = "Keystore has been corrupted or tampered with";
const WRONG_PASSWORD: &str = "Provided keystore password was incorrect";

/// A keystore failure, by the Java exception ES raises for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeystoreError {
    /// `SecurityException("Provided keystore password was incorrect")`
    /// (the GCM tag did not verify).
    WrongPassword,
    /// `SecurityException` (e.g. "Keystore has been corrupted or tampered with").
    Security(String),
    /// `IllegalStateException` (format too old / too new, corrupt flag, ...).
    IllegalState(String),
    /// `IllegalArgumentException` (an invalid setting name).
    IllegalArgument(String),
    /// Lucene `CorruptIndexException` (header / footer / checksum).
    Corrupt(String),
    /// `IOException` (truncated file, read or write failure).
    Io(String),
    /// `UserException(ExitCodes.CONFIG)` raised by `save`.
    Config(String),
}

impl KeystoreError {
    pub fn message(&self) -> &str {
        match self {
            KeystoreError::WrongPassword => WRONG_PASSWORD,
            KeystoreError::Security(m)
            | KeystoreError::IllegalState(m)
            | KeystoreError::IllegalArgument(m)
            | KeystoreError::Corrupt(m)
            | KeystoreError::Io(m)
            | KeystoreError::Config(m) => m,
        }
    }

    /// The ES exception `type` (`ElasticsearchException.getExceptionName`).
    pub fn es_type(&self) -> &'static str {
        match self {
            KeystoreError::WrongPassword | KeystoreError::Security(_) => "security_exception",
            KeystoreError::IllegalState(_) => "illegal_state_exception",
            KeystoreError::IllegalArgument(_) => "illegal_argument_exception",
            KeystoreError::Corrupt(_) => "corrupt_index_exception",
            KeystoreError::Io(_) => "i_o_exception",
            KeystoreError::Config(_) => "user_exception",
        }
    }

    /// The Java class name, as a CLI prints an unexpected exception.
    pub fn java_class(&self) -> &'static str {
        match self {
            KeystoreError::WrongPassword | KeystoreError::Security(_) => "java.lang.SecurityException",
            KeystoreError::IllegalState(_) => "java.lang.IllegalStateException",
            KeystoreError::IllegalArgument(_) => "java.lang.IllegalArgumentException",
            KeystoreError::Corrupt(_) => "org.apache.lucene.index.CorruptIndexException",
            KeystoreError::Io(_) => "java.io.IOException",
            KeystoreError::Config(_) => "org.elasticsearch.cli.UserException",
        }
    }

    /// The exception as ES renders it (`reload_exception`).
    pub fn to_json(&self) -> Value {
        let mut v = json!({"type": self.es_type(), "reason": self.message()});
        if *self == KeystoreError::WrongPassword {
            // `CipherInputStream` wraps the GCM failure in an IOException.
            v["caused_by"] = json!({
                "type": "i_o_exception",
                "reason": "javax.crypto.AEADBadTagException: Tag mismatch",
                "caused_by": {"type": "a_e_a_d_bad_tag_exception", "reason": "Tag mismatch"}
            });
        }
        v
    }
}

impl std::fmt::Display for KeystoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for KeystoreError {}

/// An Elasticsearch keystore: loaded from disk ([`KeyStore::load`]) and then
/// [`KeyStore::decrypt`]ed, or new ([`KeyStore::create`]).
pub struct KeyStore {
    format_version: i32,
    has_password: bool,
    /// The encrypted data block of a loaded keystore.
    data: Vec<u8>,
    /// The entries, once decrypted.
    entries: Option<BTreeMap<String, Vec<u8>>>,
}

impl std::fmt::Debug for KeyStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never the values.
        f.debug_struct("KeyStore")
            .field("format_version", &self.format_version)
            .field("has_password", &self.has_password)
            .field("entries", &self.entries.as_ref().map(|e| e.keys().cloned().collect::<Vec<_>>()))
            .finish()
    }
}

impl Drop for KeyStore {
    fn drop(&mut self) {
        if let Some(entries) = self.entries.as_mut() {
            for v in entries.values_mut() {
                v.iter_mut().for_each(|b| *b = 0);
            }
        }
    }
}

/// `KeyStoreWrapper.keystorePath`.
pub fn keystore_path(conf: &Path) -> PathBuf {
    conf.join(KEYSTORE_FILENAME)
}

/// Lucene's description of the input, as its exceptions name the resource.
fn resource(path: &Path) -> String {
    format!("BufferedChecksumIndexInput(NIOFSIndexInput(path=\"{}\"))", path.display())
}

impl KeyStore {
    /// `KeyStoreWrapper.create`: a new, empty current-format keystore with a
    /// `keystore.seed`.
    pub fn create() -> Result<KeyStore, KeystoreError> {
        let mut ks = KeyStore { format_version: CURRENT_VERSION, has_password: false, data: Vec::new(), entries: Some(BTreeMap::new()) };
        ks.add_bootstrap_seed()?;
        Ok(ks)
    }

    /// `KeyStoreWrapper.load`: `None` when `conf` has no keystore. The
    /// entries stay encrypted until [`KeyStore::decrypt`].
    pub fn load(conf: &Path) -> Result<Option<KeyStore>, KeystoreError> {
        let path = keystore_path(conf);
        if !path.exists() {
            return Ok(None);
        }
        let bytes = std::fs::read(&path).map_err(|e| io_error(&path, e))?;
        KeyStore::parse(&bytes, &path).map(Some)
    }

    /// Parse a keystore file's bytes (header, data block, footer and checksum).
    pub fn parse(bytes: &[u8], path: &Path) -> Result<KeyStore, KeystoreError> {
        let res = resource(path);
        let mut r = Reader::new(bytes, &res);
        // CodecUtil.checkHeader
        let magic = r.be_i32()?;
        if magic != CODEC_MAGIC {
            return Err(KeystoreError::Corrupt(format!("codec header mismatch: actual header={magic} vs expected header={CODEC_MAGIC} (resource={res})")));
        }
        let codec = r.string()?;
        if codec != KEYSTORE_FILENAME {
            return Err(KeystoreError::Corrupt(format!("codec mismatch: actual codec={codec} vs expected codec={KEYSTORE_FILENAME} (resource={res})")));
        }
        let version = r.be_i32()?;
        if version < MIN_FORMAT_VERSION {
            return Err(KeystoreError::IllegalState(format!(
                "The Elasticsearch keystore [{}] format is too old. You should delete and recreate it in order to upgrade.",
                path.display()
            )));
        }
        if version > CURRENT_VERSION {
            return Err(KeystoreError::IllegalState(format!(
                "The Elasticsearch keystore [{}] format is too new. Are you trying to downgrade? You should delete and recreate it in order to downgrade.",
                path.display()
            )));
        }
        let flag = r.byte()?;
        if flag != 0 && flag != 1 {
            return Err(KeystoreError::IllegalState(format!("hasPassword boolean is corrupt: {flag:02x}")));
        }
        let len = if version < LE_VERSION { r.be_i32()? } else { r.le_i32()? };
        let len = usize::try_from(len).map_err(|_| KeystoreError::Corrupt(format!("negative data length [{len}] (resource={res})")))?;
        let data = r.take(len)?.to_vec();
        // CodecUtil.checkFooter
        let remaining = bytes.len() - r.pos;
        if remaining < FOOTER_LENGTH {
            return Err(KeystoreError::Corrupt(format!("misplaced codec footer (file truncated?): remaining={remaining}, expected={FOOTER_LENGTH}, fp={} (resource={res})", r.pos)));
        }
        if remaining > FOOTER_LENGTH {
            return Err(KeystoreError::Corrupt(format!("misplaced codec footer (file extended?): remaining={remaining}, expected={FOOTER_LENGTH}, fp={} (resource={res})", r.pos)));
        }
        let footer = r.be_i32()?;
        if footer != FOOTER_MAGIC {
            return Err(KeystoreError::Corrupt(format!("codec footer mismatch (file truncated?): actual footer={footer} vs expected footer={FOOTER_MAGIC} (resource={res})")));
        }
        let algorithm = r.be_i32()?;
        if algorithm != 0 {
            return Err(KeystoreError::Corrupt(format!("codec footer mismatch: unknown algorithmID: {algorithm} (resource={res})")));
        }
        let actual = u64::from(crc32(&bytes[..r.pos]));
        let expected = r.be_i64()? as u64;
        if expected & 0xFFFF_FFFF_0000_0000 != 0 {
            return Err(KeystoreError::Corrupt(format!("Illegal CRC-32 checksum: {expected} (resource={res})")));
        }
        if expected != actual {
            return Err(KeystoreError::Corrupt(format!("checksum failed (hardware problem?) : expected={expected:x} actual={actual:x} (resource={res})")));
        }
        Ok(KeyStore { format_version: version, has_password: flag == 1, data, entries: None })
    }

    pub fn format_version(&self) -> i32 {
        self.format_version
    }

    /// Whether [`KeyStore::decrypt`] needs a non-empty password.
    pub fn has_password(&self) -> bool {
        self.has_password
    }

    pub fn is_loaded(&self) -> bool {
        self.entries.is_some()
    }

    /// `KeyStoreWrapper.decrypt` (once): `""` when the keystore has no password.
    pub fn decrypt(&mut self, password: &str) -> Result<(), KeystoreError> {
        if self.entries.is_some() {
            return Err(KeystoreError::IllegalState("Keystore has already been decrypted".into()));
        }
        let corrupted = || KeystoreError::Security(CORRUPTED.into());
        let le = self.format_version >= LE_VERSION;
        let mut r = Reader::new(&self.data, "");
        let salt = read_array(&mut r, le).ok_or_else(corrupted)?;
        let iv = read_array(&mut r, le).ok_or_else(corrupted)?;
        let mut encrypted = read_array(&mut r, le).ok_or_else(corrupted)?;
        if r.pos != self.data.len() {
            return Err(corrupted());
        }
        let key = derive_key(password, &salt, kdf_iterations(self.format_version), cipher_key_len(self.format_version));
        // ES always writes a 96-bit IV (Java would accept others; none exist).
        let nonce = ring::aead::Nonce::try_assume_unique_for_key(&iv).map_err(|_| corrupted())?;
        let algorithm = if key.len() == 32 { &ring::aead::AES_256_GCM } else { &ring::aead::AES_128_GCM };
        let unbound = ring::aead::UnboundKey::new(algorithm, &key).map_err(|_| corrupted())?;
        let opening = ring::aead::LessSafeKey::new(unbound);
        let plain = opening.open_in_place(nonce, ring::aead::Aad::from(&salt[..]), &mut encrypted).map_err(|_| KeystoreError::WrongPassword)?;
        let entries = parse_entries(plain, self.format_version).ok_or_else(corrupted);
        plain.iter_mut().for_each(|b| *b = 0);
        self.entries = Some(entries?);
        Ok(())
    }

    fn entries(&self) -> &BTreeMap<String, Vec<u8>> {
        self.entries.as_ref().expect("keystore is not loaded")
    }

    fn entries_mut(&mut self) -> &mut BTreeMap<String, Vec<u8>> {
        self.entries.as_mut().expect("keystore is not loaded")
    }

    /// The entry names, sorted.
    pub fn setting_names(&self) -> Vec<String> {
        self.entries().keys().cloned().collect()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries().contains_key(name)
    }

    /// An entry's raw bytes (`getFile`).
    pub fn get_bytes(&self, name: &str) -> Option<&[u8]> {
        self.entries().get(name).map(Vec::as_slice)
    }

    /// An entry as text (`getString`: UTF-8, malformed input replaced).
    pub fn get_string(&self, name: &str) -> Option<String> {
        self.get_bytes(name).map(|b| String::from_utf8_lossy(b).into_owned())
    }

    /// SHA-256 of an entry's value (`getSHA256Digest`).
    pub fn sha256_digest(&self, name: &str) -> Option<Vec<u8>> {
        self.get_bytes(name).map(|b| ring::digest::digest(&ring::digest::SHA256, b).as_ref().to_vec())
    }

    /// `setString`.
    pub fn set_string(&mut self, name: &str, value: &str) -> Result<(), KeystoreError> {
        self.set_file(name, value.as_bytes())
    }

    /// `setFile`.
    pub fn set_file(&mut self, name: &str, value: &[u8]) -> Result<(), KeystoreError> {
        validate_setting_name(name)?;
        if let Some(mut old) = self.entries_mut().insert(name.to_string(), value.to_vec()) {
            old.iter_mut().for_each(|b| *b = 0);
        }
        Ok(())
    }

    /// `remove`: whether the entry existed.
    pub fn remove(&mut self, name: &str) -> bool {
        match self.entries_mut().remove(name) {
            Some(mut old) => {
                old.iter_mut().for_each(|b| *b = 0);
                true
            }
            None => false,
        }
    }

    /// `addBootstrapSeed`: 20 random characters as `keystore.seed`.
    fn add_bootstrap_seed(&mut self) -> Result<(), KeystoreError> {
        let rng = ring::rand::SystemRandom::new();
        let limit = 256 - 256 % SEED_CHARS.len();
        let mut seed = String::with_capacity(20);
        while seed.len() < 20 {
            let mut b = [0u8; 32];
            rng.fill(&mut b).map_err(|_| KeystoreError::Io("no secure random source".into()))?;
            for x in b {
                if (x as usize) < limit && seed.len() < 20 {
                    seed.push(SEED_CHARS[x as usize % SEED_CHARS.len()] as char);
                }
            }
        }
        self.set_string(SEED_SETTING, &seed)
    }

    /// Whether [`KeyStore::upgrade`] would rewrite the file.
    pub fn needs_upgrade(&self) -> bool {
        self.format_version != CURRENT_VERSION || !self.contains(SEED_SETTING)
    }

    /// `KeyStoreWrapper.upgrade`: rewrite in the current format, with a seed.
    pub fn upgrade(&mut self, conf: &Path, password: &str) -> Result<bool, KeystoreError> {
        if !self.needs_upgrade() {
            return Ok(false);
        }
        if !self.contains(SEED_SETTING) {
            self.add_bootstrap_seed()?;
        }
        self.save(conf, password)?;
        Ok(true)
    }

    /// The current-format file for these entries under `password`, with a
    /// fresh random salt and IV.
    pub fn encode(&self, password: &str) -> Result<Vec<u8>, KeystoreError> {
        let rng = ring::rand::SystemRandom::new();
        let mut salt = [0u8; SALT_LEN];
        let mut iv = [0u8; IV_LEN];
        rng.fill(&mut salt).map_err(|_| KeystoreError::Io("no secure random source".into()))?;
        rng.fill(&mut iv).map_err(|_| KeystoreError::Io("no secure random source".into()))?;
        self.encode_with(password, &salt, &iv)
    }

    /// [`KeyStore::encode`] with a given salt and IV (deterministic, for tests).
    pub(crate) fn encode_with(&self, password: &str, salt: &[u8; SALT_LEN], iv: &[u8; IV_LEN]) -> Result<Vec<u8>, KeystoreError> {
        let entries = self.entries();
        let mut plain = Vec::new();
        plain.extend_from_slice(&(entries.len() as i32).to_be_bytes());
        for (name, value) in entries {
            let n = name.as_bytes();
            let n_len = u16::try_from(n.len()).map_err(|_| KeystoreError::IllegalArgument(format!("Setting name [{name}] is too long")))?;
            plain.extend_from_slice(&n_len.to_be_bytes());
            plain.extend_from_slice(n);
            plain.extend_from_slice(&(value.len() as i32).to_be_bytes());
            plain.extend_from_slice(value);
        }
        let key = derive_key(password, salt, kdf_iterations(CURRENT_VERSION), cipher_key_len(CURRENT_VERSION));
        let unbound = ring::aead::UnboundKey::new(&ring::aead::AES_256_GCM, &key).map_err(|_| KeystoreError::Io("cannot create the keystore cipher".into()))?;
        let sealing = ring::aead::LessSafeKey::new(unbound);
        sealing
            .seal_in_place_append_tag(ring::aead::Nonce::assume_unique_for_key(*iv), ring::aead::Aad::from(&salt[..]), &mut plain)
            .map_err(|_| KeystoreError::Io("cannot encrypt the keystore".into()))?;
        let encrypted = plain;
        let mut out = Vec::with_capacity(64 + SALT_LEN + IV_LEN + encrypted.len());
        out.extend_from_slice(&CODEC_MAGIC.to_be_bytes());
        out.push(KEYSTORE_FILENAME.len() as u8);
        out.extend_from_slice(KEYSTORE_FILENAME.as_bytes());
        out.extend_from_slice(&CURRENT_VERSION.to_be_bytes());
        out.push(u8::from(!password.is_empty()));
        let data_len = 4 + SALT_LEN + 4 + IV_LEN + 4 + encrypted.len();
        out.extend_from_slice(&(data_len as i32).to_le_bytes());
        out.extend_from_slice(&(SALT_LEN as i32).to_le_bytes());
        out.extend_from_slice(salt);
        out.extend_from_slice(&(IV_LEN as i32).to_le_bytes());
        out.extend_from_slice(iv);
        out.extend_from_slice(&(encrypted.len() as i32).to_le_bytes());
        out.extend_from_slice(&encrypted);
        out.extend_from_slice(&FOOTER_MAGIC.to_be_bytes());
        out.extend_from_slice(&0i32.to_be_bytes());
        let crc = u64::from(crc32(&out));
        out.extend_from_slice(&crc.to_be_bytes());
        Ok(out)
    }

    /// `KeyStoreWrapper.save`: write `conf/elasticsearch.keystore` through a
    /// temporary file (mode `rw-rw----`) renamed into place.
    pub fn save(&self, conf: &Path, password: &str) -> Result<(), KeystoreError> {
        let bytes = self.encode(password)?;
        let tmp = conf.join(format!("{KEYSTORE_FILENAME}.tmp.{}", std::process::id()));
        let write = || -> std::io::Result<()> {
            use std::io::Write;
            let mut opts = std::fs::OpenOptions::new();
            opts.write(true).create(true).truncate(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                opts.mode(0o660);
            }
            let mut f = opts.open(&tmp)?;
            f.write_all(&bytes)?;
            f.sync_all()?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o660))?;
            }
            Ok(())
        };
        if let Err(e) = write() {
            let _ = std::fs::remove_file(&tmp);
            if e.kind() == std::io::ErrorKind::PermissionDenied {
                return Err(KeystoreError::Config(format!(
                    "unable to create temporary keystore at [{}], write permissions required for [{}] or run [elasticsearch-keystore upgrade]",
                    conf.join(format!("{KEYSTORE_FILENAME}.tmp")).display(),
                    conf.display()
                )));
            }
            return Err(io_error(&tmp, e));
        }
        if let Err(e) = std::fs::rename(&tmp, keystore_path(conf)) {
            let _ = std::fs::remove_file(&tmp);
            return Err(io_error(&keystore_path(conf), e));
        }
        Ok(())
    }
}

fn io_error(path: &Path, e: std::io::Error) -> KeystoreError {
    match e.kind() {
        std::io::ErrorKind::PermissionDenied => KeystoreError::Io(format!("{}", path.display())),
        _ => KeystoreError::Io(format!("{}: {e}", path.display())),
    }
}

/// `KeyStoreWrapper.validateSettingName`.
pub fn validate_setting_name(name: &str) -> Result<(), KeystoreError> {
    if name.is_empty() || !name.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.')) {
        return Err(KeystoreError::IllegalArgument(format!(
            "Setting name [{name}] does not match the allowed setting name pattern [[A-Za-z0-9_\\-.]+]"
        )));
    }
    Ok(())
}

fn kdf_iterations(version: i32) -> u32 {
    if version < HIGHER_KDF_ITERATION_COUNT_VERSION { LEGACY_KDF_ITERS } else { KDF_ITERS }
}

fn cipher_key_len(version: i32) -> usize {
    if version < CIPHER_KEY_BITS_256_VERSION { 16 } else { 32 }
}

/// PBKDF2-HMAC-SHA512 of the UTF-8 password (`PBEKeySpec` +
/// `PBKDF2WithHmacSHA512`).
fn derive_key(password: &str, salt: &[u8], iterations: u32, len: usize) -> Vec<u8> {
    let mut key = vec![0u8; len];
    let iterations = std::num::NonZeroU32::new(iterations).expect("non-zero iterations");
    ring::pbkdf2::derive(ring::pbkdf2::PBKDF2_HMAC_SHA512, iterations, salt, password.as_bytes(), &mut key);
    key
}

/// `KeyStoreWrapper.readByteArray`: i32 length + bytes; `None` when malformed.
fn read_array(r: &mut Reader<'_>, le: bool) -> Option<Vec<u8>> {
    let len = if le { r.le_i32() } else { r.be_i32() }.ok()?;
    let len = usize::try_from(len).ok()?;
    r.take(len).ok().map(<[u8]>::to_vec)
}

/// The decrypted `DataOutputStream` entries; `None` when malformed.
fn parse_entries(plain: &[u8], version: i32) -> Option<BTreeMap<String, Vec<u8>>> {
    let mut r = Reader::new(plain, "");
    let count = r.be_i32().ok()?;
    let mut entries = BTreeMap::new();
    for _ in 0..count.max(0) {
        let name = r.java_utf().ok()?;
        if version == V3_VERSION {
            // The legacy entry type (`STRING` / `FILE`).
            r.java_utf().ok()?;
        }
        let len = usize::try_from(r.be_i32().ok()?).ok()?;
        let value = r.take(len).ok()?.to_vec();
        entries.insert(name, value);
    }
    (r.pos == plain.len()).then_some(entries)
}

/// A bounds-checked cursor over a byte slice.
struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
    res: &'a str,
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8], res: &'a str) -> Reader<'a> {
        Reader { b, pos: 0, res }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], KeystoreError> {
        let end = self.pos.checked_add(n).filter(|e| *e <= self.b.len());
        match end {
            Some(end) => {
                let b: &'a [u8] = self.b;
                let s = &b[self.pos..end];
                self.pos = end;
                Ok(s)
            }
            None => Err(KeystoreError::Io(format!("read past EOF: {}", self.res))),
        }
    }

    fn byte(&mut self) -> Result<u8, KeystoreError> {
        Ok(self.take(1)?[0])
    }

    fn be_i32(&mut self) -> Result<i32, KeystoreError> {
        let s = self.take(4)?;
        Ok(i32::from_be_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn le_i32(&mut self) -> Result<i32, KeystoreError> {
        let s = self.take(4)?;
        Ok(i32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn be_i64(&mut self) -> Result<i64, KeystoreError> {
        let s = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(s);
        Ok(i64::from_be_bytes(a))
    }

    /// Lucene `readVInt`.
    fn vint(&mut self) -> Result<u32, KeystoreError> {
        let mut out: u32 = 0;
        for shift in (0..35).step_by(7) {
            let b = self.byte()?;
            out |= u32::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Ok(out);
            }
        }
        Err(KeystoreError::Corrupt(format!("Invalid vInt detected (too many bits) (resource={})", self.res)))
    }

    /// Lucene `readString`: vint length + UTF-8.
    fn string(&mut self) -> Result<String, KeystoreError> {
        let len = self.vint()? as usize;
        let s = self.take(len)?;
        Ok(String::from_utf8_lossy(s).into_owned())
    }

    /// Java `DataInput.readUTF`: BE u16 length + modified UTF-8 (setting
    /// names are ASCII).
    fn java_utf(&mut self) -> Result<String, KeystoreError> {
        let s = self.take(2)?;
        let len = usize::from(u16::from_be_bytes([s[0], s[1]]));
        let s = self.take(len)?;
        std::str::from_utf8(s).map(str::to_string).map_err(|_| KeystoreError::Io("malformed input".into()))
    }
}

/// CRC-32 (IEEE 802.3, as `java.util.zip.CRC32`).
pub(crate) fn crc32(data: &[u8]) -> u32 {
    static TABLE: LazyLock<[u32; 256]> = LazyLock::new(|| {
        let mut t = [0u32; 256];
        for (i, slot) in t.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
            *slot = c;
        }
        t
    });
    let mut c = !0u32;
    for &b in data {
        c = TABLE[((c ^ u32::from(b)) & 0xff) as usize] ^ (c >> 8);
    }
    !c
}

/// Lower-case hex.
pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ------------------------------------------------------------- startup

/// The config directory of a node: `path.conf` (set by `config::load` from
/// `ES_PATH_CONF`), default `config`.
pub fn conf_dir(settings: &Settings) -> PathBuf {
    PathBuf::from(crate::config::get_str(settings, "path.conf").unwrap_or("config"))
}

/// The secure settings a node is running with, per config directory: the
/// names loaded at startup and the values of the last load or reload.
#[derive(Default)]
struct Live {
    startup: BTreeSet<String>,
    current: BTreeMap<String, String>,
}

static LIVE: LazyLock<parking_lot::Mutex<HashMap<PathBuf, Live>>> = LazyLock::new(Default::default);

/// ES's `SettingsModule` check: a secure-only setting (see
/// `elasticsearch_engine::settings_filter::is_secure_only`) must come from
/// the keystore, never from `elasticsearch.yml`, `-E` or the environment.
pub fn reject_insecure(settings: &Settings) -> Result<(), ConfigError> {
    match settings.keys().find(|k| elasticsearch_engine::settings_filter::is_secure_only(k)) {
        Some(k) => Err(ConfigError(format!(
            "Setting [{k}] is a secure setting and must be stored inside the Elasticsearch keystore, but was found inside elasticsearch.yml"
        ))),
        None => Ok(()),
    }
}

/// Node start (`cli::main`, after `config::load`): reject secure settings
/// given as plain settings, load the keystore of `path.conf` (`KeyStoreWrapper.bootstrap`:
/// created when missing, upgraded when old; a password-protected one reads
/// its password from `ES_KEYSTORE_PASSPHRASE_FILE`, `KEYSTORE_PASSWORD` or
/// standard input) and add its entries to `settings` as secure settings.
pub fn load_into_settings(settings: &mut Settings, env: &BTreeMap<String, String>) -> Result<(), ConfigError> {
    reject_insecure(settings)?;
    let conf = conf_dir(settings);
    let keystore = bootstrap(&conf, || startup_password(env)).map_err(|e| ConfigError(e.to_string()))?;
    if let Some(ks) = keystore {
        inject(settings, &conf, &ks)?;
    }
    Ok(())
}

/// `KeyStoreWrapper.bootstrap`. A missing keystore is created (and saved
/// when `conf` exists; a read-only config directory only logs a warning,
/// where ES would fail to start); `None` when there is no config directory.
pub fn bootstrap<F>(conf: &Path, password: F) -> Result<Option<KeyStore>, KeystoreError>
where
    F: FnOnce() -> Result<String, KeystoreError>,
{
    match KeyStore::load(conf)? {
        None => {
            if !conf.is_dir() {
                return Ok(None);
            }
            let ks = KeyStore::create()?;
            if let Err(e) = ks.save(conf, "") {
                eprintln!("warning: could not create the elasticsearch keystore in [{}]: {e}", conf.display());
            }
            Ok(Some(ks))
        }
        Some(mut ks) => {
            let pw = if ks.has_password() { password()? } else { String::new() };
            ks.decrypt(&pw)?;
            if ks.needs_upgrade() {
                if let Err(e) = ks.upgrade(conf, &pw) {
                    eprintln!("warning: could not upgrade the elasticsearch keystore in [{}]: {e}", conf.display());
                }
            }
            Ok(Some(ks))
        }
    }
}

/// The startup password of a protected keystore: the first line of
/// `ES_KEYSTORE_PASSPHRASE_FILE` (what ES's systemd entrypoint feeds the
/// server), else `KEYSTORE_PASSWORD` (what ES's Docker entrypoint feeds it),
/// else a line of standard input (ES's prompt).
fn startup_password(env: &BTreeMap<String, String>) -> Result<String, KeystoreError> {
    if let Some(f) = env.get("ES_KEYSTORE_PASSPHRASE_FILE").filter(|f| !f.is_empty()) {
        let text = std::fs::read_to_string(f).map_err(|e| KeystoreError::Io(format!("{f}: {e}")))?;
        return Ok(text.lines().next().unwrap_or("").to_string());
    }
    if let Some(p) = env.get("KEYSTORE_PASSWORD") {
        return Ok(p.clone());
    }
    read_secret(PROMPT)
}

/// Read one line from standard input without echo when it is a terminal
/// (`Terminal.readSecret`). The prompt goes to standard error on a
/// terminal only.
pub fn read_secret(prompt: &str) -> Result<String, KeystoreError> {
    use std::io::{BufRead, IsTerminal, Write};
    let stdin = std::io::stdin();
    let tty = stdin.is_terminal();
    if tty {
        let mut err = std::io::stderr();
        let _ = write!(err, "{prompt}");
        let _ = err.flush();
    }
    let _echo = if tty { EchoOff::new() } else { None };
    let mut line = String::new();
    let n = stdin.lock().read_line(&mut line).map_err(|e| KeystoreError::Io(e.to_string()))?;
    drop(_echo);
    if tty {
        eprintln!();
    }
    if n == 0 {
        return Err(KeystoreError::IllegalState("unable to read from standard input; is standard input open and a tty attached?".into()));
    }
    while line.ends_with('\n') || line.ends_with('\r') {
        line.pop();
    }
    Ok(line)
}

/// Terminal echo disabled until dropped.
struct EchoOff {
    #[cfg(unix)]
    saved: libc::termios,
}

impl EchoOff {
    #[cfg(unix)]
    fn new() -> Option<EchoOff> {
        // SAFETY: tcgetattr/tcsetattr on the process's stdin descriptor with
        // a zero-initialised termios that tcgetattr fills.
        unsafe {
            let mut t: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(libc::STDIN_FILENO, &mut t) != 0 {
                return None;
            }
            let saved = t;
            t.c_lflag &= !libc::ECHO;
            if libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &t) != 0 {
                return None;
            }
            Some(EchoOff { saved })
        }
    }

    #[cfg(not(unix))]
    fn new() -> Option<EchoOff> {
        None
    }
}

impl Drop for EchoOff {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            // SAFETY: restores the attributes read in `new`.
            unsafe {
                libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &self.saved);
            }
        }
    }
}

/// Add a decrypted keystore's entries to `settings` as secure settings and
/// remember them as the node's live secure settings. Entries that are not
/// UTF-8 (binary file settings) stay out of the flat map but are still
/// registered as secure names.
pub fn inject(settings: &mut Settings, conf: &Path, ks: &KeyStore) -> Result<(), ConfigError> {
    let names = ks.setting_names();
    if let Some(k) = names.iter().find(|k| settings.contains_key(*k)) {
        return Err(ConfigError(format!(
            "Setting [{k}] is a non-secure setting and must be stored inside elasticsearch.yml, but was found inside the Elasticsearch keystore"
        )));
    }
    elasticsearch_engine::settings_filter::register_secure(names.iter().cloned());
    let mut current = BTreeMap::new();
    for name in &names {
        if let Some(text) = ks.get_bytes(name).and_then(|b| std::str::from_utf8(b).ok()) {
            settings.insert(name.clone(), Value::String(text.to_string()));
            current.insert(name.clone(), text.to_string());
        }
    }
    LIVE.lock().insert(live_key(conf), Live { startup: names.into_iter().collect(), current });
    Ok(())
}

fn live_key(conf: &Path) -> PathBuf {
    std::fs::canonicalize(conf).unwrap_or_else(|_| conf.to_path_buf())
}

/// Replace, in a flat copy of a node's settings, the secure settings it
/// started with by the current ones (after `reload_secure_settings`).
pub fn overlay_secure(settings: &Settings, flat: &mut BTreeMap<String, String>) {
    let live = LIVE.lock();
    let Some(l) = live.get(&live_key(&conf_dir(settings))) else { return };
    for k in &l.startup {
        flat.remove(k);
    }
    for (k, v) in &l.current {
        flat.insert(k.clone(), v.clone());
    }
}

/// What one node's reload read (`NodesReloadSecureSettingsResponse.NodeResponse`).
#[derive(Debug, Clone)]
pub struct Reloaded {
    pub names: Vec<String>,
    pub path: String,
    /// SHA-256 of the keystore file, lower-case hex.
    pub digest: String,
    /// Last modification, epoch milliseconds.
    pub last_modified: Option<i64>,
}

/// `TransportNodesReloadSecureSettingsAction.nodeOperation`: re-read and
/// decrypt the keystore of `conf` and make its entries the live secure
/// settings.
pub fn reload(conf: &Path, password: &str) -> Result<Reloaded, KeystoreError> {
    let path = keystore_path(conf);
    if !path.exists() {
        return Err(KeystoreError::IllegalState("Keystore is missing".into()));
    }
    let bytes = std::fs::read(&path).map_err(|e| io_error(&path, e))?;
    let mut ks = KeyStore::parse(&bytes, &path)?;
    ks.decrypt(password)?;
    let names = ks.setting_names();
    elasticsearch_engine::settings_filter::register_secure(names.iter().cloned());
    let current: BTreeMap<String, String> =
        names.iter().filter_map(|n| ks.get_bytes(n).and_then(|b| std::str::from_utf8(b).ok()).map(|t| (n.clone(), t.to_string()))).collect();
    LIVE.lock().entry(live_key(conf)).or_default().current = current;
    let last_modified = std::fs::metadata(&path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as i64);
    Ok(Reloaded {
        names,
        path: path.display().to_string(),
        digest: hex(ring::digest::digest(&ring::digest::SHA256, &bytes).as_ref()),
        last_modified,
    })
}

#[cfg(test)]
mod tests;
