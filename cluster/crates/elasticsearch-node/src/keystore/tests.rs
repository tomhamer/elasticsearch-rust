//! Keystore format tests. `testdata/format-v{3,4,5,6}-*.keystore` are the
//! keystores Elasticsearch's own `KeyStoreWrapperTests` /
//! `UpgradeKeyStoreCommandTests` load (written by real ES releases);
//! `format-v7-python-kat.keystore` was written by an independent Python
//! implementation (`hashlib.pbkdf2_hmac` + `cryptography`'s AESGCM +
//! `zlib.crc32`) of the version 7 format with a fixed salt and IV, which
//! also decodes the ES fixtures above.

use super::*;

const V3: &[u8] = include_bytes!("testdata/format-v3-elasticsearch.keystore");
const V4: &[u8] = include_bytes!("testdata/format-v4-elasticsearch.keystore");
const V5: &[u8] = include_bytes!("testdata/format-v5-with-password-elasticsearch.keystore");
const V6: &[u8] = include_bytes!("testdata/format-v6-elasticsearch.keystore");
const V7_KAT: &[u8] = include_bytes!("testdata/format-v7-python-kat.keystore");

fn p() -> PathBuf {
    PathBuf::from("/conf/elasticsearch.keystore")
}

fn open(bytes: &[u8], password: &str) -> KeyStore {
    let mut ks = KeyStore::parse(bytes, &p()).unwrap();
    ks.decrypt(password).unwrap();
    ks
}

#[test]
fn reads_keystores_written_by_elasticsearch() {
    // ES `testLegacyV3` (and the v4 upgrade fixture: the same entries).
    for (bytes, version) in [(V3, 3), (V4, 4)] {
        let ks = open(bytes, "");
        assert_eq!(ks.format_version(), version);
        assert!(!ks.has_password());
        assert_eq!(ks.setting_names(), vec!["file_setting", "keystore.seed", "string_setting"]);
        assert_eq!(ks.get_string("string_setting").as_deref(), Some("string_value"));
        assert_eq!(ks.get_bytes("file_setting"), Some(&b"file_value"[..]));
        assert_eq!(ks.get_string("keystore.seed").as_deref(), Some("5ehK4*A3US#G-E9jQrq*"));
        assert!(ks.needs_upgrade());
    }
    // ES `testLegacyV5` / `testLegacyV6`: password-protected.
    let ks = open(V5, "keystorepassword");
    assert_eq!((ks.format_version(), ks.has_password()), (5, true));
    assert_eq!(ks.setting_names(), vec!["keystore.seed"]);
    let ks = open(V6, "keystorepassword");
    assert_eq!((ks.format_version(), ks.has_password()), (6, true));
    assert_eq!(ks.setting_names(), vec!["keystore.seed", "string"]);
    assert_eq!(ks.get_string("string").as_deref(), Some("value"));
    assert_eq!(ks.sha256_digest("string").unwrap(), ring::digest::digest(&ring::digest::SHA256, b"value").as_ref().to_vec());
}

#[test]
fn version_7_known_answer() {
    let ks = open(V7_KAT, "kat-password");
    assert_eq!((ks.format_version(), ks.has_password()), (7, true));
    assert_eq!(ks.setting_names(), vec!["bootstrap.password", "keystore.seed", "s3.client.default.access_key"]);
    assert_eq!(ks.get_string("s3.client.default.access_key").as_deref(), Some("AKIA-KAT"));
    // The writer reproduces the independent implementation byte for byte.
    let salt: [u8; SALT_LEN] = std::array::from_fn(|i| i as u8);
    let iv: [u8; IV_LEN] = std::array::from_fn(|i| 100 + i as u8);
    assert_eq!(ks.encode_with("kat-password", &salt, &iv).unwrap(), V7_KAT);
}

#[test]
fn wrong_password_and_tampering() {
    let mut ks = KeyStore::parse(V6, &p()).unwrap();
    let e = ks.decrypt("nope").unwrap_err();
    assert_eq!(e, KeystoreError::WrongPassword);
    assert_eq!(e.to_string(), "Provided keystore password was incorrect");
    assert_eq!(e.to_json()["type"], "security_exception");
    // Unprotected keystore opened with a password: the tag does not verify.
    assert_eq!(KeyStore::parse(V4, &p()).unwrap().decrypt("x").unwrap_err(), KeystoreError::WrongPassword);
    // Already decrypted.
    let mut ks = open(V4, "");
    assert!(matches!(ks.decrypt(""), Err(KeystoreError::IllegalState(m)) if m == "Keystore has already been decrypted"));

    // A flipped ciphertext byte with a recomputed checksum fails the tag.
    let mut bad = V6.to_vec();
    let at = bad.len() - FOOTER_LENGTH - 5;
    bad[at] ^= 1;
    fix_crc(&mut bad);
    assert_eq!(KeyStore::parse(&bad, &p()).unwrap().decrypt("keystorepassword").unwrap_err(), KeystoreError::WrongPassword);

    // A broken data block (salt length past the end): corrupted.
    let mut bad = V6.to_vec();
    let data_start = 4 + 1 + KEYSTORE_FILENAME.len() + 4 + 1 + 4;
    bad[data_start..data_start + 4].copy_from_slice(&1000i32.to_le_bytes());
    fix_crc(&mut bad);
    assert_eq!(KeyStore::parse(&bad, &p()).unwrap().decrypt("keystorepassword").unwrap_err(), KeystoreError::Security(CORRUPTED.into()));
}

fn fix_crc(bytes: &mut [u8]) {
    let n = bytes.len() - 8;
    let crc = u64::from(crc32(&bytes[..n]));
    bytes[n..].copy_from_slice(&crc.to_be_bytes());
}

#[test]
fn checksum_header_and_version_errors() {
    let mut bad = V6.to_vec();
    bad[40] ^= 0xff;
    let e = KeyStore::parse(&bad, &p()).unwrap_err();
    assert!(matches!(&e, KeystoreError::Corrupt(m) if m.starts_with("checksum failed (hardware problem?) : expected=a7e9f603 actual=")), "{e:?}");
    assert_eq!(e.es_type(), "corrupt_index_exception");

    let mut bad = V6.to_vec();
    bad[0] = 0;
    assert!(matches!(KeyStore::parse(&bad, &p()), Err(KeystoreError::Corrupt(m)) if m.starts_with("codec header mismatch: actual header=")));

    let e = KeyStore::parse(&V6[..V6.len() - 3], &p()).unwrap_err();
    assert!(matches!(&e, KeystoreError::Corrupt(m) if m.starts_with("misplaced codec footer (file truncated?)")), "{e:?}");
    let mut longer = V6.to_vec();
    longer.push(0);
    assert!(matches!(KeyStore::parse(&longer, &p()), Err(KeystoreError::Corrupt(m)) if m.starts_with("misplaced codec footer (file extended?)")));

    for (version, word) in [(2, "too old"), (8, "too new")] {
        let mut bad = V6.to_vec();
        let at = 4 + 1 + KEYSTORE_FILENAME.len();
        bad[at..at + 4].copy_from_slice(&(version as i32).to_be_bytes());
        fix_crc(&mut bad);
        let e = KeyStore::parse(&bad, &p()).unwrap_err();
        assert!(matches!(&e, KeystoreError::IllegalState(m) if m.contains(word) && m.contains("/conf/elasticsearch.keystore")), "{e:?}");
    }

    let mut bad = V6.to_vec();
    bad[4 + 1 + KEYSTORE_FILENAME.len() + 4] = 7;
    fix_crc(&mut bad);
    assert_eq!(KeyStore::parse(&bad, &p()).unwrap_err(), KeystoreError::IllegalState("hasPassword boolean is corrupt: 07".into()));
}

#[test]
fn round_trip_with_and_without_password() {
    let dir = tempfile::tempdir().unwrap();
    let mut ks = KeyStore::create().unwrap();
    assert_eq!(ks.setting_names(), vec![SEED_SETTING]);
    assert_eq!(ks.get_string(SEED_SETTING).unwrap().len(), 20);
    ks.set_string("s3.client.default.secret_key", "s3cr3t-ünïcode").unwrap();
    ks.set_file("gcs.client.default.credentials_file", &[0, 159, 146, 150, 255]).unwrap();
    for password in ["", "correct horse"] {
        ks.save(dir.path(), password).unwrap();
        let mut back = KeyStore::load(dir.path()).unwrap().unwrap();
        assert_eq!(back.format_version(), CURRENT_VERSION);
        assert_eq!(back.has_password(), !password.is_empty());
        back.decrypt(password).unwrap();
        assert_eq!(back.setting_names(), ks.setting_names());
        assert_eq!(back.get_string("s3.client.default.secret_key").as_deref(), Some("s3cr3t-ünïcode"));
        assert_eq!(back.get_bytes("gcs.client.default.credentials_file"), Some(&[0u8, 159, 146, 150, 255][..]));
        assert!(!back.needs_upgrade());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(keystore_path(dir.path())).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o660);
    }
    assert!(ks.remove("s3.client.default.secret_key"));
    assert!(!ks.remove("s3.client.default.secret_key"));
    assert!(matches!(ks.set_string("bad name", "x"), Err(KeystoreError::IllegalArgument(m))
        if m == "Setting name [bad name] does not match the allowed setting name pattern [[A-Za-z0-9_\\-.]+]"));
    assert!(KeyStore::load(&dir.path().join("missing")).unwrap().is_none());
}

#[test]
fn upgrade_rewrites_old_formats() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(keystore_path(dir.path()), V5).unwrap();
    let mut ks = KeyStore::load(dir.path()).unwrap().unwrap();
    ks.decrypt("keystorepassword").unwrap();
    assert!(ks.upgrade(dir.path(), "keystorepassword").unwrap());
    let mut back = KeyStore::load(dir.path()).unwrap().unwrap();
    assert_eq!((back.format_version(), back.has_password()), (CURRENT_VERSION, true));
    back.decrypt("keystorepassword").unwrap();
    assert_eq!(back.get_string(SEED_SETTING).as_deref(), Some("A_Egi6NbWku?H$kWzeFB"));
    assert!(!back.upgrade(dir.path(), "keystorepassword").unwrap());
}

#[test]
fn crc32_matches_java() {
    assert_eq!(crc32(b""), 0);
    assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
}

#[test]
fn startup_injects_secure_settings_and_rejects_them_in_yml() {
    let dir = tempfile::tempdir().unwrap();
    let mut ks = KeyStore::create().unwrap();
    ks.set_string("s3.client.default.access_key", "AKIA-STARTUP").unwrap();
    ks.set_string("s3.client.default.secret_key", "startup-secret").unwrap();
    ks.save(dir.path(), "pw").unwrap();
    let conf = dir.path().to_string_lossy().to_string();
    let mut settings = Settings::new();
    settings.insert("path.conf".into(), Value::String(conf.clone()));
    settings.insert("s3.client.default.endpoint".into(), Value::String("localhost:9000".into()));
    let env: BTreeMap<String, String> = [("KEYSTORE_PASSWORD".to_string(), "pw".to_string())].into_iter().collect();
    load_into_settings(&mut settings, &env).unwrap();
    assert_eq!(settings["s3.client.default.access_key"], "AKIA-STARTUP");
    assert!(settings.contains_key(SEED_SETTING));
    assert!(elasticsearch_engine::settings_filter::is_filtered(SEED_SETTING));

    // The live overlay follows a reload.
    let mut flat: BTreeMap<String, String> = settings.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string())).collect();
    let mut ks = KeyStore::load(dir.path()).unwrap().unwrap();
    ks.decrypt("pw").unwrap();
    ks.set_string("s3.client.default.access_key", "AKIA-ROTATED").unwrap();
    ks.remove("s3.client.default.secret_key");
    ks.save(dir.path(), "pw").unwrap();
    assert_eq!(reload(dir.path(), "wrong").unwrap_err(), KeystoreError::WrongPassword);
    let r = reload(dir.path(), "pw").unwrap();
    assert_eq!(r.names, vec!["keystore.seed", "s3.client.default.access_key"]);
    assert_eq!(r.digest.len(), 64);
    overlay_secure(&settings, &mut flat);
    assert_eq!(flat["s3.client.default.access_key"], "AKIA-ROTATED");
    assert!(!flat.contains_key("s3.client.default.secret_key"));
    assert_eq!(flat["s3.client.default.endpoint"], "localhost:9000");

    // Wrong startup password.
    let env: BTreeMap<String, String> = [("KEYSTORE_PASSWORD".to_string(), "nope".to_string())].into_iter().collect();
    let mut s = Settings::new();
    s.insert("path.conf".into(), Value::String(conf.clone()));
    assert_eq!(load_into_settings(&mut s, &env).unwrap_err().0, "Provided keystore password was incorrect");

    // A secure setting in elasticsearch.yml / -E.
    let mut s = Settings::new();
    s.insert("s3.client.default.secret_key".into(), Value::String("x".into()));
    assert_eq!(
        load_into_settings(&mut s, &BTreeMap::new()).unwrap_err().0,
        "Setting [s3.client.default.secret_key] is a secure setting and must be stored inside the Elasticsearch keystore, but was found inside elasticsearch.yml"
    );
    let mut s = Settings::new();
    s.insert("bootstrap.password".into(), Value::String("x".into()));
    assert!(reject_insecure(&s).is_err());

    // A keystore entry that is also a plain setting.
    let mut s = Settings::new();
    s.insert("path.conf".into(), Value::String(conf));
    s.insert("keystore.seed".into(), Value::String("x".into()));
    let mut ks = KeyStore::load(dir.path()).unwrap().unwrap();
    ks.decrypt("pw").unwrap();
    assert!(inject(&mut s, dir.path(), &ks).unwrap_err().0.contains("is a non-secure setting and must be stored inside elasticsearch.yml"));
}

#[test]
fn bootstrap_creates_a_missing_keystore_and_upgrades_old_ones() {
    let dir = tempfile::tempdir().unwrap();
    let ks = bootstrap(dir.path(), || panic!("no password needed")).unwrap().unwrap();
    assert_eq!(ks.setting_names(), vec![SEED_SETTING]);
    let mut again = KeyStore::load(dir.path()).unwrap().unwrap();
    again.decrypt("").unwrap();
    assert_eq!(again.get_string(SEED_SETTING), ks.get_string(SEED_SETTING));
    // No config directory: no keystore, nothing written.
    assert!(bootstrap(&dir.path().join("absent"), || panic!()).unwrap().is_none());
    // An ES 6.x (v4) keystore is upgraded in place.
    let old = tempfile::tempdir().unwrap();
    std::fs::write(keystore_path(old.path()), V4).unwrap();
    let ks = bootstrap(old.path(), || panic!()).unwrap().unwrap();
    assert_eq!(ks.get_string("string_setting").as_deref(), Some("string_value"));
    assert_eq!(KeyStore::load(old.path()).unwrap().unwrap().format_version(), CURRENT_VERSION);
    // A protected keystore asks for its password.
    let prot = tempfile::tempdir().unwrap();
    std::fs::write(keystore_path(prot.path()), V6).unwrap();
    let ks = bootstrap(prot.path(), || Ok("keystorepassword".to_string())).unwrap().unwrap();
    assert_eq!(ks.get_string("string").as_deref(), Some("value"));
    assert_eq!(reload(&dir.path().join("absent"), "").unwrap_err(), KeystoreError::IllegalState("Keystore is missing".into()));
}
