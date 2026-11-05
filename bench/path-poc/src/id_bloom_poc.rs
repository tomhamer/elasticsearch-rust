//! Isolated negative-ID lookup experiment. No on-disk format changes.
use rustic_core::{codec::{directory::FileSlice, terms::{TermDictReader, TermDictWriter}}, term_dictionary::PostingsLocation, types::Term};
use std::time::Instant;

fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
fn hash(bytes: &[u8]) -> u64 {
    let mut h = bytes.len() as u64 ^ 0x9e3779b97f4a7c15;
    for part in bytes.chunks(8) {
        let mut word = [0u8; 8]; word[..part.len()].copy_from_slice(part);
        h = mix(h ^ u64::from_le_bytes(word));
    }
    h
}
struct Bloom(Vec<u64>);
impl Bloom {
    fn new(n: usize) -> Self { Self(vec![0; (n * 12).div_ceil(64).max(1).next_power_of_two()]) }
    fn address(&self, key: &[u8]) -> (usize, u64) {
        let h = hash(key);
        let mask = (0..5).fold(0u64, |m, i| m | (1 << ((h >> (i * 6)) & 63)));
        ((h >> 32) as usize & (self.0.len() - 1), mask)
    }
    fn insert(&mut self, key: &[u8]) { let (i, mask) = self.address(key); self.0[i] |= mask; }
    fn contains(&self, key: &[u8]) -> bool { let (i, mask) = self.address(key); self.0[i] & mask == mask }
}

fn main() {
    let n = 1_000_000u64;
    let key = |i: u64| Term::new_text(7, &format!("auto_{:016x}", mix(i))).key();
    let mut keys: Vec<_> = (0..n).map(|i| key(i * 2)).collect(); keys.sort();
    let mut writer = TermDictWriter::new();
    for (i, key) in keys.iter().enumerate() { writer.add(key, &PostingsLocation { offset:i as u64, doc_freq:1, total_term_freq:1 }); }
    let reader = TermDictReader::open(FileSlice::from_vec(writer.finish(n as u32))).unwrap();
    let start = Instant::now();
    let mut bloom = Bloom::new(n as usize);
    let mut cursor = reader.cursor();
    while cursor.advance().unwrap() { bloom.insert(cursor.key()); }
    println!("build_s={:.6} bytes={}", start.elapsed().as_secs_f64(), bloom.0.len() * 8);
    assert!(keys.iter().all(|key| bloom.contains(key)), "false negative");
    for (name, queries) in [("miss", (0..n).map(|i| key(i * 2 + 1)).collect::<Vec<_>>()), ("hit", keys)] {
        let mut elapsed = [0.0; 2];
        let mut checked = [0usize; 2];
        for round in 0..4 {
            for filtered in [round % 2 == 0, round % 2 != 0] {
                let start = Instant::now();
                let count = queries.iter().filter(|key| (!filtered || bloom.contains(key)) && reader.lookup(key).unwrap().is_some()).count();
                elapsed[usize::from(filtered)] += start.elapsed().as_secs_f64();
                checked[usize::from(filtered)] = std::hint::black_box(count);
            }
        }
        assert_eq!(checked[0], checked[1]);
        assert_eq!(checked[0], if name == "hit" { n as usize } else { 0 });
        println!("{name} plain_s={:.6} filtered_s={:.6} speedup={:.3} bloom_positive={}", elapsed[0]/4.0, elapsed[1]/4.0, elapsed[0]/elapsed[1], queries.iter().filter(|k| bloom.contains(k)).count());
    }
}
