//! EC2-only mechanism POC. Reads existing WAL frames without modifying them.
//! Measures raw vs LZ4 bytes with one sync_data per original record.
use std::{fs::{self, OpenOptions}, io::Write, path::PathBuf, time::Instant};

fn main() {
    let mut args = std::env::args().skip(1);
    let output = PathBuf::from(args.next().expect("new output directory"));
    fs::create_dir(&output).expect("output must not already exist");
    let mut frames = Vec::new();
    for path in args {
        let data = fs::read(path).unwrap();
        let mut pos = 0;
        while pos < data.len() {
            assert!(pos + 8 <= data.len(), "torn frame");
            let len = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
            let end = pos + 8 + len;
            assert!(end <= data.len());
            frames.push(data[pos..end].to_vec());
            pos = end;
        }
    }
    assert!(!frames.is_empty());
    let start = Instant::now();
    let compressed: Vec<_> = frames.iter().map(|x| lz4_flex::compress_prepend_size(x)).collect();
    let encode_ms = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    for (raw, packed) in frames.iter().zip(&compressed) {
        assert_eq!(*raw, lz4_flex::decompress_size_prepended(packed).unwrap());
    }
    let decode_ms = start.elapsed().as_secs_f64() * 1000.0;
    let bytes: usize = frames.iter().map(Vec::len).sum();
    let packed_bytes: usize = compressed.iter().map(Vec::len).sum();
    println!("{{\"frames\":{},\"raw_bytes\":{},\"lz4_bytes\":{},\"encode_ms\":{},\"decode_ms\":{}}}",
        frames.len(), bytes, packed_bytes, encode_ms, decode_ms);
    for round in 0..3 {
        for packed in if round % 2 == 0 { [false, true] } else { [true, false] } {
            let name = format!("round-{round}-{}.poc", if packed {"lz4"} else {"raw"});
            let path = output.join(name);
            let mut file = OpenOptions::new().create_new(true).write(true).open(&path).unwrap();
            let start = Instant::now();
            for raw in &frames {
                // Include compression CPU in the measured write path.
                if packed { file.write_all(&lz4_flex::compress_prepend_size(raw)).unwrap(); }
                else { file.write_all(raw).unwrap(); }
                file.sync_data().unwrap();
            }
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            println!("{{\"round\":{round},\"compressed\":{packed},\"write_and_sync_ms\":{ms}}}");
            drop(file);
            fs::remove_file(path).unwrap();
        }
    }
}
