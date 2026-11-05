//! EC2-only mechanism POC. Reads existing WAL frames without modifying them.
//! Measures raw vs LZ4 bytes with one sync_data per original record.
use std::{fs::{self, OpenOptions}, io::Write, path::PathBuf, time::Instant};

// Isolated wire DTOs matching the current uncompressed WAL schemas.
#[derive(serde::Deserialize)]
enum ReplicaOp {
    Index { id:String, source_json:String, routing:Option<String>, seq_no:i64, version:u64, primary_term:u64 },
    Delete { id:String, seq_no:i64, version:u64, primary_term:u64 },
    NoOp { seq_no:i64, primary_term:u64, reason:String },
}
#[derive(serde::Serialize)]
enum TranslogOp {
    Index { id:String, source_json:String, seq_no:u64, version:u64, routing:Option<String> },
    Delete { id:String, seq_no:u64, version:u64 },
}
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
            
            let body=&data[pos+8..end];
            assert_eq!(crc32fast::hash(body),u32::from_le_bytes(data[pos+4..pos+8].try_into().unwrap()));
            let decoded;
            let body=if body.starts_with(b"RWZ\x01") {
                decoded=lz4_flex::decompress_size_prepended(&body[4..]).unwrap(); &decoded[..]
            } else { body };
            let kind=u32::from_le_bytes(body[..4].try_into().unwrap());
            if kind==0 {
                let ops: Vec<ReplicaOp>=bincode::deserialize_from(&mut &body[4..]).unwrap();
                for op in ops {
                    let op=match op {
                        ReplicaOp::Index{id,source_json,routing,seq_no,version,..}=>TranslogOp::Index{id,source_json,routing,seq_no:seq_no.try_into().unwrap(),version},
                        ReplicaOp::Delete{id,seq_no,version,..}=>TranslogOp::Delete{id,seq_no:seq_no.try_into().unwrap(),version},
                        ReplicaOp::NoOp{..}=>continue,
                    };
                    let bytes=bincode::serialize(&op).unwrap();
                    let mut record=Vec::with_capacity(bytes.len()+8);
                    record.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                    record.extend_from_slice(&crc32fast::hash(&bytes).to_le_bytes());
                    record.extend_from_slice(&bytes);
                    frames.push(record);
                }
            } else { assert!(kind==1 || kind==2); }

            pos = end;
        }
    }
    assert!(!frames.is_empty());
    // Model append_many's 5K-document shard batches, preserving original
    // framed bytes exactly. This is not a production format/recovery test.
    let original_records = frames.len();
    let frames: Vec<Vec<u8>> = frames.chunks(5000).map(|chunk| chunk.concat()).collect();
    println!("{{\"original_records\":{original_records},\"records_per_group\":5000}}");

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
