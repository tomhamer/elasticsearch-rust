//! EC2 diagnostic: pass every original source record through the real mapper.
//! No indexing, HTTP timing or Rally acceptance claim.
use elasticsearch_engine::{analysis::AnalysisRegistry, document, mapping::{FieldLookup, Mappings}};
use serde_json::{json, Value};
use std::{fs::{self, File}, io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write}, path::PathBuf};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(args.len(), 4, "source.json mapping.json output-directory");
    let source = PathBuf::from(&args[1]);
    let mapping: Value = serde_json::from_slice(&fs::read(&args[2]).unwrap()).unwrap();
    let out = PathBuf::from(&args[3]);
    fs::create_dir(&out).unwrap();
    let size = source.metadata().unwrap().len();
    let mut handles = Vec::new();
    for part in 0..16u64 {
        let (source, mapping, out) = (source.clone(), mapping.clone(), out.clone());
        handles.push(std::thread::spawn(move || {
            let reg = AnalysisRegistry::default();
            let mut mappings = Mappings::parse(mapping.get("mappings").unwrap_or(&mapping), &reg).unwrap();
            mappings.assign_core_ids();
            let lookup = FieldLookup::build(&mappings, &reg);
            let mut file = File::open(source).unwrap();
            let (lo, hi) = (size * part / 16, size * (part + 1) / 16);
            let mut skip = false;
            if lo > 0 {
                file.seek(SeekFrom::Start(lo-1)).unwrap();
                let mut byte = [0]; file.read_exact(&mut byte).unwrap();
                skip = byte[0] != b'\n';
            }
            let mut input = BufReader::new(file);
            let mut line = String::new();
            let mut offset = lo;
            if skip { offset += input.read_line(&mut line).unwrap() as u64; }
            let mut errors = BufWriter::new(File::create(out.join(format!("errors-{part}.jsonl"))).unwrap());
            let (mut records, mut failed) = (0u64, 0u64);
            while offset < hi {
                line.clear();
                let n = input.read_line(&mut line).unwrap();
                if n == 0 { break; }
                let source: Value = match elasticsearch_engine::jsonpos::parse_source_value(&line) {
                    Ok(value) => value,
                    Err(error) => {
                        failed += 1;
                        writeln!(errors, "{}", json!({"offset":offset,"error":error.to_string(),"raw_source":line})).unwrap();
                        records += 1;
                        offset += n as u64;
                        continue;
                    }
                };
                if let Err(error) = document::parse_document(&source, &offset.to_string(), &mappings, &lookup) {
                    failed += 1;
                    writeln!(errors, "{}", json!({"offset":offset,"error":error.to_string(),"source":source})).unwrap();
                }
                records += 1;
                offset += n as u64;
            }
            errors.flush().unwrap();
            (records, failed)
        }));
    }
    let (mut records, mut failed) = (0, 0);
    for handle in handles { let (n, e) = handle.join().unwrap(); records += n; failed += e; }
    let result = json!({"records":records,"mapping_errors":failed,"source_bytes":size});
    fs::write(out.join("summary.json"), format!("{result}\n")).unwrap();
    println!("{result}");
}
