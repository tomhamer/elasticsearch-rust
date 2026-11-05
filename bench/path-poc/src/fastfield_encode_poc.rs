//! EC2-only real-column encoding POC. Source loading is outside the timing.
use rustic_core::{codec::{directory::MmapDirectory, fastfields::encode_fast_fields}, fast_fields::FastFields, segment::Segment};
use std::{path::PathBuf, time::Instant};
fn main() {
    let mut args=std::env::args().skip(1);
    let out=PathBuf::from(args.next().expect("output dir"));
    std::fs::create_dir_all(&out).unwrap();
    for (i,path) in args.enumerate() {
        let path=PathBuf::from(path);
        let id=path.file_stem().unwrap().to_str().unwrap().strip_prefix("seg_").unwrap().parse().unwrap();
        let dir=MmapDirectory::open(path.parent().unwrap()).unwrap();
        let seg=Segment::open(&dir,id).unwrap();
        let cols=seg.fast_fields();
        let mut fields=FastFields::new(seg.num_docs);
        for (field,tag) in cols.fields() {
            macro_rules! copy_col {
                ($get:ident,$target:ident) => {{
                    let c=cols.$get(field).unwrap();
                    if c.skips_points() { fields.skip_points(field); }
                    let target=fields.$target(field);
                    for d in 0..c.num_docs() { if let Some(v)=c.get(d) { target.set(d,v); } }
                }};
            }
            match tag {
                1=>copy_col!(get_i64,i64_column_mut),
                2=>copy_col!(get_u64,u64_column_mut),
                3=>copy_col!(get_f64,f64_column_mut),
                _=>{}
            }
        }
        let expected=encode_fast_fields(&fields);
        let mut times=Vec::new();
        for _ in 0..5 {
            let start=Instant::now();
            let encoded=encode_fast_fields(&fields);
            times.push(start.elapsed().as_secs_f64()*1000.0);
            assert_eq!(encoded,expected);
        }
        std::fs::write(out.join(format!("columns-{i}.bin")), &expected).unwrap();
        println!("{{\"segment\":{i},\"documents\":{},\"bytes\":{},\"encode_ms\":{:?}}}",seg.num_docs,expected.len(),times);
    }
}
