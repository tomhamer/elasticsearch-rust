//! Inspect real sealed numeric columns and validate a compact sparse prototype.
//! Estimates codec bytes; this does not modify segments or establish server gains.
use rustic_core::{codec::{directory::MmapDirectory, fastfields::{ColumnReader, FastValue}}, segment::Segment};
use serde_json::json;

fn inspect<T: FastValue>(field:u16, tag:u8, col:ColumnReader<'_,T>) -> serde_json::Value {
    let n=col.num_docs() as usize;
    let mut present=vec![0u64;n.div_ceil(64)];
    let mut values=Vec::new();
    let mut docs=Vec::new();
    for d in 0..n {
        if let Some(v)=col.get(d as u32) {
            present[d/64]|=1<<(d%64); values.push(v); docs.push(d as u32);
        }
    }
    let mut rank=Vec::new(); let mut count=0u32;
    for &word in &present {rank.push(count);count+=word.count_ones();}
    let get=|d:usize| -> Option<T> {
        let word=present[d/64]; let bit=d%64;
        if word&(1<<bit)==0 {return None;}
        let at=rank[d/64]+(word&((1u64<<bit)-1)).count_ones();
        Some(values[at as usize])
    };
    for d in 0..n {
        assert_eq!(get(d).map(T::to_le),col.get(d as u32).map(T::to_le));
    }
    docs.sort_unstable_by(|&a,&b|get(a as usize).unwrap().sort_cmp(&get(b as usize).unwrap()).then(a.cmp(&b)));
    let bounds:Vec<T>=values.iter().copied().filter(|v|!v.is_nan()).step_by((values.len()/4).max(1)).take(5).collect();
    for &a in &bounds {for &b in &bounds {
        if a>b {continue;}
        let lo=docs.partition_point(|&d|get(d as usize).unwrap()<a);
        let hi=docs.partition_point(|&d|get(d as usize).unwrap()<=b);
        assert_eq!((hi-lo) as u32,col.count_in_range(a,b));
    }}
    let old=8+16*n.div_ceil(256)+8*n+if col.has_points(){4*n}else{0}+if values.len()<n{8*n.div_ceil(64)}else{0};
    let compact=12+16*n.div_ceil(256)+12*n.div_ceil(64)+values.len()*(8+if col.has_points(){4}else{0});
    json!({"field":field,"tag":tag,"documents":n,"present":values.len(),"dense_bytes":old,
        "compact_bytes":compact,"chosen_bytes":old.min(compact),"ratio":old as f64/compact as f64})
}

fn main(){
    let mut rows=Vec::new();let mut file_bytes=0u64;
    for path in std::env::args().skip(1) {
        let path=std::path::Path::new(&path);
        let id=path.file_stem().unwrap().to_str().unwrap().strip_prefix("seg_").unwrap().parse::<u32>().unwrap();
        let dir=MmapDirectory::open(path.parent().unwrap()).unwrap();
        let seg=Segment::open(&dir,id).unwrap();
        file_bytes+=path.metadata().unwrap().len();
        let fields=seg.fast_fields();
        for (field,tag) in fields.fields(){
            let row=match tag {
                1=>inspect(field,tag,fields.get_i64(field).unwrap()),
                2=>inspect(field,tag,fields.get_u64(field).unwrap()),
                3=>inspect(field,tag,fields.get_f64(field).unwrap()),
                _=>continue,
            };
            rows.push(row);
        }
    }
    assert!(!rows.is_empty());
    let dense:u64=rows.iter().map(|r|r["dense_bytes"].as_u64().unwrap()).sum();
    let chosen:u64=rows.iter().map(|r|r["chosen_bytes"].as_u64().unwrap()).sum();
    println!("{}",json!({"scope":"real sealed columns; prototype byte estimate and exact values/ranges, not server timing",
        "segment_bytes":file_bytes,"numeric_dense_bytes":dense,"numeric_chosen_bytes":chosen,"ratio":dense as f64/chosen as f64,"columns":rows}));
}
