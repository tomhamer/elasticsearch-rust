//! Isolate scalar count-only auto-date bucket collection on actual timestamps.
//! This preserves the production rounding choice; it does not time HTTP,
//! doc-value retrieval, sub-aggregations, missing values or multi-valued fields.
use elasticsearch_engine::{aggs::auto_date, date::{CalUnit, Tz}};
use std::{collections::HashSet, hint::black_box, time::Instant};

fn collect(values: &[i64], direct: bool, tz: Tz) -> Vec<u64> {
    let lo = *values.iter().min().unwrap();
    let hi = *values.iter().max().unwrap();
    let choice = auto_date::choose(lo, hi, 10, CalUnit::Second, 0, tz);
    let n = (choice.bucket_of(hi) + 1) as usize;
    let mut counts = vec![0; n];
    if direct {
        for &value in values { counts[choice.bucket_of(value) as usize] += 1; }
    } else {
        let vals: Vec<_> = values.iter().enumerate().map(|(doc, &v)| (v, 0usize, doc as u32, 1f32)).collect();
        let mut groups = vec![vec![Vec::new()]; n];
        let mut seen = HashSet::new();
        for (v, segment, doc, score) in vals {
            let bucket = choice.bucket_of(v) as usize;
            if seen.insert((bucket, segment, doc)) { groups[bucket][segment].push((doc, score)); }
        }
        for (i, group) in groups.iter_mut().enumerate() {
            group[0].sort_by_key(|x| x.0);
            counts[i] = group[0].len() as u64;
        }
    }
    counts
}

fn main() {
    let file = std::env::args().nth(1).expect("timestamp array JSON");
    let values: Vec<i64> = serde_json::from_slice(&std::fs::read(file).unwrap()).unwrap();
    assert!(!values.is_empty());
    let mut results = Vec::new();
    for zone in ["UTC", "America/New_York"] {
        let tz = Tz::parse(zone).unwrap();
        assert_eq!(collect(&values, false, tz), collect(&values, true, tz));
        for pass in 0..10 {
            for direct in if pass % 2 == 0 { [false, true] } else { [true, false] } {
                let start = Instant::now();
                black_box(collect(black_box(&values), direct, tz));
                results.push(serde_json::json!({"zone":zone,"pass":pass,"direct_counts":direct,"ms":start.elapsed().as_secs_f64()*1000.0}));
            }
        }
    }
    println!("{}", serde_json::json!({"scope":"scalar count-only bucket collection component, not server throughput", "documents":values.len(),"results":results}));
}
