//! Store bytes of a noaa-shaped index (the Rally `noaa` track's mapping:
//! dates, floats, keyword measurements that are mostly absent, two
//! `double_range`s, station keywords and a `geo_point`), by section and by
//! field, for the noaa store gap (9.28 vs ES 6.59 GB: 276 vs 196 B/doc).
//!
//! `cargo test --release -p elasticsearch-engine --lib noaa_store_microbench -- --ignored --nocapture`
//! (`NOAA_DOCS`, default 1,000,000; force-merged to one segment).

use crate::index::WriteReq;
use crate::node::Node;
use serde_json::{json, Map, Value};

const FLOATS: &[&str] = &["AWND", "EVAP", "MDPR", "MDTN", "MDTX", "MNPN", "MXPN", "PRCP", "TAVG", "THIC", "TMAX", "TMIN", "TOBS", "WESD", "WESF", "WSF2", "WSF5", "WSFG", "WSFI"];
const KEYWORDS: &[&str] = &["AWDR", "DAPR", "DASF", "DATN", "DATX", "DWPR", "MDSF", "PGTM", "PSUN", "SN31", "SN32", "SN33", "SN35", "SN36", "SN51", "SN52", "SN53", "SN55", "SN56", "SN57", "SNOW", "SNWD", "SX31", "SX32", "SX33", "SX35", "SX36", "SX51", "SX52", "SX53", "SX55", "SX56", "SX57", "TSUN", "WDF2", "WDF5", "WDFG", "WDMV", "WT01", "WT02", "WT03", "WT04", "WT05", "WT06", "WT07", "WT08", "WT09", "WT10", "WT11", "WT17", "WT18"];

fn mapping() -> Value {
    let mut props = Map::new();
    for f in FLOATS { props.insert((*f).into(), json!({"type": "float"})); }
    for f in KEYWORDS { props.insert((*f).into(), json!({"type": "keyword"})); }
    props.insert("MDTRANGE".into(), json!({"type": "double_range"}));
    props.insert("TRANGE".into(), json!({"type": "double_range"}));
    props.insert("date".into(), json!({"type": "date"}));
    let mut station = Map::new();
    for f in ["country", "country_code", "gsn_flag", "hcn_crn_flag", "id", "name", "state", "state_code", "wmo_id"] { station.insert(f.into(), json!({"type": "keyword"})); }
    station.insert("elevation".into(), json!({"type": "float"}));
    station.insert("location".into(), json!({"type": "geo_point"}));
    props.insert("station".into(), json!({"properties": station}));
    json!({"dynamic": "strict", "properties": props})
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn below(&mut self, n: u64) -> u64 { self.next() % n }
    fn chance(&mut self, pct: u64) -> bool { self.below(100) < pct }
}

/// One station-day: what noaa's documents hold (TMAX / TMIN / PRCP most
/// days, snow measurements in some, weather types rarely, the station
/// block always).
fn doc(i: u64, stations: u64, r: &mut Rng) -> Value {
    let s = i % stations;
    let day = i / stations;
    let mut d = Map::new();
    d.insert("date".into(), json!(format!("2016-{:02}-{:02}T00:00:00", 1 + (day / 28) % 12, 1 + day % 28)));
    let tenth = |r: &mut Rng, lo: i64, span: u64| (lo * 10 + r.below(span * 10) as i64) as f64 / 10.0;
    // TMAX never below TMIN (a day's range), so TRANGE / MDTRANGE are valid
    // ranges, as in the corpus.
    let tmin = tenth(r, -20, 30);
    let tmax = (tmin * 10.0 + r.below(200) as f64).round() / 10.0;
    if r.chance(75) {
        d.insert("TMIN".into(), json!(tmin));
        d.insert("TMAX".into(), json!(tmax));
        d.insert("TRANGE".into(), json!({"gte": tmin, "lte": tmax}));
    }
    if r.chance(90) { d.insert("PRCP".into(), json!(if r.chance(60) { 0.0 } else { tenth(r, 0, 40) })); }
    if r.chance(30) { d.insert("TAVG".into(), json!(tenth(r, -5, 30))); }
    if r.chance(20) { d.insert("TOBS".into(), json!(tenth(r, -10, 35))); }
    if r.chance(15) { d.insert("AWND".into(), json!(tenth(r, 0, 12))); }
    if r.chance(10) { for f in ["WSF2", "WSF5"] { d.insert(f.into(), json!(tenth(r, 0, 25))); } for f in ["WDF2", "WDF5"] { d.insert(f.into(), json!((10 * r.below(36)).to_string())); } }
    if r.chance(40) { d.insert("SNOW".into(), json!(if r.chance(85) { "0".to_string() } else { r.below(300).to_string() })); }
    if r.chance(35) { d.insert("SNWD".into(), json!(if r.chance(80) { "0".to_string() } else { r.below(900).to_string() })); }
    for f in ["WT01", "WT02", "WT03", "WT08", "WT11"] { if r.chance(4) { d.insert(f.into(), json!("1")); } }
    if r.chance(1) { d.insert("DAPR".into(), json!(r.below(9).to_string())); d.insert("MDPR".into(), json!(tenth(r, 0, 60))); }
    if r.chance(1) { d.insert("MDTRANGE".into(), json!({"gte": tmin, "lte": tmax})); d.insert("MDTN".into(), json!(tmin)); d.insert("MDTX".into(), json!(tmax)); }
    let mut st = Map::new();
    st.insert("id".into(), json!(format!("US{}000{:05}", ["W", "1", "C"][(s % 3) as usize], s)));
    st.insert("name".into(), json!(format!("STATION {} {}", s, ["SPRINGS", "AIRPORT", "CITY"][(s % 3) as usize])));
    st.insert("country".into(), json!("United States"));
    st.insert("country_code".into(), json!("US"));
    st.insert("state".into(), json!(format!("STATE {}", s % 50)));
    st.insert("state_code".into(), json!(format!("S{}", s % 50)));
    st.insert("elevation".into(), json!((s * 37 % 3000) as f64 / 10.0));
    st.insert("location".into(), json!({"lat": 25.0 + (s % 97) as f64 * 0.25, "lon": -125.0 + (s % 233) as f64 * 0.25}));
    if s % 20 == 0 { st.insert("gsn_flag".into(), json!("GSN")); }
    if s % 10 == 0 { st.insert("hcn_crn_flag".into(), json!("HCN")); }
    if s % 10 == 3 { st.insert("wmo_id".into(), json!((70000 + s).to_string())); }
    d.insert("station".into(), Value::Object(st));
    Value::Object(d)
}

#[test]
#[ignore]
fn noaa_store_microbench() {
    let n: u64 = std::env::var("NOAA_DOCS").ok().and_then(|v| v.parse().ok()).unwrap_or(1_000_000);
    let tmp = tempfile::tempdir().unwrap();
    let node = Node::open(tmp.path()).unwrap();
    node.create_index("noaa", &json!({"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"}, "mappings": mapping()})).unwrap();
    let svc = node.get_index("noaa").unwrap();
    let mut r = Rng(0x9e37_79b9_7f4a_7c15);
    let mut batch = Vec::with_capacity(10_000);
    let mut source_bytes = 0u64;
    for i in 0..n {
        let source = doc(i, 12_000, &mut r).to_string();
        source_bytes += source.len() as u64;
        batch.push(WriteReq::Index { id: None, source, routing: None, cond: Default::default() });
        if batch.len() == 10_000 || i + 1 == n {
            for x in svc.write(std::mem::take(&mut batch), false) { x.unwrap(); }
        }
    }
    svc.force_merge(1, false).unwrap();
    svc.refresh().unwrap();
    let names: std::collections::HashMap<u16, String> = svc.lookup().fields.iter().filter_map(|(name, f)| Some((f.core_id?, name.clone()))).collect();
    let sh = svc.local_shards().remove(0);
    let core = sh.shard.core.read();
    let (mut terms, mut postings, mut fast, mut store, mut extra) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let mut per: std::collections::BTreeMap<String, [u64; 8]> = Default::default();
    for seg in core.segments() {
        let du = seg.disk_usage().unwrap();
        terms += du.terms; postings += du.postings; fast += du.fast; store += du.store; extra += du.extra;
        for (id, u) in &du.fields {
            let name = names.get(id).cloned().unwrap_or_else(|| format!("#{id}"));
            let e = per.entry(name).or_default();
            for (k, v) in [u.terms, u.postings, u.doc_values, u.doc_values_terms, u.points, u.norms, u.stored, u.geo + u.shape + u.other_extra].into_iter().enumerate() { e[k] += v; }
        }
    }
    let total = terms + postings + fast + store + extra;
    let b = |x: u64| x as f64 / n as f64;
    eprintln!("{n} docs ({} segment(s)), _source {:.1} B/doc: total {:.1} B/doc (ES full corpus 196, ours 276)", core.segments().len(), b(source_bytes), b(total));
    eprintln!("  terms {:.1}  postings {:.1}  fast (doc values, points, norms) {:.1}  stored {:.1}  extra (geo, ranges) {:.1}  B/doc", b(terms), b(postings), b(fast), b(store), b(extra));
    let mut rows: Vec<(String, [u64; 8])> = per.into_iter().collect();
    rows.sort_by_key(|(_, v)| std::cmp::Reverse(v.iter().sum::<u64>()));
    eprintln!("  {:<22} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}  (B/doc)", "field", "terms", "post", "dv", "dvterm", "points", "norms", "stored", "extra");
    for (name, v) in rows.iter().take(30) {
        eprintln!("  {:<22} {:>7.2} {:>7.2} {:>7.2} {:>7.2} {:>7.2} {:>7.2} {:>7.2} {:>7.2}", name, b(v[0]), b(v[1]), b(v[2]), b(v[3]), b(v[4]), b(v[5]), b(v[6]), b(v[7]));
    }
}
