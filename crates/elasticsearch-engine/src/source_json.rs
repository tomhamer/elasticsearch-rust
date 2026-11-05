//! Lossless JSON structure for stored source. Scalar tokens stay raw so Java's
//! lone UTF-16 surrogate escapes never pass through a Rust Unicode string.
use std::collections::BTreeMap;
use serde::Serialize;
use serde_json::{value::RawValue, Value};

#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum Source {
    Object(BTreeMap<String, Source>),
    Array(Vec<Source>),
    Scalar(Box<RawValue>),
}
/// A single-pass reader of JSON text serde has already validated, building
/// the same tree as [`Source::from_raw`] (objects keep the last of duplicate
/// keys, scalars keep their exact text). `None`: defer to `from_raw`.
struct Scan<'a> { b: &'a [u8], at: usize }

/// A scalar's text, already validated JSON, as a raw value (no re-parse:
/// `RawValue::from_string` validates again).
fn raw_unchecked(text: &str) -> Box<RawValue> {
    // SAFETY: `RawValue` is `#[repr(transparent)]` over `str` (serde_json's
    // own `from_owned` is this transmute); the text is valid JSON, checked
    // by the scan that read it.
    unsafe { std::mem::transmute::<Box<str>, Box<RawValue>>(Box::from(text)) }
}

/// `t` is a JSON number (`-?(0|[1-9]d*)(.d+)?([eE][+-]?d+)?`).
fn json_number(t: &str) -> bool {
    let b = t.as_bytes();
    let mut i = usize::from(b.first() == Some(&b'-'));
    let digits = |i: &mut usize| { let s = *i; while *i < b.len() && b[*i].is_ascii_digit() { *i += 1; } *i - s };
    match b.get(i) {
        Some(b'0') => i += 1,
        Some(b'1'..=b'9') => { digits(&mut i); }
        _ => return false,
    }
    if b.get(i) == Some(&b'.') {
        i += 1;
        if digits(&mut i) == 0 { return false; }
    }
    if matches!(b.get(i), Some(b'e' | b'E')) {
        i += 1;
        if matches!(b.get(i), Some(b'+' | b'-')) { i += 1; }
        if digits(&mut i) == 0 { return false; }
    }
    i == b.len()
}

impl<'a> Scan<'a> {
    fn ws(&mut self) { while self.at < self.b.len() && matches!(self.b[self.at], b' ' | b'\t' | b'\n' | b'\r') { self.at += 1; } }
    /// The text of the string starting at `self.at` (quotes included) and
    /// whether it has escapes.
    fn string(&mut self) -> Option<(&'a str, bool)> {
        let start = self.at; let mut i = start + 1; let mut esc = false;
        loop {
            let run = memchr::memchr2(b'"', b'\\', self.b.get(i..)?)?;
            // JSON strings hold no raw control characters.
            if self.b[i..i + run].iter().any(|&c| c < 0x20) { return None; }
            i += run;
            if self.b[i] == b'"' { break; }
            esc = true;
            match *self.b.get(i + 1)? {
                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => i += 2,
                b'u' => {
                    let hex = std::str::from_utf8(self.b.get(i + 2..i + 6)?).ok()?;
                    let unit = u16::from_str_radix(hex, 16).ok()?;
                    // Surrogate escapes: the per-level parse decides.
                    if (0xD800..0xE000).contains(&unit) || !hex.bytes().all(|c| c.is_ascii_hexdigit()) { return None; }
                    i += 6;
                }
                _ => return None,
            }
        }
        self.at = i + 1;
        // SAFETY: `b` is a `str`'s bytes and both ends are ASCII quotes.
        Some((unsafe { std::str::from_utf8_unchecked(&self.b[start..i + 1]) }, esc))
    }
    fn value(&mut self, depth: usize) -> Option<Source> {
        if depth > 128 { return None; }
        self.ws();
        match *self.b.get(self.at)? {
            b'{' => {
                self.at += 1;
                // Members collected, then built into the map at once (a
                // stable sort: of equal keys the last wins, as inserting
                // them in order does).
                let mut m: Vec<(String, Source)> = Vec::new();
                self.ws();
                if self.b.get(self.at) == Some(&b'}') { self.at += 1; return Some(Source::Object(BTreeMap::new())); }
                loop {
                    self.ws();
                    if self.b.get(self.at) != Some(&b'"') { return None; }
                    let (k, esc) = self.string()?;
                    let key = if esc { serde_json::from_str::<String>(k).ok()? } else { k[1..k.len() - 1].to_string() };
                    self.ws();
                    if self.b.get(self.at) != Some(&b':') { return None; }
                    self.at += 1;
                    let v = self.value(depth + 1)?;
                    m.push((key, v));
                    self.ws();
                    match *self.b.get(self.at)? { b',' => self.at += 1, b'}' => { self.at += 1; return Some(Source::Object(BTreeMap::from_iter(m))); } _ => return None }
                }
            }
            b'[' => {
                self.at += 1;
                let mut a = Vec::new();
                self.ws();
                if self.b.get(self.at) == Some(&b']') { self.at += 1; return Some(Source::Array(a)); }
                loop {
                    a.push(self.value(depth + 1)?);
                    self.ws();
                    match *self.b.get(self.at)? { b',' => self.at += 1, b']' => { self.at += 1; return Some(Source::Array(a)); } _ => return None }
                }
            }
            b'"' => {
                let (t, _) = self.string()?;
                Some(Source::Scalar(raw_unchecked(t)))
            }
            _ => {
                let start = self.at;
                while self.at < self.b.len() && !matches!(self.b[self.at], b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r') { self.at += 1; }
                // SAFETY: a `str`'s bytes, cut at ASCII delimiters.
                let t = unsafe { std::str::from_utf8_unchecked(&self.b[start..self.at]) };
                if !matches!(t, "true" | "false" | "null") && !json_number(t) { return None; }
                // `from_raw` parses non-strings as a `Value`, which rejects
                // out-of-range numbers: check the ones that could be.
                if (t.len() > 300 || t.contains(['e', 'E'])) && serde_json::from_str::<Value>(t).is_err() { return None; }
                Some(Source::Scalar(raw_unchecked(t)))
            }
        }
    }
}

/// `raw` cut just past its first JSON value, when more text follows it
/// (`None`: no complete first value, or only whitespace after it).
///
/// An ingest document's source is read by ES with `XContentParser.map()`
/// (`IngestService.newIngestDocument` -> `IndexRequest.sourceAsMap` ->
/// `IndexSource.sourceAsMap` -> `XContentHelper.parseToType`), which stops
/// at the root object's closing brace and closes the parser without
/// reading on: whatever follows the first object on the bulk line is never
/// tokenized, and the pipeline's output source is re-serialized from the
/// map, so it is dropped. (Without a pipeline, `DocumentParser.validateEnd`
/// does reject it.)
pub fn leading_value(raw: &str) -> Option<&str> {
    let mut values = serde_json::Deserializer::from_str(raw).into_iter::<&RawValue>();
    values.next()?.ok()?;
    let end = values.byte_offset();
    raw[end..].bytes().any(|b| !matches!(b, b' ' | b'\t' | b'\n' | b'\r')).then(|| &raw[..end])
}

impl Source {
    /// [`Source::parse`] of an ingest document's source: text after the
    /// first complete value is ignored, as ES's `XContentParser.map()` does
    /// (see [`leading_value`]). Input without such trailing text parses
    /// (and fails) exactly as [`Source::parse`].
    pub fn parse_leading(raw: &str) -> serde_json::Result<Self> {
        Self::parse(raw).or_else(|e| match leading_value(raw) {
            Some(first) => Self::parse(first),
            None => Err(e),
        })
    }

    pub fn parse(raw: &str) -> serde_json::Result<Self> {
        // One pass that also validates (strictly: anything it is unsure
        // serde accepts, such as a surrogate escape, it leaves to serde);
        // anything else takes serde's validation and the per-level route,
        // which report it exactly (an error, a key that fails to decode, a
        // number to range-check, the nesting limit).
        let mut scan = Scan { b: raw.as_bytes(), at: 0 };
        if let Some(src) = scan.value(0) {
            scan.ws();
            if scan.at == scan.b.len() {
                return Ok(src);
            }
        }
        let raw: Box<RawValue> = serde_json::from_str(raw)?;
        Self::from_raw(raw,0)
    }
    fn from_raw(raw: Box<RawValue>,depth:usize) -> serde_json::Result<Self> {
        if depth>128 {return Err(<serde_json::Error as serde::de::Error>::custom("source nesting limit exceeded"));}
        Ok(match raw.get().as_bytes().first() {
            Some(b'{') => {
                let children: BTreeMap<String, Box<RawValue>> = serde_json::from_str(raw.get())?;
                Self::Object(children.into_iter().map(|(k,v)| Ok((k, Self::from_raw(v,depth+1)?))).collect::<serde_json::Result<_>>()?)
            }
            Some(b'[') => {
                let children: Vec<Box<RawValue>> = serde_json::from_str(raw.get())?;
                Self::Array(children.into_iter().map(|v|Self::from_raw(v,depth+1)).collect::<serde_json::Result<_>>()?)
            }
            _ => {
                if !raw.get().starts_with('"') { let _:Value=serde_json::from_str(raw.get())?; }
                Self::Scalar(raw)
            },
        })
    }
    pub fn text(&self) -> String { serde_json::to_string(self).expect("validated raw JSON") }
    pub fn from_value(v: &Value) -> Self { Self::parse(&v.to_string()).expect("valid Value") }
    pub fn equivalent(&self, other:&Self)->bool {
        match (self,other) {
            (Self::Object(a),Self::Object(b))=>a.len()==b.len() && a.iter().all(|(k,v)|b.get(k).is_some_and(|o|v.equivalent(o))),
            (Self::Array(a),Self::Array(b))=>a.len()==b.len() && a.iter().zip(b).all(|(a,b)|a.equivalent(b)),
            (Self::Scalar(a),Self::Scalar(b))=>match (serde_json::from_str::<Value>(a.get()),serde_json::from_str::<Value>(b.get())) {
                (Ok(a),Ok(b))=>a==b,
                _ if a.get().starts_with('"') && b.get().starts_with('"')=>decode_string(a.get())==decode_string(b.get()),
                _=>false,
            },
            _=>false,
        }
    }
    pub fn insert_path(&mut self, path: &[&str], value: Self) {
        if path.is_empty() { *self=value; return; }
        match self {
            Self::Object(m) => {
                if path.len()==1 { m.insert(path[0].into(),value); }
                else if let Some(child)=m.get_mut(path[0]) { child.insert_path(&path[1..],value); }
            }
            Self::Array(a) => if let Ok(i)=path[0].parse::<usize>() { if let Some(child)=a.get_mut(i) { child.insert_path(&path[1..],value); } },
            _ => {}
        }
    }
    pub fn merge(&mut self, patch: &Self) {
        match (self, patch) {
            (Self::Object(dst), Self::Object(src)) => {
                for (k,v) in src {
                    match dst.get_mut(k) {
                        Some(old @ Self::Object(_)) if matches!(v, Self::Object(_)) => old.merge(v),
                        _ => { dst.insert(k.clone(), v.clone()); }
                    }
                }
            }
            (dst, src) => *dst = src.clone(),
        }
    }
    pub fn filter(&self, includes: &[String], excludes: &[String]) -> Self {
        fn selected(path: &str, patterns: &[String]) -> bool {
            patterns.iter().any(|p| crate::mapping::wildcard_match(p,path) || path.starts_with(&format!("{p}.")))
        }
        fn walk(v: &Source, path: &str, inc: &[String], exc: &[String]) -> Option<Source> {
            if !path.is_empty() && selected(path, exc) { return None; }
            let included = inc.is_empty() || selected(path, inc);
            match v {
                Source::Object(m) => {
                    let out: BTreeMap<_,_> = m.iter().filter_map(|(k,c)| {
                        let child = if path.is_empty() { k.clone() } else { format!("{path}.{k}") };
                        walk(c,&child,inc,exc).map(|v|(k.clone(),v))
                    }).collect();
                    if out.is_empty() && !included && !path.is_empty() { None } else { Some(Source::Object(out)) }
                }
                Source::Array(a) if a.iter().any(|v|matches!(v,Source::Object(_))) => {
                    let a: Vec<_> = a.iter().filter_map(|v| {
                        if matches!(v,Source::Object(_)) { walk(v,path,inc,exc) } else { included.then(||v.clone()) }
                    }).filter(|v| !matches!(v,Source::Object(m) if m.is_empty()) || inc.is_empty()).collect();
                    (!a.is_empty()).then_some(Source::Array(a))
                }
                _ => included.then(||v.clone()),
            }
        }
        walk(self,"",includes,excludes).unwrap_or_else(||Self::Object(BTreeMap::new()))
    }
}

/// Keep the ordinary source/filter semantics and allocation path unchanged.
/// Only source rejected by Value needs the lossless structural representation.
pub fn filter_raw(raw: &str, includes: &[String], excludes: &[String]) -> serde_json::Result<String> {
    Ok(Source::parse(raw)?.filter(includes,excludes).text())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lossless_nested_filter_and_merge() {
        let mut src=Source::parse(r#"{"nested":{"body":"\ud83d x \ude02 pair \ud83d\ude02 literal \\ud83d","n":1},"rows":[{"body":"\ud800","secret":1},{"body":"\udfff"}],"secret":2}"#).unwrap();
        let filtered=src.filter(&["nested.body".into(),"rows.body".into()],&[]).text();
        assert!(filtered.contains(r#""body":"\ud83d x \ude02 pair \ud83d\ude02 literal \\ud83d""#));
        assert!(filtered.contains(r#""rows":[{"body":"\ud800"},{"body":"\udfff"}]"#));
        assert!(!filtered.contains("secret"));
        src.merge(&Source::from_value(&serde_json::json!({"nested":{"n":2}})));
        assert!(src.text().contains(r#""body":"\ud83d x \ude02 pair \ud83d\ude02 literal \\ud83d""#));
        assert!(src.text().contains(r#""n":2"#));
    }
    #[test]
    fn lazy_response_sources_pass_through_verbatim() {
        let src=r#"{"title":"T\u00e9 \"q\"","z":1,"a":[{"n":1.5e3}],"lone":"\ud800"}"#;
        let resp=format!(r#"{{"took":3,"hits":{{"total":{{"value":2,"relation":"eq"}},"hits":[{{"_index":"i","_id":"1","_score":1.25,"_source":{src}}},{{"_index":"i","_id":"2","_score":0.5,"_source":{{"b":[1,2]}},"fields":{{"f":["x"]}}}}]}}}}"#);
        let mut rs=ResponseSources::default();
        let v=rs.parse_lazy(resp.as_bytes()).unwrap();
        assert_eq!(v["hits"]["hits"][0]["_source"],serde_json::json!(0));
        assert_eq!(v["hits"]["hits"][1]["_source"],serde_json::json!(1));
        assert_eq!(v["hits"]["hits"][0]["_score"],serde_json::json!(1.25));
        let out=String::from_utf8(rs.render(&v)).unwrap();
        // Byte-identical: sources verbatim, other keys in their order.
        assert_eq!(out,resp);
        // A merged response that drops or reorders hits keeps each source.
        let merged=serde_json::json!({"hits":{"hits":[v["hits"]["hits"][1].clone(),v["hits"]["hits"][0].clone()]}});
        let out=String::from_utf8(rs.render(&merged)).unwrap();
        assert!(out.starts_with(r#"{"hits":{"hits":[{"_index":"i","_id":"2","_score":0.5,"_source":{"b":[1,2]}"#),"{out}");
        assert!(out.contains(&format!(r#""_source":{src}}}]"#)),"{out}");
    }
    /// Parts parsed with tables of their own join one table: a merge of
    /// their values renders as one parsed serially does.
    #[test]
    fn part_tables_absorb_without_renumbering() {
        let a=r#"{"hits":{"hits":[{"_index":"i","_id":"1","_source":{"a":"\u00e9"}},{"_index":"i","_id":"2","_source":{"b":[1,2]}}]}}"#;
        let b=r#"{"hits":{"hits":[{"_index":"i","_id":"3","_source":{"c":3}}]},"aggregations":{"t":{"hits":{"hits":[{"_source":{"d":4}}]}}}}"#;
        let lone=r#"{"hits":{"hits":[{"_index":"i","_id":"4","_source":{"e":"\ud800"},"fields":{"f":["\ud800"]}}]}}"#;
        let (mut serial,mut joined)=(ResponseSources::default(),ResponseSources::default());
        let sv:Vec<Value>=[a,b,lone].iter().map(|r|serial.parse_lazy(r.as_bytes()).unwrap()).collect();
        let pv:Vec<Value>=[a,b,lone].iter().enumerate().map(|(i,r)|{
            let mut t=ResponseSources::for_part(i as u32);
            let v=t.parse_lazy(r.as_bytes()).unwrap();
            joined.absorb(t);
            v
        }).collect();
        let merge=|v:&[Value]|serde_json::json!({"hits":{"hits":[v[2]["hits"]["hits"][0].clone(),v[1]["hits"]["hits"][0].clone(),v[0]["hits"]["hits"][1].clone(),v[0]["hits"]["hits"][0].clone()]},"aggregations":v[1]["aggregations"].clone()});
        assert_eq!(String::from_utf8(joined.render(&merge(&pv))).unwrap(),String::from_utf8(serial.render(&merge(&sv))).unwrap());
        assert_eq!(joined.source_text(&pv[1]["hits"]["hits"][0]["_source"]),serial.source_text(&sv[1]["hits"]["hits"][0]["_source"]));
        // The lazy render (no part needed the tree) agrees too.
        let (mut s2,mut j2)=(ResponseSources::default(),ResponseSources::default());
        let sv:Vec<Value>=[a,b].iter().map(|r|s2.parse_lazy(r.as_bytes()).unwrap()).collect();
        let pv:Vec<Value>=[a,b].iter().enumerate().map(|(i,r)|{let mut t=ResponseSources::for_part(i as u32);let v=t.parse_lazy(r.as_bytes()).unwrap();j2.absorb(t);v}).collect();
        let m=|v:&[Value]|serde_json::json!({"hits":{"hits":[v[1]["hits"]["hits"][0].clone(),v[0]["hits"]["hits"][0].clone()]}});
        assert_eq!(j2.render_lazy(&m(&pv)),s2.render_lazy(&m(&sv)));
        assert!(j2.render_lazy(&m(&pv)).is_some());
    }
    /// A whole hit kept as text renders verbatim in its hit list's place,
    /// in the lazy and in the tree render.
    #[test]
    fn hit_texts_render_verbatim() {
        let mut rs=ResponseSources::default();
        let v=rs.parse_lazy(br#"{"hits":{"hits":[{"_index":"i","_id":"1","_source":{"a":1}}]}}"#).unwrap();
        let text=r#"{"_index":"i","_id":"2","_score":1.0,"_source":{"b":"\u00e9 \ud83d\ude00","c":1E2}}"#;
        let t=rs.push_hit_text(text.to_string());
        let merged=serde_json::json!({"hits":{"hits":[t,v["hits"]["hits"][0].clone()]},"n":t});
        let out=String::from_utf8(rs.render(&merged)).unwrap();
        assert_eq!(out,format!(r#"{{"hits":{{"hits":[{text},{{"_index":"i","_id":"1","_source":{{"a":1}}}}]}},"n":{}}}"#,t));
        // A part that needed the tree.
        rs.parse(br#"{"hits":{"hits":[{"_index":"i","_id":"3","_source":{"x":"\ud800"}}]}}"#).unwrap();
        let out=String::from_utf8(rs.render(&merged)).unwrap();
        assert!(out.contains(r#""_id":"2""#)&&out.contains(r#""c":1E2"#),"{out}");
    }
    /// Only document sources are tokens: an aggregation or a suggester named
    /// `_source`, or a `_source` key inside some other value, stays JSON.
    #[test]
    fn lazy_response_sources_only_at_hits_docs_and_options() {
        let resp=r#"{"hits":{"hits":[{"_id":"1","_source":{"a":1},"inner_hits":{"x":{"hits":{"hits":[{"_id":"2","_source":{"b":2}}]}}}}]},"aggregations":{"_source":{"value":3},"t":{"hits":{"hits":[{"_source":{"c":3}}]}},"n":{"buckets":[{"key":"k","_source":7}]}},"suggest":{"_source":[{"text":"x","options":[{"text":"y","_source":{"d":4}}]}]},"docs":[{"_id":"3","_source":{"e":5}}]}"#;
        let mut rs=ResponseSources::default();
        let v=rs.parse_lazy(resp.as_bytes()).unwrap();
        assert_eq!(rs.sources.len(),5,"{v}");
        assert_eq!(v["aggregations"]["_source"],serde_json::json!({"value":3}));
        assert_eq!(v["aggregations"]["n"]["buckets"][0]["_source"],serde_json::json!(7));
        assert!(v["suggest"]["_source"].is_array());
        assert!(v["suggest"]["_source"][0]["options"][0]["_source"].is_u64());
        assert!(v["hits"]["hits"][0]["inner_hits"]["x"]["hits"]["hits"][0]["_source"].is_u64());
        assert_eq!(String::from_utf8(rs.render(&v)).unwrap(),resp);
        // A merged value whose non-hit `_source` is a valid ordinal keeps it.
        let m=serde_json::json!({"aggregations":{"_source":0},"n":{"_source":1}});
        assert_eq!(String::from_utf8(rs.render(&m)).unwrap(),m.to_string());
        // The lossless fallback path (a lone surrogate) takes the same positions.
        let lone=resp.replace(r#""e":5"#,r#""e":"\ud800""#);
        let mut rs=ResponseSources::default();
        let v=rs.parse(lone.as_bytes()).unwrap();
        assert_eq!(rs.sources.len(),5);
        assert_eq!(v["aggregations"]["_source"],serde_json::json!({"value":3}));
        assert!(String::from_utf8(rs.render(&v)).unwrap().contains(r#""e":"\ud800""#));
        // Ordinals resolve to their sources directly.
        assert!(rs.source_text(&v["docs"][0]["_source"]).unwrap().contains(r#""e":"\ud800""#));
        assert_eq!(rs.source_text(&serde_json::json!("x")), None);
        // A bulk update item's `get` is a document; an aggregation named
        // `get` is not.
        let bulk=r#"{"items":[{"update":{"_id":"1","get":{"_source":{"f":1}}}}],"aggregations":{"get":{"_source":{"value":1}}}}"#;
        let mut rs=ResponseSources::default();
        let v=rs.parse_lazy(bulk.as_bytes()).unwrap();
        assert_eq!(rs.sources.len(),1);
        assert!(v["items"][0]["update"]["get"]["_source"].is_u64());
        assert_eq!(v["aggregations"]["get"]["_source"],serde_json::json!({"value":1}));
        assert_eq!(String::from_utf8(rs.render(&v)).unwrap(),bulk);
    }

    /// Data shaped like a hit list (a scripted metric's state, a hit's
    /// `fields`, `highlight` or `sort`, a bucket key, a partial state) keeps
    /// its `_source`s: only response structure is searched for documents.
    #[test]
    fn lazy_response_sources_skip_opaque_values() {
        // Partial state on a `doc_count`-carrying result (significant_terms)
        // is data.
        let mut rs=ResponseSources::default();
        let v=rs.parse_lazy(br#"{"aggregations":{"s":{"doc_count":4,"__p":{"hits":[{"_source":"sp"}]}}}}"#).unwrap();
        assert!(rs.sources.is_empty()&&v["aggregations"]["s"]["__p"]["hits"][0]["_source"]=="sp","{v}");
        let resp=r#"{"hits":{"hits":[{"_index":"i","_id":"1","_source":{"a":1},"fields":{"hits":[{"_source":"f"}]},"highlight":{"docs":[{"_source":"h"}]},"sort":[{"options":[{"_source":"s"}]}]}]},"aggregations":{"m":{"value":{"docs":[{"_source":"x"}]}},"p":{"__p":{"states":[{"hits":[{"_source":"y"}]}]}},"c":{"after_key":{"hits":[{"_source":"z"}]},"buckets":[{"key":{"docs":[{"_source":"k"}]},"doc_count":2,"value":{"hits":{"hits":[{"_source":{"t":1}}]}},"v":{"value":{"options":[{"_source":"w"}]}},"meta":{"hits":{"hits":[{"_source":{"u":1}}]}}}]},"g":{"meta":{"hits":[{"_source":0}]},"doc_count":3,"value":{"hits":{"hits":[{"_source":{"q":2}}]}}}}}"#;
        let check=|rs:&ResponseSources,v:&Value|{
            // The hit's `_source` and the bucket's sub-aggregations named
            // `value` and `meta` (`top_hits`, under a `doc_count`) only.
            assert_eq!(rs.sources.len(),4,"{v}");
            assert!(v["aggregations"]["c"]["buckets"][0]["meta"]["hits"]["hits"][0]["_source"].is_u64());
            // A single-bucket aggregation's own `meta` (before its
            // `doc_count`, after it in the sorted lossless tree) stays data; its sub-aggregation named `value` does not.
            assert_eq!(v["aggregations"]["g"]["meta"]["hits"][0]["_source"],0);
            assert!(v["aggregations"]["g"]["value"]["hits"]["hits"][0]["_source"].is_u64());
            assert_eq!(v["hits"]["hits"][0]["fields"]["hits"][0]["_source"],"f");
            assert_eq!(v["hits"]["hits"][0]["highlight"]["docs"][0]["_source"],"h");
            assert_eq!(v["hits"]["hits"][0]["sort"][0]["options"][0]["_source"],"s");
            assert_eq!(v["aggregations"]["m"]["value"]["docs"][0]["_source"],"x");
            assert_eq!(v["aggregations"]["p"]["__p"]["states"][0]["hits"][0]["_source"],"y");
            assert_eq!(v["aggregations"]["c"]["after_key"]["hits"][0]["_source"],"z");
            assert_eq!(v["aggregations"]["c"]["buckets"][0]["key"]["docs"][0]["_source"],"k");
            assert_eq!(v["aggregations"]["c"]["buckets"][0]["v"]["value"]["options"][0]["_source"],"w");
            assert!(v["aggregations"]["c"]["buckets"][0]["value"]["hits"]["hits"][0]["_source"].is_u64());
        };
        let mut rs=ResponseSources::default();
        let v=rs.parse_lazy(resp.as_bytes()).unwrap();
        check(&rs,&v);
        assert_eq!(String::from_utf8(rs.render(&v)).unwrap(),resp);
        // The lossless fallback (a lone surrogate) takes the same positions.
        let lone=resp.replace(r#""a":1"#,r#""a":"\ud800""#);
        let mut rs=ResponseSources::default();
        let v=rs.parse(lone.as_bytes()).unwrap();
        check(&rs,&v);
        let out=String::from_utf8(rs.render(&v)).unwrap();
        assert!(out.contains(r#""a":"\ud800""#)&&out.contains(r#"{"docs":[{"_source":"x"}]}"#),"{out}");
    }

    #[test]
    fn filter_matches_value_semantics() {
        let v=serde_json::json!({"a":[{"b":1,"c":2},{},3],"empty":{},"nested":{"a":[null,"x"]}});
        for inc in [vec![],vec!["a.b".into()],vec!["a".into()],vec!["*".into()],vec!["nested.*".into()]] {
            for exc in [vec![],vec!["a.c".into()],vec!["a".into()]] {
                let raw=Source::from_value(&v).filter(&inc,&exc).text();
                assert_eq!(serde_json::from_str::<Value>(&raw).unwrap(),crate::search::filter_source(&v,&inc,&exc));
            }
        }
    }
}

/// A per-response-merge side table. Sources are opaque to the hit sorter and
/// reducer; only generated ordinal tokens travel through Value. No string
/// sentinel can collide with document content, and the table is never global.
///
/// A table made for one part of a merge ([`Self::for_part`]) numbers its
/// ordinals from its own base, so parts parsed apart (each shard's answer
/// on the task that received it) join one table ([`Self::absorb`]) without
/// renumbering a token.
#[derive(Default)]
pub struct ResponseSources { sources:Vec<Source>, tree:bool, base:u64, parts:Vec<(u64,Vec<Source>)>, hits:Vec<RawText> }

/// Text known to be JSON, written as it is.
struct RawText(Box<str>);
impl Serialize for RawText {
    fn serialize<S:serde::Serializer>(&self,s:S)->Result<S::Ok,S::Error>{
        // How `RawValue` serializes itself: serde_json writes the field verbatim.
        use serde::ser::SerializeStruct;
        const TOKEN:&str="$serde_json::private::RawValue";
        let mut out=s.serialize_struct(TOKEN,1)?;
        out.serialize_field(TOKEN,&*self.0)?;
        out.end()
    }
}

/// Bit of a whole-hit token ([`ResponseSources::push_hit_text`]).
const HIT_TOKEN:u64=1<<31;
impl ResponseSources {
    /// A table for part `part` of a merge: its ordinals cannot collide with
    /// those of another part's table, or of a default table.
    pub fn for_part(part:u32)->Self{Self{base:(u64::from(part)+1)<<32,..Self::default()}}
    /// Take over `other`'s sources (a table of another part): the values
    /// parsed with it resolve through `self` from now on.
    pub fn absorb(&mut self,other:Self){
        self.tree|=other.tree;
        if !other.sources.is_empty(){self.parts.push((other.base,other.sources));}
        self.parts.extend(other.parts);
    }
    /// A whole hit kept as its text: the token to put in the hit's place
    /// (an element of a hit list), rendered as `text` verbatim. `text` must
    /// be a JSON object (it is not checked again: the caller built it from
    /// JSON it read).
    pub fn push_hit_text(&mut self,text:String)->Value{
        self.hits.push(RawText(text.into_boxed_str()));
        Value::from(self.base+HIT_TOKEN+(self.hits.len()-1) as u64)
    }
    /// The hit text `token` stands for.
    fn hit(&self,token:u64)->Option<&RawText>{
        let (base,i)=(token&!0xffff_ffff,token&0xffff_ffff);
        if base!=self.base||i&HIT_TOKEN==0 {return None;}
        self.hits.get(usize::try_from(i&!HIT_TOKEN).ok()?)
    }
    /// The source ordinal `token` stands for.
    fn get(&self,token:u64)->Option<&Source>{
        let (base,i)=(token&!0xffff_ffff,usize::try_from(token&0xffff_ffff).ok()?);
        if base==self.base {return self.sources.get(i);}
        self.parts.iter().find(|(b,_)|*b==base).and_then(|(_,sources)|sources.get(i))
    }
    pub fn parse(&mut self,bytes:&[u8])->serde_json::Result<Value>{
        if let Ok(v)=serde_json::from_slice(bytes){return Ok(v);}
        let raw:Box<RawValue>=serde_json::from_slice(bytes)?;
        let mut root=Source::parse(raw.get())?;
        self.tree=true;
        fn take(node:&mut Source,sources:&mut Vec<Source>,base:u64,at:At){
            if at==At::Opaque {return;}
            match node {
                Source::Object(m)=>{
                    let doc=at==At::Hit && m.contains_key("_index") && m.contains_key("_id");
                    let single=m.contains_key("doc_count");
                    for (k,v) in m {
                    if k=="_source" && at==At::Hit {
                        let token=Source::from_value(&Value::from(base+sources.len() as u64));
                        sources.push(std::mem::replace(v,token));
                    } else if doc && k=="fields" {
                        if let Source::Object(fields)=v {
                            for field in fields.values_mut() {
                                if serde_json::from_str::<Value>(&field.text()).is_err() {
                                    let token=Source::from_value(&Value::from(base+sources.len() as u64));
                                    sources.push(std::mem::replace(field,token));
                                }
                            }
                        }
                    } else {take(v,sources,base,at.member(k,single));}
                }},
                Source::Array(a)=>for v in a {take(v,sources,base,at.element());},
                _=>{}
            }
        }
        take(&mut root,&mut self.sources,self.base,At::Other);
        serde_json::from_str(&root.text())
    }
    /// [`Self::parse`] that never builds a tree of a hit's `_source`: every
    /// `_source` value is kept as its raw JSON text (only validated) and
    /// replaced by its ordinal, so a merge of shard responses neither
    /// unescapes nor re-escapes document text. For merges that do not read
    /// `_source` (they see ordinals); falls back to [`Self::parse`] when the
    /// rest of the response is not plain `Value` JSON.
    pub fn parse_lazy(&mut self,bytes:&[u8])->serde_json::Result<Value>{
        use serde::de::DeserializeSeed;
        let n=self.sources.len();
        let mut de=serde_json::Deserializer::from_slice(bytes);
        match Lazy(&mut self.sources,At::Other,self.base).deserialize(&mut de).and_then(|v| de.end().map(|_| v)) {
            Ok(v)=>Ok(v),
            Err(_)=>{self.sources.truncate(n);self.parse(bytes)}
        }
    }
    /// The source an ordinal token (a document's parsed `_source`) stands
    /// for, as JSON text; `None` for anything else.
    pub fn source_text(&self, token: &Value) -> Option<String> {
        self.get(token.as_u64()?).map(Source::text)
    }
    /// [`Self::render`] of what [`Self::parse_lazy`] read without the tree
    /// parse: sources restored, everything else exactly as `v` holds it
    /// (a shard's raw scripted aggregation envelopes included); `None` when
    /// some part needed the tree.
    pub fn render_lazy(&self,v:&Value)->Option<Vec<u8>>{
        (!self.tree).then(||serde_json::to_vec(&Restored{v,sources:self,at:At::Other,raw:false}).expect("validated raw JSON"))
    }
    /// Serialize `v` with every `_source` ordinal replaced by its source,
    /// in one pass (keys keep `v`'s order). Responses that needed the
    /// lossless tree parse (hit `fields` ordinals) or carry raw scripted
    /// aggregation values take the tree render.
    pub fn render(&self,v:&Value)->Vec<u8>{
        let scripted=v.get("aggregations").is_some_and(crate::aggs::has_raw_scripted_value);
        if self.sources.is_empty()&&self.parts.is_empty()&&self.hits.is_empty()&&!scripted{return serde_json::to_vec(v).unwrap();}
        // One pass, keys in `v`'s order, also with raw aggregation values (a
        // scripted metric's, a `top_hits` hit's: written verbatim where
        // they sit). Only a part that needed the lossless tree parse takes
        // the tree render (keys sorted).
        if !self.tree{return serde_json::to_vec(&Restored{v,sources:self,at:At::Other,raw:scripted}).expect("validated raw JSON");}
        let mut root=Source::from_value(v);
        if let (Some(aggregations),Source::Object(object))=(v.get("aggregations"),&mut root){object.insert("aggregations".into(),crate::aggs::render_scripted_values(aggregations));}
        fn restore(node:&mut Source,sources:&ResponseSources,at:At){
            if at==At::Opaque {return;}
            match node {
                Source::Object(m)=>{
                    let doc=at==At::Hit && m.contains_key("_index") && m.contains_key("_id");
                    let single=m.contains_key("doc_count");
                    for (k,v) in m {
                    if k=="_source" && at==At::Hit {
                        if let Source::Scalar(raw)=v {
                            if let Ok(i)=serde_json::from_str::<u64>(raw.get()) {
                                if let Some(source)=sources.get(i){*v=source.clone();}
                            }
                        }
                    } else if doc && k=="fields" {
                        if let Source::Object(fields)=v {
                            for field in fields.values_mut() {
                                if let Source::Scalar(raw)=field {
                                    if let Ok(i)=serde_json::from_str::<u64>(raw.get()) {
                                        if let Some(value)=sources.get(i) {*field=value.clone();}
                                    }
                                }
                            }
                        }
                    } else {restore(v,sources,at.member(k,single));}
                }},
                Source::Array(a)=>for v in a {
                    // A whole hit kept as text ([`ResponseSources::push_hit_text`]).
                    if at.element()==At::Hit && !sources.hits.is_empty() {
                        if let Source::Scalar(raw)=v {
                            if let Some(hit)=serde_json::from_str::<u64>(raw.get()).ok().and_then(|i|sources.hit(i)).and_then(|h|Source::parse(&h.0).ok()) {*v=hit;continue;}
                        }
                    }
                    restore(v,sources,at.element());
                },
                _=>{}
            }
        }
        restore(&mut root,self,At::Other);root.text().into_bytes()
    }
}

/// Where a JSON value sits, for which `_source` members are document
/// sources: only those of hits (`hits.hits[*]`, also inner and `top_hits`
/// hits), multi-get docs (`docs[*]`), suggestion options (`options[*]`)
/// and bulk update results (`items[*].<op>.get`). A `_source` key anywhere else (an aggregation or a
/// suggester named `_source`, a field of a document) is ordinary JSON.
/// Values that are data, never response structure ([`At::Opaque`]: a hit's
/// members other than `_source` and `inner_hits`, an aggregation result's
/// metric `value`, `meta`, partial state `__p` and keys) are never searched
/// at all, so a scripted metric's state or a hit's `fields` shaped like a
/// hit list keep their `_source`s. Only in a bucket are `value`, `meta`
/// and `__p` sub-aggregation names; in a single-bucket aggregation result
/// (one with a `doc_count`) `value` is, while its `meta` and `__p` are its
/// own.
#[derive(Clone, Copy, PartialEq, Eq)]
enum At {
    Other,
    /// Data: nothing inside is a document source (terminal).
    Opaque,
    /// A response's `aggregations`: aggregation names.
    Aggs,
    /// An aggregation result.
    Agg,
    /// An aggregation's `buckets` (a list, or keyed by bucket key).
    Buckets,
    /// One bucket: its key, `doc_count` and sub-aggregations by name.
    Bucket,
    /// The value of a `hits` / `docs` / `options` key.
    HitList,
    /// An element of such a list, or a bulk update item's `get`.
    Hit,
    /// A bulk response's `items`.
    Items,
    /// One of its elements.
    Item,
    /// An item's operation result (`{"update": {...}}`'s value).
    ItemOp,
}
impl At {
    /// Where the value of member `k` of an object at `self` sits; `single`:
    /// the object has a `doc_count`, so as an aggregation result it is a
    /// single-bucket one (`filter`, `global`, `nested`, `sampler`,
    /// `children`, `reverse_nested`, `missing`), whose members past its own
    /// `meta` are sub-aggregations, as a bucket's are.
    fn member(self, k: &str, single: bool) -> At {
        match (self, k) {
            (At::Opaque, _) => At::Opaque,
            (At::Hit, "inner_hits") => At::Other,
            (At::Hit, _) => At::Opaque,
            (At::Aggs, _) => At::Agg,
            // Keyed buckets.
            (At::Buckets, _) => At::Bucket,
            (At::Bucket, "key") => At::Opaque,
            // Any other member of a bucket is a sub-aggregation, whatever
            // its name (`value`, `meta`, `hits`, ...).
            (At::Bucket, _) => At::Agg,
            // Partial state (`__p`) is data on any aggregation result, a
            // `doc_count`-carrying one included (significant_terms,
            // matrix_stats partials).
            (At::Agg, "meta" | "key" | "after_key" | "__p") => At::Opaque,
            (At::Agg, "value") if !single => At::Opaque,
            (At::Agg, "buckets") => At::Buckets,
            (At::Agg, "hits") => At::HitList,
            (At::Agg, _) => At::Agg,
            (At::Item, _) => At::ItemOp,
            // A bulk update item's `get` (the updated document).
            (At::ItemOp, "get") => At::Hit,
            (_, "aggregations") => At::Aggs,
            (_, "hits" | "docs" | "options") => At::HitList,
            (_, "items") => At::Items,
            _ => At::Other,
        }
    }
    /// Where the elements of an array at `self` sit.
    fn element(self) -> At {
        match self {
            At::HitList => At::Hit,
            At::Items => At::Item,
            At::Buckets => At::Bucket,
            At::Opaque | At::Agg => self,
            _ => At::Other,
        }
    }
}

/// [`ResponseSources::parse_lazy`]'s visitor: a `Value` whose hits'
/// `_source` members are pushed to the side table as raw text.
struct Lazy<'a>(&'a mut Vec<Source>, At, u64);
impl<'de> serde::de::DeserializeSeed<'de> for Lazy<'_> {
    type Value=Value;
    fn deserialize<D:serde::Deserializer<'de>>(self,d:D)->Result<Value,D::Error>{d.deserialize_any(self)}
}
impl<'de> serde::de::Visitor<'de> for Lazy<'_> {
    type Value=Value;
    fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result{f.write_str("any JSON value")}
    fn visit_bool<E:serde::de::Error>(self,v:bool)->Result<Value,E>{Ok(Value::Bool(v))}
    fn visit_i64<E:serde::de::Error>(self,v:i64)->Result<Value,E>{Ok(Value::from(v))}
    fn visit_u64<E:serde::de::Error>(self,v:u64)->Result<Value,E>{Ok(Value::from(v))}
    fn visit_f64<E:serde::de::Error>(self,v:f64)->Result<Value,E>{Ok(serde_json::Number::from_f64(v).map_or(Value::Null,Value::Number))}
    fn visit_str<E:serde::de::Error>(self,v:&str)->Result<Value,E>{Ok(Value::String(v.to_owned()))}
    fn visit_string<E:serde::de::Error>(self,v:String)->Result<Value,E>{Ok(Value::String(v))}
    fn visit_unit<E:serde::de::Error>(self)->Result<Value,E>{Ok(Value::Null)}
    fn visit_none<E:serde::de::Error>(self)->Result<Value,E>{Ok(Value::Null)}
    fn visit_some<D:serde::Deserializer<'de>>(self,d:D)->Result<Value,D::Error>{d.deserialize_any(self)}
    fn visit_seq<A:serde::de::SeqAccess<'de>>(self,mut seq:A)->Result<Value,A::Error>{
        let (sources,at,base)=(self.0,self.1.element(),self.2);
        let mut out=Vec::with_capacity(seq.size_hint().unwrap_or(0));
        while let Some(v)=seq.next_element_seed(Lazy(&mut *sources,at,base))? {out.push(v);}
        Ok(Value::Array(out))
    }
    fn visit_map<A:serde::de::MapAccess<'de>>(self,mut map:A)->Result<Value,A::Error>{
        let (sources,at,base)=(self.0,self.1,self.2);
        // A hit's few members (a coordinator's query hits are 3-5, and it
        // adds one): sized up front, not grown (rehashed) member by member.
        let mut out=if at==At::Hit {serde_json::Map::with_capacity(8)} else {serde_json::Map::new()};
        while let Some(k)=map.next_key::<String>()? {
            let v=if k=="_source" && at==At::Hit {
                let raw:Box<RawValue>=map.next_value()?;
                sources.push(Source::Scalar(raw));
                Value::from(base+(sources.len()-1) as u64)
            } else {
                // A single-bucket aggregation writes `doc_count` before its
                // sub-aggregations (its own `meta` comes first, and stays
                // data either way).
                let single=out.contains_key("doc_count");
                map.next_value_seed(Lazy(&mut *sources,at.member(&k,single),base))?
            };
            out.insert(k,v);
        }
        Ok(Value::Object(out))
    }
}

/// [`ResponseSources::render`]'s view of a merged response.
/// `raw`: write the raw values aggregation results carry
/// (`aggs::scripted::render_values`' envelopes) as their text, and leave out
/// the `__p` envelopes.
struct Restored<'a>{v:&'a Value,sources:&'a ResponseSources,at:At,raw:bool}
impl Serialize for Restored<'_> {
    fn serialize<S:serde::Serializer>(&self,s:S)->Result<S::Ok,S::Error>{
        use serde::ser::{SerializeMap,SerializeSeq};
        match self.v {
            _ if self.at==At::Opaque => self.v.serialize(s),
            // A whole hit kept as text ([`ResponseSources::push_hit_text`]).
            Value::Number(n) if self.at==At::Hit && n.as_u64().and_then(|i|self.sources.hit(i)).is_some() => {
                self.sources.hit(n.as_u64().unwrap_or_default()).expect("checked above").serialize(s)
            }
            Value::Object(m)=>{
                let text=|raw:&str| RawValue::from_string(raw.to_string()).map_err(<S::Error as serde::ser::Error>::custom);
                let envelope=self.raw && self.at!=At::Other;
                // A `top_hits` hit kept as text.
                if let Some(hit)=m.get("__p").and_then(|p|p.get("raw_json")).and_then(Value::as_str).filter(|_|envelope && self.at==At::Hit) {
                    return text(hit)?.serialize(s);
                }
                // A scripted metric's raw value.
                let metric=m.get("__p").and_then(|p|p.get("utf16_metric_value")).and_then(Value::as_str).filter(|_|envelope);
                let mut out=s.serialize_map(None)?;
                let single=m.contains_key("doc_count");
                for (k,v) in m {
                    if envelope && k=="__p" {continue;}
                    if let (Some(raw),"value")=(metric,k.as_str()) {out.serialize_entry(k,&text(raw)?)?;continue;}
                    match v.as_u64().and_then(|i| self.sources.get(i)).filter(|_| k=="_source" && self.at==At::Hit) {
                        Some(src)=>out.serialize_entry(k,src)?,
                        None=>out.serialize_entry(k,&Restored{v,sources:self.sources,at:self.at.member(k,single),raw:self.raw})?,
                    }
                }
                if let Some(raw)=metric.filter(|_|!m.contains_key("value")) {out.serialize_entry("value",&text(raw)?)?;}
                out.end()
            }
            Value::Array(a)=>{
                let mut out=s.serialize_seq(Some(a.len()))?;
                let at=self.at.element();
                for v in a {out.serialize_element(&Restored{v,sources:self.sources,at,raw:self.raw})?;}
                out.end()
            }
            v=>v.serialize(s),
        }
    }
}

/// A response's `aggregations` with their raw values (a scripted metric's,
/// a `top_hits` hit kept as text) written verbatim, in one pass: every key
/// keeps its order.
pub fn render_aggregations(aggregations: &Value) -> Vec<u8> {
    serde_json::to_vec(&Restored{v:aggregations,sources:&ResponseSources::default(),at:At::Aggs,raw:true}).expect("validated raw JSON")
}

/// The members of the JSON object `text`, in order, each value as its raw
/// text (lossless: no reordering, no re-escaping); `None` for anything but
/// an object.
pub fn raw_members(text: &str) -> Option<Vec<(String, Box<RawValue>)>> {
    struct Members;
    impl<'de> serde::de::Visitor<'de> for Members {
        type Value = Vec<(String, Box<RawValue>)>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("a JSON object") }
        fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            let mut out = Vec::new();
            while let Some(entry) = map.next_entry::<String, Box<RawValue>>()? { out.push(entry); }
            Ok(out)
        }
    }
    let mut d = serde_json::Deserializer::from_str(text);
    let members = serde::Deserializer::deserialize_map(&mut d, Members).ok()?;
    d.end().ok()?;
    Some(members)
}

/// The elements of the JSON array `text`, each as its raw text.
pub fn raw_elements(text: &str) -> Option<Vec<Box<RawValue>>> {
    serde_json::from_str(text).ok()
}

/// The JSON object `text` with member `key` set to the JSON `value`: in
/// place when it has one (unless `keep`), else appended; every other member
/// as it was (lossless).
pub fn with_member(text: &str, key: &str, value: &str, keep: bool) -> Option<String> {
    let mut members = raw_members(text)?;
    let value = RawValue::from_string(value.to_string()).ok()?;
    match members.iter_mut().find(|(k, _)| k == key) {
        Some(_) if keep => {}
        Some((_, v)) => *v = value,
        None => members.push((key.to_string(), value)),
    }
    let mut out = String::with_capacity(text.len() + key.len() + 16);
    out.push('{');
    for (i, (k, v)) in members.iter().enumerate() {
        if i > 0 { out.push(','); }
        out.push_str(&serde_json::to_string(k).ok()?);
        out.push(':');
        out.push_str(v.get());
    }
    out.push('}');
    Some(out)
}

#[cfg(test)]
mod raw_member_tests {
    use super::*;
    /// Members keep their order and text (lone surrogates, number text).
    #[test]
    fn with_member_is_lossless_and_keeps_order() {
        let t = r#"{"took":3,"aggregations":{"p":{"values":{"1.0":1,"5.0":2,"25.0":3}}},"hits":{"hits":[{"_source":{"z":1.50,"a":"\ud800"}}]},"status":500}"#;
        assert_eq!(with_member(t, "status", "200", false).unwrap(), t.replace("500", "200"));
        assert_eq!(with_member(t, "status", "200", true).unwrap(), t);
        assert_eq!(with_member(r#"{"b":1,"a":2}"#, "status", "200", false).unwrap(), r#"{"b":1,"a":2,"status":200}"#);
        assert!(with_member("[1]", "status", "200", false).is_none());
        let a = raw_elements(r#"[{"b":1,"a":"\ud800"},{"x":[1.50]}]"#).unwrap();
        assert_eq!(a.iter().map(|v| v.get()).collect::<Vec<_>>(), vec![r#"{"b":1,"a":"\ud800"}"#, r#"{"x":[1.50]}"#]);
    }
}

pub fn decode_string(raw: &str) -> Vec<u16> {
    let mut out=Vec::new();
    let mut chars=raw[1..raw.len()-1].chars();
    while let Some(c)=chars.next() {
        if c!='\\' { let mut buf=[0;2]; out.extend_from_slice(c.encode_utf16(&mut buf)); continue; }
        let c=chars.next().expect("validated JSON escape");
        if c=='u' {
            let mut unit=0u16;
            for _ in 0..4 { unit=(unit<<4)|chars.next().unwrap().to_digit(16).unwrap() as u16; }
            out.push(unit);
        } else { out.push(match c {'"'=>34,'\\'=>92,'/'=>47,'b'=>8,'f'=>12,'n'=>10,'r'=>13,'t'=>9,_=>unreachable!()}); }
    }
    out
}

/// Parse update metadata separately from source-valued doc/upsert fields.
pub fn update_request(raw:&str)->serde_json::Result<(Value,Option<Source>)>{
    if let Ok(v)=serde_json::from_str(raw){return Ok((v,None));}
    let source=Source::parse(raw)?;
    let mut metadata=source.clone();
    if let Source::Object(m)=&mut metadata {
        for key in ["doc","upsert"] {if m.contains_key(key){m.insert(key.into(),Source::Object(BTreeMap::new()));}}
        // Keep script parameter values lossless; compile metadata still passes
        // through Value and retains its ordinary type validation.
        if let Some(Source::Object(script))=m.get_mut("script") {
            for key in ["source","inline"] {
                if let Some(Source::Scalar(raw))=script.get(key) {
                    if raw.get().starts_with('"') && serde_json::from_str::<String>(raw.get()).is_err() {
                        let display=String::from_utf16_lossy(&decode_string(raw.get()));
                        script.insert(key.into(),Source::from_value(&Value::String(display)));
                    }
                }
            }
            if matches!(script.get("params"),Some(Source::Object(_))) {
                script.insert("params".into(),Source::Object(BTreeMap::new()));
            }
        }
    }
    Ok((serde_json::from_str(&metadata.text())?,Some(source)))
}
impl Source {
    pub fn get(&self,key:&str)->Option<&Self>{match self {Self::Object(m)=>m.get(key),_=>None}}
    pub fn retain_leaves(&self,allowed:&dyn Fn(&str)->bool)->Self{
        fn walk(v:&Source,p:&str,f:&dyn Fn(&str)->bool)->Option<Source>{
            match v {
                Source::Object(m)=>{
                    let out:BTreeMap<_,_>=m.iter().filter_map(|(k,v)|{let path=if p.is_empty(){k.clone()}else{format!("{p}.{k}")};walk(v,&path,f).map(|v|(k.clone(),v))}).collect();
                    (!out.is_empty() || (m.is_empty() && !p.is_empty() && f(p))).then_some(Source::Object(out))
                }
                Source::Array(a)=>{
                    let out:Vec<_>=a.iter().filter_map(|v|walk(v,p,f)).collect();
                    (!out.is_empty() || (a.is_empty() && f(p))).then_some(Source::Array(out))
                }
                _=>f(p).then(||v.clone()),
            }
        }
        walk(self,"",allowed).unwrap_or_else(||Self::Object(BTreeMap::new()))
    }
}

impl Source {
    /// Apply a structural selector which only copies/removes nodes. Every
    /// original scalar is replaced, so no user scalar can be mistaken for an
    /// internal ordinal. Do not use for arithmetic/value-transforming callbacks.
    pub fn select_structure(&self,select:impl FnOnce(&Value)->Value)->Self{
        fn shape(s:&Source,leaves:&mut Vec<Source>)->Value{
            match s {
                Source::Object(m)=>Value::Object(m.iter().map(|(k,v)|(k.clone(),shape(v,leaves))).collect()),
                Source::Array(a)=>Value::Array(a.iter().map(|s|shape(s,leaves)).collect()),
                _=>{let id=leaves.len();leaves.push(s.clone());Value::from(id as u64)}
            }
        }
        fn restore(v:Value,leaves:&[Source])->Source{
            match v {
                Value::Object(m)=>Source::Object(m.into_iter().map(|(k,v)|(k,restore(v,leaves))).collect()),
                Value::Array(a)=>Source::Array(a.into_iter().map(|v|restore(v,leaves)).collect()),
                Value::Number(n)=>leaves[n.as_u64().expect("selector must preserve scalar tokens") as usize].clone(),
                _=>unreachable!("selector must preserve scalar tokens"),
            }
        }
        let mut leaves=Vec::new();let v=shape(self,&mut leaves);restore(select(&v),&leaves)
    }
}

#[cfg(test)]
mod response_field_tests {
    use super::*;
    #[test]
    fn opaque_script_fields_survive_merge_without_hiding_collapse_keys() {
        let mut table=ResponseSources::default();
        let first=table.parse(br#"{"hits":{"hits":[{"_index":"i","_id":"1","fields":{"raw":["\ud800"],"group":[7]}}]}}"#).unwrap();
        let second=table.parse(br#"{"hits":{"hits":[{"_index":"i","_id":"2","fields":{"raw":["\udfff"],"group":[8]}}]}}"#).unwrap();
        assert_eq!(first["hits"]["hits"][0]["fields"]["group"][0],7);
        let merged=serde_json::json!({"hits":{"hits":[second["hits"]["hits"][0],first["hits"]["hits"][0]]},"other":{"fields":{"number":0}}});
        let bytes=table.render(&merged);
        let actual=Source::parse(std::str::from_utf8(&bytes).unwrap()).unwrap();
        let expected=Source::parse(r#"{"hits":{"hits":[{"_index":"i","_id":"2","fields":{"raw":["\udfff"],"group":[8]}},{"_index":"i","_id":"1","fields":{"raw":["\ud800"],"group":[7]}}]},"other":{"fields":{"number":0}}}"#).unwrap();
        assert!(actual.equivalent(&expected));
    }
    /// Raw aggregation values (a `top_hits` hit kept as text, a scripted
    /// metric's raw value) render in one pass: every other key keeps its
    /// order (keyed percentiles stay 1, 5, 25).
    #[test]
    fn raw_aggregation_values_render_in_order() {
        let hit=crate::search::hit_value(r#"{"_index":"i","_id":"1","_source":{"z":1,"a":"\ud800"}}"#).unwrap();
        let v=serde_json::json!({"took":1,"aggregations":{"p":{"values":{"1.0":1,"5.0":2,"25.0":3}},"top":{"hits":{"hits":[hit]}},"m":{"value":null,"__p":{"utf16_metric_value":"\"\\ud800\""}}},"hits":{"hits":[]}});
        let out=String::from_utf8(ResponseSources::default().render(&v)).unwrap();
        assert!(out.starts_with(r#"{"took":1,"aggregations":{"p":{"values":{"1.0":1,"5.0":2,"25.0":3}},"top":{"hits":{"hits":[{"_index":"i","_id":"1","_source":{"z":1,"a":"\ud800"}}]}},"m":{"value":"\ud800"}}"#),"{out}");
        assert!(!out.contains("__p"),"{out}");
    }
    /// A lazily parsed shard response merged with one that needed the
    /// lossless parse renders both: raw `_source`s and opaque `fields`.
    #[test]
    fn lazy_and_lossless_shard_responses_render_together() {
        let mut table=ResponseSources::default();
        let first=table.parse_lazy(br#"{"hits":{"hits":[{"_index":"i","_id":"1","_source":{"t":"a"}}]}}"#).unwrap();
        let second=table.parse_lazy(br#"{"hits":{"hits":[{"_index":"i","_id":"2","_source":{"t":"\ud800"},"fields":{"raw":["\udfff"]}}]}}"#).unwrap();
        let merged=serde_json::json!({"hits":{"hits":[second["hits"]["hits"][0],first["hits"]["hits"][0]]}});
        let bytes=table.render(&merged);
        let actual=Source::parse(std::str::from_utf8(&bytes).unwrap()).unwrap();
        let expected=Source::parse(r#"{"hits":{"hits":[{"_index":"i","_id":"2","_source":{"t":"\ud800"},"fields":{"raw":["\udfff"]}},{"_index":"i","_id":"1","_source":{"t":"a"}}]}}"#).unwrap();
        assert!(actual.equivalent(&expected),"{}",String::from_utf8_lossy(&bytes));
    }
}

/// Separate search script-field metadata from original UTF-16 script values.
pub fn search_request(raw: &str) -> serde_json::Result<(Value, Option<Source>)> {
    if let Ok(value)=serde_json::from_str(raw) {return Ok((value,None));}
    let original=Source::parse(raw)?;
    let mut metadata=original.clone();
    take_search_scripts(&mut metadata);
    if let Source::Object(root)=&mut metadata {
        for key in ["script_fields", "runtime_mappings"] {
        if let Some(Source::Object(fields))=root.get_mut(key) {
            for field in fields.values_mut() {
                let (value,_)=update_request(&field.text())?;
                *field=Source::from_value(&value);
            }
        }
    }
        }
    if let Source::Object(root)=&mut metadata {
        if let Some(sort)=root.get_mut("sort") {
            fn normalize(entry: &mut Source) -> serde_json::Result<()> {
                if let Source::Object(fields)=entry {
                    if let Some(spec)=fields.get_mut("_script") {
                        let (value,_)=update_request(&spec.text())?;
                        *spec=Source::from_value(&value);
                    }
                }
                Ok(())
            }
            match sort {
                Source::Array(entries)=>for entry in entries {normalize(entry)?;},
                entry=>normalize(entry)?,
            }
        }
    }
    Ok((serde_json::from_str(&metadata.text())?,Some(original)))
}

/// The lossless form of a search body when [`search_request`] needs one
/// (`Some` exactly when it returns `Some`), without parsing a body that
/// cannot need it. Only a `\u` escape (a lone surrogate) makes the
/// ordinary parse fail where the lossless one succeeds; a body without one
/// is either plain JSON or invalid for both. Search bodies can be large
/// (a `terms` query with tens of thousands of values).
pub fn lossless_search_source(raw: &str) -> Option<Source> {
    if !raw.contains("\\u") {
        return None;
    }
    search_request(raw).ok().and_then(|(_, source)| source)
}

/// Preserve original script-field subtrees when a coordinator adjusts pagination.
pub fn search_body(original: &[u8], rewritten: &Value) -> Vec<u8> {
    let Ok(text)=std::str::from_utf8(original) else {return rewritten.to_string().into_bytes()};
    let Some(raw)=lossless_search_source(text) else {return rewritten.to_string().into_bytes()};
    let mut result=Source::from_value(rewritten);
    let scripts=search_scripts(&raw);
    search_script_slots(&mut result,|slot| {
        if let Source::Scalar(token)=slot {
            if let Ok(index)=serde_json::from_str::<usize>(token.get()) {
                if let Some(script)=scripts.get(index) {*slot=script.clone();}
            }
        }
    });
    if let Source::Object(root)=&mut result {
        for key in ["script_fields", "runtime_mappings", "sort"] {
            if root.contains_key(key) {
                if let Some(fields)=raw.get(key) {root.insert(key.into(),fields.clone());}
            }
        }
    }
    result.text().into_bytes()
}

#[cfg(test)]
mod raw_search_tests {
    use super::*;
    #[test]
    fn raw_sort_scripts_survive_coordinator_rewrites() {
        for sort in [
            r#"{"_script":{"type":"number","script":{"source":"(int)params.s.charAt(0)","params":{"s":"\ud800"}},"order":"asc"}}"#,
            r#"[{"_script":{"type":"number","script":{"source":"(int)'\udfff'.charAt(0)"}}},{"_doc":"asc"}]"#,
        ] {
            let input=format!(r#"{{"size":1,"sort":{sort}}}"#);
            let (mut metadata,original)=search_request(&input).unwrap();
            metadata["size"]=Value::from(12);
            let output=search_body(input.as_bytes(),&metadata);
            let output=Source::parse(std::str::from_utf8(&output).unwrap()).unwrap();
            assert!(output.get("sort").unwrap().equivalent(original.unwrap().get("sort").unwrap()));
            assert_eq!(output.get("size").unwrap().text(),"12");
        }
        assert!(search_request(r#"{"sort":{"_script":{"type":"\ud800","script":"1"}}}"#).is_err());
    }
    #[test]
    fn lossless_search_source_agrees_with_search_request() {
        for body in [
            r#"{"size":1}"#,
            r#"{"query":{"terms":{"f":["a","b"]}}}"#,
            r#"{"script_fields":{"f":{"script":{"source":"params.s","params":{"s":"\ud800"}}}}}"#,
            r#"{"q":"\u0041"}"#,
            r#"{"sort":{"_script":{"type":"\ud800","script":"1"}}}"#,
            r#"{"broken":"#,
            r#"{"broken":"\ud800"#,
        ] {
            let full = search_request(body).ok().and_then(|(_, s)| s).map(|s| s.text());
            assert_eq!(lossless_search_source(body).map(|s| s.text()), full, "{body}");
        }
        let deep = format!("{}{}", "[".repeat(200), "]".repeat(200));
        assert_eq!(lossless_search_source(&deep).map(|s| s.text()), search_request(&deep).ok().and_then(|(_, s)| s).map(|s| s.text()));
    }

    #[test]
    fn search_rewrite_preserves_raw_scripts_and_validates_other_metadata() {
        let input=r#"{"size":1,"script_fields":{"f":{"script":{"source":"params.s","params":{"s":"\ud800"}}}},"runtime_mappings":{"r":{"type":"keyword","script":{"source":"emit('\udfff')"}}}}"#;
        let (mut metadata,original)=search_request(input).unwrap();
        assert!(original.is_some());
        metadata["size"]=Value::from(20);
        let output=search_body(input.as_bytes(),&metadata);
        let raw=Source::parse(std::str::from_utf8(&output).unwrap()).unwrap();
        assert!(raw.get("script_fields").unwrap().equivalent(original.as_ref().unwrap().get("script_fields").unwrap()));
        assert!(raw.get("runtime_mappings").unwrap().equivalent(original.as_ref().unwrap().get("runtime_mappings").unwrap()));
        assert_eq!(raw.get("size").unwrap().text(),"20");
        assert!(search_request(r#"{"unknown":"\ud800"}"#).is_err());
    }
}

/// Script slots are opaque only at known search DSL script positions. Every
/// such position is replaced, including invalid user numbers; user scalars
/// elsewhere never become references. The table lives with one request.
pub fn search_script_slots(request:&mut Source, mut visit:impl FnMut(&mut Source)) {
    fn many(node:&mut Source, f:&mut dyn FnMut(&mut Source)) {
        match node {Source::Array(a)=>for item in a {f(item)},item=>f(item)}
    }
    fn query(node:&mut Source,visit:&mut dyn FnMut(&mut Source)) {
        let Source::Object(queries)=node else{return};
        for (kind,body) in queries {
            let Source::Object(body)=body else{continue};
            if matches!(kind.as_str(),"script"|"script_score") {if let Some(script)=body.get_mut("script"){visit(script);}}
            match kind.as_str() {
                "bool"=>for key in ["must","filter","should","must_not"] {if let Some(v)=body.get_mut(key){many(v,&mut |q|query(q,visit));}},
                "boosting"=>for key in ["positive","negative"] {if let Some(v)=body.get_mut(key){query(v,visit);}},
                "dis_max"=>if let Some(v)=body.get_mut("queries"){many(v,&mut |q|query(q,visit));},
                "constant_score"|"knn"=>if let Some(v)=body.get_mut("filter"){many(v,&mut |q|query(q,visit));},
                "pinned"=>if let Some(v)=body.get_mut("organic"){query(v,visit);},
                "script_score"|"nested"|"has_child"|"has_parent"|"function_score"=>if let Some(v)=body.get_mut("query"){query(v,visit);},
                _=>{}
            }
            if kind=="function_score" {
                fn function(body:&mut BTreeMap<String,Source>,visit:&mut dyn FnMut(&mut Source)) {
                    if let Some(Source::Object(score))=body.get_mut("script_score") {if let Some(script)=score.get_mut("script"){visit(script);}}
                    if let Some(filter)=body.get_mut("filter"){query(filter,visit);}
                }
                function(body,visit);
                if let Some(Source::Array(functions))=body.get_mut("functions") {for f in functions {if let Source::Object(f)=f {function(f,visit);}}}
            }
        }
    }
    fn aggs(node:&mut Source,visit:&mut dyn FnMut(&mut Source)) {
        let Source::Object(named)=node else{return};
        for agg in named.values_mut() {
            let Source::Object(agg)=agg else{continue};
            for (kind,body) in agg {
                if matches!(kind.as_str(),"aggs"|"aggregations") {aggs(body,visit);continue;}
                if kind=="filter" {query(body,visit);continue;}
                let Source::Object(body)=body else{continue};
                if kind=="filters" {
                    if let Some(filters)=body.get_mut("filters") {match filters {Source::Object(m)=>for q in m.values_mut(){query(q,visit)},Source::Array(a)=>for q in a {query(q,visit)},_=>{}}}
                } else if kind!="meta" {
                    if kind=="scripted_metric" {if let Some(params)=body.get_mut("params"){visit(params);}}
                    for key in ["script","init_script","map_script","combine_script","reduce_script"] {if let Some(v)=body.get_mut(key){visit(v);}}
                }
            }
        }
    }
    let Source::Object(root)=request else{return};
    for key in ["query","post_filter"] {if let Some(v)=root.get_mut(key){query(v,&mut visit);}}
    for key in ["aggs","aggregations"] {if let Some(v)=root.get_mut(key){aggs(v,&mut visit);}}
    if let Some(rescore)=root.get_mut("rescore") {many(rescore,&mut |rescore| {
        if let Source::Object(r)=rescore {
            if let Some(Source::Object(q))=r.get_mut("query") {if let Some(q)=q.get_mut("rescore_query"){query(q,&mut visit);}}
            if let Some(Source::Object(script))=r.get_mut("script") {if let Some(s)=script.get_mut("script"){visit(s);}}
        }
    });}
}

pub fn take_search_scripts(request:&mut Source)->Vec<Source> {
    let mut scripts=Vec::new();
    search_script_slots(request,|slot| {
        if matches!(slot,Source::Scalar(raw) if raw.get()=="null") {return;}
        let token=Source::from_value(&Value::from(scripts.len() as u64));
        scripts.push(std::mem::replace(slot,token));
    });
    scripts
}

pub fn search_scripts(request:&Source)->Vec<Source> {take_search_scripts(&mut request.clone())}

#[cfg(test)]
mod search_slot_tests {
    use super::*;
    #[test]
    fn metric_params_and_optional_null_scripts_round_trip() {
        let text=r#"{"aggs":{"m":{"scripted_metric":{"params":{"s":"\ud800"},"init_script":null,"map_script":"state.n=params.s.length()","combine_script":"state.n","reduce_script":"states[0]"}}}}"#;
        let (metadata,raw)=search_request(text).unwrap();
        assert!(metadata["aggs"]["m"]["scripted_metric"]["init_script"].is_null());
        assert_eq!(search_scripts(raw.as_ref().unwrap()).len(),4);
        let restored=Source::parse(std::str::from_utf8(&search_body(text.as_bytes(),&metadata)).unwrap()).unwrap();
        assert!(restored.equivalent(raw.as_ref().unwrap()));
    }
    #[test]
    fn raw_query_scripts_remain_distinct_and_do_not_replace_term_values() {
        let raw=r#"{"query":{"bool":{"filter":[{"term":{"script":0}},{"script":{"script":{"source":"params.s","params":{"s":"\ud800"}}}},{"script":{"script":{"source":"params.s","params":{"s":"\udfff"}}}}]}},"aggs":{"script":{"sum":{"script":{"source":"params.s.length()","params":{"s":"\ud800"}}}}},"rescore":{"script":{"script":{"source":"params.s.length()","params":{"s":"\udfff"}}}}}"#;
        let (metadata,original)=search_request(raw).unwrap();
        let original=original.unwrap();let scripts=search_scripts(&original);
        assert_eq!(scripts.len(),4);
        assert_eq!(metadata["query"]["bool"]["filter"][0]["term"]["script"],0);
        let mut rewritten=metadata.clone();rewritten["from"]=Value::from(3);
        let result=Source::parse(std::str::from_utf8(&search_body(raw.as_bytes(),&rewritten)).unwrap()).unwrap();
        assert!(result.get("query").unwrap().equivalent(original.get("query").unwrap()));
        assert!(result.get("aggs").unwrap().equivalent(original.get("aggs").unwrap()));
        assert!(result.get("rescore").unwrap().equivalent(original.get("rescore").unwrap()));
    }
}

#[cfg(test)]
mod leading_value_tests {
    use super::*;

    #[test]
    fn ingest_source_ignores_text_after_the_first_value() {
        assert_eq!(leading_value(r#"{"a":1}{"b":2}"#), Some(r#"{"a":1}"#));
        assert_eq!(leading_value(r#" {"a":"}{"} x"#), Some(r#" {"a":"}{"}"#));
        assert_eq!(leading_value(r#"{"a":1}"#), None);
        assert_eq!(leading_value(r#"{"a":1}  "#), None);
        assert_eq!(leading_value(r#"{"a":"#), None);
        assert!(Source::parse(r#"{"a":1}}"#).is_err());
        assert_eq!(Source::parse_leading(r#"{"a":1}}"#).unwrap().text(), r#"{"a":1}"#);
        assert_eq!(Source::parse_leading(r#"{"a":1} trailing ] garbage"#).unwrap().text(), r#"{"a":1}"#);
        // Errors inside the first value are unchanged.
        assert_eq!(Source::parse_leading(r#"{"a":}"#).unwrap_err().to_string(), Source::parse(r#"{"a":}"#).unwrap_err().to_string());
    }
}

#[cfg(test)]
mod scan_tests {
    use super::*;

    fn slow(raw: &str) -> serde_json::Result<Source> {
        Source::from_raw(serde_json::from_str(raw)?, 0)
    }

    /// The single pass builds what the per-level route builds, and errors
    /// the same way.
    #[test]
    fn scanned_sources_match_the_per_level_parse() {
        let deep_ok = format!("{}1{}", "[".repeat(100), "]".repeat(100));
        let cases = [
            r#"{"b":1,"a":{"y":[1, 2.50, -3e2,"s",true,false,null,{}],"x":"\u00e9\ud800 \"q\""},"a":{"dup":1}}"#,
            " \n{ \"k\" : [ ] , \"e\\u0041\":{ } }\t",
            r#"[1e400]"#,
            r#"{"x":12345678901234567890123}"#,
            r#""top""#,
            "42",
            &deep_ok,
            r#"{"\ud800":1}"#,
            // Invalid or unusual input: the same error (or value) either way.
            "[1,]", r#"{"a":1,}"#, "01", "1.", "-", "-0", "1e", "1e+", "+1", ".5", r#""\x""#, "\"a\u{1}b\"", r#"{"a":tru}"#,
            "[1] x", " ", "", r#""\u12""#, r#""\u12zz""#, r#""\ud83d\ude00""#, r#""a\/b\b\f\n\r\t""#, "\u{feff}1", r#"{"a" 1}"#,
            "[1 2]", "nul", "true ", " [ ] ", r#"{"a":{"b":[1e5,-2.5E-3,0.0]}}"#, "1.5e400",
        ];
        for c in cases {
            match (Source::parse(c), slow(c)) {
                (Ok(a), Ok(b)) => assert_eq!(a.text(), b.text(), "{c}"),
                (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string(), "{c}"),
                (a, b) => panic!("{c}: {a:?} vs {b:?}"),
            }
        }
        // Key order and duplicates (last wins) are kept.
        let Source::Object(m) = Source::parse(r#"{"z":1,"a":2,"z":3}"#).unwrap() else { panic!() };
        assert_eq!(m.keys().collect::<Vec<_>>(), ["a", "z"]);
        let Source::Scalar(z) = &m["z"] else { panic!() };
        assert_eq!(z.get(), "3");
    }
}
