//! Lossless source bridge for Painless. UTF-16 strings use code units, never
//! replacement characters or sentinel strings that could collide with user data.
use super::value::{Val, Key};
use crate::source_json::Source;
use crate::error::{EsError, Result};
use std::{rc::Rc, collections::BTreeMap};

use crate::source_json::decode_string;
pub fn from_source(src: &Source) -> Val {
    match src {
        Source::Object(m)=>Val::new_map(m.iter().map(|(k,v)|(Key::str(k),from_source(v))).collect()),
        Source::Array(a)=>Val::new_list(a.iter().map(from_source).collect()),
        // A string without escapes is its own text (what parsing gives).
        Source::Scalar(raw) if raw.get().len()>=2 && raw.get().starts_with('"') && !raw.get().contains('\\') => Val::str(&raw.get()[1..raw.get().len()-1]),
        Source::Scalar(raw)=>match serde_json::from_str(raw.get()) {
            Ok(v)=>Val::from_json(&v),
            Err(_)=>Val::utf16(decode_string(raw.get())),
        }
    }
}
pub fn to_source(val: &Val) -> Result<Source> {
    fn walk(val:&Val, stack:&mut Vec<usize>) -> Result<Source> {
        let addr=match val {Val::Map(m)=>Some(Rc::as_ptr(m) as usize),Val::List(a)|Val::Set(a)=>Some(Rc::as_ptr(a) as usize),Val::Array{items,..}=>Some(Rc::as_ptr(items) as usize),_=>None};
        if let Some(addr)=addr { if stack.contains(&addr) {return Err(EsError::illegal_argument("Iterable object is self-referencing itself"));} stack.push(addr); }
        let out=match val {
            Val::Char(c)=>walk(&Val::utf16(vec![*c]),stack)?,
            Val::Builder(b)=>walk(&Val::utf16(b.borrow().to_vec()),stack)?,
            Val::Utf16(units)=> utf16_scalar(units)?,
            Val::Map(m) if m.borrow().contains_key(&Key::str(super::doc::GEO_SHAPE_KEY))=>Source::from_value(&val.to_json()),
            Val::Map(m)=>Source::Object(m.ordered().iter().map(|(k,v)|Ok((key_string(k)?,walk(v,stack)?))).collect::<Result<BTreeMap<_,_>>>()?),
            Val::List(a)|Val::Set(a)=>Source::Array(a.ordered().iter().map(|v|walk(v,stack)).collect::<Result<_>>()?),
            Val::Array{items,..}=>Source::Array(items.borrow().iter().map(|v|walk(v,stack)).collect::<Result<_>>()?),
            Val::Str(s)=>Source::Scalar(serde_json::value::RawValue::from_string(serde_json::to_string(&**s).expect("string JSON")).expect("valid JSON")),
            _=>match val.to_json() {
                // A scalar's JSON text is its raw value (what parsing it back keeps).
                json@(serde_json::Value::Null|serde_json::Value::Bool(_)|serde_json::Value::Number(_)|serde_json::Value::String(_))=>Source::Scalar(serde_json::value::RawValue::from_string(json.to_string()).expect("valid JSON")),
                json=>Source::from_value(&json),
            },
        };
        if addr.is_some(){stack.pop();} Ok(out)
    }
    walk(val,&mut Vec::new())
}

/// The maps [`partial_ctx`] built for the partially observed objects of the
/// source (by address), by key.
pub(crate) struct Made {
    ptr: usize,
    kids: std::collections::HashMap<String, Made>,
}

/// `ctx` for a script that observes only `paths` of the source object
/// `src` ([`super::ctx_access`]): those keys, converted as [`from_source`]
/// converts them; a partially observed object is a map of just its
/// observed keys.
pub(crate) fn partial_ctx(src: &BTreeMap<String, Source>, paths: &super::ctx_access::CtxPaths) -> (Val, Made) {
    let mut map = indexmap::IndexMap::with_capacity(paths.children.len());
    let mut kids = std::collections::HashMap::new();
    for (key, child) in &paths.children {
        let Some(value) = src.get(key) else { continue };
        match value {
            Source::Object(m) if !child.full => {
                let (v, made) = partial_ctx(m, child);
                kids.insert(key.clone(), made);
                map.insert(Key::str(key), v);
            }
            v => { map.insert(Key::str(key), from_source(v)); }
        }
    }
    let val = Val::new_map(map);
    let ptr = match &val { Val::Map(m) => Rc::as_ptr(m) as usize, _ => unreachable!("a map") };
    (val, Made { ptr, kids })
}

enum Change {
    Set(Source),
    Into(Vec<(String, Change)>),
}

/// Write a script's `ctx` built by [`partial_ctx`] back into `context`:
/// what [`to_source`] of the whole `ctx` would have produced. The
/// observed keys take their new values (a partially observed object still
/// in place is merged into); the rest of the source keeps its structure
/// with each scalar as a conversion round trip leaves it. An error leaves
/// `context` unchanged.
pub(crate) fn merge_partial(context: &mut Source, val: &Val, made: &Made) -> Result<()> {
    fn plan(val: &Val, made: &Made) -> Result<Vec<(String, Change)>> {
        let Val::Map(m) = val else { unreachable!("a partial ctx is a map") };
        let m = m.ordered();
        let mut out = Vec::with_capacity(m.len());
        for (k, v) in m.iter() {
            let key = key_string(k)?;
            let change = match (v, made.kids.get(&key)) {
                (Val::Map(child), Some(kid)) if Rc::as_ptr(child) as usize == kid.ptr => Change::Into(plan(v, kid)?),
                _ => Change::Set(to_source(v)?),
            };
            out.push((key, change));
        }
        Ok(out)
    }
    fn apply(map: &mut BTreeMap<String, Source>, changes: Vec<(String, Change)>) {
        let mut done = std::collections::HashSet::with_capacity(changes.len());
        for (key, change) in changes {
            match change {
                Change::Set(v) => { map.insert(key.clone(), v); }
                Change::Into(sub) => match map.get_mut(&key) {
                    Some(Source::Object(child)) => apply(child, sub),
                    _ => unreachable!("partial maps are built for objects"),
                },
            }
            done.insert(key);
        }
        for (key, v) in map.iter_mut() {
            if !done.contains(key) { normalize(v); }
        }
    }
    let changes = plan(val, made)?;
    let Source::Object(map) = context else { unreachable!("a partial ctx is built from an object") };
    apply(map, changes);
    Ok(())
}

/// A source subtree as converting it to a script value and back leaves it
/// (`to_source(&from_source(v))`), changing only what that changes.
fn normalize(v: &mut Source) {
    let round_trip = |v: &mut Source| { if let Ok(n) = to_source(&from_source(v)) { *v = n; } };
    match v {
        Source::Object(m) if m.contains_key(super::doc::GEO_SHAPE_KEY) => round_trip(v),
        Source::Object(m) => for c in m.values_mut() { normalize(c); },
        Source::Array(a) => for c in a.iter_mut() { normalize(c); },
        Source::Scalar(raw) => if !stable_scalar(raw.get()) { round_trip(v) },
    }
}

/// A scalar the round trip keeps as written: literals, strings without
/// escapes, integers in their canonical `i64` form, and doubles written as
/// they format.
fn stable_scalar(t: &str) -> bool {
    match t.as_bytes().first() {
        Some(b'"') => !t.contains('\\'),
        Some(b't' | b'f' | b'n') => matches!(t, "true" | "false" | "null"),
        Some(_) => {
            let digits = t.strip_prefix('-').unwrap_or(t);
            if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
                // An integer stays one while it fits `i64` (wider ones become doubles).
                (digits == "0" || !digits.starts_with('0')) && t != "-0" && t.parse::<i64>().is_ok()
            } else {
                // A double stays as written when written as it formats.
                // (Formatted into a stack buffer: no allocation per value.)
                serde_json::from_str::<f64>(t).ok().is_some_and(|d| {
                    let mut buf = [0u8; 40];
                    let mut out = std::io::Cursor::new(&mut buf[..]);
                    serde_json::to_writer(&mut out, &d).is_ok() && &out.get_ref()[..out.position() as usize] == t.as_bytes()
                })
            }
        }
        None => false,
    }
}

/// A map key as a source field name (a string key is its own text).
fn key_string(k: &Key) -> Result<String> {
    match k {
        Key::Str(s) => Ok(s.to_string()),
        _ => String::from_utf16(&k.to_val().string_units()).map_err(|_| EsError::illegal_argument("Broken surrogate pair in field name")),
    }
}

/// A UTF-16 string as the source scalar `"\uXXXX..."` (every unit escaped,
/// lowercase hex: the form a script's strings have always been written
/// back in), without formatting machinery or a re-parse of the literal.
fn utf16_scalar(units: &[u16]) -> Result<Source> {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut raw = Vec::with_capacity(units.len() * 6 + 2);
    raw.push(b'"');
    for &u in units {
        raw.extend_from_slice(&[b'\\', b'u', HEX[(u >> 12) as usize & 15], HEX[(u >> 8) as usize & 15], HEX[(u >> 4) as usize & 15], HEX[u as usize & 15]]);
    }
    raw.push(b'"');
    let raw = String::from_utf8(raw).expect("ASCII escapes");
    let raw = serde_json::value::RawValue::from_string(raw).map_err(|e| EsError::illegal_argument(e.to_string()))?;
    Ok(Source::Scalar(raw))
}

#[cfg(test)]
mod partial_ctx_tests {
    use super::*;
    use crate::script::ctx_access::CtxPaths;

    fn paths(spec: &[&str]) -> CtxPaths {
        let mut p = CtxPaths::default();
        for path in spec {
            let mut node = &mut p;
            for k in path.split('.') { node = node.children.entry(k.to_string()).or_default(); }
            node.full = true;
        }
        p
    }

    /// Changing a partial ctx and merging it back gives what converting
    /// all of ctx, changing it the same way and converting back gives.
    #[test]
    fn a_partial_ctx_merges_back_as_the_whole_conversion() {
        let doc = r#"{"a":{"b":{"c":"old","d":[1,2.50,"x\u00e9",{"e":1e2}]},"f":-0,"g":18446744073709551616,"h":0.1},
            "k":{"l":true,"m":null,"n":"plain","o":-12,"p":1.5,"q":"\ud800"},"r":[],"s":{}}"#;
        let edits: &[(&[&str], &[&str], fn(&Val))] = &[
            (&["a.b.c"], &["a", "b"], |m| { let Val::Map(m) = m else { panic!() }; m.borrow_mut().insert(Key::str("c"), Val::str("new")); }),
            (&["a.b.c", "k.n"], &["a", "b"], |m| { let Val::Map(m) = m else { panic!() }; m.borrow_mut().insert(Key::str("z"), Val::Int(7)); }),
            (&["a.b"], &["a"], |m| { let Val::Map(m) = m else { panic!() }; m.borrow_mut().insert(Key::str("b"), Val::Long(1 << 40)); }),
            (&["k"], &[], |m| { let Val::Map(m) = m else { panic!() }; m.borrow_mut().insert(Key::str("new"), Val::Double(2.0)); }),
        ];
        for (spec, at, edit) in edits {
            let src = Source::parse(doc).unwrap();
            // The whole conversion.
            let whole = from_source(&src);
            let mut node = whole.clone();
            for k in *at { node = match &node { Val::Map(m) => m.borrow().get(&Key::str(k)).cloned().unwrap(), _ => panic!() }; }
            edit(&node);
            let want = to_source(&whole).unwrap();
            // The partial one.
            let Source::Object(map) = &src else { panic!() };
            let (val, made) = partial_ctx(map, &paths(spec));
            let mut node = val.clone();
            for k in *at { node = match &node { Val::Map(m) => m.borrow().get(&Key::str(k)).cloned().unwrap(), _ => panic!() }; }
            edit(&node);
            let mut got = src.clone();
            merge_partial(&mut got, &val, &made).unwrap();
            assert_eq!(got.text(), want.text(), "{spec:?}");
        }
    }

    #[test]
    fn stable_scalars_are_the_round_trip_fixed_points() {
        for t in ["true", "false", "null", "\"x\"", "\"\\n\"", "0", "-0", "7", "-12", "9223372036854775807", "9223372036854775808",
            "1.5", "2.50", "1e2", "-0.0", "0.1", "1.0E-7", "1e-7", "123456789.123", "\"\\ud800\""] {
            let src = Source::parse(t).unwrap();
            let round = to_source(&from_source(&src)).unwrap();
            if stable_scalar(t) { assert_eq!(round.text(), t, "{t} is stable"); }
        }
        assert!(stable_scalar("1.5") && stable_scalar("-12") && stable_scalar("\"x\""));
        assert!(!stable_scalar("2.50") && !stable_scalar("-0") && !stable_scalar("1e2"));
    }
}

#[cfg(test)]
mod utf16_scalar_tests {
    use super::*;

    /// The same scalar as the escaped literal parsed as a source.
    #[test]
    fn utf16_scalars_match_the_parsed_escaped_literal() {
        for s in ["", "value-123", "héllo ☃", "\u{1F600} pair", "quote \" back\\slash"] {
            let units: Vec<u16> = s.encode_utf16().collect();
            let mut raw = String::from("\"");
            for u in &units { raw.push_str(&format!("\\u{u:04x}")); }
            raw.push('"');
            let want = Source::parse(&raw).unwrap();
            let (Source::Scalar(a), Source::Scalar(b)) = (utf16_scalar(&units).unwrap(), want) else { panic!("scalar") };
            assert_eq!(a.get(), b.get(), "{s}");
        }
        // Scalars and keys convert as the generic JSON route does.
        let doc = r#"{"a":"plain","b":"esc\"aped \u00e9","c":12,"d":-1.5e3,"e":true,"f":null,"g":[1,"x",{"h":"y"}],"k\u00e9y":{"z":9007199254740993}}"#;
        let src = Source::parse(doc).unwrap();
        let val = from_source(&src);
        assert_eq!(val.to_json(), serde_json::from_str::<serde_json::Value>(doc).unwrap());
        let back = to_source(&val).unwrap();
        let slow = Source::from_value(&val.to_json());
        assert_eq!(back.text(), slow.text());
        assert!(back.equivalent(&src));
        // A lone surrogate is escaped as written too.
        let Source::Scalar(lone) = utf16_scalar(&[0xd800, 0x61]).unwrap() else { panic!() };
        assert_eq!(lone.get(), "\"\\ud800\\u0061\"");
    }
}

/// Java UTF8 replacement decoding consumes the valid prefix of a malformed
/// sequence as one replacement, including a three-byte surrogate encoding.
pub(crate) fn decode_java_utf8(bytes:&[u8])->Vec<u16> {
    let mut out=Vec::new();let mut i=0;
    while i<bytes.len() {
        let lead=bytes[i];
        if lead<0x80 {out.push(lead as u16);i+=1;continue;}
        let count=match lead {0xc2..=0xdf=>2,0xe0..=0xef=>3,0xf0..=0xf4=>4,_=>{out.push(0xfffd);i+=1;continue;}};
        let mut used=1;let mut cp=(lead & (0x7f >> count)) as u32;
        while used<count && i+used<bytes.len() {
            let next=bytes[i+used];
            if !(0x80..=0xbf).contains(&next) || (used==1 && ((lead==0xe0 && next<0xa0)||(lead==0xf0 && next<0x90)||(lead==0xf4 && next>0x8f))) {break;}
            cp=(cp<<6)|(next as u32&63);used+=1;
        }
        if used!=count || (0xd800..=0xdfff).contains(&cp) {out.push(0xfffd);}
        else if cp<=0xffff {out.push(cp as u16);}
        else {let n=cp-0x10000;out.extend([0xd800+(n>>10) as u16,0xdc00+(n&1023) as u16]);}
        i+=used;
    }
    out
}

fn simple_fold(cp: u32) -> u32 {
    let table = super::simple_case::SIMPLE_CASE_FOLD;
    table.binary_search_by_key(&cp, |p| p.0).map(|i| table[i].1).unwrap_or(cp)
}

fn compare_case(a: &[u16], b: &[u16]) -> i32 {
    fn including(s: &[u16], i: usize) -> (u32, usize) {
        let u=s[i];
        if (0xd800..=0xdbff).contains(&u) && s.get(i+1).is_some_and(|v|(0xdc00..=0xdfff).contains(v)) {
            (0x10000+((u as u32-0xd800)<<10)+(s[i+1] as u32-0xdc00),1)
        } else if (0xdc00..=0xdfff).contains(&u) && i>0 && (0xd800..=0xdbff).contains(&s[i-1]) {
            (0x10000+((s[i-1] as u32-0xd800)<<10)+(u as u32-0xdc00),0)
        } else {(u as u32,0)}
    }
    let (mut i,mut j)=(0,0);
    while i<a.len() && j<b.len() {
        // Equal leading units remain equal even if one string ends inside
        // a pair. Resolve surrounding code points only at a difference.
        if a[i]!=b[j] && simple_fold(a[i] as u32)!=simple_fold(b[j] as u32) {
            let (ca,advance_a)=including(a,i); let (cb,advance_b)=including(b,j);
            let diff=simple_fold(ca) as i32-simple_fold(cb) as i32;
            if diff!=0 {return diff;}
            i+=advance_a;j+=advance_b;
        }
        i+=1;j+=1;
    }
    a.len() as i32-b.len() as i32
}

pub fn string_method(recv:&Val, units:&[u16], name:&str, args:&[Val]) -> super::error::SResult<Val> {
    use super::error::illegal_arg;
    let _argument_copies=args.iter().map(super::allocation::Allocation::iterable).collect::<super::error::SResult<Vec<_>>>()?;
    super::allocation::Scope::collect_dead();
    let output=super::allocation::Allocation::new();
    let retain_text=|units:u64| output.reserve(units.saturating_mul(16).saturating_add(256));
    let string_arg=|n:usize,sequence:bool|->super::error::SResult<Vec<u16>> {
        match &args[n] {
            Val::Str(_)|Val::Utf16(_)=>Ok(args[n].string_units()),
            Val::Builder(_) if sequence=>Ok(args[n].string_units()),
            Val::Null=>Err(super::error::npe("argument is null")),
            _=>Err(super::error::class_cast("Expected string argument")),
        }
    };
    let index=|n:usize|->super::error::SResult<usize>{
        let i=args[n].to_i64()?;
        if i<0 || i as usize>units.len(){Err(illegal_arg("String index out of range"))} else {Ok(i as usize)}
    };
    let result=match (name,args.len()) {
        ("charAt",1)=>{
            let i=args[0].to_i64()?;
            let unit=usize::try_from(i).ok().and_then(|i|units.get(i)).ok_or_else(||super::error::ScriptErr::new("string_index_out_of_bounds_exception",format!("Index {i} out of bounds for length {}",units.len())))?;
            Val::Char(*unit)
        },
        ("codePointAt",1)|("codePointBefore",1)=>{
            let i=args[0].to_i64()?;
            let before=name=="codePointBefore";
            if i<0 || (!before && i as usize>=units.len()) || (before && (i==0 || i as usize>units.len())) {
                return Err(super::error::ScriptErr::new("string_index_out_of_bounds_exception","String index out of range"));
            }
            let i=i as usize;
            let at=if before{i-1}else{i};
            let u=units[at];
            let pair=if before && (0xdc00..=0xdfff).contains(&u) && at>0 && (0xd800..=0xdbff).contains(&units[at-1]) {
                Some((units[at-1],u))
            }else if !before && (0xd800..=0xdbff).contains(&u) && units.get(at+1).is_some_and(|v|(0xdc00..=0xdfff).contains(v)) {
                Some((u,units[at+1]))
            }else{None};
            Val::Int(pair.map(|(hi,lo)|0x10000+(((hi as i32)-0xd800)<<10)+(lo as i32)-0xdc00).unwrap_or(u as i32))
        }
        ("offsetByCodePoints",2)=>{
            let mut i=index(0)?;let offset=args[1].to_i64()?;
            if offset>=0 {for _ in 0..offset {if i==units.len(){return Err(illegal_arg("String index out of range"));}let pair=(0xd800..=0xdbff).contains(&units[i]) && units.get(i+1).is_some_and(|u|(0xdc00..=0xdfff).contains(u));i+=if pair{2}else{1};}}
            else {for _ in offset..0 {if i==0{return Err(illegal_arg("String index out of range"));}i-=1;if (0xdc00..=0xdfff).contains(&units[i]) && i>0 && (0xd800..=0xdbff).contains(&units[i-1]) {i-=1;}}}
            Val::Int(i as i32)
        }
        ("toLowerCase",0|1)|("toUpperCase",0|1)=>{retain_text((units.len() as u64).saturating_mul(3))?;super::locale_case::map(units,name=="toUpperCase",args)?},
        ("codePointCount",2)=>{
            let a=index(0)?;let b=index(1)?;
            if a>b{return Err(illegal_arg("String index out of range"));}
            let mut n=0;let mut i=a;
            while i<b {let pair=(0xd800..=0xdbff).contains(&units[i]) && i+1<b && (0xdc00..=0xdfff).contains(&units[i+1]);i+=if pair{2}else{1};n+=1;}
            Val::Int(n)
        }
        ("trim",0)|("strip",0)|("stripLeading",0)|("stripTrailing",0)|("isBlank",0)=>{
            // Character.isWhitespace excludes non-breaking spaces and NEL.
            let white=|u:u16| if name=="trim" {u<=0x20} else {matches!(u,0x09..=0x0d|0x1c..=0x20|0x1680|0x2000..=0x2006|0x2008..=0x200a|0x2028|0x2029|0x205f|0x3000)};
            if name=="isBlank" {Val::Bool(units.iter().all(|u|white(*u)))} else {
                let mut a=0;let mut b=units.len();
                if name!="stripTrailing" {while a<b && white(units[a]){a+=1;}}
                if name!="stripLeading" {while b>a && white(units[b-1]){b-=1;}}
                {retain_text((b-a) as u64)?;Val::utf16(units[a..b].to_vec())}
            }
        }
        ("replace",2)=>{
            let (old,new)=if matches!((&args[0],&args[1]),(Val::Char(_),Val::Char(_))) {(args[0].string_units(),args[1].string_units())}else{(string_arg(0,true)?,string_arg(1,true)?)};retain_text((units.len() as u64).saturating_add((units.len() as u64).saturating_add(1).saturating_mul(new.len() as u64)))?;let mut out=Vec::new();let mut i=0;
            if old.is_empty(){out.extend_from_slice(&new);for u in units {out.push(*u);out.extend_from_slice(&new);}}
            else {while i<units.len(){if units[i..].starts_with(&old){out.extend_from_slice(&new);i+=old.len();}else{out.push(units[i]);i+=1;}}}
            Val::utf16(out)
        }
        ("equalsIgnoreCase",1)=>{
            if args[0].is_null() { Val::Bool(false) } else {
                let other=string_arg(0,false)?;
                Val::Bool(units.len()==other.len() && compare_case(units,&other)==0)
            }
        }
        ("compareToIgnoreCase",1)=>{
            let other=string_arg(0,false)?;
            Val::Int(compare_case(units,&other))
        }
        ("regionMatches",4|5)=>{
            let shift=usize::from(args.len()==5);
            let ignore=shift==1 && args[0].as_bool()?;
            let offset=args[shift].to_i64()?;
            let other=string_arg(shift+1,false)?;
            let other_offset=args[shift+2].to_i64()?;
            let count=args[shift+3].to_i64()?;
            if offset<0 || other_offset<0 || offset as i128>units.len() as i128-count as i128 || other_offset as i128>other.len() as i128-count as i128 {
                Val::Bool(false)
            } else if count<=0 { Val::Bool(true) } else {
                let a=&units[offset as usize..(offset+count) as usize];
                let b=&other[other_offset as usize..(other_offset+count) as usize];
                Val::Bool(if ignore { compare_case(a,b)==0 } else { a==b })
            }
        }
        ("compareTo",1)=>{
            let other=string_arg(0,false)?;
            Val::Int(units.iter().zip(&other).find_map(|(a,b)|(a!=b).then_some(*a as i32-*b as i32)).unwrap_or(units.len() as i32-other.len() as i32))
        }
        ("repeat",1)=>{
            let n=args[0].to_i64()?;if n<0{return Err(illegal_arg("count is negative"));}
            if units.is_empty() || n==0 {return Ok(Val::utf16(Vec::new()));}
            let n=usize::try_from(n).map_err(|_|illegal_arg("repeat count too large"))?;
            let len=units.len().checked_mul(n).ok_or_else(||illegal_arg("repeat count too large"))?;
            retain_text(len as u64)?;
            let mut out=Vec::new();out.try_reserve_exact(len).map_err(|_|illegal_arg("repeat count too large"))?;
            for _ in 0..n {out.extend_from_slice(units);}Val::utf16(out)
        }
        ("encodeBase64"|"sha1"|"sha256"|"sha512",0)=>{
            // Java's UTF8 encoder replaces each malformed UTF16 unit with '?'.
            // This is an explicit encoding boundary, not source serialization.
            let mut bytes=Vec::new();
            for ch in char::decode_utf16(units.iter().copied()) {
                match ch {
                    Ok(ch)=>{let mut buf=[0u8;4];bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());}
                    Err(_)=>bytes.push(b'?'),
                }
            }
            use base64::Engine;
            use sha2::Digest;
            Val::string(match name {
                "encodeBase64"=>base64::engine::general_purpose::STANDARD.encode(&bytes),
                "sha1"=>super::methods::hex(&super::methods::sha1(&bytes)),
                "sha256"=>super::methods::hex(&sha2::Sha256::digest(&bytes)),
                _=>super::methods::hex(&sha2::Sha512::digest(&bytes)),
            })
        }
        ("decodeBase64",0)=>{
            // Base64's alphabet is ASCII; a lone surrogate cannot be a digit.
            let text=String::from_utf16(units).map_err(|_|illegal_arg("Illegal base64 character"))?;
            use base64::Engine;
            let bytes=base64::engine::general_purpose::STANDARD.decode(text.as_bytes()).map_err(|e|illegal_arg(e.to_string()))?;
            Val::utf16(decode_java_utf8(&bytes))
        }
        ("chars",0)=>Val::stream_from_iter(units.iter().map(|u|Ok(Val::Int(*u as i32))))?,
        ("codePoints",0)=>Val::stream_from_iter(char::decode_utf16(units.iter().copied()).map(|c|Ok(Val::Int(match c {Ok(c)=>c as i32,Err(e)=>e.unpaired_surrogate() as i32}))))?,
        ("toCharArray",0)=>{
            let allocation=super::allocation::Allocation::values(units.len())?;
            Val::new_array(super::ast::Ty::Char,units.iter().copied().map(Val::Char).collect()).allocated(allocation)
        },
        ("length",0)=>Val::Int(units.len() as i32),
        ("isEmpty",0)=>Val::Bool(units.is_empty()),
        ("toString",0)=>recv.clone(),
        ("hashCode",0)=>Val::Int(units.iter().fold(0i32,|h,c|h.wrapping_mul(31).wrapping_add(*c as i32))),
        ("contentEquals",1)=>Val::Bool(units==string_arg(0,true)?),
        ("getChars",4)=>{
            let a=index(0)?;let b=index(1)?;
            if a>b {return Err(illegal_arg("String index out of range"));}
            let Val::Array{elem: super::ast::Ty::Char,items}=&args[2] else {return Err(if args[2].is_null(){super::error::npe("destination is null")}else{super::error::class_cast("Expected char[]")});};
            let offset=args[3].to_i64()?;
            let mut dest=items.borrow_mut();
            if offset<0 || (offset as usize).checked_add(b-a).is_none_or(|end|end>dest.len()) {return Err(illegal_arg("Array index out of range"));}
            for (i,u) in units[a..b].iter().enumerate() {dest[offset as usize+i]=Val::Char(*u);}
            Val::Null
        }
        ("splitOnToken",1|2)=>{
            let allocation=super::allocation::Allocation::values(units.len().saturating_add(1).saturating_mul(4))?;
            let token=string_arg(0,false)?;
            if token.is_empty(){return Err(illegal_arg("Cannot splitOnToken with an empty token"));}
            let limit=if args.len()==2{args[1].to_i64()?}else{0};
            let mut parts=Vec::new();let mut start=0;let mut i=0;
            while i+token.len()<=units.len() && (limit<=0 || parts.len()+1<limit as usize) {
                if units[i..].starts_with(&token) {parts.push(Val::utf16(units[start..i].to_vec()));i+=token.len();start=i;} else {i+=1;}
            }
            parts.push(Val::utf16(units[start..].to_vec()));
            Val::new_array(super::ast::Ty::String,parts).allocated(allocation)
        }
        ("equals",1)=>Val::Bool(matches!(args[0],Val::Str(_)|Val::Utf16(_)) && units==args[0].string_units()),
        ("substring",1|2)|("subSequence",2)=>{
            let a=index(0)?;let b=if args.len()==2{index(1)?}else{units.len()};
            if a>b{return Err(illegal_arg("String index out of range"));}
            {retain_text((b-a) as u64)?;Val::utf16(units[a..b].to_vec())}
        }
        ("concat",1)=>{let other=string_arg(0,false)?;retain_text((units.len() as u64).saturating_add(other.len() as u64))?;let mut s=units.to_vec();s.extend(other);Val::utf16(s)}
        ("contains",1)=>{let n=string_arg(0,true)?;Val::Bool(n.is_empty() || units.windows(n.len()).any(|w|w==n))}
        ("startsWith",1|2)=>{
            let prefix=string_arg(0,false)?;
            let offset=if args.len()==2 {args[1].to_i64()?} else {0};
            Val::Bool(offset>=0 && (offset as usize)<=units.len()
                && units[offset as usize..].starts_with(&prefix))
        },
        ("endsWith",1)=>Val::Bool(units.ends_with(&string_arg(0,false)?)),
        ("indexOf"|"lastIndexOf",1|2)=>{
            let needle=string_arg(0,false)?;
            let reverse=name=="lastIndexOf";
            let from=if args.len()==2 {args[1].to_i64()?} else if reverse {units.len() as i64} else {0};
            let found=if reverse {
                if from<0 || needle.len()>units.len() {None} else {
                    let end=(from as usize).min(units.len()-needle.len());
                    (0..=end).rev().find(|&i|units[i..].starts_with(&needle))
                }
            } else {
                let start=(from.max(0) as usize).min(units.len());
                if needle.is_empty(){Some(start)}else if from>units.len() as i64 {None}else{
                    (start..units.len()).find(|&i|units[i..].starts_with(&needle))
                }
            };
            Val::Int(found.map(|i|i as i32).unwrap_or(-1))
        }
        _=>return Err(illegal_arg(format!("Unsupported UTF-16 string method [{name}/{}]",args.len()))),
    };
    super::allocation::Scope::adopt(&result)?;
    Ok(result)

}

/// Retain literal code units separately from the Unicode diagnostic display.
/// The table is addressed by tokenizer positions, not by matching placeholder
/// text, and this program is not inserted into the ordinary string cache.
pub fn compile_raw_literal_source(units:&[u16], ctx:super::ScriptContext) -> Result<std::sync::Arc<super::runtime::Program>> {
    if let Ok(text) = String::from_utf16(units) { return super::program(&text, ctx); }
    // The exact units distinguish raw strings from each other and from any
    // valid Unicode source. Use the same admission/cache policy as Painless.
    let key = (format!("\0utf16\0{units:?}"), ctx);
    super::cached(key, || compile_raw_uncached(units, ctx))
}
fn compile_raw_uncached(units:&[u16], ctx:super::ScriptContext) -> Result<std::sync::Arc<super::runtime::Program>> {
    let mut display=String::new();let mut positions=std::collections::HashMap::new();let mut i=0;
    while i<units.len() {
        positions.insert(display.len(),i);
        let u=units[i];
        if (0xd800..=0xdbff).contains(&u) && units.get(i+1).is_some_and(|v|(0xdc00..=0xdfff).contains(v)) {
            let cp=0x10000+(((u as u32)-0xd800)<<10)+units[i+1] as u32-0xdc00;
            display.push(char::from_u32(cp).unwrap());i+=2;
        }else{display.push(char::from_u32(u as u32).unwrap_or('�'));i+=1;}
    }
    let tokens=super::lexer::lex(&display).map_err(|e|EsError::illegal_argument(format!("{e:?}")))?;
    let mut literals=std::collections::HashMap::new();
    for token in tokens {
        let regex=matches!(token.tok,super::lexer::Tok::Regex(..));
        if !regex && !matches!(token.tok,super::lexer::Tok::Str(_)){continue;}
        let start=positions[&token.pos];let quote=units[start];let mut i=start+1;let mut literal=Vec::new();
        while i<units.len() && units[i]!=quote {
            if units[i]==92 && i+1<units.len() {if regex{literal.push(92);}i+=1;literal.push(if regex{units[i]}else{match units[i] {110=>10,116=>9,114=>13,u=>u}});}
            else{literal.push(units[i]);}i+=1;
        }
        if String::from_utf16(&literal).is_err(){literals.insert(token.pos,literal);}
    }
    super::compile::compile_program_literals(&display,ctx,&literals).map(std::sync::Arc::new).map_err(|e|EsError::illegal_argument(format!("{e:?}")))
}

#[cfg(test)]
mod opaque_source_tests {
    use super::*;
    #[test]
    fn lossless_bridge_keeps_opaque_geo_shape_serialization() {
        let shape=Val::new_map(vec![(Key::str(super::super::doc::GEO_SHAPE_KEY),Val::Bool(true))].into_iter().collect());
        let value=Val::new_list(vec![shape,Val::utf16(vec![0xd800])]);
        let actual=to_source(&value).unwrap();
        assert!(actual.equivalent(&Source::parse(r#"["<unserializable>","\ud800"]"#).unwrap()));
    }
}
