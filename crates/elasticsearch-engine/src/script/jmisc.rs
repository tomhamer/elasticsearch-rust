//! Smaller library objects: `UUID`, `StringJoiner`, `Iterator`.

use super::error::{illegal_arg, illegal_state, unsupported, SResult, ScriptErr};
use super::runtime::Rt;
use super::value::{List, Val};
use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::fmt::Write as _;
use std::rc::Rc;

pub enum Obj {
    Locale { language: Vec<u16>, country: Vec<u16>, variant: Vec<u16> },
    Uuid(u64, u64),
    Random(RefCell<rustic_core::vector::int4::JavaRandom>),
    Joiner { sep: Vec<u16>, prefix: Vec<u16>, suffix: Vec<u16>, parts: RefCell<Vec<Vec<u16>>>, empty: RefCell<Option<Vec<u16>>>, allocation: super::allocation::Allocation },
    /// `Iterator` over a list/set snapshot; `remove()` edits `src`.
    Iter { items: Vec<Val>, pos: Cell<usize>, src: Option<List>, allocation: super::allocation::Allocation },
    /// A `Comparator` object (e.g. `Comparator.comparing(...)`); `compare(a, b)` via the closure.
    Cmp(Rc<RefCell<dyn FnMut(&mut Rt, &Val, &Val) -> SResult<i64>>>, &'static str),
    /// Fields API `field('name')`: the doc's values for one field.
    Field { name: String, class: &'static str, values: Vec<Val> },
    /// A `Collectors.*(..)` collector (evaluated by `methods::collector`).
    Collector { kind: String, args: Vec<Val> },
    /// x-pack `version` field value (`new Version('1.2.3')`), ordered like the field.
    Version(String),
    /// A doc's `dense_vector` value (fields API `DenseVector`).
    DenseVector(Vec<f32>),
}

impl std::fmt::Debug for Obj {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple(self.type_name()).finish()
    }
}

impl Obj {
    pub fn type_name(&self) -> &'static str {
        match self {
            Obj::Locale {..} => "java.util.Locale",
            Obj::Uuid(..) => "java.util.UUID",
            Obj::Random(_) => "java.util.Random",
            Obj::Joiner { .. } => "java.util.StringJoiner",
            Obj::Iter { .. } => "java.util.Iterator",
            Obj::Cmp(_, name) => name,
            Obj::Field { class, .. } => class,
            Obj::Collector { .. } => "java.util.stream.Collectors$CollectorImpl",
            Obj::Version(_) => "org.elasticsearch.xpack.versionfield.Version",
            Obj::DenseVector(_) => "org.elasticsearch.script.field.vectors.FloatDenseVector",
        }
    }
    /// Items when the object is iterable (`for (x : field('f'))`).
    pub fn iter_items(&self) -> Option<Vec<Val>> {
        match self {
            Obj::Field { values, .. } => Some(values.clone()),
            _ => None,
        }
    }
    pub fn string_units(&self) -> Option<Vec<u16>> {
        if let Obj::Locale{language,country,variant}=self {
            let mut out=language.clone();
            if !country.is_empty() || !variant.is_empty() {out.push(95);out.extend_from_slice(country);}
            if !variant.is_empty() {out.push(95);out.extend_from_slice(variant);}
            return Some(out);
        }
        let Obj::Joiner { sep, prefix, suffix, parts, empty, .. } = self else { return None; };
        let parts = parts.borrow();
        if parts.is_empty() {
            if let Some(value) = &*empty.borrow() { return Some(value.clone()); }
        }
        let mut out = prefix.clone();
        for (i, part) in parts.iter().enumerate() {
            if i != 0 { out.extend_from_slice(sep); }
            out.extend_from_slice(part);
        }
        out.extend_from_slice(suffix);
        Some(out)
    }
    pub fn write_java(&self, out: &mut String) {
        match self {
            Obj::Locale {..} => out.push_str(&String::from_utf16_lossy(&self.string_units().unwrap())),
            Obj::Uuid(hi, lo) => {
                let _ = write!(out, "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}", hi >> 32, (hi >> 16) & 0xffff, hi & 0xffff, lo >> 48, lo & 0xffff_ffff_ffff);
            }
            Obj::Joiner { .. } => out.push_str(&String::from_utf16_lossy(&self.string_units().unwrap())),
            Obj::Random(_) => out.push_str("java.util.Random"),
            Obj::Iter { .. } => out.push_str("java.util.Iterator"),
            Obj::Cmp(_, name) => out.push_str(name),
            Obj::Field { class, .. } => out.push_str(class),
            Obj::Collector { .. } => out.push_str("java.util.stream.Collectors$CollectorImpl"),
            Obj::Version(v) => out.push_str(v),
            Obj::DenseVector(_) => out.push_str("org.elasticsearch.script.field.vectors.FloatDenseVector"),
        }
    }
    pub fn hash(&self) -> i32 {
        match self {
            Obj::Uuid(hi, lo) => {
                let x = hi ^ lo;
                ((x >> 32) as i32) ^ (x as i32)
            }
            _ => 0,
        }
    }
    pub fn equals(&self, o: &Obj) -> bool {
        match (self, o) {
            (Obj::Uuid(a, b), Obj::Uuid(c, d)) => a == c && b == d,
            (Obj::Version(a), Obj::Version(b)) => a == b,
            (Obj::Locale{language:a,country:b,variant:c},Obj::Locale{language:x,country:y,variant:z})=>a==x && b==y && c==z,
            _ => std::ptr::eq(self, o),
        }
    }
    pub fn compare(&self, o: &Obj) -> SResult<Ordering> {
        match (self, o) {
            (Obj::Uuid(a, b), Obj::Uuid(c, d)) => Ok(((*a as i64), (*b as i64)).cmp(&((*c as i64), (*d as i64)))),
            (Obj::Version(a), Obj::Version(b)) => Ok(version_cmp(a, b)),
            _ => Err(super::error::class_cast(format!("Cannot cast {} to java.lang.Comparable", self.type_name()))),
        }
    }
    /// Invoke a `Comparator` object: `compare(a, b)` as an int.
    pub fn cmp_call(&self, rt: &mut Rt, a: &Val, b: &Val) -> SResult<i64> {
        match self {
            Obj::Cmp(f, _) => f.borrow_mut()(rt, a, b),
            _ => Err(super::error::class_cast(format!("Cannot cast {} to java.util.Comparator", self.type_name()))),
        }
    }
}

/// `Collections.reverseOrder()` / `reverseOrder(cmp)`.
pub fn reverse_order(inner: Option<Val>) -> Val {
    let name = if inner.is_some() { "java.util.Collections$ReverseComparator2" } else { "java.util.Collections$ReverseComparator" };
    Val::Obj(Rc::new(Obj::Cmp(
        Rc::new(RefCell::new(move |rt: &mut Rt, a: &Val, b: &Val| match &inner {
            None => Ok(match b.compare(a)? {
                Ordering::Less => -1,
                Ordering::Equal => 0,
                Ordering::Greater => 1,
            }),
            Some(f) => rt.call_fn(f, &[b.clone(), a.clone()])?.to_i64(),
        })),
        name,
    )))
}

/// `Comparator.comparing(keyExtractor[, keyComparator])` (and the
/// `comparingInt/Long/Double` variants): compares the extracted keys.
pub fn comparing(key: Val, key_cmp: Option<Val>) -> SResult<Val> {
    if key.is_null() {
        return Err(super::error::npe(""));
    }
    if matches!(key_cmp, Some(Val::Null)) {
        return Err(super::error::npe(""));
    }
    Ok(Val::Obj(Rc::new(Obj::Cmp(
        Rc::new(RefCell::new(move |rt: &mut Rt, a: &Val, b: &Val| {
            let ka = rt.call_fn(&key, &[a.clone()])?;
            let kb = rt.call_fn(&key, &[b.clone()])?;
            match &key_cmp {
                Some(c) => rt.call_fn(c, &[ka, kb])?.to_i64(),
                None => Ok(match ka.compare(&kb)? {
                    Ordering::Less => -1,
                    Ordering::Equal => 0,
                    Ordering::Greater => 1,
                }),
            }
        })),
        "java.util.Comparator$$Lambda",
    ))))
}

/// `Comparator.naturalOrder()`.
pub fn natural_order() -> Val {
    Val::Obj(Rc::new(Obj::Cmp(
        Rc::new(RefCell::new(|_: &mut Rt, a: &Val, b: &Val| {
            Ok(match a.compare(b)? {
                Ordering::Less => -1,
                Ordering::Equal => 0,
                Ordering::Greater => 1,
            })
        })),
        "java.util.Comparators$NaturalOrderComparator",
    )))
}

/// A `Field` object for the fields API.
pub fn new_field(name: String, kind: super::doc::Kind, values: Vec<Val>) -> Val {
    use super::doc::Kind;
    let class = match kind {
        Kind::Long => "org.elasticsearch.script.field.LongDocValuesField",
        Kind::Unsigned => "org.elasticsearch.script.field.UnsignedLongDocValuesField",
        Kind::Double => "org.elasticsearch.script.field.DoubleDocValuesField",
        Kind::Date => "org.elasticsearch.script.field.DateMillisDocValuesField",
        Kind::Bool => "org.elasticsearch.script.field.BooleanDocValuesField",
        Kind::Keyword => "org.elasticsearch.script.field.KeywordDocValuesField",
        Kind::Ip => "org.elasticsearch.script.field.IpDocValuesField",
        Kind::GeoPoint => "org.elasticsearch.script.field.GeoPointDocValuesField",
        Kind::Unmapped => "org.elasticsearch.script.field.EmptyField",
        _ => "org.elasticsearch.script.field.DocValuesField",
    };
    Val::Obj(Rc::new(Obj::Field { name, class, values }))
}

/// Wrap a two-arg lambda/method-ref as a `Comparator` object.
pub fn new_cmp(f: Val, name: &'static str) -> SResult<Val> {
    let Val::Lambda(l) = f else {
        return Err(super::error::class_cast(format!("Cannot cast {} to java.util.Comparator", f.type_name())));
    };
    Ok(Val::Obj(Rc::new(Obj::Cmp(Rc::new(RefCell::new(move |rt: &mut Rt, a: &Val, b: &Val| {
        rt.call_fn(&Val::Lambda(l), &[a.clone(), b.clone()]).map(|v| v.to_i64().unwrap_or(0))
    })), name))))
}

pub fn new_iter(values: impl IntoIterator<Item = Val>, src: Option<List>) -> SResult<Val> {
    let allocation = super::allocation::Allocation::values(0)?;
    let mut items = Vec::new();
    for value in values { allocation.push(&mut items, value)?; }
    Ok(Val::Obj(Rc::new(Obj::Iter { items, pos: Cell::new(0), src, allocation })))
}

fn random_uuid() -> Val {
    let mut b = [0u8; 16];
    let _ = getrandom::getrandom(&mut b);
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    let hi = u64::from_be_bytes(b[..8].try_into().unwrap());
    let lo = u64::from_be_bytes(b[8..].try_into().unwrap());
    Val::Obj(Rc::new(Obj::Uuid(hi, lo)))
}

fn parse_uuid(s: &str) -> SResult<Val> {
    let bad = || illegal_arg(format!("Invalid UUID string: {s}"));
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 5 || s.len() > 36 {
        return Err(bad());
    }
    let p = |i: usize| u64::from_str_radix(parts[i], 16).map_err(|_| bad());
    let hi = (p(0)? << 32) | ((p(1)? & 0xffff) << 16) | (p(2)? & 0xffff);
    let lo = ((p(3)? & 0xffff) << 48) | (p(4)? & 0xffff_ffff_ffff);
    Ok(Val::Obj(Rc::new(Obj::Uuid(hi, lo))))
}

pub const CLASSES: &[&str] = &["UUID", "StringJoiner", "Version", "Random", "Locale"];

/// Semantic-version ordering as the `version` field type sorts (numeric
/// dotted parts, a pre-release suffix sorts before the release).
pub fn version_cmp(a: &str, b: &str) -> Ordering {
    let split = |s: &str| -> (Vec<u64>, Option<String>) {
        let (main, pre) = s.split_once('-').map(|(m, p)| (m, Some(p.to_string()))).unwrap_or((s, None));
        (main.split('.').map(|p| p.parse::<u64>().unwrap_or(0)).collect(), pre)
    };
    let (am, ap) = split(a);
    let (bm, bp) = split(b);
    let n = am.len().max(bm.len());
    for i in 0..n {
        let o = am.get(i).copied().unwrap_or(0).cmp(&bm.get(i).copied().unwrap_or(0));
        if o != Ordering::Equal {
            return o;
        }
    }
    match (ap, bp) {
        (None, None) => Ordering::Equal,
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (Some(x), Some(y)) => x.cmp(&y),
    }
}

pub fn call_static(_rt: &mut Rt, class: &str, method: &str, args: &[Val]) -> SResult<Option<Val>> {
    Ok(Some(match (class, method, args.len()) {
        ("Locale", "forLanguageTag", 1) => {
            let tag = String::from_utf16(&locale_text(&args[0])?).map_err(|_|illegal_arg("Ill-formed language tag"))?;
            let mut parts=tag.split('-');
            locale(parts.next().unwrap_or(""),parts.next().unwrap_or(""),"")
        }
        ("UUID", "randomUUID", 0) => random_uuid(),
        ("UUID", "fromString", 1) => parse_uuid(&args[0].to_java_string())?,
        ("UUID", "nameUUIDFromBytes", 1) => return Err(unsupported("nameUUIDFromBytes")),
        _ => return Ok(None),
    }))
}

fn reserve_joiner(object:&Obj,extra_units:usize,append:bool)->SResult<()> {
    let Obj::Joiner{sep,prefix,suffix,parts,empty,allocation}=object else {return Ok(())};
    let parts=parts.borrow();
    let slots=parts.capacity().max(parts.len().saturating_add(usize::from(append))).saturating_mul(2).max(4);
    let units=parts.iter().fold(sep.capacity().saturating_add(prefix.capacity()).saturating_add(suffix.capacity()),|n,v|n.saturating_add(v.capacity()))
        .saturating_add(empty.borrow().as_ref().map_or(0,Vec::capacity)).saturating_add(extra_units);
    allocation.reserve((units as u64).saturating_mul(4).saturating_add((slots as u64).saturating_mul(std::mem::size_of::<Vec<u16>>() as u64)).saturating_add(256))
}

pub fn construct(class: &str, args: &[Val]) -> Option<SResult<Val>> {
    if !matches!(class,"Locale"|"StringJoiner"|"Version") {return construct_inner(class,args);}
    Some((|| {
        let allocation=super::allocation::Allocation::new();
        let mut bytes=0u64;
        if allocation.enabled() {for arg in args {bytes=bytes.saturating_add(super::allocation::serialization_bound(arg)?);}}
        allocation.reserve(bytes)?;
        let result=construct_inner(class,args).ok_or_else(||illegal_state("unsupported library constructor"))??;
        super::allocation::Scope::adopt(&result)?;
        Ok(result)
    })())
}
fn construct_inner(class: &str, args: &[Val]) -> Option<SResult<Val>> {
    Some(Ok(match (class, args.len()) {
        ("Locale", 1..=3) => {
            return Some((|| {
                let language=locale_text(&args[0])?;
                let country=if args.len()>1 {locale_text(&args[1])?}else{Vec::new()};
                let variant=if args.len()>2 {locale_text(&args[2])?}else{Vec::new()};
                Ok(locale_units(language,country,variant))
            })());
        }
        ("Random", 0 | 1) => {
            let seed = if args.is_empty() {
                let mut bytes = [0u8; 8];
                if let Err(error) = getrandom::getrandom(&mut bytes) {
                    return Some(Err(illegal_state(format!("Cannot seed Random: {error}"))));
                }
                i64::from_ne_bytes(bytes)
            } else {
                match args[0].to_i64() { Ok(seed) => seed, Err(error) => return Some(Err(error)) }
            };
            Val::Obj(Rc::new(Obj::Random(RefCell::new(rustic_core::vector::int4::JavaRandom::new(seed)))))
        },
        ("StringJoiner", 1 | 3) => return Some((|| {
            let sep=joiner_arg(&args[0])?;
            let (prefix,suffix)=if args.len()==3 {(joiner_arg(&args[1])?,joiner_arg(&args[2])?)} else {(Vec::new(),Vec::new())};
            let allocation=super::allocation::Allocation::new();
            allocation.reserve(((sep.capacity()+prefix.capacity()+suffix.capacity()) as u64).saturating_mul(2).saturating_add(256))?;
            Ok(Val::Obj(Rc::new(Obj::Joiner { sep,prefix,suffix,parts:RefCell::new(Vec::new()),empty:RefCell::new(None),allocation })))
        })()),
        ("Version", 1) => Val::Obj(Rc::new(Obj::Version(args[0].to_java_string()))),
        ("UUID", 2) => match (args[0].to_i64(), args[1].to_i64()) {
            (Ok(a), Ok(b)) => Val::Obj(Rc::new(Obj::Uuid(a as u64, b as u64))),
            (Err(e), _) | (_, Err(e)) => return Some(Err(e)),
        },
        _ => return None,
    }))
}

pub fn call_method(rt: &mut Rt, recv: &Val, name: &str, args: &[Val]) -> SResult<Option<Val>> {
    let Val::Obj(o) = recv else { return Ok(None) };
    let n = args.len();
    Ok(Some(match (&**o, name, n) {
        (Obj::Locale{language,..},"getLanguage",0)=>Val::utf16(language.clone()),
        (Obj::Locale{country,..},"getCountry",0)=>Val::utf16(country.clone()),
        (Obj::Locale{variant,..},"getVariant",0)=>Val::utf16(variant.clone()),
        (_, "toString", 0) => Val::utf16(recv.string_units()),
        (_, "equals", 1) => Val::Bool(recv.equals(&args[0])),
        (_, "hashCode", 0) => Val::Int(recv.java_hash()),
        (Obj::Uuid(..) | Obj::Version(_), "compareTo", 1) => Val::Int(recv.compare(&args[0])? as i32),
        (Obj::Random(random), "nextLong", 0) => {
            let mut random = random.borrow_mut();
            // Java sign-extends BOTH next(32) results before addition.
            let high = (random.next_int() as i64) << 32;
            Val::Long(high.wrapping_add(random.next_int() as i64))
        },
        (Obj::Random(random), "nextInt", 0) => Val::Int(random.borrow_mut().next_int()),
        (Obj::Random(random), "nextInt", 1) => {
            let bound = args[0].to_i64()?;
            if bound <= 0 || bound > i32::MAX as i64 { return Err(illegal_arg("bound must be positive")); }
            Val::Int(random.borrow_mut().next_int_bounded(bound as i32))
        },
        (Obj::Random(random), "setSeed", 1) => {
            *random.borrow_mut() = rustic_core::vector::int4::JavaRandom::new(args[0].to_i64()?);
            Val::Null
        },
        (Obj::DenseVector(v), "getVector", 0) => {
            let allocation = super::allocation::Allocation::values(v.len())?;
            Val::new_array(super::ast::Ty::Float, v.iter().map(|x| Val::Double(*x as f64)).collect()).allocated(allocation)
        },
        (Obj::DenseVector(v), "getMagnitude", 0) => Val::Double(v.iter().map(|x| x * x).sum::<f32>().sqrt() as f64),
        (Obj::DenseVector(v), "getDims" | "size", 0) => Val::Int(v.len() as i32),
        (Obj::DenseVector(v), "isEmpty", 0) => Val::Bool(v.is_empty()),
        (Obj::Uuid(hi, _), "getMostSignificantBits", 0) => Val::Long(*hi as i64),
        (Obj::Uuid(_, lo), "getLeastSignificantBits", 0) => Val::Long(*lo as i64),
        (Obj::Uuid(hi, _), "version", 0) => Val::Int(((hi >> 12) & 0xf) as i32),
        (Obj::Uuid(_, lo), "variant", 0) => Val::Int(((lo >> 62) & 0x3) as i32),
        (Obj::Joiner { parts, .. }, "add", 1) => {
            let _copy=super::allocation::Allocation::text_copy(&args[0])?;
            let text=args[0].string_units();
            reserve_joiner(o,text.len(),true)?;
            parts.borrow_mut().push(text);
            recv.clone()
        }
        (Obj::Joiner { empty, .. }, "setEmptyValue", 1) => {
            let _copy=super::allocation::Allocation::text_copy(&args[0])?;
            let text=joiner_arg(&args[0])?;
            reserve_joiner(o,text.len(),false)?;
            *empty.borrow_mut() = Some(text);
            recv.clone()
        }
        (Obj::Joiner { .. }, "length", 0) => {
            let _copy=super::allocation::Allocation::text_copy(recv)?;
            Val::Int(recv.string_units().len() as i32)
        },
        (Obj::Joiner { parts, .. }, "merge", 1) => {
            let Val::Obj(other) = &args[0] else {
                return Err(if args[0].is_null() { super::error::npe("other StringJoiner is null") } else { super::error::class_cast("Expected StringJoiner") });
            };
            let Obj::Joiner { sep, parts: other_parts, .. } = &**other else {
                return Err(super::error::class_cast("Expected StringJoiner"));
            };
            let text={
                let source=other_parts.borrow();
                if source.is_empty() {return Ok(Some(recv.clone()));}
                let units=source.iter().fold(sep.len().saturating_mul(source.len().saturating_sub(1)),|n,v|n.saturating_add(v.len()));
                reserve_joiner(o,units,true)?;
                source.join(sep.as_slice())
            };
            parts.borrow_mut().push(text);
            recv.clone()
        }
        (Obj::Iter { items, pos, .. }, "hasNext", 0) => Val::Bool(pos.get() < items.len()),
        (Obj::Iter { items, pos, .. }, "next", 0) => {
            let i = pos.get();
            let v = items.get(i).cloned().ok_or_else(|| ScriptErr::new("no_such_element_exception", ""))?;
            pos.set(i + 1);
            v
        }
        (Obj::Iter { items, pos, src, .. }, "remove", 0) => {
            let i = pos.get();
            if i == 0 {
                return Err(illegal_state(""));
            }
            let Some(src) = src else { return Err(unsupported("remove")) };
            let target = &items[i - 1];
            let mut b = src.borrow_mut();
            if let Some(p) = b.iter().position(|x| x.ref_equals(target)) {
                b.remove(p);
            }
            Val::Null
        }
        (Obj::Cmp(..), "compare", 2) => Val::Int(o.cmp_call(rt, &args[0], &args[1])?.clamp(i32::MIN as i64, i32::MAX as i64) as i32),
        (Obj::Cmp(..), "reversed", 0) => reverse_order(Some(recv.clone())),
        // `UnsignedLongDocValuesField` (values are two's-complement longs).
        (Obj::Field { class, values, .. }, "getValue", 1) if class.ends_with("UnsignedLongDocValuesField") => values.first().cloned().unwrap_or_else(|| Val::Long(args[0].to_i64().unwrap_or(0))),
        (Obj::Field { class, values, .. }, "getValue", 2) if class.ends_with("UnsignedLongDocValuesField") => {
            let i = args[0].to_i64()?;
            values.get(i.max(0) as usize).filter(|_| i >= 0).cloned().unwrap_or_else(|| Val::Long(args[1].to_i64().unwrap_or(0)))
        }
        (Obj::Field { class, values, .. }, "getValues", 0) if class.ends_with("UnsignedLongDocValuesField") => {
            let allocation = super::allocation::Allocation::values(values.len())?;
            Val::new_list(values.clone()).allocated(allocation)
        },
        (Obj::Field { class, values, .. }, "asBigIntegers", 0) if class.ends_with("UnsignedLongDocValuesField") => {
            let allocation = super::allocation::Allocation::values(values.len().saturating_mul(4))?;
            Val::new_list(values.iter().map(|v| super::jbig::biginteger_u64(v.to_i64().unwrap_or(0) as u64)).collect()).allocated(allocation)
        }
        (Obj::Field { class, values, .. }, "asBigInteger", 1) if class.ends_with("UnsignedLongDocValuesField") => match values.first() {
            Some(v) => super::jbig::biginteger_u64(v.to_i64()? as u64),
            None => args[0].clone(),
        },
        (Obj::Field { class, values, .. }, "asBigInteger", 2) if class.ends_with("UnsignedLongDocValuesField") => {
            let i = args[0].to_i64()?;
            match values.get(i.max(0) as usize).filter(|_| i >= 0) {
                Some(v) => super::jbig::biginteger_u64(v.to_i64()? as u64),
                None => args[1].clone(),
            }
        }
        (Obj::Field { name, .. }, "getName", 0) => Val::str(name),
        (Obj::Field { values, .. }, "isEmpty", 0) => Val::Bool(values.is_empty()),
        (Obj::Field { values, .. }, "size", 0) => Val::Int(values.len() as i32),
        (Obj::Field { values, .. }, "get", 1) => values.first().cloned().unwrap_or_else(|| args[0].clone()),
        (Obj::Field { values, .. }, "get", 2) => {
            let i = args[0].to_i64()?;
            if i < 0 {
                return Err(super::error::out_of_bounds(format!("Index {i} out of bounds for length {}", values.len())));
            }
            values.get(i as usize).cloned().unwrap_or_else(|| args[1].clone())
        }
        (Obj::Field { values, .. }, "iterator", 0) => new_iter(values.iter().cloned(), None)?,
        (Obj::Iter { items, pos, .. }, "forEachRemaining", 1) => {
            let rest: Vec<Val> = items[pos.get().min(items.len())..].to_vec();
            pos.set(items.len());
            for v in rest {
                rt.call_fn(&args[0], &[v])?;
            }
            Val::Null
        }
        _ => return Ok(None),
    }))
}

pub fn instance_of(v: &Val, class: &str) -> Option<bool> {
    let Val::Obj(o) = v else { return None };
    Some(match (&**o, class) {
        (_, "Object") => true,
        (Obj::Uuid(..), "UUID" | "Comparable") => true,
        (Obj::Random(_), "Random") => true,
        (Obj::Joiner { .. }, "StringJoiner") => true,
        (Obj::Iter { .. }, "Iterator") => true,
        (Obj::Cmp(..), "Comparator") => true,
        (Obj::Field { .. }, "Field" | "DocValuesField") => true,
        (Obj::Collector { .. }, "Collector") => true,
        (Obj::Version(_), "Version" | "Comparable") => true,
        (Obj::DenseVector(_), "DenseVector") => true,
        _ => false,
    })
}

fn joiner_arg(value: &Val) -> SResult<Vec<u16>> {
    if value.is_null() { return Err(super::error::npe("StringJoiner argument is null")); }
    match value {
        Val::Str(_) | Val::Utf16(_) | Val::Builder(_) => Ok(value.string_units()),
        _ => Err(super::error::class_cast("Expected CharSequence")),
    }
}

fn locale_text(v:&Val)->SResult<Vec<u16>> {
    match v {Val::Str(_)|Val::Utf16(_)=>Ok(v.string_units()),Val::Null=>Err(super::error::npe("locale argument is null")),_=>Err(super::error::class_cast("Expected String"))}
}
fn locale_units(language:Vec<u16>,country:Vec<u16>,variant:Vec<u16>)->Val {
    let lower=language.into_iter().map(|u|if (65..=90).contains(&u) {u+32}else{u}).collect();
    let upper=country.into_iter().map(|u|if (97..=122).contains(&u) {u-32}else{u}).collect();
    Val::Obj(Rc::new(Obj::Locale{language:lower,country:upper,variant}))
}
pub(super) fn locale(language:&str,country:&str,variant:&str)->Val {
    locale_units(language.encode_utf16().collect(),country.encode_utf16().collect(),variant.encode_utf16().collect())
}
pub(super) fn locale_constant(name:&str)->Option<Val> {
    let (language,country)=match name {
        "ROOT"=>("",""),"ENGLISH"=>("en",""),"US"=>("en","US"),"UK"=>("en","GB"),
        "FRENCH"=>("fr",""),"FRANCE"=>("fr","FR"),"GERMAN"=>("de",""),"GERMANY"=>("de","DE"),
        "ITALIAN"=>("it",""),"ITALY"=>("it","IT"),"JAPANESE"=>("ja",""),"JAPAN"=>("ja","JP"),
        "KOREAN"=>("ko",""),"KOREA"=>("ko","KR"),"CHINESE"=>("zh",""),
        "CHINA"|"PRC"|"SIMPLIFIED_CHINESE"=>("zh","CN"),"TAIWAN"|"TRADITIONAL_CHINESE"=>("zh","TW"),
        "CANADA"=>("en","CA"),"CANADA_FRENCH"=>("fr","CA"),_=>return None,
    };Some(locale(language,country,""))
}
