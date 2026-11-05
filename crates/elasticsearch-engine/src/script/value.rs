//! Runtime values of the Painless interpreter.
//!
//! Numbers, booleans and dates are inline (no allocation); strings and
//! collections are reference counted. Values never leave the executor that
//! created them except as JSON (see [`Val::to_json`]).

use super::ast::Ty;
use super::error::{class_cast, illegal_arg, npe, ScriptErr, SResult};
use chrono::{Datelike, Timelike};
use indexmap::IndexMap;
use serde_json::{Map, Value};
use std::cell::{Cell, Ref, RefCell, RefMut};
use std::fmt::Write as _;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

/// Java capacity is observable and independent of Rust's allocation strategy.
#[derive(Debug)]
pub struct JBuilder {
    units: Vec<u16>,
    allocation: super::allocation::Allocation,
    pub capacity: usize,
}
impl JBuilder {
    pub fn new(units: Vec<u16>, has_argument: bool) -> SResult<Self> {
        let allocation=super::allocation::Allocation::values(0)?;
        allocation.reserve((units.capacity() as u64).saturating_mul(2).saturating_add(256))?;
        let capacity=if has_argument {units.len().saturating_add(16)} else {16};
        Ok(Self {units,capacity,allocation})
    }
    pub fn reserve_length(&mut self, length: usize) -> SResult<()> {
        let capacity=if length>self.units.capacity(){length.max(self.units.capacity().saturating_mul(2)).max(4)}else{self.units.capacity()};
        self.allocation.reserve((capacity as u64).saturating_mul(2).saturating_add(256))?;
        if capacity>self.units.capacity() {
            self.units.try_reserve_exact(capacity-self.units.len()).map_err(|_|ScriptErr::breaker(crate::error::EsError::new(429,"circuit_breaking_exception","script builder allocation failed")))?;
        }
        Ok(())
    }
    pub fn update_capacity(&mut self) {
        if self.units.len()>self.capacity {self.capacity=self.capacity.saturating_mul(2).saturating_add(2).max(self.units.len());}
    }
}
impl std::ops::Deref for JBuilder {type Target=Vec<u16>;fn deref(&self)->&Vec<u16>{&self.units}}
impl std::ops::DerefMut for JBuilder {fn deref_mut(&mut self)->&mut Vec<u16>{&mut self.units}}

pub type List = Rc<JList>;
pub type MapRef = Rc<JMap>;

/// Which Java collection class a list/set value is (iteration order and
/// mutability follow it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollKind {
    /// `ArrayList` / `LinkedList` / arrays.
    List,
    /// `HashSet`: Java hash-bucket iteration order.
    HashSet,
    /// `LinkedHashSet` (and set views/copies whose order is fixed).
    LinkedSet,
    /// `TreeSet`: natural order or `cmp`.
    TreeSet,
}

/// Which Java map class a map value is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapKind {
    /// `HashMap` (and Painless map literals): Java hash-bucket order.
    Hash,
    /// `LinkedHashMap`: insertion order.
    Linked,
    /// Maps built from JSON / internally: insertion order, reported as
    /// `java.util.HashMap`.
    Json,
    /// `TreeMap`: natural key order or `cmp`.
    Tree,
}

/// A Java list or set (`Val::List`, `Val::Set`, `Val::Array` items).
#[derive(Debug)]
pub struct JList {
    data: Rc<RefCell<Vec<Val>>>,
    allocation: super::allocation::Allocation,
    registered: Cell<bool>,
    pub kind: CollKind,
    /// `TreeSet` comparator (`None` = natural order).
    pub cmp: Option<Val>,
    /// `List.of` / `Collections.unmodifiable*` / `Set.of`.
    pub immutable: bool,
    /// Java `HashMap` table capacity backing a `HashSet`.
    cap: Cell<usize>,
    /// Physical order may differ from Java iteration order.
    dirty: Cell<bool>,
}

/// Eager stream storage owns its request reservation, including when aliased.
#[derive(Debug)]
pub struct JStream {
    pub(super) values: Vec<Val>,
    pub(super) allocation: super::allocation::Allocation,
}
impl std::ops::Deref for JStream {
    type Target = Vec<Val>;
    fn deref(&self) -> &Self::Target { &self.values }
}

/// A Java map (`Val::Map`).
#[derive(Debug)]
pub struct JMap {
    data: Rc<RefCell<IndexMap<Key, Val>>>,
    allocation: super::allocation::Allocation,
    registered: Cell<bool>,
    pub kind: MapKind,
    /// `TreeMap` comparator (`None` = natural order).
    pub cmp: Option<Val>,
    pub immutable: bool,
    cap: Cell<usize>,
    dirty: Cell<bool>,
}

/// Java `HashMap.hash` + bucket index.
fn bucket(h: i32, cap: usize) -> usize {
    let h = h ^ ((h as u32) >> 16) as i32;
    (h as u32 as usize) & (cap - 1)
}

/// Java `HashMap.tableSizeFor`.
pub fn table_size_for(n: usize) -> usize {
    n.max(1).next_power_of_two()
}

/// Java's table capacity after adding `n` keys one by one.
pub fn growth_capacity(n: usize) -> usize {
    let mut cap = 16;
    while n as f64 > cap as f64 * 0.75 {
        cap *= 2;
    }
    cap
}

/// Java's capacity for `new HashMap(m)` / `new HashSet(c)` of `n` elements.
pub fn copy_capacity(n: usize, set: bool) -> usize {
    if set {
        table_size_for(((n as f32 / 0.75) as usize + 1).max(16))
    } else if n == 0 {
        16
    } else {
        table_size_for((n as f32 / 0.75 + 1.0) as usize).max(16)
    }
}

impl JList {
    pub fn new(kind: CollKind, v: Vec<Val>) -> JList {
        let cap = if kind == CollKind::HashSet { copy_capacity(v.len(), true) } else { 16 };
        let data = Rc::new(RefCell::new(v));
        JList { data, allocation: super::allocation::Allocation::new(), registered: Cell::new(false), kind, cmp: None, immutable: false, cap: Cell::new(cap), dirty: Cell::new(kind != CollKind::List && kind != CollKind::LinkedSet) }
    }
    pub fn register_mutation(&self) -> SResult<()> {
        if self.allocation.enabled() && !self.registered.get() {
            super::allocation::Scope::register_list(&self.data)?;
            self.registered.set(true);
        }
        Ok(())
    }
    pub fn reserve_growth(&self, additional: usize) -> SResult<()> {
        if !self.allocation.enabled() { return Ok(()) }
        self.register_mutation()?;
        let mut data = self.data.borrow_mut();
        let needed = data.len().checked_add(additional).ok_or_else(|| illegal_arg("collection too large"))?;
        let capacity = if needed > data.capacity() { needed.max(data.capacity().saturating_mul(2)).max(4) } else { data.capacity() };
        self.allocation.reserve((capacity as u64).saturating_mul(std::mem::size_of::<Val>() as u64).saturating_add(256))?;
        if capacity > data.capacity() {
            let extra = capacity - data.len();
            data.try_reserve_exact(extra).map_err(|_| ScriptErr::breaker(crate::error::EsError::new(429, "circuit_breaking_exception", "script collection allocation failed")))?;
        }
        Ok(())
    }
    pub fn with_cmp(mut self, cmp: Option<Val>) -> JList {
        self.cmp = cmp.filter(|c| !c.is_null());
        self
    }
    pub fn immutable(mut self) -> JList {
        self.immutable = true;
        self
    }
    pub fn with_cap(self, cap: usize) -> JList {
        self.cap.set(cap.max(16));
        self
    }
    /// Raw storage (physical order).
    #[inline]
    pub fn borrow(&self) -> Ref<'_, Vec<Val>> {
        self.data.borrow()
    }
    /// Raw mutable storage; marks the Java order for recomputation.
    #[inline]
    pub fn borrow_mut(&self) -> RefMut<'_, Vec<Val>> {
        if self.kind != CollKind::List {
            self.dirty.set(true);
            if self.kind == CollKind::HashSet {
                // Grow like `HashMap.resize` (elements are added one by one).
                let n = self.data.try_borrow().map(|d| d.len()).unwrap_or(0) + 1;
                while n as f64 > self.cap.get() as f64 * 0.75 {
                    self.cap.set(self.cap.get() * 2);
                }
            }
        }
        self.data.borrow_mut()
    }
    /// Contents in Java iteration order.
    pub fn ordered(&self) -> Ref<'_, Vec<Val>> {
        if self.dirty.get() {
            if let Ok(mut d) = self.data.try_borrow_mut() {
                self.fix(&mut d);
                self.dirty.set(false);
            }
        }
        self.data.borrow()
    }
    fn fix(&self, d: &mut Vec<Val>) {
        match self.kind {
            CollKind::HashSet => {
                let n = d.len();
                while n as f64 > self.cap.get() as f64 * 0.75 {
                    self.cap.set(self.cap.get() * 2);
                }
                let cap = self.cap.get();
                d.sort_by_key(|v| bucket(v.java_hash(), cap));
            }
            CollKind::TreeSet if self.cmp.is_none() => d.sort_by(|a, b| a.compare(b).unwrap_or(std::cmp::Ordering::Equal)),
            _ => {}
        }
    }
    pub fn len(&self) -> usize {
        self.data.borrow().len()
    }
    pub fn is_set(&self) -> bool {
        self.kind != CollKind::List
    }
}

impl JMap {
    pub fn new(kind: MapKind, m: IndexMap<Key, Val>) -> JMap {
        let cap = copy_capacity(m.len(), false);
        let dirty = matches!(kind, MapKind::Hash | MapKind::Tree) && !m.is_empty();
        let data = Rc::new(RefCell::new(m));
        JMap { data, allocation: super::allocation::Allocation::new(), registered: Cell::new(false), kind, cmp: None, immutable: false, cap: Cell::new(cap), dirty: Cell::new(dirty) }
    }
    pub fn register_mutation(&self) -> SResult<()> {
        if self.allocation.enabled() && !self.registered.get() {
            super::allocation::Scope::register_map(&self.data)?;
            self.registered.set(true);
        }
        Ok(())
    }
    pub fn reserve_entry(&self, key: &Key) -> SResult<()> {
        if self.allocation.enabled() {
            self.register_mutation()?;
            let data = self.data.borrow();
            let needed = data.len().saturating_add(usize::from(!data.contains_key(key)));
            // IndexMap's dense entries and hash table, including resize overlap.
            let slots = data.capacity().max(needed).max(4).saturating_mul(2);
            self.allocation.reserve((slots as u64).saturating_mul((std::mem::size_of::<(Key, Val)>() + 32) as u64))?;
        }
        Ok(())
    }
    pub fn insert_checked(&self, key: Key, value: Val) -> SResult<Option<Val>> {
        self.reserve_entry(&key)?;
        Ok(self.borrow_mut().insert(key, value))
    }
    pub fn with_cmp(mut self, cmp: Option<Val>) -> JMap {
        self.cmp = cmp.filter(|c| !c.is_null());
        self
    }
    pub fn immutable(mut self) -> JMap {
        self.immutable = true;
        self
    }
    pub fn with_cap(self, cap: usize) -> JMap {
        self.cap.set(cap.max(16));
        self
    }
    /// Raw storage (physical order).
    #[inline]
    pub fn borrow(&self) -> Ref<'_, IndexMap<Key, Val>> {
        self.data.borrow()
    }
    /// Raw mutable storage; marks the Java order for recomputation.
    #[inline]
    pub fn borrow_mut(&self) -> RefMut<'_, IndexMap<Key, Val>> {
        if matches!(self.kind, MapKind::Hash | MapKind::Tree) {
            self.dirty.set(true);
            if self.kind == MapKind::Hash {
                let n = self.data.try_borrow().map(|d| d.len()).unwrap_or(0) + 1;
                while n as f64 > self.cap.get() as f64 * 0.75 {
                    self.cap.set(self.cap.get() * 2);
                }
            }
        }
        self.data.borrow_mut()
    }
    /// Entries in Java iteration order.
    pub fn ordered(&self) -> Ref<'_, IndexMap<Key, Val>> {
        if self.dirty.get() {
            if let Ok(mut d) = self.data.try_borrow_mut() {
                match self.kind {
                    MapKind::Hash => {
                        let cap = self.cap.get();
                        d.sort_by_cached_key(|k, _| bucket(k.to_val().java_hash(), cap));
                    }
                    MapKind::Tree if self.cmp.is_none() => d.sort_by(|a, _, b, _| a.to_val().compare(&b.to_val()).unwrap_or(std::cmp::Ordering::Equal)),
                    _ => {}
                }
                self.dirty.set(false);
            }
        }
        self.data.borrow()
    }
    pub fn len(&self) -> usize {
        self.data.borrow().len()
    }
}

#[derive(Clone, Debug)]
pub enum Val {
    Null,
    Bool(bool),
    Int(i32),
    Long(i64),
    Double(f64),
    Char(u16),
    Str(Rc<str>),
    /// Java string containing unpaired UTF-16 code units.
    Utf16(Rc<[u16]>),
    List(List),
    /// A `Set` (insertion ordered, deduplicated on insert).
    Set(List),
    Map(MapRef),
    /// `ZonedDateTime` in UTC, epoch millis.
    Date(i64),
    /// `Instant`, epoch millis.
    Instant(i64),
    /// `DayOfWeek` (1 = Monday).
    DayOfWeek(u8),
    /// `Month` (1 = January).
    Month(u8),
    /// `doc['field']`: the field's values.
    DocValues(Rc<DocValues>),
    /// The `doc` map itself.
    Doc,
    /// A compiled lambda (index into the program's lambda table).
    Lambda(u32),
    /// `Class::method` reference.
    MethodRef(Rc<(String, String)>),
    Stream(Rc<JStream>),
    Optional(Option<Rc<Val>>),
    Entry(Rc<(Val, Val)>),
    /// A mutable string builder.
    Builder(Rc<RefCell<JBuilder>>),
    /// A thrown exception object (`catch (Exception e)`): (type, message).
    Exception(Rc<(String, String)>),
    /// `ChronoUnit` constant (index into [`CHRONO_UNITS`]).
    ChronoUnit(u8),
    /// A Java array (distinct from `List` for toString purposes).
    Array { elem: Ty, items: List },
    /// `java.util.regex.Pattern`.
    Pattern(std::sync::Arc<super::regex::PatternData>),
    /// `java.util.regex.Matcher`.
    Matcher(Rc<RefCell<super::regex::MatcherState>>),
    /// A `java.time` value other than UTC `ZonedDateTime`/millisecond
    /// `Instant` (see `jtime`).
    Time(Rc<super::jtime::JTime>),
    /// `BigInteger` / `BigDecimal` / `MathContext` / `RoundingMode` (see `jbig`).
    Big(Rc<super::jbig::Big>),
    /// Other library objects: `UUID`, `StringJoiner`, ... (see `jmisc`).
    Obj(Rc<super::jmisc::Obj>),
}

pub const CHRONO_UNITS: [&str; 16] = [
    "NANOS", "MICROS", "MILLIS", "SECONDS", "MINUTES", "HOURS", "HALF_DAYS", "DAYS", "WEEKS", "MONTHS", "YEARS", "DECADES", "CENTURIES", "MILLENNIA", "ERAS", "FOREVER",
];

/// `doc['field']` contents.
#[derive(Debug)]
pub struct DocValues {
    pub field: String,
    pub values: Vec<Val>,
}

/// A hashable map key (Java `equals`/`hashCode` semantics for the common
/// key types; `Integer` and `Long` keys are distinct as in Java).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Key {
    Null,
    Bool(bool),
    Int(i32),
    Long(i64),
    Double(u64),
    Char(u16),
    Str(Rc<str>),
    Date(i64),
    /// Any other value, compared with Java `equals` / `hashCode`.
    Obj(KeyObj),
}

/// A map key of a non-primitive type.
#[derive(Clone, Debug)]
pub struct KeyObj(pub Box<Val>);

impl PartialEq for KeyObj {
    fn eq(&self, o: &Self) -> bool {
        self.0.equals(&o.0)
    }
}
impl Eq for KeyObj {}
impl Hash for KeyObj {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.0.java_hash().hash(h)
    }
}

impl Key {
    pub fn from_val(v: &Val) -> Key {
        match v {
            Val::Null => Key::Null,
            Val::Bool(b) => Key::Bool(*b),
            Val::Int(i) => Key::Int(*i),
            Val::Long(l) => Key::Long(*l),
            Val::Double(d) => Key::Double(d.to_bits()),
            Val::Char(c) => Key::Char(*c),
            Val::Str(s) => Key::Str(s.clone()),
            Val::Date(d) => Key::Date(*d),
            other => Key::Obj(KeyObj(Box::new(other.clone()))),
        }
    }
    pub fn to_val(&self) -> Val {
        match self {
            Key::Null => Val::Null,
            Key::Bool(b) => Val::Bool(*b),
            Key::Int(i) => Val::Int(*i),
            Key::Long(l) => Val::Long(*l),
            Key::Double(d) => Val::Double(f64::from_bits(*d)),
            Key::Char(c) => Val::Char(*c),
            Key::Str(s) => Val::Str(s.clone()),
            Key::Date(d) => Val::Date(*d),
            Key::Obj(o) => (*o.0).clone(),
        }
    }
    pub fn as_string(&self) -> String {
        match self {
            Key::Str(s) => s.to_string(),
            k => k.to_val().to_java_string(),
        }
    }
    pub fn str(s: &str) -> Key {
        Key::Str(Rc::from(s))
    }
}

impl Val {
    /// Clone without the out-of-line drop/clone glue for inline variants.
    #[inline(always)]
    pub fn fast_clone(&self) -> Val {
        match self {
            Val::Double(d) => Val::Double(*d),
            Val::Long(l) => Val::Long(*l),
            Val::Int(i) => Val::Int(*i),
            Val::Bool(b) => Val::Bool(*b),
            Val::Null => Val::Null,
            other => other.clone(),
        }
    }

    /// True for variants without heap data (dropping them is a no-op).
    #[inline(always)]
    pub fn is_inline(&self) -> bool {
        matches!(self, Val::Null | Val::Bool(_) | Val::Int(_) | Val::Long(_) | Val::Double(_) | Val::Char(_) | Val::Date(_) | Val::Instant(_) | Val::DayOfWeek(_) | Val::Month(_) | Val::ChronoUnit(_) | Val::Lambda(_) | Val::Doc)
    }

    pub fn str(s: &str) -> Val {
        Val::Str(Rc::from(s))
    }
    pub fn string(s: String) -> Val {
        Val::Str(Rc::from(s))
    }
    pub fn new_list(v: Vec<Val>) -> Val {
        Val::List(Rc::new(JList::new(CollKind::List, v)))
    }
    /// An insertion-ordered map (JSON-derived / internal values).
    pub fn new_map(m: IndexMap<Key, Val>) -> Val {
        Val::Map(Rc::new(JMap::new(MapKind::Json, m)))
    }
    /// A `HashMap` (Painless map literals, `new HashMap()`).
    pub fn new_hash_map(m: IndexMap<Key, Val>) -> Val {
        let n = m.len();
        Val::Map(Rc::new(JMap::new(MapKind::Hash, m).with_cap(growth_capacity(n))))
    }
    /// A set of the given kind (`vals` already deduplicated).
    pub fn new_set(kind: CollKind, vals: Vec<Val>) -> Val {
        Val::Set(Rc::new(JList::new(kind, vals)))
    }
    pub(super) fn stream_from_iter(iter: impl IntoIterator<Item = SResult<Val>>) -> SResult<Val> {
        let allocation = super::allocation::Allocation::values(0)?;
        let mut values = Vec::new();
        for value in iter {
            let value = value?;
            allocation.push(&mut values, value)?;
        }
        Ok(Val::Stream(Rc::new(JStream { values, allocation })))
    }
    pub(super) fn stream_from_values(values: Vec<Val>, allocation: super::allocation::Allocation) -> Val {
        Val::Stream(Rc::new(JStream { values, allocation }))
    }
    pub(super) fn allocated(mut self, allocation: super::allocation::Allocation) -> Val {
        match &mut self {
            Val::List(v) | Val::Set(v) | Val::Array { items: v, .. } => Rc::get_mut(v).expect("new collection").allocation = allocation,
            Val::Map(v) => Rc::get_mut(v).expect("new collection").allocation = allocation,
            _ => unreachable!("allocation belongs to collection"),
        }
        self
    }
    pub fn new_array(elem: Ty, items: Vec<Val>) -> Val {
        Val::Array { elem, items: Rc::new(JList::new(CollKind::List, items)) }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Val::Null => "null",
            Val::Bool(_) => "java.lang.Boolean",
            Val::Int(_) => "java.lang.Integer",
            Val::Long(_) => "java.lang.Long",
            Val::Double(_) => "java.lang.Double",
            Val::Char(_) => "java.lang.Character",
            Val::Str(_) | Val::Utf16(_) => "java.lang.String",
            Val::List(l) => if l.immutable { "java.util.ImmutableCollections$ListN" } else { "java.util.ArrayList" },
            Val::Set(l) => match l.kind {
                CollKind::TreeSet => "java.util.TreeSet",
                CollKind::LinkedSet => "java.util.LinkedHashSet",
                _ => "java.util.HashSet",
            },
            Val::Date(_) => "java.time.ZonedDateTime",
            Val::Instant(_) => "java.time.Instant",
            Val::DayOfWeek(_) => "java.time.DayOfWeek",
            Val::Month(_) => "java.time.Month",
            Val::DocValues(_) => "org.elasticsearch.index.fielddata.ScriptDocValues",
            Val::Doc => "org.elasticsearch.search.lookup.LeafDocLookup",
            Val::Lambda(_) | Val::MethodRef(_) => "java.util.function.Function",
            Val::Stream(_) => "java.util.stream.Stream",
            Val::Optional(_) => "java.util.Optional",
            Val::Entry(_) => "java.util.Map$Entry",
            Val::Builder(_) => "java.lang.StringBuilder",
            Val::Exception(_) => "java.lang.Exception",
            Val::ChronoUnit(_) => "java.time.temporal.ChronoUnit",
            Val::Pattern(_) => "java.util.regex.Pattern",
            Val::Matcher(_) => "java.util.regex.Matcher",
            Val::Time(t) => t.type_name(),
            Val::Big(b) => b.type_name(),
            Val::Obj(o) => o.type_name(),
            Val::Map(m) => match m.kind {
                MapKind::Tree => "java.util.TreeMap",
                MapKind::Linked => "java.util.LinkedHashMap",
                MapKind::Hash | MapKind::Json => "java.util.HashMap",
            },
            Val::Array { elem, .. } => match elem {
                Ty::Int => "[I",
                Ty::Long => "[J",
                Ty::Double => "[D",
                Ty::Float => "[F",
                Ty::Boolean => "[Z",
                Ty::Byte => "[B",
                Ty::Short => "[S",
                Ty::Char => "[C",
                Ty::String => "[Ljava.lang.String;",
                Ty::Array(_) => "[[",
                _ => "[Ljava.lang.Object;",
            },
        }
    }

    /// Java type name as printed in `cannot cast def [..]` messages.
    pub fn def_name(&self) -> &'static str {
        match self {
            Val::Int(_) => "int",
            Val::Long(_) => "long",
            Val::Double(_) => "double",
            Val::Char(_) => "char",
            Val::Bool(_) => "boolean",
            other => other.type_name(),
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Val::Null)
    }

    pub fn is_number(&self) -> bool {
        matches!(self, Val::Int(_) | Val::Long(_) | Val::Double(_) | Val::Char(_))
    }

    /// Numeric value as f64 (numbers and chars only).
    #[inline]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Val::Double(d) => Some(*d),
            Val::Long(l) => Some(*l as f64),
            Val::Int(i) => Some(*i as f64),
            Val::Char(c) => Some(*c as u32 as f64),
            _ => None,
        }
    }

    #[inline]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Val::Long(l) => Some(*l),
            Val::Int(i) => Some(*i as i64),
            Val::Double(d) => Some(*d as i64),
            Val::Char(c) => Some(*c as i64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> SResult<bool> {
        match self {
            Val::Bool(b) => Ok(*b),
            Val::Null => Err(npe("Cannot invoke \"Object.getClass()\" because \"value\" is null")),
            other => Err(class_cast(format!("cannot cast def [{}] to boolean", other.def_name()))),
        }
    }

    pub fn to_f64(&self) -> SResult<f64> {
        if self.is_null() {
            return Err(npe("Cannot invoke \"Object.getClass()\" because \"value\" is null"));
        }
        self.as_f64().ok_or_else(|| class_cast(format!("Cannot cast {} to java.lang.Number", self.type_name())))
    }

    pub fn to_i64(&self) -> SResult<i64> {
        self.as_i64().ok_or_else(|| class_cast(format!("Cannot cast {} to java.lang.Number", self.type_name())))
    }

    pub fn to_index(&self) -> SResult<i64> {
        match self {
            Val::Int(i) => Ok(*i as i64),
            Val::Long(l) => Ok(*l),
            Val::Char(c) => Ok(*c as i64),
            other => Err(class_cast(format!("Cannot cast {} to java.lang.Integer", other.type_name()))),
        }
    }

    /// Java `toString()` / string concatenation rendering.
    pub fn to_java_string(&self) -> String {
        let mut s = String::new();
        self.write_java(&mut s);
        s
    }

    pub fn write_java(&self, out: &mut String) {
        match self {
            Val::Null => out.push_str("null"),
            Val::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Val::Int(i) => {
                let _ = write!(out, "{i}");
            }
            Val::Long(l) => {
                let _ = write!(out, "{l}");
            }
            Val::Double(d) => out.push_str(&java_double_string(*d)),
            Val::Char(c) => out.push(char::from_u32(*c as u32).unwrap_or('\u{fffd}')),
            Val::Str(s) => out.push_str(s),
            Val::Utf16(s) => out.push_str(&String::from_utf16_lossy(s)),
            Val::List(l) | Val::Set(l) => {
                out.push('[');
                for (i, v) in l.ordered().iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    v.write_java(out);
                }
                out.push(']');
            }
            Val::Stream(l) => {
                out.push('[');
                for (i, v) in l.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    v.write_java(out);
                }
                out.push(']');
            }
            Val::Map(m) => {
                out.push('{');
                for (i, (k, v)) in m.ordered().iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    k.to_val().write_java(out);
                    out.push('=');
                    v.write_java(out);
                }
                out.push('}');
            }
            Val::Date(ms) => out.push_str(&zoned_string(*ms)),
            Val::Instant(ms) => out.push_str(&instant_string(*ms)),
            Val::DayOfWeek(d) => out.push_str(DAYS[(*d as usize).clamp(1, 7) - 1]),
            Val::Month(m) => out.push_str(MONTHS[(*m as usize).clamp(1, 12) - 1]),
            Val::DocValues(d) => {
                out.push('[');
                for (i, v) in d.values.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    v.write_java(out);
                }
                out.push(']');
            }
            Val::Doc => out.push_str("doc"),
            Val::Lambda(_) | Val::MethodRef(_) => out.push_str("lambda"),
            Val::Optional(o) => match o {
                Some(v) => {
                    out.push_str("Optional[");
                    v.write_java(out);
                    out.push(']');
                }
                None => out.push_str("Optional.empty"),
            },
            Val::Entry(e) => {
                e.0.write_java(out);
                out.push('=');
                e.1.write_java(out);
            }
            Val::Builder(b) => out.push_str(&String::from_utf16_lossy(&b.borrow())),
            Val::Exception(e) => {
                let suffix = format!(".{}", e.0);
                let class = super::whitelist::WHITELIST.iter().find(|w| w.0.ends_with(&suffix)).map(|w| w.0).unwrap_or(e.0.as_str());
                if e.1.is_empty() {
                    out.push_str(class);
                } else {
                    let _ = write!(out, "{}: {}", class, e.1);
                }
            }
            Val::ChronoUnit(u) => {
                let n = CHRONO_UNITS[(*u as usize).min(CHRONO_UNITS.len() - 1)];
                let mut c = n.chars();
                if let Some(f) = c.next() {
                    out.push(f);
                    out.push_str(&c.as_str().to_lowercase().replace('_', ""));
                }
            }
            Val::Pattern(p) => out.push_str(&p.source),
            Val::Matcher(m) => {
                let m = m.borrow();
                let _ = write!(out, "java.util.regex.Matcher[pattern={} region=0,{} lastmatch=]", m.pat.source, m.input.len());
            }
            Val::Time(t) => t.write_java(out),
            Val::Big(b) => b.write_java(out),
            Val::Obj(o) => o.write_java(out),
            Val::Array { elem, .. } => {
                // Java array toString: [I@<hash> for int[], [Ljava.lang.String;@<hash> for String[], etc.
                let prefix = match elem {
                    Ty::Int => "[I@",
                    Ty::Long => "[J@",
                    Ty::Double => "[D@",
                    Ty::Float => "[F@",
                    Ty::Boolean => "[Z@",
                    Ty::Byte => "[B@",
                    Ty::Short => "[S@",
                    Ty::Char => "[C@",
                    Ty::String => "[Ljava.lang.String;@",
                    Ty::Array(_) => "[[@",
                    _ => "[Ljava.lang.Object;@",
                };
                out.push_str(prefix);
                // Use a deterministic hash based on the array's identity
                let h = self as *const Val as usize;
                let _ = write!(out, "{:x}", h & 0xffff_ffff);
            }
        }
    }

    /// JSON rendering (script_fields, `_source` writes, results). A
    /// container that contains itself renders `null` at the repeat (use
    /// [`Val::try_to_json`] where ES rejects self-references).
    pub fn to_json(&self) -> Value {
        self.json_inner(&mut Vec::new(), false).unwrap_or(Value::Null)
    }

    /// [`Val::to_json`], failing on self-referencing containers (ES's
    /// `ensureNoSelfReferences`: "Iterable object is self-referencing itself").
    pub fn try_to_json(&self) -> Result<Value, ()> {
        self.json_inner(&mut Vec::new(), true)
    }

    fn json_inner(&self, stack: &mut Vec<usize>, strict: bool) -> Result<Value, ()> {
        let addr = match self {
            Val::List(l) | Val::Set(l) => Some(Rc::as_ptr(l) as *const () as usize),
            Val::Map(m) => Some(Rc::as_ptr(m) as *const () as usize),
            Val::Array { items, .. } => Some(Rc::as_ptr(items) as *const () as usize),
            _ => None,
        };
        if let Some(a) = addr {
            if stack.contains(&a) {
                return if strict { Err(()) } else { Ok(Value::Null) };
            }
            stack.push(a);
        }
        let mut sub = |v: &Val| v.json_inner(stack, strict);
        let out = match self {
            Val::Null => Value::Null,
            Val::Bool(b) => Value::Bool(*b),
            Val::Int(i) => Value::from(*i),
            Val::Long(l) => Value::from(*l),
            Val::Double(d) => serde_json::Number::from_f64(*d).map(Value::Number).unwrap_or_else(|| Value::String(java_double_string(*d))),
            Val::Char(c) => Value::String(String::from_utf16_lossy(&[*c])),
            Val::Str(s) => Value::String(s.to_string()),
            Val::List(l) | Val::Set(l) => Value::Array(l.ordered().iter().map(&mut sub).collect::<Result<_, _>>()?),
            Val::Array { items, .. } => Value::Array(items.borrow().iter().map(&mut sub).collect::<Result<_, _>>()?),
            Val::Stream(l) => Value::Array(l.iter().map(&mut sub).collect::<Result<_, _>>()?),
            Val::DocValues(d) => Value::Array(d.values.iter().map(&mut sub).collect::<Result<_, _>>()?),
            Val::Map(m) => {
                if m.borrow().contains_key(&Key::str(super::doc::GEO_SHAPE_KEY)) {
                    // ES `GeoShapeValue` has no x-content form.
                    Value::String("<unserializable>".into())
                } else {
                    let mut o = Map::new();
                    for (k, v) in m.ordered().iter() {
                        o.insert(k.as_string(), sub(v)?);
                    }
                    Value::Object(o)
                }
            }
            Val::Date(ms) => Value::String(zoned_string(*ms)),
            Val::Instant(ms) => Value::String(instant_string(*ms)),
            Val::Optional(o) => match o.as_ref() {
                Some(v) => sub(v)?,
                None => Value::Null,
            },
            Val::Entry(e) => {
                let mut o = Map::new();
                o.insert(e.0.to_java_string(), sub(&e.1)?);
                Value::Object(o)
            }
            other => Value::String(other.to_java_string()),
        };
        if addr.is_some() {
            stack.pop();
        }
        Ok(out)
    }

    /// JSON → value (Jackson's mapping: ints that fit are `Integer`).
    pub fn utf16(units: Vec<u16>) -> Val {
        match String::from_utf16(&units) { Ok(s)=>Val::string(s), Err(_)=>Val::Utf16(Rc::from(units)) }
    }
    pub fn string_units(&self) -> Vec<u16> {
        fn sequence(values: &[Val]) -> Vec<u16> {
            let mut out = vec![91];
            for (i, value) in values.iter().enumerate() {
                if i > 0 { out.extend([44, 32]); }
                out.extend(value.string_units());
            }
            out.push(93);
            out
        }
        match self {
            Val::Char(c) => vec![*c],
            Val::Utf16(s) => s.to_vec(),
            Val::Builder(b) => b.borrow().to_vec(),
            Val::Obj(obj) => obj.string_units().unwrap_or_else(|| self.to_java_string().encode_utf16().collect()),
            Val::List(l) | Val::Set(l) => sequence(&l.ordered()),
            Val::Stream(values) => sequence(values),
            Val::DocValues(d) => sequence(&d.values),
            Val::Map(m) => {
                let mut out = vec![123];
                for (i, (key, value)) in m.ordered().iter().enumerate() {
                    if i > 0 { out.extend([44, 32]); }
                    out.extend(key.to_val().string_units());
                    out.push(61);
                    out.extend(value.string_units());
                }
                out.push(125);
                out
            }
            Val::Entry(e) => {
                let mut out = e.0.string_units();
                out.push(61);
                out.extend(e.1.string_units());
                out
            }
            Val::Optional(Some(value)) => {
                let mut out: Vec<u16> = "Optional[".encode_utf16().collect();
                out.extend(value.string_units());
                out.push(93);
                out
            }
            _ => self.to_java_string().encode_utf16().collect(),
        }
    }
    pub fn from_json(v: &Value) -> Val {
        match v {
            Value::Null => Val::Null,
            Value::Bool(b) => Val::Bool(*b),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    if i >= i32::MIN as i64 && i <= i32::MAX as i64 {
                        Val::Int(i as i32)
                    } else {
                        Val::Long(i)
                    }
                } else if let Some(u) = n.as_u64() {
                    Val::Double(u as f64)
                } else {
                    Val::Double(n.as_f64().unwrap_or(0.0))
                }
            }
            Value::String(s) => Val::str(s),
            Value::Array(a) => Val::new_list(a.iter().map(Val::from_json).collect()),
            Value::Object(o) => {
                let mut m = IndexMap::with_capacity(o.len());
                for (k, v) in o {
                    m.insert(Key::str(k), Val::from_json(v));
                }
                Val::new_map(m)
            }
        }
    }

    /// Java `equals` (`==` on def values): numbers compare numerically.
    pub fn equals(&self, o: &Val) -> bool {
        match (self, o) {
            (Val::Null, Val::Null) => true,
            (Val::Null, _) | (_, Val::Null) => false,
            (Val::Bool(a), Val::Bool(b)) => a == b,
            (Val::Str(a), Val::Str(b)) => a == b,
            (Val::Utf16(a), Val::Utf16(b)) => a == b,
            (Val::Char(a), Val::Char(b)) => a == b,
            (Val::Int(a), Val::Int(b)) => a == b,
            (Val::Long(a), Val::Long(b)) => a == b,
            (a, b) if a.is_number() && b.is_number() => match (a, b) {
                (Val::Double(_), _) | (_, Val::Double(_)) => a.as_f64() == b.as_f64(),
                _ => a.as_i64() == b.as_i64(),
            },
            (Val::List(a), Val::List(b)) | (Val::Set(a), Val::Set(b)) => {
                if Rc::ptr_eq(a, b) {
                    return true;
                }
                let (a, b) = (a.borrow(), b.borrow());
                if a.len() != b.len() {
                    return false;
                }
                if matches!(self, Val::Set(_)) {
                    a.iter().all(|x| b.iter().any(|y| x.equals(y)))
                } else {
                    a.iter().zip(b.iter()).all(|(x, y)| x.equals(y))
                }
            }
            (Val::Array { items: a, .. }, Val::Array { items: b, .. }) => {
                if Rc::ptr_eq(a, b) {
                    return true;
                }
                let (a, b) = (a.borrow(), b.borrow());
                a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.equals(y))
            }
            (Val::Map(a), Val::Map(b)) => {
                if Rc::ptr_eq(a, b) {
                    return true;
                }
                let (a, b) = (a.borrow(), b.borrow());
                a.len() == b.len() && a.iter().all(|(k, v)| b.get(k).is_some_and(|w| v.equals(w)))
            }
            (Val::Date(a), Val::Date(b)) | (Val::Instant(a), Val::Instant(b)) => a == b,
            (Val::DayOfWeek(a), Val::DayOfWeek(b)) | (Val::Month(a), Val::Month(b)) => a == b,
            (Val::Entry(a), Val::Entry(b)) => a.0.equals(&b.0) && a.1.equals(&b.1),
            (Val::Optional(a), Val::Optional(b)) => match (a, b) {
                (None, None) => true,
                (Some(x), Some(y)) => x.equals(y),
                _ => false,
            },
            (Val::DocValues(a), Val::DocValues(b)) => Rc::ptr_eq(a, b),
            (Val::Stream(a), Val::Stream(b)) => Rc::ptr_eq(a, b),
            (Val::Lambda(a), Val::Lambda(b)) => a == b,
            (Val::ChronoUnit(a), Val::ChronoUnit(b)) => a == b,
            (Val::Time(_), _) | (_, Val::Time(_)) => super::jtime::equals(self, o),
            (Val::Big(a), Val::Big(b)) => a.equals(b),
            (Val::Obj(a), Val::Obj(b)) => a.equals(b),
            _ => false,
        }
    }

    /// Reference equality (`===`).
    pub fn ref_equals(&self, o: &Val) -> bool {
        match (self, o) {
            (Val::List(a), Val::List(b)) | (Val::Set(a), Val::Set(b)) => Rc::ptr_eq(a, b),
            (Val::Array { items: a, .. }, Val::Array { items: b, .. }) => Rc::ptr_eq(a, b),
            (Val::Map(a), Val::Map(b)) => Rc::ptr_eq(a, b),
            (Val::Str(a), Val::Str(b)) => Rc::ptr_eq(a, b) || a == b,
            _ => self.equals(o),
        }
    }

    /// Java `hashCode` for strings/numbers (used by `hashCode()` calls).
    pub fn java_hash(&self) -> i32 {
        match self {
            Val::Str(s) => s.encode_utf16().fold(0i32, |h, c| h.wrapping_mul(31).wrapping_add(c as i32)),
            Val::Utf16(s) => s.iter().fold(0i32, |h,c|h.wrapping_mul(31).wrapping_add(*c as i32)),
            Val::Int(i) => *i,
            Val::Long(l) => (*l ^ (*l >> 32)) as i32,
            Val::Bool(b) => {
                if *b {
                    1231
                } else {
                    1237
                }
            }
            Val::Double(d) => {
                let b = d.to_bits() as i64;
                (b ^ (b >> 32)) as i32
            }
            Val::Null => 0,
            Val::Char(c) => *c as i32,
            Val::List(l) => l.borrow().iter().fold(1i32, |h, v| h.wrapping_mul(31).wrapping_add(v.java_hash())),
            Val::Set(l) => l.borrow().iter().fold(0i32, |h, v| h.wrapping_add(v.java_hash())),
            Val::Map(m) => m.borrow().iter().fold(0i32, |h, (k, v)| h.wrapping_add(k.to_val().java_hash() ^ v.java_hash())),
            Val::Entry(e) => e.0.java_hash() ^ e.1.java_hash(),
            Val::DayOfWeek(_) | Val::Month(_) | Val::ChronoUnit(_) => self.to_java_string().encode_utf16().fold(0i32, |h, c| h.wrapping_mul(31).wrapping_add(c as i32)),
            Val::Time(t) => t.hash(),
            Val::Big(b) => b.hash(),
            Val::Obj(o) => o.hash(),
            other => {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                other.to_java_string().hash(&mut h);
                h.finish() as i32
            }
        }
    }

    /// Ordering for sorting (`Comparable` semantics).
    pub fn compare(&self, o: &Val) -> SResult<std::cmp::Ordering> {
        use std::cmp::Ordering;
        Ok(match (self, o) {
            (Val::Str(a), Val::Str(b)) => {
                // Java compareTo compares UTF-16 code units.
                a.encode_utf16().cmp(b.encode_utf16())
            }
            (a @ (Val::Str(_) | Val::Utf16(_)), b @ (Val::Str(_) | Val::Utf16(_))) => a.string_units().cmp(&b.string_units()),
            (Val::Int(a), Val::Int(b)) => a.cmp(b),
            (Val::Long(a), Val::Long(b)) => a.cmp(b),
            (a, b) if a.is_number() && b.is_number() => match (a, b) {
                (Val::Double(_), _) | (_, Val::Double(_)) => a.as_f64().unwrap_or(0.0).total_cmp(&b.as_f64().unwrap_or(0.0)),
                _ => a.as_i64().cmp(&b.as_i64()),
            },
            (Val::Bool(a), Val::Bool(b)) => a.cmp(b),
            (Val::Date(a), Val::Date(b)) | (Val::Instant(a), Val::Instant(b)) => a.cmp(b),
            (Val::DayOfWeek(a), Val::DayOfWeek(b)) | (Val::Month(a), Val::Month(b)) => a.cmp(b),
            (Val::Null, Val::Null) => Ordering::Equal,
            (Val::Null, _) | (_, Val::Null) => return Err(super::error::npe("Cannot compare null values")),
            (Val::Time(_), _) | (_, Val::Time(_)) | (Val::Date(_), Val::Instant(_)) | (Val::Instant(_), Val::Date(_)) => return super::jtime::compare(self, o),
            (Val::Big(a), Val::Big(b)) => return a.compare(b),
            (Val::Obj(a), Val::Obj(b)) => return a.compare(b),
            (a, b) => return Err(class_cast(format!("Cannot cast {} to java.lang.Comparable (compared with {})", a.type_name(), b.type_name()))),
        })
    }
}

pub const DAYS: [&str; 7] = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"];
pub const MONTHS: [&str; 12] = ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"];

/// Java `Double.toString`.
pub fn java_double_string(v: f64) -> String {
    if v.is_nan() {
        return "NaN".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if v == 0.0 {
        return if v.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let a = v.abs();
    if (1e-3..1e7).contains(&a) {
        let s = format!("{v}");
        if s.contains('.') { s } else { format!("{s}.0") }
    } else {
        // Java (JDK 19+) picks the shortest decimal with at least two
        // significant digits that is closest to the value: a one-digit
        // shortest repr (`5e-324`) becomes the closest two-digit one (`4.9E-324`).
        let s = format!("{v:e}");
        let (m, e) = s.split_once('e').unwrap_or((&s, "0"));
        if !m.contains('.') {
            let s2 = format!("{v:.1e}");
            let (m2, e2) = s2.split_once('e').unwrap_or((&s2, "0"));
            return format!("{m2}E{e2}");
        }
        format!("{m}E{e}")
    }
}

/// `ZonedDateTime.toString()` for a UTC instant (`2020-01-01T10:00Z`).
pub fn zoned_string(ms: i64) -> String {
    let dt = crate::date::from_millis(ms);
    let mut s = format!("{:04}-{:02}-{:02}T{:02}:{:02}", dt.year(), dt.month(), dt.day(), dt.hour(), dt.minute());
    let nanos = dt.nanosecond();
    if dt.second() != 0 || nanos != 0 {
        let _ = write!(s, ":{:02}", dt.second());
        if nanos != 0 {
            let _ = write!(s, ".{:03}", nanos / 1_000_000);
        }
    }
    s.push('Z');
    s
}

/// `Instant.toString()` (`2020-01-01T10:00:00Z`, millis when non-zero).
pub fn instant_string(ms: i64) -> String {
    let dt = crate::date::from_millis(ms);
    let mut s = format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}", dt.year(), dt.month(), dt.day(), dt.hour(), dt.minute(), dt.second());
    let nanos = dt.nanosecond();
    if nanos != 0 {
        let _ = write!(s, ".{:03}", nanos / 1_000_000);
    }
    s.push('Z');
    s
}
