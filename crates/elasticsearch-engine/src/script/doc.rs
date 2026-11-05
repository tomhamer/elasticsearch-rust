//! `doc['field']` access: per-leaf column readers resolved once per leaf
//! (no per-doc lookups), values as script [`Val`]s.

use super::error::{illegal_arg, illegal_state, SResult};
use super::value::{DocValues, Val};
use crate::mapping::{EsType, FieldLookup, ID_FIELD};
use rustic_core::codec::fastfields::{ColumnReader, KeywordReader};
use rustic_core::fast_fields::{F64Column, I64Column, KeywordColumn};
use rustic_core::query::LeafReader;
use rustic_core::types::DocId;
use std::rc::Rc;
use std::sync::Arc;

/// How a field's values are typed in scripts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Long,
    Unsigned,
    Double,
    Date,
    Bool,
    Keyword,
    Ip,
    GeoPoint,
    /// `geo_shape`: one opaque shape value per doc ([`geo_shape_value`]).
    GeoShape,
    /// `dense_vector`: one `DenseVector` object per doc.
    DenseVector,
    /// A `byte` / `bit` `dense_vector`: its packed rows are unpacked into
    /// the values scripts compute on (one per byte / one 0/1 per bit).
    PackedVector(crate::vectors::Packing),
    /// Text without fielddata: access is an error.
    Text,
    /// Mapped but not doc-value backed in scripts.
    Unsupported,
    Unmapped,
    /// A runtime field (index into the runtime field set).
    Runtime(usize),
    /// `rank_vectors` (element type: 0 float, 1 byte, 2 bit).
    RankVectors(u8),
}

impl Kind {
    pub fn of(ty: EsType) -> Kind {
        match ty {
            EsType::Long | EsType::Integer | EsType::Short | EsType::Byte => Kind::Long,
            EsType::UnsignedLong => Kind::Unsigned,
            EsType::Double | EsType::Float | EsType::HalfFloat | EsType::ScaledFloat => Kind::Double,
            EsType::Date | EsType::DateNanos => Kind::Date,
            EsType::Boolean => Kind::Bool,
            EsType::Keyword | EsType::ConstantKeyword | EsType::Wildcard | EsType::Flattened => Kind::Keyword,
            EsType::Ip => Kind::Ip,
            EsType::GeoPoint => Kind::GeoPoint,
            EsType::GeoShape => Kind::GeoShape,
            EsType::DenseVector => Kind::DenseVector,
            EsType::Text | EsType::MatchOnlyText => Kind::Text,
            _ => Kind::Unsupported,
        }
    }
}

/// Column reader bound to one leaf.
enum Col {
    None,
    SegI64(ColumnReader<'static, i64>),
    SegF64(ColumnReader<'static, f64>),
    SegKeyword(KeywordReader<'static>),
    MemI64(&'static I64Column),
    MemF64(&'static F64Column),
    MemKeyword(&'static KeywordColumn),
    /// Through `LeafReader`'s (allocating) accessors.
    Generic,
}

/// A single-valued f64 column of one leaf (geo_point lat / lon).
enum F64Col {
    Seg(ColumnReader<'static, f64>),
    Mem(&'static F64Column),
}

impl F64Col {
    #[inline(always)]
    fn get(&self, doc: DocId) -> Option<f64> {
        match self {
            F64Col::Seg(c) => c.get(doc),
            F64Col::Mem(c) => c.get(doc),
        }
    }
}

/// The points of `doc` as ES's doc values list them: decoded from the
/// quantized encoding, in the order of their encoded longs
/// (`LatLonDocValuesField`: latitude in the high 32 bits, longitude in the
/// low, sorted as signed longs).
pub(crate) fn sorted_geo_points(v: &rustic_core::geo::GeoValues, doc: DocId) -> Vec<(f64, f64)> {
    let mut pts: Vec<(i32, i32)> = v.iter_range(v.doc_range(doc)).map(|p| (p.lat, p.lon)).collect();
    pts.sort_by_key(|&(lat, lon)| ((lat as i64) << 32) | (lon as u32 as i64));
    pts.into_iter().map(|(lat, lon)| (rustic_core::geo::decode_lat(lat), rustic_core::geo::decode_lon(lon))).collect()
}

/// One referenced field, resolved against an index mapping and bound to
/// the current leaf.
pub struct FieldAcc {
    pub name: String,
    pub kind: Kind,
    /// The field has doc values (scripts' `doc` access needs them).
    pub doc_values: bool,
    /// `constant_keyword`: the value every document holds.
    constant: Option<String>,
    id: Option<u16>,
    lat_lon: Option<(u16, u16)>,
    /// A `geo_point`'s core geo field (its quantized doc values).
    geo_id: Option<u16>,
    col: Col,
    /// The bound leaf's `geo_point` doc values.
    geo: Option<GeoDv>,
    /// A `scaled_float`'s factor and inverse: its doc values read as ES
    /// reads them (the scaled long times the inverse factor).
    scale: Option<(f64, f64)>,
}

/// A leaf's `geo_point` doc values: the core geo field's doc-sorted
/// quantized points, or (indices that still write them) single-valued
/// lat / lon columns.
enum GeoDv {
    Points(&'static rustic_core::geo::GeoValues),
    Cols(F64Col, F64Col),
}

impl FieldAcc {
    pub fn resolve(name: &str, lookup: Option<&FieldLookup>, runtime: Option<&super::RuntimeFields>) -> FieldAcc {
        if let Some(rt) = runtime {
            if let Some(i) = rt.index_of(name) {
                return FieldAcc { name: name.to_string(), kind: Kind::Runtime(i), doc_values: true, constant: None, id: None, lat_lon: None, geo_id: None, col: Col::None, geo: None, scale: None };
            }
        }
        if name == "_id" {
            return FieldAcc { name: name.to_string(), kind: Kind::Keyword, doc_values: true, constant: None, id: Some(ID_FIELD), lat_lon: None, geo_id: None, col: Col::Generic, geo: None, scale: None };
        }
        match lookup.and_then(|l| l.get(name)) {
            // aggregate_metric_double reads as its default metric (a double).
            Some(f) if f.declared == EsType::AggregateMetricDouble => {
                FieldAcc { name: name.to_string(), kind: Kind::Double, doc_values: true, constant: None, id: crate::fieldtypes::aggs_default_metric(f).and_then(|m| f.aux_id(&m)), lat_lon: None, geo_id: None, col: Col::None, geo: None, scale: None }
            }
            Some(f) => {
                let kind = if f.ty.is_text() {
                    Kind::Text
                } else if f.declared == EsType::RankVectors {
                    Kind::RankVectors(match f.ext.as_ref().and_then(|x| x.str("element_type")) {
                        Some("byte") => 1,
                        Some("bit") => 2,
                        _ => 0,
                    })
                } else if f.declared == EsType::DenseVector {
                    crate::vectors::Packing::of(f).map_or(Kind::DenseVector, Kind::PackedVector)
                } else {
                    Kind::of(f.ty)
                };
                let lat_lon = match (f.lat_id, f.lon_id) {
                    (Some(a), Some(b)) => Some((a, b)),
                    _ => None,
                };
                FieldAcc {
                    name: name.to_string(),
                    kind,
                    doc_values: f.doc_values,
                    constant: if f.declared == EsType::ConstantKeyword { f.constant_value.as_ref().map(crate::query::dsl::scalar_string) } else { None },
                    id: f.core_id,
                    lat_lon,
                    geo_id: f.geo_id,
                    col: Col::None,
                    geo: None,
                    scale: (f.ty == EsType::ScaledFloat && f.scaling_factor.is_finite() && f.scaling_factor > 0.0).then(|| (f.scaling_factor, 1.0 / f.scaling_factor)),
                }
            }
            None => FieldAcc { name: name.to_string(), kind: Kind::Unmapped, doc_values: false, constant: None, id: None, lat_lon: None, geo_id: None, col: Col::None, geo: None, scale: None },
        }
    }

    /// Bind the column readers of `leaf`.
    pub fn bind(&mut self, leaf: &LeafReader<'static>) {
        let points = match (self.kind, self.geo_id, leaf) {
            (Kind::GeoPoint, Some(g), LeafReader::Segment(s)) => s.geo_index(g).map(|i| GeoDv::Points(i.values())),
            (Kind::GeoPoint, Some(g), LeafReader::Memtable(w, ..)) => w.geo_values(g).map(GeoDv::Points),
            _ => None,
        };
        self.geo = points.or_else(|| match (self.kind, self.lat_lon, leaf) {
            (Kind::GeoPoint, Some((la, lo)), LeafReader::Segment(s)) => {
                let ff = s.fast_fields();
                match (ff.get_f64(la), ff.get_f64(lo)) {
                    (Some(a), Some(b)) => Some(GeoDv::Cols(F64Col::Seg(a), F64Col::Seg(b))),
                    _ => None,
                }
            }
            (Kind::GeoPoint, Some((la, lo)), LeafReader::Memtable(w, ..)) => {
                let ff = w.fast_fields();
                match (ff.get_f64(la), ff.get_f64(lo)) {
                    (Some(a), Some(b)) => Some(GeoDv::Cols(F64Col::Mem(a), F64Col::Mem(b))),
                    _ => None,
                }
            }
            _ => None,
        });
        let Some(id) = self.id else {
            self.col = Col::None;
            return;
        };
        self.col = match (leaf, self.kind) {
            (LeafReader::Segment(s), Kind::Long | Kind::Date) => {
                let ff = s.fast_fields();
                match ff.get_i64(id) {
                    Some(c) => Col::SegI64(c),
                    None => Col::Generic,
                }
            }
            (LeafReader::Segment(s), Kind::Double) => {
                let ff = s.fast_fields();
                match ff.get_f64(id) {
                    Some(c) => Col::SegF64(match self.scale { Some((f, _)) => c.with_es_scaled(f), None => c }),
                    None => Col::Generic,
                }
            }
            (LeafReader::Segment(_), Kind::Keyword) if super::allocation::Allocation::new().enabled()=>Col::Generic,
            (LeafReader::Segment(s), Kind::Keyword) if id != ID_FIELD => {
                match s.keyword_reader(id) {
                    Some(c) => Col::SegKeyword(c),
                    None => Col::Generic,
                }
            }
            (LeafReader::Memtable(w, ..), Kind::Long | Kind::Date) => match w.fast_fields().get_i64(id) {
                Some(c) => Col::MemI64(c),
                None => Col::Generic,
            },
            (LeafReader::Memtable(w, ..), Kind::Double) => match w.fast_fields().get_f64(id) {
                Some(c) => Col::MemF64(c),
                None => Col::Generic,
            },
            (LeafReader::Memtable(w, ..), Kind::Keyword) if id != ID_FIELD => match w.fast_fields().get_keyword(id) {
                Some(c) => Col::MemKeyword(c),
                None => Col::Generic,
            },
            _ => Col::Generic,
        };
    }

    fn check(&self) -> SResult<()> {
        match self.kind {
            Kind::Unmapped => Err(illegal_arg(format!("No field found for [{}] in mapping", self.name))),
            // `doc` access loads fielddata for search (ES `failIfNoDocValues`).
            Kind::Unsigned if !self.doc_values => Err(illegal_arg(format!(
                "Can't load fielddata on [{}] because fielddata is unsupported on fields of type [unsigned_long]. Use doc values instead.",
                self.name
            ))),
            Kind::Text => Err(illegal_arg(format!(
                "Fielddata is disabled on [{}]. Text fields are not optimised for operations that require per-document field data like aggregations and sorting, so these operations are disabled by default. Please use a keyword field instead. Alternatively, set fielddata=true on [{}] in order to load field data by uninverting the inverted index. Note that this can use significant memory.",
                self.name, self.name
            ))),
            _ => Ok(()),
        }
    }

    fn wrap_i64(&self, v: i64) -> Val {
        if self.kind == Kind::Date { Val::Date(v) } else { Val::Long(v) }
    }

    fn reserve_decoding(&self,leaf:&LeafReader<'static>,doc:DocId)->SResult<super::allocation::Allocation> {
        super::allocation::Scope::collect_dead();
        let allocation=super::allocation::Allocation::new();
        if !allocation.enabled() {return Ok(allocation)}
        let mut metadata=0u64;
        if let Some(id)=self.id {metadata=metadata.saturating_add(leaf.field_reader_allocation_bound(id));}
        allocation.reserve(metadata)?;
        let mut bytes=256u64.saturating_add(metadata);
        if let Some(value)=&self.constant {bytes=bytes.saturating_add((value.len() as u64).saturating_mul(8));}
        if let Some(id)=self.id {bytes=bytes.saturating_add(leaf.field_values_allocation_bound(id,doc));}
        if let Some((lat,lon))=self.lat_lon {bytes=bytes.saturating_add(leaf.field_values_allocation_bound(lat,doc)).saturating_add(leaf.field_values_allocation_bound(lon,doc));}
        if let Some(GeoDv::Points(v))=&self.geo {bytes=bytes.saturating_add((v.doc_range(doc).len() as u64).saturating_mul(96));}
        // Temporary upper bound only, released after the actual vector owner
        // receives its capacity charge; it does not accumulate across docs.
        if matches!(self.kind, Kind::DenseVector | Kind::PackedVector(_)) {bytes=bytes.saturating_add(4096*256);}
        allocation.reserve(bytes)?;
        Ok(allocation)
    }

    /// First (smallest) value, without allocating for single-valued numeric
    /// columns.
    #[inline]
    pub fn first(&self, leaf: &LeafReader<'static>, doc: DocId) -> SResult<Option<Val>> {
        let _decode=self.reserve_decoding(leaf,doc)?;
        let result=self.first_decoded(leaf,doc)?;
        if let Some(value)=&result {super::allocation::Scope::adopt(value)?;}
        Ok(result)
    }
    fn first_decoded(&self,leaf:&LeafReader<'static>,doc:DocId)->SResult<Option<Val>> {
        if let Some(c) = &self.constant {
            return Ok(Some(Val::str(c)));
        }
        match &self.col {
            // Column readers report presence (`get` is `None` for docs
            // without a value).
            Col::SegF64(c) => return Ok(c.get(doc).map(Val::Double)),
            Col::SegI64(c) => return Ok(c.get(doc).map(|v| self.wrap_i64(v))),
            Col::MemF64(c) => return Ok(c.get(doc).map(|v| Val::Double(crate::aggs::es_scaled(v, self.scale)))),
            Col::MemI64(c) => return Ok(c.get(doc).map(|v| self.wrap_i64(v))),
            Col::SegKeyword(c) => return Ok(c.get(doc).map(|v| Val::str(&v))),
            Col::MemKeyword(c) => return Ok(c.get(doc).map(Val::str)),
            _ => {}
        }
        self.check()?;
        Ok(self.all_decoded(leaf, doc)?.into_iter().next())
    }

    /// The (quantized) point of a doc whose `geo_point` is single-valued
    /// and present; `None` when the doc has no point or the leaf's columns
    /// are not single-valued (callers then take the general path).
    #[inline]
    pub fn geo_point(&self, doc: DocId) -> Option<(f64, f64)> {
        match self.geo.as_ref()? {
            GeoDv::Points(v) => {
                let r = v.doc_range(doc);
                if r.len() != 1 {
                    return None;
                }
                let p = v.entry(r.start);
                Some((rustic_core::geo::decode_lat(p.lat), rustic_core::geo::decode_lon(p.lon)))
            }
            GeoDv::Cols(la, lo) => {
                let (a, b) = (la.get(doc)?, lo.get(doc)?);
                Some(rustic_core::geo::quantize(a, b))
            }
        }
    }

    /// Number of values.
    pub fn count(&self, leaf: &LeafReader<'static>, doc: DocId) -> SResult<usize> {
        match &self.col {
            Col::SegF64(..) | Col::SegI64(..) | Col::MemF64(_) | Col::MemI64(_) | Col::SegKeyword(_) | Col::MemKeyword(_) => Ok(self.first(leaf, doc)?.is_some() as usize),
            _ if self.geo_point(doc).is_some() => Ok(1),
            _ => Ok(self.all(leaf, doc)?.len()),
        }
    }

    /// All values, sorted ascending (ES doc values order).
    pub fn all(&self, leaf: &LeafReader<'static>, doc: DocId) -> SResult<Vec<Val>> {
        let _decode=self.reserve_decoding(leaf,doc)?;
        let result=self.all_decoded(leaf,doc)?;
        for value in &result {super::allocation::Scope::adopt(value)?;}
        Ok(result)
    }
    fn all_decoded(&self,leaf:&LeafReader<'static>,doc:DocId)->SResult<Vec<Val>> {
        self.check()?;
        match &self.col {
            Col::SegF64(..) | Col::SegI64(..) | Col::MemF64(_) | Col::MemI64(_) | Col::SegKeyword(_) | Col::MemKeyword(_) => {
                return Ok(self.first_decoded(leaf, doc)?.into_iter().collect());
            }
            _ => {}
        }
        if self.kind == Kind::GeoPoint {
            if let Some((a, b)) = self.geo_point(doc) {
                return Ok(vec![geo_point_val((a, b))]);
            }
            if let Some(GeoDv::Points(v)) = &self.geo {
                return Ok(sorted_geo_points(v, doc).into_iter().map(geo_point_val).collect());
            }
            if let Some((la, lo)) = self.lat_lon {
                let lats = leaf.f64_values(la, doc);
                let lons = leaf.f64_values(lo, doc);
                return Ok(lats
                    .into_iter()
                    .zip(lons)
                    .map(|(a, b)| {
                        // Doc values hold Lucene's quantized (32-bit) coordinates.
                        let (a, b) = rustic_core::geo::quantize(a, b);
                        let mut m = indexmap::IndexMap::new();
                        m.insert(super::value::Key::str("lat"), Val::Double(a));
                        m.insert(super::value::Key::str("lon"), Val::Double(b));
                        Val::new_map(m)
                    })
                    .collect());
            }
            return Ok(Vec::new());
        }
        if let Some(c) = &self.constant {
            return Ok(vec![Val::str(c)]);
        }
        let Some(id) = self.id else { return Ok(Vec::new()) };
        if self.kind == Kind::DenseVector {
            return Ok(leaf.vector(id, doc).map(|v| Val::Obj(std::rc::Rc::new(super::jmisc::Obj::DenseVector(v)))).into_iter().collect());
        }
        if let Kind::PackedVector(packing) = self.kind {
            return Ok(leaf.vector(id, doc).map(|v| Val::Obj(std::rc::Rc::new(super::jmisc::Obj::DenseVector(packing.unpack(&v))))).into_iter().collect());
        }
        if self.kind == Kind::GeoShape {
            return Ok(leaf.shape_value(id, doc).and_then(|v| geo_shape_value(&v)).into_iter().collect());
        }
        if let Kind::RankVectors(et) = self.kind {
            return Ok(leaf.keyword_values(id, doc).first().and_then(|j| rank_vectors_value(j, et)).into_iter().collect());
        }
        if self.kind == Kind::Unsigned {
            // Unsigned order, as two's-complement longs.
            let mut u = leaf.u64_values(id, doc);
            u.sort_unstable();
            return Ok(u.into_iter().map(|v| Val::Long(v as i64)).collect());
        }
        let mut out: Vec<Val> = match self.kind {
            Kind::Long => leaf.i64_values(id, doc).into_iter().map(Val::Long).collect(),
            Kind::Date => leaf.i64_values(id, doc).into_iter().map(Val::Date).collect(),
            Kind::Unsigned => leaf.u64_values(id, doc).into_iter().map(|v| Val::Long(v as i64)).collect(),
            Kind::Double => leaf.f64_values(id, doc).into_iter().map(|v| Val::Double(crate::aggs::es_scaled(v, self.scale))).collect(),
            Kind::Bool => leaf.bool_value(id, doc).map(Val::Bool).into_iter().collect(),
            Kind::Keyword => leaf.keyword_values(id, doc).into_iter().map(Val::string).collect(),
            Kind::Ip => leaf.keyword_values(id, doc).iter().map(|t| Val::string(crate::values::ip_display(t))).collect(),
            _ => Vec::new(),
        };
        if out.len() > 1 {
            out.sort_by(|a, b| a.compare(b).unwrap_or(std::cmp::Ordering::Equal));
            if self.kind == Kind::Keyword {
                out.dedup_by(|a, b| a.equals(b));
            }
        }
        Ok(out)
    }
}

/// The `doc` lookup of one executor: resolved fields + current leaf/doc.
pub struct DocCtx {
    pub lookup: Option<Arc<FieldLookup>>,
    pub runtime: Option<Arc<super::RuntimeFields>>,
    pub fields: Vec<FieldAcc>,
    pub leaf: Option<LeafReader<'static>>,
    pub doc: DocId,
    /// Lazily created executors of runtime fields referenced via `doc`.
    pub rt_execs: Vec<Option<Box<super::Executor>>>,
    /// Recursion guard for runtime fields referencing each other.
    pub depth: u32,
    /// Doc values taken from a JSON document instead of a leaf
    /// (`_scripts/painless/_execute` with `context_setup.document`).
    pub source_json: Option<serde_json::Value>,
}

pub const MISSING_VALUE: &str = "A document doesn't have a value for a field! Use doc[<field>].size()==0 to check if a document is missing a field!";

impl DocCtx {
    pub fn new(names: &[String], lookup: Option<Arc<FieldLookup>>, runtime: Option<Arc<super::RuntimeFields>>) -> DocCtx {
        let fields = names.iter().map(|n| FieldAcc::resolve(n, lookup.as_deref(), runtime.as_deref())).collect();
        let n = runtime.as_ref().map(|r| r.fields.len()).unwrap_or(0);
        DocCtx { lookup, runtime, fields, leaf: None, doc: 0, rt_execs: (0..n).map(|_| None).collect(), depth: 0, source_json: None }
    }

    pub fn bind(&mut self, leaf: LeafReader<'static>) {
        for f in &mut self.fields {
            f.bind(&leaf);
        }
        for e in self.rt_execs.iter_mut().flatten() {
            // SAFETY: same contract as the outer binding (see `Executor::bind_leaf`).
            unsafe { e.bind_leaf(&leaf) };
        }
        self.leaf = Some(leaf);
    }

    /// Slot of a dynamically named field (resolved and bound on first use).
    /// The current doc's dense vector in field `name` (script vector functions).
    pub fn vector(&mut self, name: &str) -> Option<Vec<f32>> {
        let slot = self.slot_of(name);
        let id = self.fields[slot].id?;
        let row = self.leaf.as_ref()?.vector(id, self.doc)?;
        Some(match self.fields[slot].kind {
            Kind::PackedVector(packing) => packing.unpack(&row),
            _ => row,
        })
    }

    pub fn with_vector<T>(&mut self, name: &str, f: impl FnOnce(&[f32]) -> T) -> Option<T> {
        let slot = self.slot_of(name);
        let id = self.fields[slot].id?;
        if let Kind::PackedVector(packing) = self.fields[slot].kind {
            return self.leaf.as_ref()?.with_vector(id, self.doc, |row| f(&packing.unpack(row)));
        }
        self.leaf.as_ref()?.with_vector(id, self.doc, f)
    }

    pub fn slot_of(&mut self, name: &str) -> usize {
        if let Some(i) = self.fields.iter().position(|f| f.name == name) {
            return i;
        }
        let mut f = FieldAcc::resolve(name, self.lookup.as_deref(), self.runtime.as_deref());
        if let Some(l) = &self.leaf {
            f.bind(l);
        }
        self.fields.push(f);
        self.fields.len() - 1
    }

    /// Values of `slot` read from `source_json` with the mapping's types.
    fn json_values(&self, slot: usize) -> SResult<Vec<Val>> {
        let f = &self.fields[slot];
        f.check()?;
        let src = self.source_json.as_ref().unwrap();
        let rf = self.lookup.as_ref().and_then(|l| l.get(&f.name).cloned());
        let mut out = Vec::new();
        for v in crate::highlight::source_values(src, &f.name) {
            let x = match (f.kind, v) {
                (_, serde_json::Value::Null) => continue,
                (Kind::Long | Kind::Unsigned, v) => v.as_i64().or_else(|| v.as_f64().map(|x| x as i64)).or_else(|| v.as_str().and_then(|s| s.parse().ok())).map(Val::Long),
                (Kind::Double, v) => v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok())).map(Val::Double),
                (Kind::Bool, v) => v.as_bool().or_else(|| v.as_str().map(|s| s == "true")).map(Val::Bool),
                (Kind::Date, serde_json::Value::Number(n)) => Some(Val::Date(n.as_f64().unwrap_or(0.0) as i64)),
                (Kind::Date, serde_json::Value::String(s)) => match &rf {
                    Some(r) => r.date_format.parse(s).ok().map(Val::Date),
                    None => None,
                },
                (Kind::Keyword | Kind::Ip, v) => Some(Val::string(crate::query::dsl::scalar_string(v))),
                _ => None,
            };
            if let Some(x) = x {
                out.push(x);
            }
        }
        out.sort_by(|a, b| a.compare(b).unwrap_or(std::cmp::Ordering::Equal));
        Ok(out)
    }

    fn leaf(&self) -> SResult<&LeafReader<'static>> {
        self.leaf.as_ref().ok_or_else(|| illegal_arg("cannot access doc values outside of a search context"))
    }

    fn runtime_values(&mut self, idx: usize) -> SResult<Vec<Val>> {
        if self.depth > 8 {
            return Err(illegal_arg("Cyclic dependency detected while resolving runtime fields"));
        }
        let rt = self.runtime.clone().ok_or_else(|| illegal_arg("runtime field without runtime set"))?;
        let leaf = *self.leaf()?;
        if self.rt_execs[idx].is_none() {
            let mut e = rt.fields[idx].executor(self.lookup.clone(), Some(rt.clone()));
            e.rt.doc.depth = self.depth + 1;
            // SAFETY: `leaf` is the executor's own bound leaf; same lifetime contract.
            unsafe { e.bind_leaf(&leaf) };
            self.rt_execs[idx] = Some(Box::new(e));
        }
        let doc = self.doc;
        let e = self.rt_execs[idx].as_mut().unwrap();
        rt.fields[idx].values(e, doc).map_err(|err| illegal_arg(format!("failed to evaluate runtime field [{}]: {}", rt.fields[idx].name, err.msg)))
    }

    /// First value of `slot` when it can be read straight from a bound
    /// single-valued column and the doc has one; `None` = take the general
    /// path (which also produces any error).
    #[inline]
    pub fn fast_first(&self, slot: usize) -> Option<Val> {
        let f = self.fields.get(slot)?;
        let d = self.doc;
        match &f.col {
            Col::SegF64(c) => c.get(d).map(Val::Double),
            Col::SegI64(c) => c.get(d).map(|v| f.wrap_i64(v)),
            Col::MemF64(c) => c.get(d).map(|v| Val::Double(crate::aggs::es_scaled(v, f.scale))),
            Col::MemI64(c) => c.get(d).map(|v| f.wrap_i64(v)),
            _ => None,
        }
    }

    /// [`Self::fast_first`] as an unboxed number (`None` also for dates).
    #[inline]
    pub(crate) fn fast_num(&self, slot: usize) -> Option<super::compile::Num> {
        use super::compile::Num;
        let f = self.fields.get(slot)?;
        let d = self.doc;
        match &f.col {
            Col::SegF64(c) => c.get(d).map(Num::D),
            Col::MemF64(c) => c.get(d).map(Num::D),
            Col::SegI64(c) if f.kind != Kind::Date => c.get(d).map(Num::L),
            Col::MemI64(c) if f.kind != Kind::Date => c.get(d).map(Num::L),
            _ => None,
        }
    }

    /// Expressions: the value of `slot` when it is read from a single-valued
    /// numeric / date column (`Ok(None)`: the doc has none); `Err(())`:
    /// take the general path ([`Self::values`]).
    #[inline]
    pub fn expr_single(&self, slot: usize) -> Result<Option<f64>, ()> {
        let f = &self.fields[slot];
        let d = self.doc;
        match &f.col {
            Col::SegF64(c) => Ok(c.get(d)),
            Col::MemF64(c) => Ok(c.get(d)),
            Col::SegI64(c) => Ok(c.get(d).map(|v| v as f64)),
            Col::MemI64(c) => Ok(c.get(d).map(|v| v as f64)),
            _ => Err(()),
        }
    }

    /// Expressions: whether `slot`'s point is read from single-valued lat /
    /// lon columns ([`Self::expr_geo_coord`]).
    #[inline]
    pub fn expr_geo_fast(&self, slot: usize) -> bool {
        self.fields[slot].geo.is_some()
    }

    /// Expressions: one (quantized) coordinate of the point of `slot`
    /// (requires [`Self::expr_geo_fast`]); `None` when the doc has none.
    #[inline]
    pub fn expr_geo_coord(&self, slot: usize, lat: bool) -> Option<f64> {
        self.fast_geo(slot, lat)
    }

    /// Expressions: the (quantized) point of a `geo_point` `slot` read from
    /// single-valued lat / lon columns; `Err(())`: take the general path.
    #[inline]
    pub fn expr_geo(&self, slot: usize) -> Result<Option<(f64, f64)>, ()> {
        let f = &self.fields[slot];
        if f.geo.is_none() {
            return Err(());
        }
        Ok(f.geo_point(self.doc))
    }

    /// Epoch millis of a single-valued `date` field's value.
    #[inline]
    pub fn fast_date(&self, slot: usize) -> Option<i64> {
        let f = self.fields.get(slot)?;
        match &f.col {
            Col::SegI64(c) if f.kind == Kind::Date => c.get(self.doc),
            Col::MemI64(c) if f.kind == Kind::Date => c.get(self.doc),
            _ => None,
        }
    }

    /// `lat` (or `lon`) of the single point of `slot` (see
    /// [`FieldAcc::geo_point`]).
    #[inline]
    pub fn fast_geo(&self, slot: usize, lat: bool) -> Option<f64> {
        let (a, b) = self.fields.get(slot)?.geo_point(self.doc)?;
        Some(if lat { a } else { b })
    }

    /// `doc['f'].value`.
    #[inline]
    pub fn value(&mut self, slot: usize) -> SResult<Val> {
        if let Some(v) = self.fast_first(slot) {
            return Ok(v);
        }
        self.value_slow(slot)
    }

    #[inline(never)]
    fn value_slow(&mut self, slot: usize) -> SResult<Val> {
        if let Kind::Runtime(i) = self.fields[slot].kind {
            return self.runtime_values(i)?.into_iter().next().ok_or_else(|| illegal_state(MISSING_VALUE));
        }
        if self.leaf.is_none() && self.source_json.is_some() {
            return self.json_values(slot)?.into_iter().next().ok_or_else(|| illegal_state(MISSING_VALUE));
        }
        let leaf = self.leaf()?;
        match self.fields[slot].first(leaf, self.doc)? {
            Some(v) => Ok(v),
            None => Err(illegal_state(MISSING_VALUE)),
        }
    }

    pub fn size(&mut self, slot: usize) -> SResult<usize> {
        if let Kind::Runtime(i) = self.fields[slot].kind {
            return Ok(self.runtime_values(i)?.len());
        }
        if self.leaf.is_none() && self.source_json.is_some() {
            return Ok(self.json_values(slot)?.len());
        }
        let leaf = self.leaf()?;
        self.fields[slot].count(leaf, self.doc)
    }

    pub fn values(&mut self, slot: usize) -> SResult<Val> {
        let vals = match self.fields[slot].kind {
            Kind::Runtime(i) => self.runtime_values(i)?,
            _ if self.leaf.is_none() && self.source_json.is_some() => self.json_values(slot)?,
            _ => {
                let leaf = self.leaf()?;
                self.fields[slot].all(leaf, self.doc)?
            }
        };
        let _outer=super::allocation::Allocation::values(vals.capacity())?;
        _outer.reserve((vals.capacity() as u64).saturating_mul(std::mem::size_of::<Val>() as u64).saturating_add(self.fields[slot].name.len() as u64).saturating_add(256))?;
        let value=Val::DocValues(Rc::new(DocValues { field: self.fields[slot].name.clone(), values: vals }));
        super::allocation::Scope::adopt(&value)?;
        Ok(value)
    }

    /// The fields API's view of `slot`: doc values, or (for unsigned longs
    /// without doc values, ES `SourceValueFetcherSortedUnsignedLong...`)
    /// the `_source` values in unsigned order.
    pub fn field_api_values(&mut self, slot: usize) -> SResult<Option<Vec<Val>>> {
        let f = &self.fields[slot];
        if f.kind != Kind::Unsigned || f.doc_values {
            return Ok(None);
        }
        let name = f.name.clone();
        let src = match (&self.source_json, &self.leaf) {
            (Some(j), _) => Some(j.clone()),
            (None, Some(leaf)) => leaf
                .get_doc(self.doc).map_err(|e| super::error::ScriptErr::breaker(crate::error::EsError::from(e)))?
                .and_then(|d| d.get(crate::mapping::SOURCE_FIELD).and_then(|v| v.as_text()).and_then(|s| serde_json::from_str::<serde_json::Value>(crate::synthetic::residual_json(s)).ok())),
            _ => None,
        };
        let mut u: Vec<u64> = Vec::new();
        if let Some(src) = src {
            for v in crate::highlight::source_values(&src, &name) {
                let x = match v {
                    serde_json::Value::Number(n) => n.as_u64().or_else(|| n.as_f64().filter(|f| *f >= 0.0).map(|f| f as u64)),
                    serde_json::Value::String(s) => s.parse::<u64>().ok(),
                    _ => None,
                };
                u.extend(x);
            }
        }
        u.sort_unstable();
        Ok(Some(u.into_iter().map(|v| Val::Long(v as i64)).collect()))
    }

    pub fn contains(&mut self, name: &str) -> bool {
        let s = self.slot_of(name);
        !matches!(self.fields[s].kind, Kind::Unmapped)
    }
}

/// Marker key of a `rank_vectors` doc value map (ES
/// `RankVectorsScriptDocValues`): `vectors` (arrays), `magnitudes`,
/// `element_type`.
pub const RANK_VECTORS_KEY: &str = "\u{0}rank_vectors";

/// The script value of one doc's `rank_vectors` (stored as JSON).
pub fn rank_vectors_value(json: &str, et: u8) -> Option<Val> {
    use super::value::Key;
    let vecs: Vec<Vec<f64>> = serde_json::from_str(json).ok()?;
    let mut vs = Vec::new();
    let mut mags = Vec::new();
    for v in &vecs {
        let mag = match et {
            0 => {
                let f: Vec<f32> = v.iter().map(|x| *x as f32).collect();
                (rustic_core::vector::dot(&f, &f) as f64).sqrt() as f32 as f64
            }
            1 => (v.iter().map(|x| (*x as i64) * (*x as i64)).sum::<i64>() as f64).sqrt() as f32 as f64,
            _ => (v.iter().map(|x| (*x as i64 as u8).count_ones() as u64).sum::<u64>() as f64).sqrt() as f32 as f64,
        };
        mags.push(Val::Double(mag));
        let (elem, items): (super::ast::Ty, Vec<Val>) = if et == 0 {
            (super::ast::Ty::Float, v.iter().map(|x| Val::Double(*x as f32 as f64)).collect())
        } else {
            (super::ast::Ty::Byte, v.iter().map(|x| Val::Int(*x as i64 as i8 as i32)).collect())
        };
        vs.push(Val::new_array(elem, items));
    }
    let mut m = indexmap::IndexMap::new();
    m.insert(Key::str(RANK_VECTORS_KEY), Val::Bool(true));
    m.insert(Key::str("element_type"), Val::Int(et as i32));
    m.insert(Key::str("vectors"), Val::new_list(vs));
    m.insert(Key::str("magnitudes"), Val::new_list(mags));
    Some(Val::new_map(m))
}

/// Marker key of a `geo_shape` doc value map (ES `GeoShapeValue`, which
/// scripts can query but not serialize).
pub const GEO_SHAPE_KEY: &str = "\u{0}geo_shape";

fn geo_point_val(p: (f64, f64)) -> Val {
    let mut m = indexmap::IndexMap::new();
    m.insert(super::value::Key::str("lat"), Val::Double(p.0));
    m.insert(super::value::Key::str("lon"), Val::Double(p.1));
    Val::new_map(m)
}

/// A `GeoBoundingBox` value (`top_left` / `bottom_right` points).
pub fn geo_bbox_val(top: f64, left: f64, bottom: f64, right: f64) -> Val {
    let mut m = indexmap::IndexMap::new();
    m.insert(super::value::Key::str("top_left"), geo_point_val((top, left)));
    m.insert(super::value::Key::str("bottom_right"), geo_point_val((bottom, right)));
    Val::new_map(m)
}

/// `getLabelPosition` of a `geo_shape` doc value ([`geo_shape_value`]).
pub fn geo_shape_label(v: &Val) -> Val {
    let Val::Map(m) = v else { return Val::Null };
    m.borrow().get(&super::value::Key::str("label")).cloned().unwrap_or(Val::Null)
}

/// The script value of a `geo_shape` doc (ES `GeoShapeValues.GeoShapeValue`
/// over its shape doc value): centroid, bounding box, dimension and label
/// position.
pub fn geo_shape_value(v: &rustic_core::fieldtypes::shape_dv::ShapeValue<'_>) -> Option<Val> {
    use crate::aggs::geo_shape::{lat_mercator, lon_mercator};
    let b = v.bounding_box()?;
    let (top, left, bottom, right) = (b.max_y(), b.min_x(), b.min_y(), b.max_x());
    let (x, y) = v.centroid();
    let k = super::value::Key::str;
    let mut m = indexmap::IndexMap::new();
    m.insert(k(GEO_SHAPE_KEY), Val::Bool(true));
    m.insert(k("centroid"), geo_point_val((y, x)));
    m.insert(k("bbox"), geo_bbox_val(top, left, bottom, right));
    m.insert(k("label"), v.label_position().map(|(x, y)| geo_point_val((y, x))).unwrap_or(Val::Null));
    m.insert(k("dim"), Val::Int(v.dimension().ordinal() as i32));
    m.insert(k("mw"), Val::Double(lon_mercator(right) - lon_mercator(left)));
    m.insert(k("mh"), Val::Double(lat_mercator(top) - lat_mercator(bottom)));
    Some(Val::new_map(m))
}

#[cfg(test)]
mod allocation_tests {
    use super::*;
    #[test]
    fn metric_doc_value_decode_is_bounded_by_live_values() {
        use rustic_core::{segment::SegmentWriter, types::{Document, Field, FieldType}};
        let mut field = Field::new(0, "value", FieldType::Keyword);
        field.fast = true;
        let mut writer = SegmentWriter::new(0);
        for _ in 0..1000 {
            let mut document = Document::new();
            document.add_keyword(0, "constant");
            writer.add_document(&document, &[field.clone()]);
        }
        let leaf = LeafReader::Memtable(Box::leak(Box::new(writer)), None);
        let mut access = FieldAcc { name: "value".into(), kind: Kind::Keyword, doc_values: true, constant: None, id: Some(0), lat_lon: None, geo_id: None, col: Col::Generic, geo: None, scale: None };
        let breakers = crate::memory::Breakers::leak_for_tests(1 << 30);
        breakers.apply_settings(&|key| match key {
            "indices.breaker.request.limit" => Some("64kb".into()),
            "indices.breaker.total.limit" => Some("-1".into()),
            _ => None,
        });
        let request = crate::memory::RequestScope::new("bounded metric doc values", breakers);
        crate::memory::scoped(Some(request), || {
            let scope = super::super::allocation::Scope::enter();
            access.bind(&leaf);
            for doc in 0..1000 {
                let values = access.all(&leaf, doc).unwrap();
                assert_eq!(values.len(), 1);
                drop(values);
                super::super::allocation::Scope::collect_all_dead();
                assert!(breakers.request.reserved() < 4096);
            }
            drop(scope);
            assert_eq!(breakers.request.reserved(), 0);
        });
    }

    #[test]
    fn metric_doc_keywords_break_before_owned_value_copy() {
        use rustic_core::{segment::SegmentWriter,types::{Document,Field,FieldType}};
        let mut field=Field::new(0,"value",FieldType::Keyword);
        field.fast=true;
        let mut writer=SegmentWriter::new(0);
        let mut document=Document::new();
        document.add_keyword(0,&"a".repeat(20000));
        document.add_keyword(0,&"b".repeat(20000));
        writer.add_document(&document,&[field]);
        let writer=Box::leak(Box::new(writer));
        let leaf=LeafReader::Memtable(writer,None);
        let mut access=FieldAcc {name:"value".into(),kind:Kind::Keyword,doc_values:true,constant:None,id:Some(0),lat_lon:None,geo_id:None,col:Col::Generic,geo:None,scale:None};
        access.bind(&leaf);
        let breakers=crate::memory::Breakers::leak_for_tests(1 << 30);
        for limit in ["64kb","64mb"] {
            breakers.apply_settings(&|key|match key {
                "indices.breaker.request.limit"=>Some(limit.into()),
                "indices.breaker.total.limit"=>Some("-1".into()),
                _=>None,
            });
            let request=crate::memory::RequestScope::new("metric doc values",breakers);
            let result=crate::memory::scoped(Some(request),|| {
                let _metric=super::super::allocation::Scope::enter();
                access.all(&leaf,0)
            });
            if limit=="64kb" {assert!(result.unwrap_err().breaker.is_some());}
            else {assert_eq!(result.unwrap().len(),2);}
            assert_eq!(breakers.request.reserved(),0);
        }
    }
}
