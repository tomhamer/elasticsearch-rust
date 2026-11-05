//! Scroll and point-in-time contexts that span nodes.
//!
//! A context whose indices all live on one node is that node's engine
//! context, its id wrapped with the owner (`{node}.{engine id}`, see
//! `wrap_id`). When no single node holds a copy of every target index the
//! front opens one engine context per node of the cover and combines them:
//!
//! * **Composite scroll** (`{node}.~s{key}`): kept by the coordinating node.
//!   It remembers each sub-scroll's id and the hits fetched but not yet
//!   returned, and serves every page as a k-way merge of the sub streams in
//!   the request's sort order. A sub is refilled before a page is cut while
//!   it holds fewer than a page of buffered hits, so no hit is ever returned
//!   out of order, twice, or not at all.
//! * **Composite PIT** (`{node}.~p{hex json [sub ids]}`): stateless; the id
//!   lists the per-node PIT ids. Searches fan out with the sub id in
//!   `pit.id` and are merged like any cross-node search. The `_shard_doc`
//!   tiebreak gets the sub's ordinal in its top bits so that `search_after`
//!   can be split back into a per-sub cursor.
//!
//! On indices with more than one shard a sub is one (node, shard) engine
//! context: its id is tagged `~k{shard}:{node}.{engine id}` (shard 0 keeps
//! the plain wrapped id) and every request on it carries the shard header.

use super::copies;
use super::{exec_on, fanout, forward, local_engine, unwrap_id, wrap_id, RestReq, RestResp};
use crate::node::ClusterNode;
use parking_lot::Mutex;
use serde_json::{json, Map, Value};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

pub(super) const SCROLL_PREFIX: &str = "~s";
pub(super) const PIT_PREFIX: &str = "~p";
/// Bits of a `_shard_doc` value left to the engine (`leaf << 32 | doc`).
const SUB_SHIFT: u32 = 52;
const LOCAL_MASK: i64 = (1i64 << SUB_SHIFT) - 1;

/// A buffered sub-scroll hit: the fields the merge orders by, and the hit's
/// JSON exactly as the sub returned it (its `_source` is never parsed).
struct BufHit {
    index: Option<String>,
    score: Option<f64>,
    sort: Option<Vec<Value>>,
    /// The hit's JSON text, written to the client verbatim.
    raw: Box<str>,
}

#[derive(serde::Deserialize)]
#[serde(field_identifier)]
enum HitField {
    #[serde(rename = "_index")]
    Index,
    #[serde(rename = "_score")]
    Score,
    #[serde(rename = "sort")]
    Sort,
    #[serde(other)]
    Other,
}

#[derive(Default)]
struct HitKeys {
    index: Option<String>,
    score: Option<f64>,
    sort: Option<Vec<Value>>,
}

/// Reads a hit's merge keys into `keys`. Without sort keys only `_index`
/// and `_score` matter, and the engine writes both before `_source`, so the
/// read stops as soon as it has them instead of skipping the rest of the hit.
struct KeyReader<'a> {
    keys: &'a mut HitKeys,
    need_sort: bool,
}

impl<'de> serde::de::Visitor<'de> for KeyReader<'_> {
    type Value = ();
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a search hit")
    }
    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut m: A) -> Result<(), A::Error> {
        let (mut index, mut score) = (false, false);
        while let Some(k) = m.next_key::<HitField>()? {
            match k {
                HitField::Index => { self.keys.index = m.next_value()?; index = true; }
                HitField::Score => { self.keys.score = m.next_value::<Value>()?.as_f64(); score = true; }
                HitField::Sort if self.need_sort => self.keys.sort = m.next_value()?,
                _ => { m.next_value::<serde::de::IgnoredAny>()?; }
            }
            if index && score && !self.need_sort {
                // Leaves the map unfinished; the caller ignores that error.
                return Ok(());
            }
        }
        Ok(())
    }
}

fn buf_hit(raw: Box<serde_json::value::RawValue>, need_sort: bool) -> BufHit {
    use serde::Deserializer as _;
    let mut keys = HitKeys::default();
    let _ = serde_json::Deserializer::from_str(raw.get()).deserialize_map(KeyReader { keys: &mut keys, need_sort });
    BufHit { index: keys.index, score: keys.score, sort: keys.sort, raw: raw.get().into() }
}

/// A sub response read in one pass: everything but `hits.hits` as JSON
/// (`hits.hits` left empty), and the hits unparsed beyond their merge keys.
struct SubBody {
    meta: Map<String, Value>,
    hits: VecDeque<BufHit>,
}

struct SubBodyReader {
    need_sort: bool,
    /// Reading the `hits` object rather than the top level.
    inner: bool,
}

impl<'de> serde::de::DeserializeSeed<'de> for SubBodyReader {
    type Value = SubBody;
    fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<SubBody, D::Error> {
        d.deserialize_map(self)
    }
}

impl<'de> serde::de::Visitor<'de> for SubBodyReader {
    type Value = SubBody;
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a search response")
    }
    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut m: A) -> Result<SubBody, A::Error> {
        let mut out = SubBody { meta: Map::new(), hits: VecDeque::new() };
        while let Some(k) = m.next_key::<String>()? {
            if k == "hits" && !self.inner {
                let h = m.next_value_seed(SubBodyReader { need_sort: self.need_sort, inner: true })?;
                out.hits = h.hits;
                out.meta.insert(k, Value::Object(h.meta));
            } else if k == "hits" {
                let raw: Vec<Box<serde_json::value::RawValue>> = m.next_value()?;
                out.hits = raw.into_iter().map(|r| buf_hit(r, self.need_sort)).collect();
                out.meta.insert(k, Value::Array(Vec::new()));
            } else {
                let v: Value = m.next_value()?;
                out.meta.insert(k, v);
            }
        }
        Ok(out)
    }
}

/// The hits of a sub-scroll response, unparsed beyond their merge keys, and
/// the rest of the response. `need_sort`: the merge orders by `sort` values.
///
/// Each hit is captured once as a `RawValue` borrowed from the body (serde's
/// word-at-a-time string skip) and copied once. Its keys are read without a
/// second pass over `_source`: `_index` and `_score` from the head (the
/// engine writes them first, so the read stops there) and `sort` from the
/// tail ([`tail_sort`]); a hit whose tail does not end with a top-level
/// `sort` member is read whole, as [`sub_hits_serde`] reads every hit.
fn sub_hits(body: &[u8], need_sort: bool) -> Option<SubBody> {
    use serde::de::DeserializeSeed as _;
    let mut d = serde_json::Deserializer::from_slice(body);
    let b = BorrowedBodyReader { need_sort, inner: false }.deserialize(&mut d).ok()?;
    d.end().ok()?;
    Some(b)
}

/// [`SubBodyReader`] with the hits borrowed from the body ([`sub_hits`]).
struct BorrowedBodyReader {
    need_sort: bool,
    inner: bool,
}

impl<'de> serde::de::DeserializeSeed<'de> for BorrowedBodyReader {
    type Value = SubBody;
    fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<SubBody, D::Error> {
        d.deserialize_map(self)
    }
}

impl<'de> serde::de::Visitor<'de> for BorrowedBodyReader {
    type Value = SubBody;
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a search response")
    }
    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut m: A) -> Result<SubBody, A::Error> {
        let mut out = SubBody { meta: Map::new(), hits: VecDeque::new() };
        while let Some(k) = m.next_key::<String>()? {
            if k == "hits" && !self.inner {
                let h = m.next_value_seed(BorrowedBodyReader { need_sort: self.need_sort, inner: true })?;
                out.hits = h.hits;
                out.meta.insert(k, Value::Object(h.meta));
            } else if k == "hits" {
                let raw: Vec<&'de serde_json::value::RawValue> = m.next_value()?;
                out.hits = raw.into_iter().map(|r| buf_hit_str(r.get(), self.need_sort)).collect();
                out.meta.insert(k, Value::Array(Vec::new()));
            } else {
                let v: Value = m.next_value()?;
                out.meta.insert(k, v);
            }
        }
        Ok(out)
    }
}

/// [`sub_hits`] of a page whose hits' spans the engine reported
/// ([`elasticsearch_http::encode_hit_spans`]): the hits are cut out of the
/// body without reading it, only the rest of the response is parsed.
/// `None` when the spans do not fit the body (it was edited after them):
/// the caller reads the page whole.
fn sub_hits_spanned(body: &[u8], spans: &str, need_sort: bool) -> Option<SubBody> {
    let mut nums = spans.split(',').map(|n| n.parse::<usize>().ok());
    let tail = nums.next()??;
    let lens: Vec<usize> = nums.collect::<Option<_>>()?;
    let end = body.len().checked_sub(tail)?;
    let text = std::str::from_utf8(body).ok()?;
    let mut hits = VecDeque::with_capacity(lens.len());
    let mut start = end.checked_sub(lens.iter().sum::<usize>() + lens.len().saturating_sub(1))?;
    // `[` before the first hit, `]` after the last, a comma between two.
    if body.get(start.checked_sub(1)?) != Some(&b'[') || body.get(end) != Some(&b']') {
        return None;
    }
    let first = start;
    for (n, len) in lens.iter().enumerate() {
        let hit = text.get(start..start + len)?;
        if !(hit.starts_with('{') && hit.ends_with('}')) {
            return None;
        }
        hits.push_back(buf_hit_str(hit, need_sort));
        start += len;
        if n + 1 < lens.len() {
            if body.get(start) != Some(&b',') {
                return None;
            }
            start += 1;
        }
    }
    // The rest of the response, its hits left out.
    let mut rest = Vec::with_capacity(first + body.len() - end);
    rest.extend_from_slice(&body[..first]);
    rest.extend_from_slice(&body[end..]);
    let Value::Object(meta) = serde_json::from_slice::<Value>(&rest).ok()? else { return None };
    match meta.get("hits").and_then(Value::as_object).and_then(|h| h.get("hits")) {
        Some(Value::Array(a)) if a.is_empty() => {}
        _ => return None,
    }
    Some(SubBody { meta, hits })
}

/// [`buf_hit`] of a hit's text without reading past its `_source` twice.
/// Unlike `KeyReader` with `sort` wanted, a key repeated in the hit gives
/// its first value (the engine writes each key once).
fn buf_hit_str(raw: &str, need_sort: bool) -> BufHit {
    use serde::Deserializer as _;
    let mut keys = HitKeys::default();
    let _ = serde_json::Deserializer::from_str(raw).deserialize_map(KeyReader { keys: &mut keys, need_sort: false });
    if need_sort {
        match tail_sort(raw) {
            Some(sort) => keys.sort = sort,
            None => {
                keys = HitKeys::default();
                let _ = serde_json::Deserializer::from_str(raw).deserialize_map(KeyReader { keys: &mut keys, need_sort: true });
            }
        }
    }
    BufHit { index: keys.index, score: keys.score, sort: keys.sort, raw: raw.into() }
}

/// The value of the hit's top-level `sort` when it is the hit's last
/// member (where the engine writes it): the last `"sort":` of the text,
/// opened after `,` or `{`, whose value ends right before the hit's closing
/// brace. An unescaped `"sort":` is never inside a string, and a member
/// whose value is followed by the closing brace of the whole hit belongs
/// to the hit itself. `None`: not found that way (read the hit whole).
fn tail_sort(raw: &str) -> Option<Option<Vec<Value>>> {
    let text = raw.trim_end();
    let body = text.strip_suffix('}')?;
    let at = body.rfind("\"sort\":")?;
    if !matches!(body[..at].trim_end().chars().last(), Some(',' | '{')) {
        return None;
    }
    let value = &body[at + "\"sort\":".len()..];
    let mut stream = serde_json::Deserializer::from_str(value).into_iter::<Option<Vec<Value>>>();
    let sort = stream.next()?.ok()?;
    value[stream.byte_offset()..].trim().is_empty().then_some(sort)
}

/// [`sub_hits`] through serde: every hit is scanned once into a
/// `RawValue` and again for its keys (with `sort` wanted, past its whole
/// `_source`, which the engine writes before `sort`).
fn sub_hits_serde(body: &[u8], need_sort: bool) -> Option<SubBody> {
    use serde::de::DeserializeSeed as _;
    let mut d = serde_json::Deserializer::from_slice(body);
    let b = SubBodyReader { need_sort, inner: false }.deserialize(&mut d).ok()?;
    d.end().ok()?;
    Some(b)
}

const RAW_HITS: &str = "\u{0}raw-hits";

/// A response with `hits.hits` given as raw hits: `v` carries the
/// [`RAW_HITS`] placeholder there, replaced by the hits' own JSON.
fn with_raw_hits(status: u16, v: &Value, hits: &[BufHit]) -> RestResp {
    match raw_hits_body(v, hits) {
        Some(body) => { let mut r = RestResp::json(status, &Value::Null); r.replace_body_vec(body); r }
        None => RestResp::json(500, &json!({"error":{"type":"internal_server_error","reason":"invalid scroll response structure"},"status":500})),
    }
}

/// The bytes of [`with_raw_hits`]: `v` serialized compactly (as
/// `serde_json::to_vec` writes it) with only `hits.hits` replaced,
/// structurally, by the hits' JSON copied verbatim (searching serialized
/// JSON for a marker could replace a user aggregation value equal to it).
/// `None` when `v` or its `hits` is not an object.
fn raw_hits_body(v: &Value, hits: &[BufHit]) -> Option<Vec<u8>> {
    let object = v.as_object()?;
    let mut out = Vec::with_capacity(512 + hits.iter().map(|h| h.raw.len() + 1).sum::<usize>());
    out.push(b'{');
    for (i, (key, value)) in object.iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        serde_json::to_writer(&mut out, key).ok()?;
        out.push(b':');
        if key == "hits" {
            let inner = value.as_object()?;
            out.push(b'{');
            for (j, (k, v)) in inner.iter().enumerate() {
                if j > 0 {
                    out.push(b',');
                }
                serde_json::to_writer(&mut out, k).ok()?;
                out.push(b':');
                if k == "hits" {
                    out.push(b'[');
                    for (n, h) in hits.iter().enumerate() {
                        if n > 0 {
                            out.push(b',');
                        }
                        out.extend_from_slice(h.raw.as_bytes());
                    }
                    out.push(b']');
                } else {
                    serde_json::to_writer(&mut out, v).ok()?;
                }
            }
            out.push(b'}');
        } else if key == "aggregations" {
            serde_json::to_writer(&mut out, &elasticsearch_engine::aggs::render_scripted_values(value)).ok()?;
        } else {
            serde_json::to_writer(&mut out, value).ok()?;
        }
    }
    out.push(b'}');
    Some(out)
}


struct Sub {
    /// Wrapped (`{node}.{engine id}`) id of the node's own scroll.
    id: String,
    /// Shard engine the sub-scroll lives in.
    shard: u32,
    /// The indices it reads (for its shard failures).
    indices: Vec<String>,
    buf: VecDeque<BufHit>,
    done: bool,
    touched: Instant,
    /// Its node answered a touch with a page (a build before the touch):
    /// it is not touched again, and its age-based refill keeps it alive.
    legacy: bool,
}

struct Composite {
    subs: Vec<Sub>,
    body: Value,
    params: HashMap<String, String>,
    shards: Value,
    /// Shards that failed on the first page (not in `shards`) and shards of
    /// `shards` whose sub-scroll failed since: every page reports them (ES
    /// keeps failing a shard whose scroll context is gone).
    failed_at_start: Vec<copies::ShardFailure>,
    lost: Vec<copies::ShardFailure>,
    total: Option<(u64, bool)>,
    /// Hits per page of the client's scroll.
    size: usize,
    /// Hits per page of each sub-scroll.
    sub_page: usize,
    keep_alive: Duration,
    expires: Instant,
    /// The buffered hits on the `accounting` breaker ([`Composite::account`]).
    accounting: elasticsearch_engine::memory::Reservation,
}

impl Composite {
    /// Hold the buffered hits' bytes on the `accounting` breaker (kept
    /// between requests; released when the scroll is cleared or expires).
    fn account(&mut self) {
        let bytes: u64 = self.subs.iter().flat_map(|s| s.buf.iter()).map(|h| 96 + h.raw.len() as u64).sum::<u64>() + 1024;
        let held = self.accounting.bytes();
        if bytes > held {
            self.accounting.grow_without_breaking(bytes - held);
        } else {
            self.accounting.shrink(held - bytes);
        }
    }
}

fn composites() -> &'static Mutex<HashMap<String, Composite>> {
    static M: OnceLock<Mutex<HashMap<String, Composite>>> = OnceLock::new();
    M.get_or_init(Default::default)
}

fn purge_expired() {
    let now = Instant::now();
    composites().lock().retain(|_, c| c.expires > now);
}

/// `DELETE /_search/scroll/_all` on this node.
pub(super) fn forget_all() {
    composites().lock().clear();
}

// Only contexts observed before dispatch belong to this clear operation.
fn forget_snapshot<T>(contexts: &mut HashMap<String, T>, ids: Vec<String>) {
    for id in ids { contexts.remove(&id); }
}

/// `DELETE /_search/scroll/_all` on this node: drop the composite scrolls
/// it coordinates and clear every shard engine's scroll contexts (shard 0
/// and each `~k` engine). Returns the engine contexts freed; a composite
/// is only bookkeeping over engine contexts, which are what ES counts.
pub(super) async fn clear_all_local(node: &ClusterNode, req: &RestReq) -> elasticsearch_engine::error::Result<u64> {
    let existing: Vec<String> = composites().lock().keys().cloned().collect();
    let mut requests = Vec::new();
    for (k, _) in node.all_shard_engines() {
        let mut r = json_req(req, "DELETE", "/_search/scroll/_all", &Value::Null)?;
        r.set_shard(k)?;
        requests.push(r);
    }
    let mut freed = 0;
    for r in requests {
        freed += freed_count(&local_engine(node, &r).await)?;
    }
    let mut contexts = composites().lock();
    forget_snapshot(&mut contexts, existing);
    Ok(freed)
}

/// ES `SearchService.validateKeepAlives` against `search.max_keep_alive`
/// (cluster setting, default 24h): the 400 of a keep-alive too large.
fn keep_alive_too_large(node: &ClusterNode, s: &str) -> Option<RestResp> {
    let ms = elasticsearch_engine::date::parse_time_value(s)?;
    let max = {
        let cs = node.engine.cluster_settings.read();
        cs.1.get("search.max_keep_alive").or(cs.0.get("search.max_keep_alive"))
            .and_then(|v| v.as_str().map(str::to_string).or_else(|| v.as_i64().map(|n| n.to_string())))
            .and_then(|v| elasticsearch_engine::date::parse_time_value(&v))
            .unwrap_or(86_400_000)
    };
    (ms > max).then(|| RestResp::error(400, "illegal_argument_exception", &format!(
        "Keep alive for request ({}) is too large. It must be less than ({}). This limit can be set by changing the [search.max_keep_alive] cluster level setting.",
        fmt_time(ms), fmt_time(max)
    )))
}

/// ES `TimeValue.toString` of `ms` (`1d`, `1.5h`, `300ms`).
fn fmt_time(ms: i64) -> String {
    let units = [(86_400_000, "d"), (3_600_000, "h"), (60_000, "m"), (1000, "s")];
    for (u, s) in units {
        if ms >= u && ms % u == 0 {
            return format!("{}{s}", ms / u);
        }
    }
    for (u, s) in units {
        if ms >= u {
            return format!("{:.1}{s}", ms as f64 / u as f64);
        }
    }
    format!("{ms}ms")
}

fn keep_alive(s: Option<&str>) -> Duration {
    s.and_then(elasticsearch_engine::date::parse_time_value).map(|ms| Duration::from_millis(ms.max(0) as u64)).unwrap_or(Duration::from_secs(300))
}

// ------------------------------------------------- cluster async searches

/// Async searches run by the front itself (`{node}.~a{engine-style id}`):
/// a cross-shard or cross-node `_async_search` executes the ordinary
/// cluster search (`multi_shard_fan_out`, DFS round included) in the
/// background and is kept here by the coordinating node, which answers
/// get / status / delete for it (other nodes forward by the id's owner).
pub(super) const ASYNC_PREFIX: &str = "~a";

pub(super) struct ClusterAsync {
    pub s: Arc<elasticsearch_engine::async_search::AsyncSearch>,
    /// The failed search's ES `error` object, kept verbatim (the engine
    /// state only carries type / reason / status).
    pub error: Mutex<Option<Value>>,
    pub warnings: Mutex<Vec<(String, String)>>,
}

fn asyncs() -> &'static Mutex<HashMap<String, Arc<ClusterAsync>>> {
    static M: OnceLock<Mutex<HashMap<String, Arc<ClusterAsync>>>> = OnceLock::new();
    M.get_or_init(Default::default)
}

/// Store under the unwrapped (`~a...`) id.
pub(super) fn async_put(key: String, c: Arc<ClusterAsync>) {
    let mut m = asyncs().lock();
    m.retain(|_, c| !c.s.expired());
    m.insert(key, c);
}

/// Drop expired cluster async searches (also done on every access).
pub(super) fn reap_asyncs() {
    asyncs().lock().retain(|_, c| !c.s.expired());
}

pub(super) fn async_get(key: &str) -> Option<Arc<ClusterAsync>> {
    let mut m = asyncs().lock();
    m.retain(|_, c| !c.s.expired());
    m.get(key).cloned()
}

pub(super) fn async_remove(key: &str) -> Option<Arc<ClusterAsync>> {
    let mut m = asyncs().lock();
    m.retain(|_, c| !c.s.expired());
    m.remove(key)
}

/// The async-search response for `c`, with `id` (when stored) set to the
/// wrapped cluster id and a failure's original `error` object.
pub(super) fn async_render(c: &ClusterAsync, with_id: bool, wrapped: &str, typed: bool) -> RestResp {
    let (status, mut v) = match c.s.response(with_id) {
        Ok(response) => response,
        Err(error) => return RestResp::json(error.status, &error.to_json()),
    };
    if with_id {
        v["id"] = json!(wrapped);
    }
    if let (Some(e), Some(o)) = (c.error.lock().clone(), v.as_object_mut()) {
        if o.contains_key("error") {
            o.insert("error".into(), e);
        }
    }
    if !typed { if let Some(response) = v.get_mut("response") { elasticsearch_http::untype_search_keys(response); } }
    let mut response = RestResp::json(status, &v);
    response.extend_headers(c.warnings.lock().iter().cloned());
    response
}

/// `GET /_async_search/status/{id}` for a cluster async search.
pub(super) fn async_status(c: &ClusterAsync, wrapped: &str) -> RestResp {
    let mut v = c.s.status();
    v["id"] = json!(wrapped);
    RestResp::json(200, &v)
}

/// One sub-context to open: (node, shard engine, indices).
pub(super) type Group = (String, u32, Vec<String>);

/// Groups of a copy plan: one per (first copy, shard), as a search groups
/// its shard requests (`shard_fan_out`).
pub(super) fn plan_groups(plan: &copies::Plan) -> Vec<Group> {
    let mut out: BTreeMap<(String, u32), Vec<String>> = BTreeMap::new();
    for key in plan.copies.keys() {
        let (Some((i, k)), Some(first)) = (copies::split_key(key), plan.candidates(key).first().cloned()) else { continue };
        out.entry((first, k)).or_default().push(i);
    }
    out.into_iter().map(|((n, k), idx)| (n, k, idx)).collect()
}

/// A group's nodes in order: its own, then the other nodes the plan lists
/// for every one of its keys (ES opens a scroll / PIT on a shard's next
/// copy when one fails).
fn group_candidates(plan: Option<&copies::Plan>, n: &str, k: u32, idx: &[String]) -> Vec<String> {
    let mut out = vec![n.to_string()];
    let Some(plan) = plan else { return out };
    let lists: Vec<Vec<String>> = idx.iter().map(|i| plan.candidates(&format!("{i}#{k}"))).collect();
    if let Some(first) = lists.first() {
        out.extend(first.iter().filter(|c| c.as_str() != n && lists.iter().all(|l| l.contains(*c))).cloned());
    }
    out
}

/// `r` on each of `cands` in turn until one answers or fails with a
/// request error; the answer and the node that gave it.
/// Also the nodes that failed as nodes (not tried again by a split). A
/// forward gives up when its node leaves the cluster.
async fn exec_first(node: &Arc<ClusterNode>, cands: Vec<String>, k: u32, r: elasticsearch_engine::error::Result<RestReq>) -> (RestResp, String, Vec<String>) {
    let r = match r { Ok(r) => r, Err(e) => return (RestResp::json(e.status, &e.to_json()), String::new(), Vec::new()) };
    let mut best: Option<(RestResp, String)> = None;
    let mut dead = Vec::new();
    for c in cands {
        let resp = if c == node.local.id {
            exec_shard(node, &c, k, Ok(r.clone())).await
        } else {
            let mut fwd = r.clone();
            if k > 0 {
                if let Err(e) = fwd.set_shard(k) { return (RestResp::json(e.status, &e.to_json()), c, dead); }
            }
            super::copy_request(node, &fwd, &c, None).await
        };
        if resp.is_success() || copies::is_request_error(&resp) {
            return (resp, c, dead);
        }
        if copies::node_failure(&resp) {
            dead.push(c.clone());
        }
        if best.as_ref().map_or(true, |(prev, _)| copies::replaces(prev, &resp)) {
            best = Some((resp, c));
        }
    }
    let (resp, c) = best.unwrap_or_else(|| (RestResp::error(503, "no_shard_available_action_exception", "no active shard copy"), String::new()));
    (resp, c, dead)
}

/// One group's request on its copies; a group of several indices that no
/// candidate answered is split, each index on its own copies (as a search
/// addresses every shard on its own: no node needs to hold all of them).
/// `(answer, node, shard, indices)` per request that ran.
async fn exec_group(node: &Arc<ClusterNode>, plan: Option<&copies::Plan>, n: &str, k: u32, idx: &[String], r: elasticsearch_engine::error::Result<RestReq>, build: &(dyn Fn(u32, &[String]) -> elasticsearch_engine::error::Result<RestReq> + Sync)) -> Vec<(RestResp, String, u32, Vec<String>)> {
    let (resp, owner, dead) = exec_first(node, group_candidates(plan, n, k, idx), k, r).await;
    let Some(plan) = plan.filter(|_| idx.len() > 1 && !resp.is_success() && !copies::is_request_error(&resp)) else {
        return vec![(resp, owner, k, idx.to_vec())];
    };
    let (dead, resp, owner) = (&dead, &resp, &owner);
    let singles = idx.iter().map(|i| async move {
        let one = std::slice::from_ref(i);
        // Nodes that just failed as nodes are not waited for again.
        let cands: Vec<String> = plan.candidates(&format!("{i}#{k}")).into_iter().filter(|c| !dead.contains(c)).collect();
        if cands.is_empty() {
            return (resp.clone(), owner.clone(), k, one.to_vec());
        }
        let (resp, owner, _) = exec_first(node, cands, k, build(k, one)).await;
        (resp, owner, k, one.to_vec())
    });
    futures::future::join_all(singles).await
}

fn tag(k: u32, wrapped: &str) -> String {
    if k == 0 { wrapped.to_string() } else { format!("~k{k}:{wrapped}") }
}

/// A PIT sub id over shard engines carries the number of shards it covers
/// (`~n{n}:`), since each shard engine reports its index's full shard count.
fn tag_shards(n: usize, id: String) -> String {
    format!("~n{n}:{id}")
}

fn untag_shards(id: &str) -> (Option<usize>, &str) {
    if let Some((n, rest)) = id.strip_prefix("~n").and_then(|r| r.split_once(':')) {
        if let Ok(n) = n.parse() {
            return (Some(n), rest);
        }
    }
    (None, id)
}

fn untag(id: &str) -> (u32, &str) {
    let (_, id) = untag_shards(id);
    if let Some(rest) = id.strip_prefix("~k") {
        if let Some((k, w)) = rest.split_once(':') {
            if let Ok(k) = k.parse() {
                return (k, w);
            }
        }
    }
    (0, id)
}

/// `exec_on` against one shard engine of `owner`.
async fn exec_shard(node: &Arc<ClusterNode>, owner: &str, k: u32, req: elasticsearch_engine::error::Result<RestReq>) -> RestResp {
    let mut req = match req {Ok(req)=>req,Err(e)=>return RestResp::json(e.status,&e.to_json())};
    if k > 0 {
        if let Err(e) = req.set_shard(k) {return RestResp::json(e.status,&e.to_json());}
    }
    exec_on(node, owner, req).await
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn unhex(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(s.get(i..i + 2)?, 16).ok()).collect()
}

/// A JSON request derived from `base` (keeps auth and other headers).
pub(super) fn json_req(base: &RestReq, method: &str, uri: &str, body: &Value) -> elasticsearch_engine::error::Result<RestReq> {
    let mut r = base.clone();
    r.set_method(method)?;
    r.set_uri(uri)?;
    if body.is_null() { r.set_body(&[])?; } else { r.set_json(body)?; }
    r.retain_headers(|(k, _)| !["content-type", "content-length", "accept-encoding"].iter().any(|h| k.eq_ignore_ascii_case(h)))?;
    if !r.body.is_empty() {
        r.push_header("content-type", &base.json_content_type())?;
    }
    Ok(r)
}

/// Construct generated URI bytes under admission before deriving the request.
/// Complete this synchronous call before awaiting any request execution.
pub(super) fn json_req_formatted(base:&RestReq,method:&str,uri:std::fmt::Arguments<'_>,body:&Value)->elasticsearch_engine::error::Result<RestReq> {
    let uri=super::SharedText::try_format(uri)?;
    json_req(base,method,&uri,body)
}

pub(super) fn query_without(q:&str,drop:&[&str])->elasticsearch_engine::error::Result<super::SharedText> {
    super::SharedText::try_format(format_args!("{}",super::FilteredQuery{query:q,drop,append:""}))
}
/// Stream selected decoded parameters into the admitted URI formatter.
pub(super) struct SelectedParams<'a>(pub &'a HashMap<String,String>,pub &'a [&'a str]);
impl std::fmt::Display for SelectedParams<'_> {
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result {
        let mut sep="?";
        for key in self.1 {if let Some(value)=self.0.get(*key){write!(f,"{sep}{key}={}",super::UriComponent(value))?;sep="&";}}
        Ok(())
    }
}
pub(super) struct QueryPairs<'a>(pub &'a [(String,String)]);
impl std::fmt::Display for QueryPairs<'_> {
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result {
        for (i,(key,value)) in self.0.iter().enumerate(){if i>0 {f.write_str("&")?;}f.write_str(key)?;if !value.is_empty(){write!(f,"={value}")?;}}
        Ok(())
    }
}

fn hit_cmp(a: &BufHit, ai: usize, b: &BufHit, bi: usize, keys: Option<&[fanout::SortKey]>) -> Ordering {
    let ord = match keys {
        Some(keys) => fanout::sort_order(a.sort.as_deref(), b.sort.as_deref(), keys),
        None => fanout::score_order(a.score, b.score),
    };
    ord.then_with(|| a.index.cmp(&b.index)).then(ai.cmp(&bi))
}

/// Which subs must fetch a page before the next `n` hits can be cut.
///
/// The merge may take a sub's hit only while every sub that is not done
/// still has a buffered hit to compare it with; a done sub's empty buffer
/// is its end. Running the merge over the buffers, the first point where a
/// live sub has none left blocks it: that sub, and every live sub with
/// fewer buffered hits than the page still needs, is refilled (in one
/// parallel round). A refill adds a whole sub page (at least `n` hits) or
/// marks the sub done, so after it no sub can block this page; the hits
/// taken before the block point are unchanged by it (they sort before the
/// refilled sub's next hit). Pages are therefore exactly those of the
/// former rule (refill every live sub holding fewer than `n` hits), which
/// fetched a page from every sub it could and left up to two pages per
/// sub fetched and unreturned when the client stopped scrolling. ES fetches
/// only the hits a page returns (`SearchScrollQueryThenFetchAsyncAction`:
/// query every shard, merge, then fetch the page's docs; from knowledge of
/// the 9.5.5 source, not re-read here).
fn refills_needed(subs: &[Sub], n: usize, keys: Option<&[fanout::SortKey]>) -> Vec<bool> {
    let mut pos = vec![0usize; subs.len()];
    let mut taken = 0;
    while taken < n {
        if subs.iter().zip(&pos).any(|(s, &p)| !s.done && p >= s.buf.len()) {
            let left = n - taken;
            return subs.iter().zip(&pos).map(|(s, &p)| !s.done && s.buf.len().saturating_sub(p) < left).collect();
        }
        let mut best: Option<usize> = None;
        for (i, s) in subs.iter().enumerate() {
            let Some(h) = s.buf.get(pos[i]) else { continue };
            if best.map_or(true, |b| hit_cmp(h, i, &subs[b].buf[pos[b]], b, keys) == Ordering::Less) {
                best = Some(i);
            }
        }
        let Some(b) = best else { break };
        pos[b] += 1;
        taken += 1;
    }
    vec![false; subs.len()]
}

/// Pop the next `n` hits of the merged sub streams.
fn take(subs: &mut [Sub], n: usize, keys: Option<&[fanout::SortKey]>) -> Vec<BufHit> {
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let mut best: Option<usize> = None;
        for (i, s) in subs.iter().enumerate() {
            let Some(h) = s.buf.front() else { continue };
            if best.map_or(true, |b| hit_cmp(h, i, &subs[b].buf[0], b, keys) == Ordering::Less) {
                best = Some(i);
            }
        }
        let Some(b) = best else { break };
        out.extend(subs[b].buf.pop_front());
    }
    out
}

fn total_json(total: Option<(u64, bool)>, params: &HashMap<String, String>) -> Option<Value> {
    let (v, gte) = total?;
    Some(if params.get("rest_total_hits_as_int").map(|v| v == "true").unwrap_or(false) {
        json!(v)
    } else {
        json!({"value": v, "relation": if gte { "gte" } else { "eq" }})
    })
}

fn parse_total(t: &Value) -> Option<(u64, bool)> {
    match t {
        Value::Number(n) => Some((n.as_u64().unwrap_or(0), false)),
        Value::Object(o) => Some((o.get("value").and_then(Value::as_u64).unwrap_or(0), o.get("relation").and_then(Value::as_str) == Some("gte"))),
        _ => None,
    }
}

fn max_score(hits: &[BufHit]) -> Value {
    hits.iter().filter_map(|h| h.score).fold(None, |m: Option<f64>, s| Some(m.map_or(s, |m| m.max(s)))).map(|m| json!(m)).unwrap_or(Value::Null)
}

// Public close operations must preserve engine refusal statuses, including 429.
fn freed_count(response: &RestResp) -> elasticsearch_engine::error::Result<u64> {
    let value = response.body_json().unwrap_or(Value::Null);
    if !response.is_success() && !(response.status == 404 && value["succeeded"] == true && value["num_freed"].as_u64() == Some(0)) {
        return Err(elasticsearch_engine::error::EsError::from_error_json(response.status, &value["error"]));
    }
    value["num_freed"].as_u64().ok_or_else(|| elasticsearch_engine::error::EsError::internal("clear context response omitted num_freed"))
}

/// Free the engine contexts behind tagged wrapped ids, summing `num_freed`.
async fn clear_subs(node: &Arc<ClusterNode>, req: &RestReq, ids: &[String]) -> elasticsearch_engine::error::Result<u64> {
    let mut requests = Vec::new();
    for id in ids {
        let (k, id) = untag(id);
        let Some((owner, inner)) = unwrap_id(id) else { continue };
        let mut r = json_req(req, "DELETE", "/_search/scroll", &json!({"scroll_id": [inner]}))?;
        if k>0 {r.set_shard(k)?;}
        requests.push((owner.to_string(), r));
    }
    let futs = requests.into_iter().map(|(owner,r)| async move {exec_shard(node,&owner,0,Ok(r)).await});
    futures::future::join_all(futs).await.iter().map(freed_count).sum()
}

// ------------------------------------------------------------------ scroll

/// First page of a scroll over indices no single node holds.
///
/// `shard_counts` (index → number of shards) turns a `slice` into the
/// per-shard slice ES runs on each shard; empty leaves the body alone.
/// `split`: the groups are single shard engines (`plan_groups`), whose
/// `_shards` get normalized to the shards each one covers.
fn continuation_shards(initial: &Value) -> Value {
    let mut shards = initial.clone();
    let skipped = initial["skipped"].as_u64().unwrap_or(0);
    for key in ["total", "successful"] {
        if let Some(count) = initial[key].as_u64() {
            shards[key] = json!(count.saturating_sub(skipped));
        }
    }
    shards["skipped"] = json!(0);
    shards
}

/// `plan`: the copies each group may fall back to (and the shards with no
/// copy, reported failed); `None` for groups that have no alternative.
#[allow(clippy::too_many_arguments)]
pub(super) async fn start_scroll(node: &Arc<ClusterNode>, req: &RestReq, mut groups: Vec<Group>, split: bool, body: &Value, params: &HashMap<String, String>, shard_counts: &HashMap<String, u32>, plan: Option<&copies::Plan>) -> RestResp {
    let (from, size) = fanout::from_size(body, params);
    let sub_page = from + size;
    let query = match query_without(req.query(), &["from", "size"]) {Ok(query)=>query,Err(error)=>return super::err(error)};
    let mut nb = fanout::node_body(body, from, size);
    if from > 0 {
        // Not allowed with scroll: let the engine reject it.
        nb["from"] = json!(from);
    }
    let sliced = !shard_counts.is_empty() && fanout::slice_spec(body).is_some();
    if sliced {
        groups = groups.into_iter().flat_map(|(n, k, idx)| idx.into_iter().map(move |i| (n.clone(), k, vec![i]))).collect();
    }
    let searched_shards: usize = groups.iter().map(|(_, _, indices)| indices.len()).sum();
    let build = |k: u32, idx: &[String]| -> elasticsearch_engine::error::Result<RestReq> {
        let b = match idx.first().and_then(|i| shard_counts.get(i)) {
            Some(&cnt) if sliced => fanout::slice_for_shard(&nb, k, cnt),
            _ => nb.clone(),
        };
        let mut r = json_req(req, "POST", &super::shard_read_uri(node, req, idx, "_search", query.trim_start_matches('?'))?, &b)?;
        r.set_read_copies(idx)?;
        r.push_header(crate::shard_engines::PREFILTER_SHARDS_HEADER, &searched_shards.to_string())?;
        Ok(r)
    };
    let mut reqs = match groups.iter().map(|(_, k, idx)| build(*k, idx)).collect::<elasticsearch_engine::error::Result<Vec<_>>>() {Ok(r)=>r,Err(e)=>return RestResp::json(e.status,&e.to_json())};
    // DFS on the first page only: each engine keeps the global statistics
    // in its scroll context for the later pages (as ES's scroll keeps the
    // DFS-scored query).
    if super::dfs_wanted(req, body, params, reqs.len()) {
        let round = reqs.iter().zip(groups.iter()).map(|(r, (n, k, _))| {
            let (node, n, k, r) = (node.clone(), n.clone(), *k, super::dfs_collect_req(r));
            async move { exec_shard(&node, &n, k, r).await }
        });
        let merged = match super::dfs_round(round).await {
            Ok(m) => m,
            Err(r) => return r,
        };
        for r in reqs.iter_mut() {
            if let Err(e)=r.set_dfs(super::Dfs::Use(merged.clone())) {return RestResp::json(e.status,&e.to_json());}
        }
    }
    let dfs = reqs.first().map(|r| r.dfs.clone()).unwrap_or_default();
    let build = |k: u32, idx: &[String]| {
        let mut r = build(k, idx)?;
        r.set_dfs(dfs.clone())?;
        Ok(r)
    };
    let futs = reqs.into_iter().zip(groups.iter()).map(|(r, (n, k, idx))| exec_group(node, plan, n, *k, idx, Ok(r), &build));
    let resps: Vec<(RestResp, String, u32, Vec<String>)> = futures::future::join_all(futs).await.into_iter().flatten().collect();
    let now = Instant::now();
    let mut parts = Vec::new();
    let mut subs = Vec::new();
    // ES: the first page is a search like any other; a shard that failed
    // is reported (`_shards.failures`) and the scroll goes on without it.
    // A request error fails it as a whole.
    let mut failed = None;
    let mut failures: Vec<copies::ShardFailure> = plan.map_or_else(Vec::new, |p| p.unavailable.iter().map(|(i, k)| copies::ShardFailure::unavailable(i, *k)).collect());
    let keys = fanout::sort_keys(body, params);
    for (r, owner, shard, idx) in resps {
        // `v` keeps hits.hits empty: the merge below only needs totals,
        // shards and aggregations.
        let (mut v, hits) = match sub_hits(&r.body, keys.is_some()) {
            Some(b) => (Value::Object(b.meta), b.hits),
            None => (Value::Null, VecDeque::new()),
        };
        if split {
            fanout::normalize_part_shards(&mut v, idx.len());
        }
        match v["_scroll_id"].as_str() {
            Some(id) if r.is_success() => {
                subs.push(Sub { id: id.to_string(), shard, indices: idx.clone(), done: hits.is_empty() || hits.len() < sub_page, buf: hits, touched: now, legacy: false });
                parts.push(v);
            }
            _ => {
                let r = if r.is_success() { RestResp::error(500, "illegal_state_exception", "sub-search returned no scroll id") } else { r };
                if copies::is_request_error(&r) {
                    failed.get_or_insert(r);
                    continue;
                }
                failures.extend(copies::ShardFailure::from_response(&r, &idx, shard, &owner));
            }
        }
    }
    if failed.is_none() && !failures.is_empty() {
        if subs.is_empty() {
            failed = Some(copies::phase_failure("query", "all shards failed", &failures));
        } else if !copies::allow_partial(node, params) {
            failed = Some(copies::phase_failure("query", "Partial shards failure", &failures));
        }
    }
    if let Some(f) = failed {
        let ids: Vec<String> = subs.iter().map(|s| tag(s.shard, &s.id)).collect();
        let _ = clear_subs(node, req, &ids).await;
        return f;
    }
    // Totals, shards, aggregations, suggest: the usual cross-node merge;
    // hits come from the sub streams.
    let mut meta_body = if body.is_object() { body.clone() } else { json!({}) };
    meta_body["from"] = json!(0);
    meta_body["size"] = json!(0);
    let mut meta = fanout::merge_search_with_source(&parts, &meta_body, params, node.engine.max_buckets(), &req.body);
    if let Some(e) = fanout::take_agg_breaking() {
        let ids: Vec<String> = subs.iter().map(|s| tag(s.shard, &s.id)).collect();
        let _ = clear_subs(node, req, &ids).await;
        return RestResp::json(e.status, &e.to_json());
    }
    if let Some(e) = fanout::take_agg_error() {
        let ids: Vec<String> = subs.iter().map(|s| tag(s.shard, &s.id)).collect();
        let _ = clear_subs(node, req, &ids).await;
        return RestResp::json(e.status, &e.to_json());
    }
    let mut page = take(&mut subs, sub_page, keys.as_deref());
    let page: Vec<BufHit> = page.drain(..).skip(from).collect();
    meta["hits"]["hits"] = json!(RAW_HITS);
    let shards = continuation_shards(&meta["_shards"]);
    meta["_shards"] = copies::context_shards(&meta["_shards"], &failures, &[]);
    let ka = keep_alive(params.get("scroll").map(String::as_str));
    let key = format!("{SCROLL_PREFIX}{:016x}{:016x}", rand::random::<u64>(), rand::random::<u64>());
    let mut c = Composite {
        subs,
        body: body.clone(),
        params: params.clone(),
        shards,
        failed_at_start: failures,
        lost: Vec::new(),
        total: parse_total(&meta["hits"]["total"]),
        size,
        sub_page,
        keep_alive: ka,
        expires: now + ka,
        accounting: elasticsearch_engine::memory::process_breakers().reserve_without_breaking(elasticsearch_engine::memory::Kind::Accounting, 0),
    };
    c.account();
    purge_expired();
    composites().lock().insert(key.clone(), c);
    let mut out = Map::new();
    out.insert("_scroll_id".into(), json!(wrap_id(node, &key)));
    if let Value::Object(m) = meta {
        out.extend(m);
    }
    with_raw_hits(200, &Value::Object(out), &page)
}

/// The ES answer for a scroll context that is gone: 404
/// `search_phase_execution_exception` caused by `search_context_missing`.
/// Rendered here, not by the engine: `inner` is a cluster composite id
/// (`~s...`), which the engine's strict scroll-id parser rejects with 400.
async fn missing(_node: &Arc<ClusterNode>, _req: &RestReq, inner: &str) -> RestResp {
    use elasticsearch_engine::error::EsError;
    let mut e = EsError::search_phase("", EsError::search_context_missing(inner));
    e.status = 404;
    RestResp::json(404, &e.to_json())
}

/// Header asking a shard engine to renew a scroll sub-context without
/// reading a page (`InternalScrollTouch`): internal, stripped from client
/// requests.
pub(super) const SCROLL_TOUCH_HEADER: &str = "x-rustic-internal-scroll-touch";

fn owner_supports_touch(state: &elasticsearch_cluster::state::ClusterState, id: &str) -> bool {
    unwrap_id(id).and_then(|(owner, _)| state.nodes.get(owner)).is_some_and(|peer|
        crate::internal_capabilities::supports(peer, crate::internal_capabilities::Capability::ScrollTouch))
}

/// Next page of a composite scroll (`inner` = `~s...`).
pub(super) async fn continue_scroll(node: &Arc<ClusterNode>, req: &RestReq, owner: &str, inner: &str, scroll: Option<String>) -> RestResp {
    if owner != node.local.id {
        return forward(node, owner, req).await;
    }
    // A keep-alive past `search.max_keep_alive` is this request's 400, as
    // ES checks it before the scroll is touched: the scroll stays valid
    // (sent on, every sub-scroll would refuse it and be lost).
    if let Some(r) = scroll.as_deref().and_then(|s| keep_alive_too_large(node, s)) {
        return r;
    }
    purge_expired();
    let Some(mut c) = composites().lock().remove(inner) else {
        return missing(node, req, inner).await;
    };
    if let Some(s) = scroll.as_deref() {
        c.keep_alive = keep_alive(Some(s));
    }
    let ka = format!("{}ms", c.keep_alive.as_millis().max(1));
    // Unknown/old owners receive ordinary age-based page refills, never a
    // touch header which would advance their cursor unexpectedly. Recheck the
    // current node identity for each request rather than caching build support.
    let state = node.state();
    for sub in &mut c.subs {
        sub.legacy |= !owner_supports_touch(&state, &sub.id);
        if !sub.legacy {
            if let Some(peer) = unwrap_id(&sub.id).and_then(|(owner, _)| state.nodes.get(owner)) {
                sub.legacy |= crate::capability_probe::verify(&node.transport, peer, true).await.is_err();
            }
        }
    }
    let now = Instant::now();
    // A page from the subs whose buffers could not fill this one; an aged
    // sub is renewed by a touch below, except a legacy one (its node pages
    // on a touch anyway), refilled once per half keep-alive.
    let aged = |s: &Sub| now.duration_since(s.touched) > c.keep_alive / 2;
    let keys = fanout::sort_keys(&c.body, &c.params);
    let short = refills_needed(&c.subs, c.size, keys.as_deref());
    let refill: Vec<usize> = (0..c.subs.len()).filter(|&i| !c.subs[i].done && (short[i] || (c.subs[i].legacy && aged(&c.subs[i])))).collect();
    let futs = refill.iter().map(|&i| {
        let (id, k) = (c.subs[i].id.clone(), c.subs[i].shard);
        let (node, ka) = (node.clone(), ka.clone());
        let req = req.clone();
        async move {
            let Some((o, sub_inner)) = unwrap_id(&id) else { return RestResp::error(500, "illegal_state_exception", "bad sub scroll id") };
            let r = json_req(&req, "POST", "/_search/scroll", &json!({"scroll_id": sub_inner, "scroll": ka}))
                .and_then(|mut r| { r.push_header(elasticsearch_http::HIT_SPANS_HEADER, "1")?; Ok(r) });
            exec_shard(&node, o, k, r).await
        }
    });
    // Every other live sub-context is renewed to the composite's keep-alive
    // (which this request may have raised): a sub that needs no page now
    // must not expire while the composite scroll is still valid.
    let renew: Vec<usize> = (0..c.subs.len()).filter(|&i| !c.subs[i].done && !c.subs[i].legacy && !refill.contains(&i)).collect();
    let touches = renew.iter().map(|&i| {
        let (id, k) = (c.subs[i].id.clone(), c.subs[i].shard);
        let (node, ka) = (node.clone(), ka.clone());
        let req = req.clone();
        async move {
            let Some((o, sub_inner)) = unwrap_id(&id) else { return RestResp::error(500, "illegal_state_exception", "bad sub scroll id") };
            let r = json_req(&req, "POST", "/_search/scroll", &json!({"scroll_id": sub_inner, "scroll": ka})).and_then(|mut r| {r.push_header(SCROLL_TOUCH_HEADER,"1")?;Ok(r)});
            exec_shard(&node, o, k, r).await
        }
    });
    let (resps, touched) = futures::future::join(futures::future::join_all(futs), futures::future::join_all(touches)).await;
    // Pages to buffer: the refills', and those of touches a node of an
    // earlier build answered with a page (it ignores the touch header and
    // advances the cursor: dropping that page would lose its hits). Even
    // after checking the advertisement, validate the answer: a renewal is
    // `"renewed": true`, anything else is a page. A failed touch is a lost
    // sub-context like a failed page: that shard's failure (below).
    let mut pages: Vec<(usize, RestResp)> = refill.iter().copied().zip(resps).collect();
    for (&i, r) in renew.iter().zip(touched) {
        if r.is_success() && r.body_json().is_some_and(|v| v.get("renewed") == Some(&Value::Bool(true))) {
            c.subs[i].touched = now;
        } else {
            // A page for a touch: an older build, touched no more (its
            // buffer would grow by a page on every request).
            c.subs[i].legacy |= r.is_success();
            pages.push((i, r));
        }
    }
    let need_sort = keys.is_some();
    let mut took = 0u64;
    let mut timed_out = false;
    let (mut request_error, mut page_failures, mut cleared) = (None, Vec::new(), Vec::new());
    let (asked, mut answered) = (pages.len(), 0usize);
    for (i, r) in pages {
        if copies::is_request_error(&r) {
            // This request's error (every sub would give it): the answer,
            // with the scroll kept as it is.
            request_error.get_or_insert(r);
            continue;
        }
        if !r.is_success() {
            let owner = unwrap_id(&c.subs[i].id).map(|(o, _)| o.to_string()).unwrap_or_default();
            let failure = copies::ShardFailure::from_response(&r, &c.subs[i].indices, c.subs[i].shard, &owner);
            let ty = r.body_json().and_then(|v| v["error"]["type"].as_str().map(str::to_string)).unwrap_or_default();
            if r.status != 404 && !(copies::node_failure(&r) && !ty.contains("timeout")) {
                // A rejection, a timeout, a shard error: this page reports
                // it, and the next page asks the same context again (ES
                // `SearchScrollAsyncAction`).
                page_failures.extend(failure);
                continue;
            }
            // A lost sub-context (its copy moved or closed, its node left,
            // the context expired) is that shard's failure: the scroll goes
            // on with the others and reports it on every page; its buffered
            // hits were fetched from it, but ES would not return them. What
            // may remain of it is cleared.
            c.lost.extend(failure);
            cleared.push(tag(c.subs[i].shard, &c.subs[i].id));
            c.subs[i].done = true;
            c.subs[i].buf.clear();
            continue;
        }
        answered += 1;
        let spans = r.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(elasticsearch_http::HIT_SPANS_HEADER)).map(|(_, v)| v.as_str());
        let SubBody { meta, hits } = spans.and_then(|spans| sub_hits_spanned(&r.body, spans, need_sort))
            .or_else(|| sub_hits(&r.body, need_sort)).unwrap_or(SubBody { meta: Map::new(), hits: VecDeque::new() });
        // Per-shard failures inside a good page: a shard the sub-context
        // lost with its copy (context gone, copy gone or closed) is failed
        // for the rest of the scroll; any other failure is this page's
        // only (the sub still returns that shard's next hits).
        for f in meta.get("_shards").and_then(|s| s.get("failures")).and_then(Value::as_array).into_iter().flatten() {
            let (Some(index), Some(shard)) = (f["index"].as_str(), f["shard"].as_u64()) else { continue };
            let owner = unwrap_id(&c.subs[i].id).map(|(o, _)| o.to_string()).unwrap_or_default();
            let ty = f["reason"]["type"].as_str().unwrap_or("");
            let gone = matches!(ty, "search_context_missing_exception" | "shard_not_found_exception" | "index_shard_closed_exception" | "no_shard_available_action_exception");
            let status = if gone { 404 } else { f.get("status").and_then(Value::as_u64).map_or(500, |s| s as u16) };
            let failure = copies::ShardFailure { index: index.to_string(), shard: shard as u32, node: Some(owner), status, reason: f["reason"].clone(), engine_error: None };
            if !gone {
                page_failures.push(failure);
            } else if !c.lost.iter().any(|l| l.index == index && l.shard as u64 == shard) {
                c.lost.push(failure);
            }
        }
        took = took.max(meta.get("took").and_then(Value::as_u64).unwrap_or(0));
        timed_out |= meta.get("timed_out").and_then(Value::as_bool).unwrap_or(false);
        let s = &mut c.subs[i];
        if let Some(id) = meta.get("_scroll_id").and_then(Value::as_str) {
            s.id = id.to_string();
        }
        s.done = hits.is_empty() || hits.len() < c.sub_page;
        s.buf.extend(hits);
        s.touched = now;
    }
    if !cleared.is_empty() {
        let (node, req) = (node.clone(), req.clone());
        tokio::spawn(async move {
            let _ = clear_subs(&node, &req, &cleared).await;
        });
    }
    if let Some(r) = request_error {
        c.expires = Instant::now() + c.keep_alive;
        c.account();
        composites().lock().insert(inner.to_string(), c);
        return r;
    }
    // Every sub asked failed, transiently, and nothing buffered is left to
    // serve: ES's all shards failed (`successfulOps == 0`), not an empty
    // page a client would take for the scroll's end. The scroll is kept;
    // the next request asks the same contexts again.
    if asked > 0 && answered == 0 && !page_failures.is_empty() && c.subs.iter().all(|s| s.buf.is_empty()) {
        let failures: Vec<copies::ShardFailure> = c.failed_at_start.iter().chain(&c.lost).chain(&page_failures).cloned().collect();
        c.expires = Instant::now() + c.keep_alive;
        c.account();
        composites().lock().insert(inner.to_string(), c);
        return copies::phase_failure("query", "all shards failed", &failures);
    }
    // Every shard failed: ES's all-shards-failed (the scroll is gone).
    if !c.lost.is_empty() && c.subs.iter().all(|s| s.done && s.buf.is_empty()) && c.lost.len() >= c.subs.iter().map(|s| s.indices.len().max(1)).sum::<usize>() {
        let failures: Vec<copies::ShardFailure> = c.failed_at_start.iter().chain(&c.lost).cloned().collect();
        return copies::phase_failure("query", "all shards failed", &failures);
    }
    let page = take(&mut c.subs, c.size, keys.as_deref());
    let params = req.params();
    let mut hits = Map::new();
    if let Some(t) = total_json(c.total, &params) {
        hits.insert("total".into(), t);
    }
    hits.insert("max_score".into(), max_score(&page));
    hits.insert("hits".into(), json!(RAW_HITS));
    let out = json!({
        "_scroll_id": wrap_id(node, inner),
        "took": took,
        "timed_out": timed_out,
        "_shards": copies::context_shards(&c.shards, &c.failed_at_start, &c.lost.iter().chain(&page_failures).cloned().collect::<Vec<_>>()),
        "hits": Value::Object(hits),
    });
    c.expires = Instant::now() + c.keep_alive;
    c.account();
    composites().lock().insert(inner.to_string(), c);
    with_raw_hits(200, &out, &page)
}

/// Free a composite scroll; `None` when no such scroll exists.
pub(super) async fn clear_scroll(node: &Arc<ClusterNode>, req: &RestReq, wrapped: &str, owner: &str, inner: &str) -> elasticsearch_engine::error::Result<u64> {
    if owner != node.local.id {
        let r = json_req(req, "DELETE", "/_search/scroll", &json!({"scroll_id": [wrapped]}))?;
        return freed_count(&forward(node, owner, &r).await);
    }
    let Some(c) = composites().lock().remove(inner) else { return Ok(0) };
    let ids: Vec<String> = c.subs.iter().map(|s| tag(s.shard, &s.id)).collect();
    match clear_subs(node, req, &ids).await {
        Ok(freed) => Ok(freed.max(1)),
        Err(error) => {
            composites().lock().insert(inner.to_string(), c);
            Err(error)
        }
    }
}

// --------------------------------------------------------------------- PIT

/// Validate a wrapped PIT using one of its native shard-context IDs; the
/// wrapped ID itself is resolved by the coordinator when the search runs.
pub(super) fn pit_validation_id(id: &str) -> Option<String> {
    let (_, inner) = unwrap_id(id)?;
    if inner.starts_with(PIT_PREFIX) {
        let subs = pit_subs(inner)?;
        let (_, sub) = untag(subs.first()?);
        Some(unwrap_id(sub)?.1.to_string())
    } else { Some(inner.to_string()) }
}

fn pit_subs(inner: &str) -> Option<Vec<String>> {
    let raw = unhex(inner.strip_prefix(PIT_PREFIX)?)?;
    serde_json::from_slice(&raw).ok()
}

/// `POST /{indices}/_pit` over indices no single node holds.
/// `split` as for `start_scroll`.
/// `plan` as for [`start_scroll`]. A shard with no copy fails the open
/// unless `allow_partial_search_results=true` (ES's point-in-time default
/// is false).
pub(super) async fn open_pit(node: &Arc<ClusterNode>, req: &RestReq, groups: Vec<Group>, split: bool, plan: Option<&copies::Plan>) -> RestResp {
    let params = req.params();
    let partial = params.get("allow_partial_search_results").map(String::as_str) == Some("true");
    let missing: Vec<(String, u32)> = plan.map_or_else(Vec::new, |p| p.unavailable.clone());
    if !missing.is_empty() && !partial {
        return super::stateless_read::missing_shards_rejected(&missing);
    }
    let query = match query_without(req.query(), &[]) {Ok(query)=>query,Err(error)=>return super::err(error)};
    let build = |_k: u32, idx: &[String]| -> elasticsearch_engine::error::Result<RestReq> {
        let mut r = req.clone();
        r.set_method("POST")?;
        r.set_uri(&super::shard_read_uri(node, req, idx, "_pit", query.trim_start_matches('?'))?)?;
        r.retain_headers(|(h, _)| !h.eq_ignore_ascii_case("accept-encoding"))?;
        r.set_read_copies(idx)?;
        Ok(r)
    };
    let futs = groups.iter().map(|(n, k, idx)| exec_group(node, plan, n, *k, idx, build(*k, idx), &build));
    let resps: Vec<(RestResp, String, u32, Vec<String>)> = futures::future::join_all(futs).await.into_iter().flatten().collect();
    let mut ids = Vec::new();
    let mut shards = [0u64; 4];
    let mut failed = None;
    for (r, _, shard, idx) in resps.iter() {
        let mut v = r.body_json().unwrap_or(Value::Null);
        if split {
            fanout::normalize_part_shards(&mut v, idx.len());
        }
        match v["id"].as_str() {
            Some(id) if r.is_success() => {
                ids.push(if split { tag_shards(idx.len(), tag(*shard, id)) } else { tag(*shard, id) });
                for (i, k) in ["total", "successful", "skipped", "failed"].iter().enumerate() {
                    shards[i] += v["_shards"][k].as_u64().unwrap_or(0);
                }
            }
            _ => {
                if failed.is_none() {
                    failed = Some(r.clone());
                }
            }
        }
    }
    if let Some(f) = failed {
        let _ = close_ids(node, req, &ids).await;
        return f;
    }
    let inner = format!("{PIT_PREFIX}{}", hex(&serde_json::to_vec(&ids).unwrap_or_default()));
    let mut out = json!({"id": wrap_id(node, &inner), "_shards": {"total": shards[0], "successful": shards[1], "skipped": shards[2], "failed": shards[3]}});
    let unavailable: Vec<copies::ShardFailure> = missing.iter().map(|(i, k)| copies::ShardFailure::unavailable(i, *k)).collect();
    copies::add_failures(&mut out, &unavailable);
    RestResp::json(200, &out)
}

async fn close_ids(node: &Arc<ClusterNode>, req: &RestReq, ids: &[String]) -> elasticsearch_engine::error::Result<u64> {
    let mut requests = Vec::new();
    for id in ids {
        let (k,id)=untag(id);
        let Some((owner,inner))=unwrap_id(id) else {continue};
        let mut r=json_req(req,"DELETE","/_pit",&json!({"id":inner}))?;
        if k>0 {r.set_shard(k)?;}
        requests.push((owner.to_string(),r));
    }
    let futs=requests.into_iter().map(|(owner,r)|async move {exec_shard(node,&owner,0,Ok(r)).await});
    futures::future::join_all(futs).await.iter().map(freed_count).sum()
}

/// `DELETE /_pit` of a composite PIT.
pub(super) async fn close_pit(node: &Arc<ClusterNode>, req: &RestReq, inner: &str) -> Option<RestResp> {
    let ids = pit_subs(inner)?;
    let freed = match close_ids(node, req, &ids).await {Ok(n)=>n,Err(e)=>return Some(RestResp::json(e.status,&e.to_json()))};
    Some(RestResp::json(if freed == 0 { 404 } else { 200 }, &json!({"succeeded": true, "num_freed": freed})))
}

/// Index of the `_shard_doc` tiebreak in a PIT search's sort values, and
/// whether it sorts descending (the implicit tiebreak is ascending).
fn tiebreak_slot(body: &Value, params: &HashMap<String, String>) -> Option<(usize, bool)> {
    let keys = fanout::sort_keys(body, params)?;
    if keys.is_empty() {
        return None;
    }
    let order = |o: &Value| -> bool {
        let o = o.get("order").unwrap_or(o);
        o.as_str().map(|s| s.eq_ignore_ascii_case("desc")).unwrap_or(false)
    };
    let names: Vec<(String, bool)> = match body.get("sort") {
        Some(Value::String(f)) => vec![(f.clone(), false)],
        Some(Value::Array(a)) => a
            .iter()
            .map(|e| match e {
                Value::String(s) => (s.clone(), false),
                Value::Object(m) => m.iter().next().map(|(k, v)| (k.clone(), order(v))).unwrap_or_default(),
                _ => Default::default(),
            })
            .collect(),
        Some(Value::Object(m)) => m.iter().map(|(k, v)| (k.clone(), order(v))).collect(),
        _ => params
            .get("sort")
            .map(|s| s.split(',').map(|p| match p.split_once(':') {
                Some((f, o)) => (f.to_string(), o.eq_ignore_ascii_case("desc")),
                None => (p.to_string(), false),
            }).collect())
            .unwrap_or_default(),
    };
    Some(match names.iter().position(|(n, _)| n == "_shard_doc") {
        Some(i) => (i, names[i].1),
        None => (keys.len(), false),
    })
}

/// Search a composite PIT (`wrapped` is the client's id).
pub(super) async fn search_pit(node: &Arc<ClusterNode>, req: &RestReq, wrapped: &str, inner: &str) -> Option<RestResp> {
    let ids = pit_subs(inner)?;
    let body = req.json().unwrap_or(json!({}));
    let params = req.params();
    if body.get("collapse").and_then(|c| c.get("inner_hits")).is_some() {
        // The expand phase would have to search the same point in time.
        return Some(RestResp::error(500, "illegal_state_exception", "collapse with inner_hits in a point in time cannot be combined across shards yet"));
    }
    // The same limits as a plain multi-shard search: each sub is sliced on
    // its own, so `slice` is fine; kNN top-k, retrievers and rank are not
    // merged here.
    let mut checked = body.clone();
    if let Some(o) = checked.as_object_mut() {
        o.remove("pit");
        o.remove("slice");
    }
    if ids.len() > 1 || ids.iter().any(|id| untag_shards(id).0.unwrap_or(1) > 1) {
        if let Some(f) = fanout::unsupported_feature(&checked, &params) {
            return Some(RestResp::error(500, "illegal_state_exception", &format!("{f} in a point in time cannot be combined across shards yet")));
        }
        if fanout::knn_needs_dfs(&checked, &params) || !fanout::knn_clauses(&checked).is_empty() {
            return Some(RestResp::error(500, "illegal_state_exception", "knn in a point in time cannot be combined across shards yet"));
        }
    }
    let (from, size) = fanout::from_size(&body, &params);
    let tb = tiebreak_slot(&body, &params);
    let slot = tb.map(|(s, _)| s);
    let desc = tb.map(|(_, d)| d).unwrap_or(false);
    let after = body.get("search_after").and_then(Value::as_array).cloned();
    let query = match query_without(req.query(), &["from", "size"]) {Ok(query)=>query,Err(error)=>return Some(super::err(error))};
    let mut reqs = Vec::new();
    let covers: Vec<Option<usize>> = ids.iter().map(|id| untag_shards(id).0).collect();
    for (i, id) in ids.iter().enumerate() {
        let (k, id) = untag(id);
        let Some((owner, sub_inner)) = unwrap_id(id) else { continue };
        let mut b = fanout::node_body(&body, from, size);
        b["pit"]["id"] = json!(sub_inner);
        // Split a global cursor back into this sub's own cursor: subs that
        // sort before the cursor's one have given all their ties (cursor past
        // every local value), later ones none (cursor before local doc 0).
        // Engine cursor sentinels preserve the other sort values while placing
        // this sub before/after every document in the requested direction.
        // Zero is a real document and must never stand in for "before first".
        if let (Some(slot), Some(after)) = (slot, after.as_ref()) {
            if let Some(t) = after.get(slot).and_then(Value::as_i64) {
                let (s, local) = ((t >> SUB_SHIFT) as usize, t & LOCAL_MASK);
                let (done, fresh) = (i64::MIN, -1);
                let v = match (i.cmp(&s), desc) {
                    (Ordering::Equal, _) => local,
                    (Ordering::Less, false) | (Ordering::Greater, true) => done,
                    _ => fresh,
                };
                b["search_after"][slot] = json!(v);
            }
        }
        let r = match json_req_formatted(req,"POST",format_args!("/_search{query}"),&b) {Ok(r)=>r,Err(e)=>return Some(RestResp::json(e.status,&e.to_json()))};
        reqs.push((owner.to_string(), k, r));
    }
    let searched_shards: usize = covers.iter().map(|count| count.unwrap_or(1)).sum();
    for (_, _, request) in &mut reqs {if let Err(e)=request.replace_header(crate::shard_engines::PREFILTER_SHARDS_HEADER,&searched_shards.to_string()) {return Some(RestResp::json(e.status,&e.to_json()))}}
    // DFS over the PIT's shard contexts (the engine's PIT search answers a
    // stats round from the same point in time).
    if super::dfs_wanted(req, &body, &params, reqs.len()) {
        let round = reqs.iter().map(|(owner, k, r)| {
            let (node, owner, k, r) = (node.clone(), owner.clone(), *k, super::dfs_collect_req(r));
            async move { exec_shard(&node, &owner, k, r).await }
        });
        let merged = match super::dfs_round(round).await {
            Ok(m) => m,
            Err(r) => return Some(r),
        };
        for (_, _, r) in reqs.iter_mut() {
            if let Err(e)=r.set_dfs(super::Dfs::Use(merged.clone())) {return Some(RestResp::json(e.status,&e.to_json()));}
        }
    }
    let owners: Vec<(String, u32)> = reqs.iter().map(|(owner, k, _)| (owner.clone(), *k)).collect();
    let futs = reqs.into_iter().map(|(owner, k, r)| {
        let node = node.clone();
        async move { exec_shard(&node, &owner, k, Ok(r)).await }
    });
    let mut parts = Vec::new();
    // ES: a shard whose PIT context is gone (or that failed) is a shard
    // failure of this search, not the search's answer.
    let mut failures: Vec<copies::ShardFailure> = Vec::new();
    let mut sources = elasticsearch_engine::source_json::ResponseSources::default();
    for (i, r) in futures::future::join_all(futs).await.into_iter().enumerate() {
        if !r.is_success() {
            if copies::is_request_error(&r) {
                return Some(r);
            }
            let (owner, k) = &owners[i];
            let covered = vec![String::new(); covers.get(i).copied().flatten().unwrap_or(1)];
            failures.extend(copies::ShardFailure::from_response(&r, &covered, *k, owner));
            continue;
        }
        // Lossless: sources pass through raw (a lone surrogate would
        // otherwise drop the part's hits).
        let mut v = match sources.parse_lazy(&r.body) {
            Ok(v) => v,
            Err(e) => return Some(RestResp::error(500, "illegal_state_exception", &e.to_string())),
        };
        if let Some(n) = covers.get(i).copied().flatten() {
            fanout::normalize_part_shards(&mut v, n);
        }
        if let (Some(slot), Some(hits)) = (slot, v["hits"]["hits"].as_array_mut()) {
            for h in hits {
                if let Some(x) = h["sort"].get(slot).and_then(Value::as_i64) {
                    h["sort"][slot] = json!(((i as i64) << SUB_SHIFT) | (x & LOCAL_MASK));
                }
            }
        }
        parts.push(v);
    }
    // Merge on the user's sort plus the (now global) tiebreak.
    let mut merge_body = body.clone();
    if let Some(slot) = slot {
        let n = fanout::sort_keys(&body, &params).map(|k| k.len()).unwrap_or(0);
        if slot == n {
            let mut s = match body.get("sort") {
                Some(Value::Array(a)) => a.clone(),
                Some(x @ Value::String(_)) | Some(x @ Value::Object(_)) => vec![x.clone()],
                _ => params.get("sort").map(|s| s.split(',').map(|p| match p.split_once(':') {
                    Some((f, o)) => json!({f: o}),
                    None => json!(p),
                }).collect()).unwrap_or_default(),
            };
            s.push(json!("_shard_doc"));
            merge_body["sort"] = Value::Array(s);
        }
    }
    if !failures.is_empty() {
        if parts.is_empty() {
            return Some(copies::phase_failure("query", "all shards failed", &failures));
        }
        if !copies::allow_partial(node, &params) {
            return Some(copies::phase_failure("query", "Partial shards failure", &failures));
        }
    }
    let mut merged = fanout::merge_search_with_source(&parts, &merge_body, &params, node.engine.max_buckets(), &req.body);
    if let Some(e) = fanout::take_agg_breaking() { return Some(RestResp::json(e.status, &e.to_json())); }
    if let Some(e) = fanout::take_agg_error() { return Some(RestResp::json(e.status, &e.to_json())); }
    copies::add_failures(&mut merged, &failures);
    let mut out = Map::new();
    out.insert("pit_id".into(), json!(wrapped));
    if let Value::Object(m) = merged {
        out.extend(m);
    }
    let mut response=RestResp::json(200,&Value::Null);response.replace_body_vec(sources.render(&Value::Object(out)));
    Some(response)
}

#[cfg(test)]
mod scroll_shard_tests {
    use super::*;
    #[test]
    fn admitted_query_formatters_preserve_order_empty_values_and_encoding() {
        assert_eq!(&*query_without("&&a=1&size=9&flag&from=2&a=3&", &["size","from"]).unwrap(),"?a=1&flag&a=3");
        assert_eq!(&*query_without("size=9", &["size"]).unwrap(),"");
        let params=HashMap::from([("typed_keys".into(),"x y/é".into()),("empty".into(),String::new())]);
        let rendered=super::super::SharedText::try_format(format_args!("{}",SelectedParams(&params,&["typed_keys","missing","empty"]))).unwrap();
        assert_eq!(&*rendered,"?typed_keys=x%20y%2F%C3%A9&empty=");
        let pairs=vec![("flag".into(),String::new()),("a".into(),"1".into()),("a".into(),"2".into())];
        let rendered=super::super::SharedText::try_format(format_args!("{}",QueryPairs(&pairs))).unwrap();
        assert_eq!(&*rendered,"flag&a=1&a=2");
    }
    #[test]
    fn query_owner_survives_scope_and_refuses_before_growth() {
        use elasticsearch_engine::memory::{Breakers,Kind,RequestScope,scoped};
        let input=format!("q={}","x".repeat(4096));
        let budget=Breakers::leak_for_tests(1<<20);
        budget.apply_settings(&|key|(key=="indices.breaker.total.use_real_memory").then(||"false".into()));
        let scope=RequestScope::new("query owner",budget);
        let query=scoped(Some(scope.clone()),||query_without(&input,&[])).unwrap();
        drop(scope);assert!(budget.get(Kind::Request).used()>0);
        drop(query);assert_eq!(budget.get(Kind::Request).used(),0);
        let small=Breakers::leak_for_tests(1024);
        small.apply_settings(&|key|(key=="indices.breaker.total.use_real_memory").then(||"false".into()));
        let scope=RequestScope::new("query refusal",small);
        assert_eq!(scoped(Some(scope.clone()),||query_without(&input,&[])).unwrap_err().status,429);
        drop(scope);assert_eq!(small.get(Kind::Request).used(),0);
    }


    #[test]
    fn raw_hits_merge_by_key_and_pass_through_unchanged() {
        let a = br#"{"_scroll_id":"a1","took":3,"timed_out":false,"hits":{"hits":[{"_index":"i","_id":"1","_score":2.0,"_source":{"t":"x \"q\" \u00e9"}},{"_index":"i","_id":"3","_score":0.5,"_source":{}}]}}"#;
        let b = br#"{"_scroll_id":"b1","took":7,"hits":{"hits":[{"_index":"i","_id":"2","_score":1.0,"_source":{"n":[1,2.5e3]}}]}}"#;
        let SubBody { meta: pa, hits: ha } = sub_hits(a, false).unwrap();
        let SubBody { meta: pb, hits: hb } = sub_hits(b, false).unwrap();
        assert_eq!((pa["_scroll_id"].as_str(), pa["took"].as_u64(), pb["took"].as_u64()), (Some("a1"), Some(3), Some(7)));
        assert_eq!(pa["hits"], json!({"hits": []}));
        let now = Instant::now();
        let mut subs = vec![Sub { id: "a".into(), shard: 0, indices: vec![], buf: ha, done: true, touched: now, legacy: false }, Sub { id: "b".into(), shard: 0, indices: vec![], buf: hb, done: true, touched: now, legacy: false }];
        let page = take(&mut subs, 3, None);
        let out = with_raw_hits(200, &json!({"hits": {"max_score": max_score(&page), "hits": RAW_HITS}}), &page);
        let text = std::str::from_utf8(&out.body).map(str::to_owned).unwrap();
        assert!(text.contains(r#"{"_index":"i","_id":"1","_score":2.0,"_source":{"t":"x \"q\" \u00e9"}}"#), "{text}");
        let v: Value = serde_json::from_str(&text).unwrap();
        let ids: Vec<&str> = v["hits"]["hits"].as_array().unwrap().iter().map(|h| h["_id"].as_str().unwrap()).collect();
        assert_eq!(ids, ["1", "2", "3"]);
        assert_eq!(v["hits"]["max_score"], 2.0);
        assert_eq!(v["hits"]["hits"][1]["_source"]["n"][1], 2500.0);
    }

    #[test]
    fn raw_hits_merge_by_sort_values_after_source() {
        let a = br#"{"_scroll_id":"a1","hits":{"total":2,"max_score":null,"hits":[{"_index":"i","_id":"1","_score":null,"_source":{"p":9},"sort":[9]},{"_index":"i","_id":"3","_score":null,"_source":{"p":1},"sort":[1]}]}}"#;
        let b = br#"{"_scroll_id":"b1","hits":{"total":1,"max_score":null,"hits":[{"_index":"i","_id":"2","_score":null,"_source":{"p":5},"sort":[5]}]}}"#;
        let keys = fanout::sort_keys(&json!({"sort": [{"p": "desc"}]}), &HashMap::new()).unwrap();
        let SubBody { meta, hits: ha } = sub_hits(a, true).unwrap();
        assert_eq!(meta["hits"], json!({"total": 2, "max_score": null, "hits": []}));
        let hb = sub_hits(b, true).unwrap().hits;
        let now = Instant::now();
        let mut subs = vec![Sub { id: "a".into(), shard: 0, indices: vec![], buf: ha, done: true, touched: now, legacy: false }, Sub { id: "b".into(), shard: 0, indices: vec![], buf: hb, done: true, touched: now, legacy: false }];
        let page = take(&mut subs, 3, Some(&keys));
        let ids: Vec<Value> = page.iter().map(|h| serde_json::from_str::<Value>(&h.raw).unwrap()["_id"].clone()).collect();
        assert_eq!(ids, [json!("1"), json!("2"), json!("3")]);
        assert_eq!(max_score(&page), Value::Null);
    }

    #[test]
    fn hit_keys_found_after_source_without_sort() {
        let raw = serde_json::value::RawValue::from_string(r#"{"_id":"1","_source":{"_score":7,"_index":"x"},"_score":2.5,"_index":"i","sort":[1]}"#.into()).unwrap();
        let hit = buf_hit(raw, false);
        assert_eq!((hit.index.as_deref(), hit.score, hit.sort.is_none()), (Some("i"), Some(2.5), true));
    }

    #[test]
    fn raw_hit_marker_in_aggregation_does_not_replace_user_data() {
        let raw = serde_json::value::RawValue::from_string(r#"{"_index":"i","_score":1,"_source":{"body":"\ud83d"}}"#.into()).unwrap();
        let hit = buf_hit(raw, false);
        let out = with_raw_hits(200, &json!({"aggregations":{"value": RAW_HITS},"hits":{"hits":RAW_HITS}}), &[hit]);
        let text = std::str::from_utf8(&out.body).map(str::to_owned).unwrap();
        assert!(text.contains(r#""aggregations":{"value":"\u0000raw-hits"}"#), "{text}");
        assert!(text.contains(r#""hits":{"hits":[{"_index":"i","_score":1,"_source":{"body":"\ud83d"}}]}"#), "{text}");
    }

    #[test]
    fn subsequent_pages_count_only_shards_with_opened_contexts() {
        let initial = json!({"total":3,"successful":3,"skipped":3,"failed":0});
        assert_eq!(continuation_shards(&initial), json!({"total":0,"successful":0,"skipped":0,"failed":0}));
        assert_eq!(initial["total"], 3);
        assert_eq!(continuation_shards(&json!({"total":3,"successful":3,"skipped":1,"failed":0})), json!({"total":2,"successful":2,"skipped":0,"failed":0}));
        assert_eq!(continuation_shards(&json!({"total":3,"successful":3,"skipped":0,"failed":0}))["total"], 3);
    }
    #[test]
    fn scroll_touch_requires_current_owner_advertisement() {
        let mut state = crate::internal_capabilities::tests::lifecycle_state();
        let mut peer = state.nodes.remove("n").unwrap();
        peer.id = "abcdefghijklmnopqrstuv".into();
        let id = format!("{}.scroll", peer.id);
        state.nodes.insert(peer.id.clone(), peer.clone());
        assert!(owner_supports_touch(&state, &id));
        peer.attributes.clear();
        peer.ephemeral_id = "restarted-old-build".into();
        state.nodes.insert(peer.id.clone(), peer);
        assert!(!owner_supports_touch(&state, &id));
        assert!(!owner_supports_touch(&state, "invalid"));
        state.nodes.clear();
        assert!(!owner_supports_touch(&state, &id));
    }

    #[test]
    fn clear_snapshot_preserves_context_created_during_dispatch() {
        let mut contexts = HashMap::from([("before".to_string(), 1)]);
        let selected = contexts.keys().cloned().collect();
        // A new scroll is registered while engine-clear requests are awaited.
        contexts.insert("during".to_string(), 2);
        forget_snapshot(&mut contexts, selected);
        assert_eq!(contexts, HashMap::from([("during".to_string(), 2)]));
    }

    #[test]
    fn context_clear_preserves_refusal_and_accepts_only_valid_missing_context() {
        assert_eq!(freed_count(&RestResp::json(404, &json!({"succeeded":true,"num_freed":0}))).unwrap(), 0);
        assert!(freed_count(&RestResp::json(404, &json!({"succeeded":true,"num_freed":1}))).is_err());
        assert!(freed_count(&RestResp::json(200, &json!({"succeeded":true}))).is_err());
        let response = RestResp::error(429, "circuit_breaking_exception", "clear refused");
        assert_eq!(freed_count(&response).unwrap_err().status, 429);
    }

    /// A hit text as the engine renders a pmc scroll hit (`sort=_doc`):
    /// `_source` first, `sort` after it, with escapes, unicode and nesting.
    fn pmc_hit(i: usize, body_len: usize) -> String {
        let mut body = String::new();
        while body.len() < body_len {
            body.push_str(&format!("line {i} of the article: \\\"quoted\\\" \\u00e9 \\\\ tab\\t end\\n"));
        }
        format!(r#"{{"_index":"pmc","_id":"{i}","_score":null,"_source":{{"name":"a{{b}}[c]","timestamp":"2015-10-30 20:08:11","pmid":"","body":"{body}","nested":{{"x":[1,2.5e3,true,null,{{"y":"}}"}}]}}}},"sort":[{i}]}}"#)
    }

    fn sub_response(hits: &[String]) -> Vec<u8> {
        format!(r#"{{"_scroll_id":"FGluY2x1ZGU","took":4,"timed_out":false,"_shards":{{"total":1,"successful":1,"skipped":0,"failed":0}},"hits":{{"total":{{"value":115000,"relation":"eq"}},"max_score":null,"hits":[{}]}}}}"#, hits.join(",")).into_bytes()
    }

    fn same(a: &SubBody, b: &SubBody) {
        assert_eq!(a.meta, b.meta);
        assert_eq!(a.hits.len(), b.hits.len());
        for (x, y) in a.hits.iter().zip(&b.hits) {
            assert_eq!((&x.index, x.score, &x.sort, &x.raw), (&y.index, y.score, &y.sort, &y.raw));
        }
    }

    /// A page read through its hit spans is the page read whole; spans
    /// that do not fit the body (edited after them, or wrong) are refused.
    #[test]
    fn spanned_read_matches_whole_read() {
        for n in [0usize, 1, 5] {
            let hits: Vec<String> = (0..n).map(|i| pmc_hit(i, 50 + i * 31)).collect();
            let body = sub_response(&hits);
            let text = String::from_utf8(body.clone()).unwrap();
            let end = text.rfind("]}}").unwrap();
            let lens: Vec<u32> = hits.iter().map(|h| h.len() as u32).collect();
            let spans = elasticsearch_http::encode_hit_spans(body.len() - end, &lens);
            for need_sort in [false, true] {
                same(&sub_hits_spanned(&body, &spans, need_sort).unwrap(), &sub_hits(&body, need_sort).unwrap());
            }
            // An edit before the hits keeps them where the spans say.
            let edited = text.replacen("FGluY2x1ZGU", "node.FGluY2x1ZGU", 1).into_bytes();
            same(&sub_hits_spanned(&edited, &spans, true).unwrap(), &sub_hits(&edited, true).unwrap());
            if n > 0 {
                let wrong = elasticsearch_http::encode_hit_spans(body.len() - end + 1, &lens);
                assert!(sub_hits_spanned(&body, &wrong, true).is_none());
                let mut short = lens.clone();
                short[0] -= 1;
                assert!(sub_hits_spanned(&body, &elasticsearch_http::encode_hit_spans(body.len() - end, &short), true).is_none());
            }
        }
        assert!(sub_hits_spanned(b"{}", "x", false).is_none());
    }

    /// The borrowed read reads exactly what the serde reader reads: the
    /// meta, every hit's text and its merge keys, with and without `sort`,
    /// for engine-shaped and unusual (escaped keys, whitespace, empty,
    /// duplicate keys, keys after `_source`) responses.
    #[test]
    fn raw_scan_matches_serde_reader() {
        let hits: Vec<String> = (0..7).map(|i| pmc_hit(i, 300 + i * 97)).collect();
        let mut bodies = vec![sub_response(&hits), sub_response(&[])];
        bodies.push(br#" { "_scroll_id" : "x" , "hits" : { "hits" : [ { "_index" : "i" , "_score" : 1.5 , "sort" : [ 3 , "a" ] } , {"_id":"2"} ] , "total" : 2 } } "#.to_vec());
        bodies.push(br#"{"hits":{"hits":[{"_source":{"_index":"no","sort":[0]},"_index":"esc","_score":2,"sort":null}],"hits":[{"_index":"second","_score":0.25,"_source":"\ud83d"}]}}"#.to_vec());
        bodies.push(br#"{"hits":{"hits":[{"_index":"a","_score":1,"_index":"b","sort":[1],"_score":3}]},"aggregations":{"t":{"buckets":[{"key":"k","doc_count":1}]}}}"#.to_vec());
        bodies.push(br#"{"hits":{"hits":[{"_index":"i","_score":1,"_source":{"sort":[5]}},{"_index":"i","_score":1,"_source":{"x":",\"sort\":[6]}"},"sort":[7, "b"]}]}}"#.to_vec());
        bodies.push(br#"{"hits":{"max_score":null,"hits":[]},"_shards":{"failures":[{"reason":{"type":"x","reason":"a \"b\""}}]}}"#.to_vec());
        for body in &bodies {
            for need_sort in [false, true] {
                if need_sort && body.windows(12).any(|w| w == b"\"_index\":\"b\"") {
                    continue; // a repeated key: the first value is read (see `buf_hit_str`)
                }
                let fast = sub_hits(body, need_sort).unwrap_or_else(|| panic!("refused {}", String::from_utf8_lossy(body)));
                let slow = sub_hits_serde(body, need_sort).unwrap();
                same(&fast, &slow);
            }
        }
        // What serde refuses, the scan does not take either (or serde decides).
        for bad in [&b"{\"hits\":{\"hits\":{}}}"[..], b"{\"hits\":[]}", b"{\"a\":1} x", b"{\"a\":\"unterminated}", b"[1]"] {
            assert!(sub_hits(bad, true).is_none() == sub_hits_serde(bad, true).is_none(), "{}", String::from_utf8_lossy(bad));
        }
    }

    /// The structural writer emits what the serde `Body` serializer wrote:
    /// the meta compactly, `hits.hits` verbatim, raw scripted aggregation
    /// values rendered.
    #[test]
    fn raw_hits_body_is_compact_serde_with_verbatim_hits() {
        let hits: Vec<BufHit> = (0..3).map(|i| buf_hit_str(&pmc_hit(i, 200), true)).collect();
        let v = json!({"_scroll_id": "s\"id", "took": 1, "timed_out": false, "_shards": {"total": 5, "successful": 5, "skipped": 0, "failed": 0},
            "hits": {"total": {"value": 3, "relation": "eq"}, "max_score": null, "hits": RAW_HITS}});
        let body = String::from_utf8(raw_hits_body(&v, &hits).unwrap()).unwrap();
        let mut expect = serde_json::to_string(&v).unwrap();
        let joined = hits.iter().map(|h| &*h.raw).collect::<Vec<_>>().join(",");
        expect = expect.replace(&serde_json::to_string(RAW_HITS).unwrap(), &format!("[{joined}]"));
        assert_eq!(body, expect);
        assert!(raw_hits_body(&json!({"hits": 1}), &hits).is_none());
        assert!(raw_hits_body(&json!([1]), &hits).is_none());
    }

    /// A composite scroll over simulated sub streams: refilling only the
    /// subs the merge would exhaust returns exactly the pages of the
    /// former rule (refill every live sub holding less than a page), for
    /// `_doc`, field and score orders, uneven and drying-up subs, and
    /// fetches no more (here far fewer) hits.
    #[test]
    fn lazy_refill_pages_equal_eager_refill_pages() {
        // Sub `s` stream: hits sorted in the merge order of `keys`.
        fn streams(subs: usize, lens: &[usize], order: u8) -> Vec<Vec<BufHit>> {
            let mut seed = 5u64;
            (0..subs).map(|s| {
                let mut keys: Vec<i64> = (0..lens[s % lens.len()]).map(|_| { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((seed >> 33) % 5000) as i64 }).collect();
                match order { 0 => keys = (0..keys.len() as i64).collect(), 1 => keys.sort(), _ => keys.sort_by(|a, b| b.cmp(a)) }
                keys.into_iter().enumerate().map(|(d, k)| {
                    let raw = format!(r#"{{"_index":"i","_id":"{s}-{d}","_score":{},"_source":{{}},"sort":[{k}]}}"#, if order == 2 { k.to_string() } else { "null".into() });
                    buf_hit_str(&raw, order != 2)
                }).collect()
            }).collect()
        }
        fn run(mut src: Vec<std::collections::VecDeque<BufHit>>, size: usize, pages: usize, keys: Option<&[fanout::SortKey]>, lazy: bool) -> (Vec<Vec<String>>, usize) {
            let now = Instant::now();
            let mut fetched = 0;
            let mut subs: Vec<Sub> = src.iter_mut().map(|s| {
                let take_n = size.min(s.len());
                let page: std::collections::VecDeque<BufHit> = s.drain(..take_n).collect();
                fetched += page.len();
                Sub { id: String::new(), shard: 0, indices: vec![], done: page.len() < size, buf: page, touched: now, legacy: false }
            }).collect();
            let mut out = vec![take(&mut subs, size, keys).iter().map(|h| h.raw.to_string()).collect::<Vec<_>>()];
            for _ in 1..pages {
                let short = refills_needed(&subs, size, keys);
                for i in 0..subs.len() {
                    let want = if lazy { short[i] } else { subs[i].buf.len() < size };
                    if !subs[i].done && want {
                        let take_n = size.min(src[i].len());
                        let page: Vec<BufHit> = src[i].drain(..take_n).collect();
                        fetched += page.len();
                        subs[i].done = page.len() < size;
                        subs[i].buf.extend(page);
                    }
                }
                out.push(take(&mut subs, size, keys).iter().map(|h| h.raw.to_string()).collect());
            }
            (out, fetched)
        }
        let doc_keys = fanout::sort_keys(&json!({}), &HashMap::from([("sort".to_string(), "_doc".to_string())]));
        let asc_keys = fanout::sort_keys(&json!({"sort": [{"n": "asc"}]}), &HashMap::new());
        for (order, keys) in [(0u8, doc_keys.as_deref()), (1, asc_keys.as_deref()), (2, None)] {
            for (subs, lens, size, pages) in [(5, vec![2600], 100, 25), (5, vec![3000, 40, 0, 777], 100, 40), (3, vec![50, 51, 49], 10, 30), (1, vec![1000], 7, 200)] {
                let lazy = run(streams(subs, &lens, order).into_iter().map(Into::into).collect(), size, pages, keys, true);
                let eager = run(streams(subs, &lens, order).into_iter().map(Into::into).collect(), size, pages, keys, false);
                assert_eq!(lazy.0, eager.0, "order {order} subs {subs} lens {lens:?}");
                assert!(lazy.1 <= eager.1, "order {order}: lazy fetched {} > eager {}", lazy.1, eager.1);
                if order == 0 && subs == 5 && lens == [2600] {
                    // pmc's shape (25 pages of 100 over 5 evenly interleaved
                    // shards): both rules refill every sub once per 5 pages.
                    eprintln!("pmc-shaped scroll: lazy refill fetched {} hits, eager {}", lazy.1, eager.1);
                }
            }
        }
    }

    /// Timing of one pmc scroll page on the coordinator (100 hits of
    /// `PMC_HIT_BYTES`, default 40,000): the serde reader + `Body` writer
    /// against the scan + structural writer. Run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn pmc_scroll_page_coordinator_timing() {
        let len: usize = std::env::var("PMC_HIT_BYTES").ok().and_then(|v| v.parse().ok()).unwrap_or(40_000);
        let n: usize = std::env::var("PMC_HITS").ok().and_then(|v| v.parse().ok()).unwrap_or(100);
        // `GEO_HITS`: geonames-shaped hits (few escapes) instead.
        let geo = std::env::var_os("GEO_HITS").is_some();
        let hits: Vec<String> = (0..n).map(|i| if geo {
            format!(r#"{{"_index":"geonames","_id":"Xk{i}aZ9","_score":1.0,"_source":{{"geonameid":{},"name":"Pic de Font Blanca {i}","asciiname":"Pic de Font Blanca","alternatenames":"Pic de Font Blanca,Pic du Port,Pico de Fuente Blanca","feature_class":"T","feature_code":"PK","country_code":"AD","cc2":"","admin1_code":"00","admin2_code":"","admin3_code":"","admin4_code":"","population":0,"elevation":"","dem":"2860","timezone":"Europe/Andorra","location":[1.53335,42.64991]}}}}"#, 2986043 + i)
        } else { pmc_hit(i, len) }).collect();
        let need_sort = std::env::var_os("NO_SORT").is_none();
        let body = sub_response(&hits);
        let meta = json!({"_scroll_id": "x", "took": 1, "timed_out": false, "hits": {"max_score": null, "hits": RAW_HITS}});
        let time = |f: &dyn Fn() -> usize| {
            let mut t: Vec<_> = (0..21).map(|_| { let s = Instant::now(); assert!(f() > 0); s.elapsed() }).collect();
            t.sort();
            t[10]
        };
        let serde_read = time(&|| sub_hits_serde(&body, need_sort).unwrap().hits.len());
        let scan_read = time(&|| sub_hits(&body, need_sort).unwrap().hits.len());
        let page: Vec<BufHit> = sub_hits(&body, need_sort).unwrap().hits.into();
        let write = time(&|| raw_hits_body(&meta, &page).unwrap().len());
        eprintln!("pmc page ({} bytes): serde read {serde_read:?}, borrowed read {scan_read:?}, write {write:?}", body.len());
    }

}
