//! Typed bulk item results for an in-process shard group (the cluster
//! coordinator's own bulk subgroup, [`crate::PreparedBulk`]).
//!
//! A plain index / create that succeeded is handed back as a
//! [`TypedWrite`] instead of the JSON object `write_json` builds: the
//! coordinator renders the final response once, from these, instead of the
//! engine building a `Value` per item that the coordinator then rebuilds,
//! merges and serializes (and frees on another thread). Every other item
//! (errors, updates, deletes through other paths, failure store results)
//! stays the JSON value the engine built. [`TypedWrite::to_value`] is the
//! exact object of the JSON path, and [`TypedWrite::write_json`] writes the
//! bytes `serde_json` writes for it; the coordinator falls back to values
//! wherever its response needs them.
use serde_json::{Map, Value};
use std::sync::Arc;

/// A successful plain write (`index` / `create`) of a bulk item.
#[derive(Clone, Debug, PartialEq)]
pub struct TypedWrite {
    /// The action key the engine answered under (`index` / `create`).
    pub action: &'static str,
    pub index: Arc<str>,
    pub id: String,
    pub version: u64,
    pub result: &'static str,
    pub seq_no: i64,
    pub primary_term: u64,
    /// `_shards.total` of the JSON path (`1 + number_of_replicas`).
    pub total_shards: usize,
    /// The index has sequence numbers disabled (`_seq_no: -2`,
    /// `_primary_term: 0`).
    pub seq_disabled: bool,
    pub status: u16,
    pub forced_refresh: bool,
}

/// One bulk item result of a typed shard group.
#[derive(Clone, Debug)]
pub enum TypedItem {
    Write(TypedWrite),
    Json(Value),
}

/// A typed shard group's answer: `took` and the item results in body order.
#[derive(Clone, Debug)]
pub struct TypedBulk {
    pub took: u64,
    pub items: Vec<TypedItem>,
}

impl TypedWrite {
    fn default_shards(&self) -> Value {
        if self.result == "noop" {
            serde_json::json!({"total": 0, "successful": 0, "failed": 0})
        } else {
            serde_json::json!({"total": self.total_shards, "successful": 1, "failed": 0})
        }
    }

    /// The item's inner object, exactly as the bulk's JSON path builds it
    /// (`write_json` plus `status`, then `forced_refresh` when the bulk's
    /// refresh was forced).
    pub fn inner_value(&self) -> Map<String, Value> {
        let mut m = Map::new();
        m.insert("_index".into(), Value::String(self.index.to_string()));
        m.insert("_id".into(), Value::String(self.id.clone()));
        m.insert("_version".into(), Value::from(self.version));
        m.insert("result".into(), Value::String(self.result.into()));
        m.insert("_shards".into(), self.default_shards());
        if self.seq_disabled {
            m.insert("_seq_no".into(), Value::from(-2));
            m.insert("_primary_term".into(), Value::from(0));
        } else {
            m.insert("_seq_no".into(), Value::from(self.seq_no));
            m.insert("_primary_term".into(), Value::from(self.primary_term));
        }
        m.insert("status".into(), Value::from(self.status));
        // Set by the bulk's refresh after the item was built: last.
        if self.forced_refresh {
            m.insert("forced_refresh".into(), Value::Bool(true));
        }
        m
    }

    /// The whole item, `{key: inner}` (`key`: the action label the
    /// response shows).
    pub fn to_value(&self, key: &str) -> Value {
        let mut outer = Map::new();
        outer.insert(key.into(), Value::Object(self.inner_value()));
        Value::Object(outer)
    }

    /// Write `{key: inner}` as `serde_json` writes [`Self::to_value`], with
    /// `_shards` replaced by `shards` when given (the coordinator's
    /// replication summary).
    pub fn write_json(&self, key: &str, shards: Option<&Value>, out: &mut Vec<u8>) {
        fn string(out: &mut Vec<u8>, s: &str) {
            serde_json::to_writer(&mut *out, s).expect("string to JSON");
        }
        fn number(out: &mut Vec<u8>, n: impl std::fmt::Display) {
            use std::io::Write as _;
            write!(out, "{n}").expect("number to JSON");
        }
        out.push(b'{');
        string(out, key);
        out.extend_from_slice(b":{\"_index\":");
        string(out, &self.index);
        out.extend_from_slice(b",\"_id\":");
        string(out, &self.id);
        out.extend_from_slice(b",\"_version\":");
        number(out, self.version);
        out.extend_from_slice(b",\"result\":");
        string(out, self.result);
        out.extend_from_slice(b",\"_shards\":");
        match shards {
            Some(shards) => serde_json::to_writer(&mut *out, shards).expect("shards to JSON"),
            None if self.result == "noop" => out.extend_from_slice(b"{\"total\":0,\"successful\":0,\"failed\":0}"),
            None => {
                out.extend_from_slice(b"{\"total\":");
                number(out, self.total_shards);
                out.extend_from_slice(b",\"successful\":1,\"failed\":0}");
            }
        }
        out.extend_from_slice(b",\"_seq_no\":");
        if self.seq_disabled {
            out.extend_from_slice(b"-2,\"_primary_term\":0");
        } else {
            number(out, self.seq_no);
            out.extend_from_slice(b",\"_primary_term\":");
            number(out, self.primary_term);
        }
        out.extend_from_slice(b",\"status\":");
        number(out, self.status);
        if self.forced_refresh {
            out.extend_from_slice(b",\"forced_refresh\":true");
        }
        out.extend_from_slice(b"}}");
    }
}

impl TypedItem {
    /// The item as the JSON path's value (`key` for a write).
    pub fn to_value(&self, key: &str) -> Value {
        match self {
            TypedItem::Write(w) => w.to_value(key),
            TypedItem::Json(v) => v.clone(),
        }
    }

    /// Whether the item failed (the JSON path's `errors` test).
    pub fn is_error(&self) -> bool {
        match self {
            TypedItem::Write(_) => false,
            TypedItem::Json(v) => v.as_object().and_then(|o| o.values().next()).map(|v| v.get("error").is_some()).unwrap_or(false),
        }
    }
}

impl TypedBulk {
    /// The engine's JSON response for these results (`{took, errors, items}`).
    pub fn to_value(&self) -> Value {
        let errors = self.items.iter().any(TypedItem::is_error);
        let items: Vec<Value> = self.items.iter().map(|i| match i {
            TypedItem::Write(w) => w.to_value(w.action),
            TypedItem::Json(v) => v.clone(),
        }).collect();
        serde_json::json!({"took": self.took, "errors": errors, "items": items})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(result: &'static str, forced: bool, seq_disabled: bool, index: &str, id: &str) -> TypedWrite {
        TypedWrite { action: "index", index: Arc::from(index), id: id.into(), version: 7, result, seq_no: 1234567890123, primary_term: 3,
            total_shards: 2, seq_disabled, status: if result == "created" { 201 } else { 200 }, forced_refresh: forced }
    }

    /// The written bytes are `serde_json`'s for the JSON path's value, for
    /// every field combination, escaped strings and replaced `_shards`.
    #[test]
    fn written_bytes_match_the_json_path() {
        let shards = serde_json::json!({"total": 2, "successful": 1, "failed": 1, "failures": [{"_shard": 0, "_node": "n\"1", "reason": {"type": "x", "reason": "é\n"}, "status": "INTERNAL_SERVER_ERROR", "primary": false}]});
        for result in ["created", "updated", "noop"] {
            for forced in [false, true] {
                for seq_disabled in [false, true] {
                    for (index, id) in [("logs-1", "abc"), ("we\"ird\\idx", "id with \"quotes\"\t\u{1}é")] {
                        let w = write(result, forced, seq_disabled, index, id);
                        for key in ["index", "create"] {
                            let mut out = Vec::new();
                            w.write_json(key, None, &mut out);
                            assert_eq!(out, serde_json::to_vec(&w.to_value(key)).unwrap(), "{result} {forced} {seq_disabled} {key}");
                            let mut patched = w.to_value(key);
                            patched[key]["_shards"] = shards.clone();
                            let mut out = Vec::new();
                            w.write_json(key, Some(&shards), &mut out);
                            assert_eq!(out, serde_json::to_vec(&patched).unwrap());
                        }
                    }
                }
            }
        }
    }
}
