//! Which implementation serves a built-in model, and the opt-in stand-ins.
//!
//! The `elasticsearch` service's built-in models are:
//!
//! * `.multilingual-e5-small`: the real model runs on the CPU with `candle`
//!   when the binary has the `candle-models` feature and the model files are
//!   installed ([`super::real`]).
//! * `.elser_model_2` and `.rerank-v1`: rustic has no implementation (the
//!   weights are not distributed outside Elastic).
//!
//! When the real model cannot run, inference fails the way ES fails for a
//! model that is not deployed (a `status_exception`), unless the node opts
//! into the built-in stand-ins with
//! `xpack.inference.builtin_fallback: lexical`. The stand-ins
//! ([`super::local`]) produce output of the same shape but different
//! semantics, so every response that used one carries a warning header, and
//! `GET _inference` / `GET _ml/trained_models/_stats` name the
//! implementation.
//!
//! An endpoint records the implementation that first produced embeddings
//! with it ([`check_pin`]); a later call that would use a different one is
//! refused, so an index never mixes vectors from two implementations.

use crate::error::EsError;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Node setting that enables the built-in stand-ins: `none` (default) or
/// `lexical`.
pub const FALLBACK_SETTING: &str = "xpack.inference.builtin_fallback";

/// Node-local record of the implementation each endpoint embedded with.
const PINS_FILE: &str = "inference-implementations.json";

/// What serves a built-in model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Implementation {
    /// The real `multilingual-e5-small` weights on `candle`.
    Candle,
    /// Signed feature hashing of words and character n-grams (E5 stand-in).
    HashingFallback,
    /// Saturated term weights / term overlap (ELSER and rerank stand-ins).
    LexicalFallback,
}

impl Implementation {
    pub fn name(self) -> &'static str {
        match self {
            Implementation::Candle => "candle",
            Implementation::HashingFallback => "hashing-fallback",
            Implementation::LexicalFallback => "lexical-fallback",
        }
    }

    pub fn is_fallback(self) -> bool {
        self != Implementation::Candle
    }
}

/// The built-in models.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Builtin {
    E5,
    Elser,
    Rerank,
}

impl Builtin {
    pub fn of(model_id: &str) -> Option<Builtin> {
        match model_id.trim_end_matches("_linux-x86_64") {
            ".multilingual-e5-small" => Some(Builtin::E5),
            ".elser_model_2" => Some(Builtin::Elser),
            ".rerank-v1" => Some(Builtin::Rerank),
            _ => None,
        }
    }

    /// The preconfigured endpoint serving the model.
    pub fn default_endpoint(self) -> &'static str {
        match self {
            Builtin::E5 => ".multilingual-e5-small-elasticsearch",
            Builtin::Elser => ".elser-2-elasticsearch",
            Builtin::Rerank => ".rerank-v1-elasticsearch",
        }
    }

    /// The model id without the platform suffix (its model directory name).
    pub fn base_id(self) -> &'static str {
        match self {
            Builtin::E5 => ".multilingual-e5-small",
            Builtin::Elser => ".elser_model_2",
            Builtin::Rerank => ".rerank-v1",
        }
    }
}

/// Parse [`FALLBACK_SETTING`]: `Ok(true)` when the stand-ins are enabled.
pub fn parse_fallback(v: Option<&Value>) -> Result<bool, String> {
    let s = match v {
        None | Some(Value::Null) => return Ok(false),
        Some(Value::String(s)) => s.trim().to_ascii_lowercase(),
        Some(other) => other.to_string(),
    };
    match s.as_str() {
        "" | "none" => Ok(false),
        "lexical" => Ok(true),
        other => Err(format!("illegal value [{other}] for setting [{FALLBACK_SETTING}]; expected one of [none, lexical]")),
    }
}

/// A node's stand-in switch and implementation records.
pub struct NodeState {
    fallback: AtomicBool,
    /// `inference_id -> implementation name`, loaded on first use.
    pins: Mutex<Option<Map<String, Value>>>,
}

/// The state of the node whose `_state` dir is `state_dir`.
pub fn node(state_dir: &Path) -> Arc<NodeState> {
    static R: OnceLock<Mutex<HashMap<PathBuf, Arc<NodeState>>>> = OnceLock::new();
    let mut r = R.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner());
    if let Some(n) = r.get(state_dir) {
        return n.clone();
    }
    r.entry(state_dir.to_path_buf()).or_insert_with(|| Arc::new(NodeState { fallback: AtomicBool::new(false), pins: Mutex::new(None) })).clone()
}

/// Enable or disable the stand-ins for the node whose `_state` dir is `state_dir`.
pub fn set_fallback(state_dir: &Path, enabled: bool) {
    node(state_dir).fallback.store(enabled, Ordering::Relaxed);
}

/// Apply [`FALLBACK_SETTING`] from the node settings (an invalid value
/// leaves the stand-ins off; the cluster node refuses to start with one).
pub fn configure(state_dir: &Path, settings: &Map<String, Value>) {
    configure_node(&node(state_dir), settings);
}

/// [`configure`] on a node's state (no registry lookup: the per-request path).
pub fn configure_node(n: &NodeState, settings: &Map<String, Value>) {
    let on = match parse_fallback(settings.get(FALLBACK_SETTING)) {
        Ok(on) => on,
        Err(e) => {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::error!("{e}; built-in model stand-ins stay disabled"));
            false
        }
    };
    if n.fallback.load(Ordering::Relaxed) != on {
        n.fallback.store(on, Ordering::Relaxed);
    }
}

pub fn fallback_enabled(state_dir: &Path) -> bool {
    node(state_dir).fallback.load(Ordering::Relaxed)
}

/// Directory holding a built-in model's files: `ES_ML_MODEL_DIR_<ID>`
/// (`ES_ML_MODEL_DIR_MULTILINGUAL_E5_SMALL`) or `<path.data>/ml-models/<id>`.
pub fn model_dir(state_dir: &Path, b: Builtin) -> PathBuf {
    let env = format!("ES_ML_MODEL_DIR_{}", b.base_id().trim_start_matches('.').to_ascii_uppercase().replace(['-', '.'], "_"));
    if let Some(d) = std::env::var_os(&env).filter(|d| !d.is_empty()) {
        return PathBuf::from(d);
    }
    state_dir.parent().unwrap_or(state_dir).join("ml-models").join(b.base_id())
}

const HINT: &str = "set [xpack.inference.builtin_fallback: lexical] to serve it with the built-in stand-in instead (its results differ from the real model)";

/// The implementation serving built-in model `b` on this node, or why none can.
pub fn resolve(state_dir: &Path, b: Builtin) -> Result<Implementation, String> {
    let fallback = fallback_enabled(state_dir);
    let why = match b {
        Builtin::E5 => match super::real::ready(&model_dir(state_dir, b)) {
            Ok(()) => return Ok(Implementation::Candle),
            Err(e) => e,
        },
        Builtin::Elser => "rustic has no ELSER v2 implementation (the model weights are not distributed outside Elastic)".to_string(),
        Builtin::Rerank => "rustic has no Elastic Rerank implementation (the model weights are not distributed outside Elastic)".to_string(),
    };
    if fallback {
        return Ok(if b == Builtin::E5 { Implementation::HashingFallback } else { Implementation::LexicalFallback });
    }
    Err(format!("model [{}] cannot run on this node: {why}; {HINT}", b.base_id()))
}

/// ES's error for inference on a model that is not deployed
/// (`TransportInternalInferModelAction`), with the reason as its cause.
pub fn not_deployed(model_id: &str, why: &str) -> EsError {
    EsError::new(409, "status_exception", format!("Trained model deployment [{model_id}] is not allocated to any nodes")).caused_by(EsError::new(409, "status_exception", why.to_string()))
}

/// The response warning of a call served by a stand-in.
pub fn warning(model_id: &str, imp: Implementation) -> String {
    format!(
        "model [{model_id}] is served by the built-in [{}] stand-in enabled by [{FALLBACK_SETTING}]; its results are not those of the real model",
        imp.name()
    )
}

/// Is `w` the warning of a call served by a stand-in ([`warning`])?
pub fn is_fallback_warning(w: &str) -> bool {
    w.contains(&format!("stand-in enabled by [{FALLBACK_SETTING}]"))
}

fn with_pins<R>(state_dir: &Path, f: impl FnOnce(&mut Map<String, Value>) -> (R, bool)) -> R {
    let n = node(state_dir);
    let mut g = n.pins.lock().unwrap_or_else(|e| e.into_inner());
    let pins = g.get_or_insert_with(|| {
        std::fs::read(state_dir.join(PINS_FILE)).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()).and_then(|v| v.as_object().cloned()).unwrap_or_default()
    });
    let (out, changed) = f(pins);
    if changed {
        let _ = std::fs::create_dir_all(state_dir);
        if let Err(e) = serde_json::to_vec_pretty(&Value::Object(pins.clone())).map_err(std::io::Error::other).and_then(|b| std::fs::write(state_dir.join(PINS_FILE), b)) {
            tracing::error!("failed to persist inference implementations: {e}");
        }
    }
    out
}

/// The implementation `inference_id` embedded with, if it has embedded.
pub fn pin_of(state_dir: &Path, inference_id: &str) -> Option<String> {
    with_pins(state_dir, |p| (p.get(inference_id).and_then(Value::as_str).map(str::to_string), false))
}

/// Record that `inference_id` embeds with `imp`, refusing a change from
/// the implementation it embedded with before: vectors of two
/// implementations are not comparable (as ES refuses changing an endpoint's
/// dimensions or similarity under existing data).
pub fn check_pin(state_dir: &Path, inference_id: &str, imp: Implementation) -> Result<(), EsError> {
    with_pins(state_dir, |p| match p.get(inference_id).and_then(Value::as_str) {
        Some(prev) if prev == imp.name() => (Ok(()), false),
        Some(prev) => {
            let e = EsError::new(
                409,
                "status_exception",
                format!(
                    "Inference endpoint [{inference_id}] produced its existing embeddings with implementation [{prev}] but would now use [{}]; embeddings from different implementations are not comparable. Use a new inference endpoint (and reindex) to change implementation",
                    imp.name()
                ),
            );
            (Err(e), false)
        }
        None => {
            p.insert(inference_id.to_string(), Value::String(imp.name().to_string()));
            (Ok(()), true)
        }
    })
}

/// Forget the implementation of `inference_id` (no data uses it any more).
pub fn clear_pin(state_dir: &Path, inference_id: &str) {
    with_pins(state_dir, |p| {
        let removed = p.shift_remove(inference_id).is_some();
        ((), removed)
    })
}

/// Forget every recorded implementation (feature reset).
pub fn clear_pins(state_dir: &Path) {
    with_pins(state_dir, |p| {
        let changed = !p.is_empty();
        p.clear();
        ((), changed)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setting_values() {
        assert_eq!(parse_fallback(None), Ok(false));
        assert_eq!(parse_fallback(Some(&Value::String("none".into()))), Ok(false));
        assert_eq!(parse_fallback(Some(&Value::String("lexical".into()))), Ok(true));
        assert!(parse_fallback(Some(&Value::String("hashing".into()))).unwrap_err().contains("expected one of [none, lexical]"));
    }

    #[test]
    fn elser_needs_opt_in_and_pins_refuse_changes() {
        let d = tempfile::tempdir().unwrap();
        let st = d.path().join("_state");
        let e = resolve(&st, Builtin::Elser).unwrap_err();
        assert!(e.contains("no ELSER v2 implementation") && e.contains(FALLBACK_SETTING), "{e}");
        set_fallback(&st, true);
        assert_eq!(resolve(&st, Builtin::Elser), Ok(Implementation::LexicalFallback));
        check_pin(&st, "ep", Implementation::HashingFallback).unwrap();
        check_pin(&st, "ep", Implementation::HashingFallback).unwrap();
        let err = check_pin(&st, "ep", Implementation::Candle).unwrap_err();
        assert_eq!(err.status, 409);
        assert!(err.reason.contains("[hashing-fallback] but would now use [candle]"), "{}", err.reason);
        // Persisted per node.
        assert!(std::fs::read_to_string(st.join(PINS_FILE)).unwrap().contains("hashing-fallback"));
        clear_pin(&st, "ep");
        check_pin(&st, "ep", Implementation::Candle).unwrap();
    }
}
