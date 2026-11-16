//! Real CPU transformer inference (feature `candle-models`): runs
//! `intfloat/multilingual-e5-small` (BERT architecture, XLM-R tokenizer)
//! with `candle`, from files under `<path.data>/ml-models/<model>/`
//! (`config.json`, `tokenizer.json`, `model.safetensors`) or the directory
//! named by `ES_ML_MODEL_DIR_MULTILINGUAL_E5_SMALL` ([`super::backend::model_dir`]).
//!
//! Without the feature, or without loadable files, the model cannot run:
//! callers report it as not deployed (see [`super::backend`]); the built-in
//! stand-ins are used only when the node explicitly enables them.

#[cfg(feature = "candle-models")]
mod imp {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use candle_transformers::models::bert::{BertModel, Config};
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex, OnceLock};
    use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

    pub struct Embedder {
        model: BertModel,
        tok: Tokenizer,
        dev: Device,
    }

    /// Loaded models, and load failures keyed by the weights' mtime (a
    /// broken install is not reloaded on every call; replacing the files
    /// retries).
    enum Slot {
        Ready(Arc<Embedder>),
        Failed(Option<std::time::SystemTime>, String),
    }

    fn cache() -> &'static Mutex<HashMap<PathBuf, Slot>> {
        static C: OnceLock<Mutex<HashMap<PathBuf, Slot>>> = OnceLock::new();
        C.get_or_init(|| Mutex::new(HashMap::new()))
    }

    fn load(dir: &Path) -> Result<Embedder, String> {
        let dev = Device::Cpu;
        let cfg: Config = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).map_err(|e| format!("config.json: {e}"))?).map_err(|e| format!("config.json: {e}"))?;
        let mut tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| format!("tokenizer.json: {e}"))?;
        tok.with_padding(Some(PaddingParams::default()));
        tok.with_truncation(Some(TruncationParams { max_length: 512, ..Default::default() })).map_err(|e| e.to_string())?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &dev) }.map_err(|e| format!("model.safetensors: {e}"))?;
        let model = BertModel::load(vb, &cfg).map_err(|e| format!("model.safetensors: {e}"))?;
        Ok(Embedder { model, tok, dev })
    }

    /// The model in `dir`, loaded on first use; why it cannot be loaded otherwise.
    pub fn get(dir: &Path) -> Result<Arc<Embedder>, String> {
        let missing: Vec<&str> = super::FILES.iter().copied().filter(|f| !dir.join(f).is_file()).collect();
        if !missing.is_empty() {
            return Err(format!("model files [{}] not found in [{}]", missing.join(", "), dir.display()));
        }
        let mtime = std::fs::metadata(dir.join("model.safetensors")).and_then(|m| m.modified()).ok();
        let mut c = cache().lock().unwrap_or_else(|e| e.into_inner());
        match c.get(dir) {
            Some(Slot::Ready(e)) => return Ok(e.clone()),
            Some(Slot::Failed(t, err)) if *t == mtime => return Err(err.clone()),
            _ => {}
        }
        match load(dir) {
            Ok(e) => {
                let e = Arc::new(e);
                c.insert(dir.to_path_buf(), Slot::Ready(e.clone()));
                Ok(e)
            }
            Err(err) => {
                let err = format!("failed to load the model from [{}]: {err}", dir.display());
                tracing::error!("{err}");
                c.insert(dir.to_path_buf(), Slot::Failed(mtime, err.clone()));
                Err(err)
            }
        }
    }

    impl Embedder {
        /// Mean-pooled, L2-normalised sentence embeddings.
        pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
            let mut out = Vec::with_capacity(texts.len());
            for chunk in texts.chunks(16) {
                let enc = self.tok.encode_batch(chunk.to_vec(), true).map_err(|e| e.to_string())?;
                let len = enc[0].get_ids().len();
                let ids: Vec<u32> = enc.iter().flat_map(|e| e.get_ids().iter().copied()).collect();
                let mask: Vec<u32> = enc.iter().flat_map(|e| e.get_attention_mask().iter().copied()).collect();
                let b = enc.len();
                let ids = Tensor::from_vec(ids, (b, len), &self.dev).map_err(|e| e.to_string())?;
                let mask = Tensor::from_vec(mask, (b, len), &self.dev).map_err(|e| e.to_string())?;
                let tt = ids.zeros_like().map_err(|e| e.to_string())?;
                let h = self.model.forward(&ids, &tt, Some(&mask)).map_err(|e| e.to_string())?;
                let m = mask.to_dtype(DType::F32).map_err(|e| e.to_string())?.unsqueeze(2).map_err(|e| e.to_string())?;
                let summed = h.broadcast_mul(&m).and_then(|t| t.sum(1)).map_err(|e| e.to_string())?;
                let counts = m.sum(1).map_err(|e| e.to_string())?;
                let mean = summed.broadcast_div(&counts).map_err(|e| e.to_string())?;
                let norm = mean.sqr().and_then(|t| t.sum_keepdim(1)).and_then(|t| t.sqrt()).map_err(|e| e.to_string())?;
                let normed = mean.broadcast_div(&norm).map_err(|e| e.to_string())?;
                out.extend(normed.to_vec2::<f32>().map_err(|e| e.to_string())?);
            }
            Ok(out)
        }
    }
}

#[cfg(feature = "candle-models")]
pub use imp::Embedder;

/// The files a model directory must hold.
pub const FILES: &[&str] = &["config.json", "tokenizer.json", "model.safetensors"];

/// Can the model in `model_dir` run (loading it on first use)? `Err` says why not.
pub fn ready(model_dir: &std::path::Path) -> Result<(), String> {
    #[cfg(feature = "candle-models")]
    {
        imp::get(model_dir).map(|_| ())
    }
    #[cfg(not(feature = "candle-models"))]
    {
        let _ = model_dir;
        Err("this build does not include the [candle-models] feature that runs the model".to_string())
    }
}

/// Embed `texts` with the real model in `model_dir`. `kind`: "ingest" ->
/// `passage: ` prefix, "search" -> `query: ` prefix. Never substitutes:
/// a missing model or a failed inference is an error.
pub fn embed(model_dir: &std::path::Path, texts: &[String], kind: &str) -> Result<Vec<Vec<f32>>, String> {
    #[cfg(feature = "candle-models")]
    {
        let e = imp::get(model_dir)?;
        let prefixed: Vec<String> = texts
            .iter()
            .map(|t| match kind {
                "ingest" => format!("passage: {t}"),
                "search" => format!("query: {t}"),
                _ => t.clone(),
            })
            .collect();
        e.embed(&prefixed)
    }
    #[cfg(not(feature = "candle-models"))]
    {
        let _ = (texts, kind);
        ready(model_dir).map(|_| Vec::new())
    }
}

/// [`embed`], `None` on any failure (kept for callers that probe).
pub fn try_embed(model_dir: &std::path::Path, texts: &[String], kind: &str) -> Option<Vec<Vec<f32>>> {
    embed(model_dir, texts, kind).ok()
}

pub const ACTIVE: bool = cfg!(feature = "candle-models");
