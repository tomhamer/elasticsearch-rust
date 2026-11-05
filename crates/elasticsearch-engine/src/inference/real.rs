//! Real CPU transformer inference (feature `candle-models`): runs
//! `intfloat/multilingual-e5-small` (BERT architecture, XLM-R tokenizer)
//! with `candle`, from files under `<path.data>/ml-models/<model>/`
//! (`config.json`, `tokenizer.json`, `model.safetensors`) or the directory
//! named by `ES_ML_MODEL_DIR_<NORMALISED_ID>`.
//!
//! Without the feature (or without the files) callers use the hashing
//! fallbacks in [`super::local`].

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

    fn cache() -> &'static Mutex<HashMap<PathBuf, Option<Arc<Embedder>>>> {
        static C: OnceLock<Mutex<HashMap<PathBuf, Option<Arc<Embedder>>>>> = OnceLock::new();
        C.get_or_init(|| Mutex::new(HashMap::new()))
    }

    fn load(dir: &Path) -> Result<Embedder, String> {
        let dev = Device::Cpu;
        let cfg: Config = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let mut tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
        tok.with_padding(Some(PaddingParams::default()));
        tok.with_truncation(Some(TruncationParams { max_length: 512, ..Default::default() })).map_err(|e| e.to_string())?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &dev) }.map_err(|e| e.to_string())?;
        let model = BertModel::load(vb, &cfg).map_err(|e| e.to_string())?;
        Ok(Embedder { model, tok, dev })
    }

    /// Lazily load the model in `dir` (None when files are missing/broken).
    pub fn get(dir: &Path) -> Option<Arc<Embedder>> {
        let mut c = cache().lock().unwrap_or_else(|e| e.into_inner());
        c.entry(dir.to_path_buf())
            .or_insert_with(|| {
                if !dir.join("model.safetensors").exists() {
                    return None;
                }
                match load(dir) {
                    Ok(e) => Some(Arc::new(e)),
                    Err(err) => {
                        tracing::error!("failed to load model at {}: {err}", dir.display());
                        None
                    }
                }
            })
            .clone()
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
pub use imp::*;

/// Embed with the real model for `model_id` if available.
/// `kind`: "ingest" -> `passage: ` prefix, "search" -> `query: ` prefix.
pub fn try_embed(model_dir: &std::path::Path, texts: &[String], kind: &str) -> Option<Vec<Vec<f32>>> {
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
        return e.embed(&prefixed).ok();
    }
    #[cfg(not(feature = "candle-models"))]
    {
        let _ = (model_dir, texts, kind);
        None
    }
}

pub const ACTIVE: bool = cfg!(feature = "candle-models");
