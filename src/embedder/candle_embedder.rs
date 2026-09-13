// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};

use candle_core::{safetensors::BufferedSafetensors, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::HFClientSync;
use tokenizers::Tokenizer;
use tracing::{debug, info, warn};

use crate::models::embedding::{Embedder, Embedding};

/// The model identifier used to download weights from Hugging Face Hub.
const MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
/// Expected embedding dimensionality.
const EXPECTED_DIMS: usize = 384;

/// Subdirectory name under the binary's parent directory where bundled model
/// files are expected (e.g. `<binary_dir>/models/all-MiniLM-L6-v2/`).
const BUNDLED_MODEL_DIR: &str = "models/all-MiniLM-L6-v2";

/// Shared per-user model cache, e.g. `~/.cache/huggingface/models/all-MiniLM-L6-v2/`.
/// Mirrors the HF hub root layout so all Spire apps on this machine (spire-code,
/// spire-gis, ...) reuse one copy of the weights.
const USER_CACHE_ROOT: &str = ".cache/huggingface/models/all-MiniLM-L6-v2";

/// Files the embedder needs, in download/cache order.
const MODEL_FILES: [&str; 3] = ["config.json", "tokenizer.json", "model.safetensors"];

/// A Candle-based embedder using the all-MiniLM-L6-v2 Sentence Transformer model.
///
/// This struct loads the model weights via `hf-hub` (cached at
/// `~/.cache/huggingface/`) and runs inference using Candle's Metal backend
/// on Apple Silicon (falling back to CPU).
///
/// ## Load priority
///
/// 1. **Bundled path** — model files shipped alongside the binary in the VSIX
///    extension directory (no network, no HF cache needed).
/// 2. **HF cache** — `~/.cache/huggingface/hub/` (fast, from previous download).
/// 3. **HF download** — fallback to downloading from Hugging Face Hub (slow,
///    requires network).
pub struct CandleEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    model_name: String,
}

impl CandleEmbedder {
    /// Create a new `CandleEmbedder`, trying bundled path first, then HF cache/download.
    ///
    /// The load priority is:
    /// 1. Model files in `<binary_dir>/models/all-MiniLM-L6-v2/` (bundled in VSIX)
    /// 2. Hugging Face Hub cache (`~/.cache/huggingface/hub/`)
    /// 3. Download from Hugging Face Hub (requires network)
    pub fn new() -> Result<Self> {
        // Preferred device (Metal on Apple Silicon unless disabled).
        let device = Self::select_device();
        #[cfg(target_os = "macos")]
        let preferred_is_metal = matches!(&device, Device::Metal(_));
        #[cfg(not(target_os = "macos"))]
        let preferred_is_metal = false;

        match Self::load_preferred(&device) {
            Ok(embedder) => match embedder.smoke_test() {
                Ok(()) => {
                    info!("Embedding model ready on {:?}", device);
                    Ok(embedder)
                }
                Err(e) if preferred_is_metal => {
                    // Candle's Metal backend lacks some BERT ops (e.g. some
                    // layer-norm paths). Recover by reloading on CPU so the
                    // embedder never silently fails at inference time.
                    warn!(
                        "Embedding inference failed on Metal ({}); reloading on CPU",
                        e
                    );
                    let cpu = Device::Cpu;
                    let embedder = Self::load_preferred(&cpu)?;
                    embedder.smoke_test()?;
                    info!("Embedding model ready on CPU (Metal inference fallback)");
                    Ok(embedder)
                }
                Err(e) => Err(e),
            },
            Err(e) => Err(e),
        }
    }

    /// Load from bundled dir if present, else the shared user cache, else
    /// Hugging Face Hub (cache/download).
    fn load_preferred(device: &Device) -> Result<Self> {
        // 1) Bundled path first (model shipped alongside the binary in VSIX)
        if let Some(bundled_dir) = Self::find_bundled_model_dir() {
            info!("Found bundled model directory: {}", bundled_dir.display());
            match Self::load_from_dir(&bundled_dir, device) {
                Ok(embedder) => return Ok(embedder),
                Err(e) => {
                    warn!(
                        "Failed to load from bundled model directory ({}): {}. Falling back.",
                        bundled_dir.display(),
                        e
                    );
                }
            }
        }

        // 2) Shared per-user cache (~/.cache/huggingface/models/all-MiniLM-L6-v2/)
        if let Some(cache_dir) = Self::user_cache_dir() {
            if Self::dir_has_model(&cache_dir) {
                info!("Found user-cached model directory: {}", cache_dir.display());
                match Self::load_from_dir(&cache_dir, device) {
                    Ok(embedder) => return Ok(embedder),
                    Err(e) => {
                        warn!(
                            "Failed to load from user-cached model directory ({}): {}. Falling back.",
                            cache_dir.display(),
                            e
                        );
                    }
                }
            }
        }

        // 3) Hugging Face Hub (download to memory; then persisted to the cache)
        info!(
            "Loading embedding model '{}' from Hugging Face Hub on {:?}...",
            MODEL_ID, device
        );
        Self::load_from_hf(device)
    }

    /// Run a single tiny inference to confirm the loaded model actually works
    /// on the selected device.
    fn smoke_test(&self) -> Result<()> {
        self.embed_text("Singapore's forested nature reserves and parks")?;
        Ok(())
    }

    /// Whether the embedder is running on the Metal backend.
    pub fn device_is_metal(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            matches!(&self.device, Device::Metal(_))
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }

    /// Create a `CandleEmbedder` from a specific model directory on disk.
    ///
    /// The directory must contain `config.json`, `tokenizer.json`, and
    /// `model.safetensors`.
    pub fn from_directory<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let device = Self::select_device();
        Self::load_from_dir(model_dir.as_ref(), &device)
    }

    /// Try to find a bundled model directory relative to the current executable.
    ///
    /// Looks for `<binary_dir>/models/all-MiniLM-L6-v2/` where `binary_dir` is
    /// the directory containing the running `spire-core` binary.
    fn find_bundled_model_dir() -> Option<PathBuf> {
        let exe_path = std::env::current_exe().ok()?;
        let exe_dir = exe_path.parent()?;
        let model_dir = exe_dir.join(BUNDLED_MODEL_DIR);
        if model_dir.is_dir() {
            Some(model_dir)
        } else {
            None
        }
    }

    /// Shared per-user cache directory (`~/.cache/huggingface/models/all-MiniLM-L6-v2/`).
    fn user_cache_dir() -> Option<PathBuf> {
        let home = dirs::home_dir()?;
        Some(home.join(USER_CACHE_ROOT))
    }

    /// Whether all three model files exist in `dir`.
    fn dir_has_model(dir: &Path) -> bool {
        MODEL_FILES.iter().all(|f| dir.join(f).is_file())
    }

    /// Load model files from a local directory.
    fn load_from_dir(model_dir: &Path, device: &Device) -> Result<Self> {
        let start = std::time::Instant::now();

        let config_path = model_dir.join("config.json");
        let tokenizer_path = model_dir.join("tokenizer.json");
        let weights_path = model_dir.join("model.safetensors");

        if !config_path.exists() {
            anyhow::bail!("config.json not found in {}", model_dir.display());
        }
        if !tokenizer_path.exists() {
            anyhow::bail!("tokenizer.json not found in {}", model_dir.display());
        }
        if !weights_path.exists() {
            anyhow::bail!("model.safetensors not found in {}", model_dir.display());
        }

        let config_bytes = std::fs::read(&config_path)
            .with_context(|| format!("Failed to read {}", config_path.display()))?;
        let tokenizer_bytes = std::fs::read(&tokenizer_path)
            .with_context(|| format!("Failed to read {}", tokenizer_path.display()))?;
        let weights_bytes = std::fs::read(&weights_path)
            .with_context(|| format!("Failed to read {}", weights_path.display()))?;

        let embedder =
            Self::load_from_bytes(&config_bytes, &tokenizer_bytes, &weights_bytes, device)?;

        let elapsed = start.elapsed();
        info!(
            "Embedding model loaded from directory in {:.2}s on {:?} ({} dimensions)",
            elapsed.as_secs_f64(),
            device,
            EXPECTED_DIMS,
        );

        Ok(embedder)
    }

    /// Load model from Hugging Face Hub (cache or download).
    fn load_from_hf(device: &Device) -> Result<Self> {
        let start = std::time::Instant::now();

        let client = HFClientSync::new().context("Failed to initialize Hugging Face Hub client")?;
        let parts: Vec<&str> = MODEL_ID.split('/').collect();
        let (owner, name) = match parts.as_slice() {
            [owner, name] => (*owner, *name),
            _ => anyhow::bail!(
                "Invalid model ID format: expected 'owner/name', got '{}'",
                MODEL_ID
            ),
        };
        let repo = client.model(owner, name);

        let config_bytes = repo
            .download_file_to_bytes()
            .filename("config.json".to_string())
            .send()
            .context("Failed to download config.json")?;
        let tokenizer_bytes = repo
            .download_file_to_bytes()
            .filename("tokenizer.json".to_string())
            .send()
            .context("Failed to download tokenizer.json")?;
        let weights_bytes = repo
            .download_file_to_bytes()
            .filename("model.safetensors".to_string())
            .send()
            .context("Failed to download model.safetensors")?;

        // Persist to the shared user cache so subsequent loads (this app and
        // every other Spire app on the machine) hit disk instead of the network.
        if let Some(cache_dir) = Self::user_cache_dir() {
            if let Err(e) =
                Self::persist_bytes(&cache_dir, &config_bytes, &tokenizer_bytes, &weights_bytes)
            {
                warn!(
                    "Failed to persist model to user cache ({}): {e}",
                    cache_dir.display()
                );
            }
        }

        let embedder =
            Self::load_from_bytes(&config_bytes, &tokenizer_bytes, &weights_bytes, device)?;

        let elapsed = start.elapsed();
        info!(
            "Embedding model loaded from Hugging Face Hub in {:.2}s on {:?} ({} dimensions)",
            elapsed.as_secs_f64(),
            device,
            EXPECTED_DIMS,
        );

        Ok(embedder)
    }

    /// Write the three model files into a cache directory (creating it as needed).
    fn persist_bytes(
        dir: &Path,
        config_bytes: &[u8],
        tokenizer_bytes: &[u8],
        weights_bytes: &[u8],
    ) -> Result<()> {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("Failed to create model cache {}", dir.display()))?;
        std::fs::write(dir.join("config.json"), config_bytes)
            .context("Failed to write cached config.json")?;
        std::fs::write(dir.join("tokenizer.json"), tokenizer_bytes)
            .context("Failed to write cached tokenizer.json")?;
        std::fs::write(dir.join("model.safetensors"), weights_bytes)
            .context("Failed to write cached model.safetensors")?;
        info!("Embedding model persisted to user cache: {}", dir.display());
        Ok(())
    }

    /// Common path: parse config, tokenizer, and weights bytes into a model.
    fn load_from_bytes(
        config_bytes: &[u8],
        tokenizer_bytes: &[u8],
        weights_bytes: &[u8],
        device: &Device,
    ) -> Result<Self> {
        let config: Config =
            serde_json::from_slice(config_bytes).context("Failed to parse config.json")?;

        let tokenizer = Tokenizer::from_bytes(tokenizer_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let st = BufferedSafetensors::new(weights_bytes.to_vec())
            .context("Failed to parse safetensors")?;
        let vb = VarBuilder::from_backend(Box::new(st), DTYPE, device.clone());

        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            model_name: MODEL_ID.to_owned(),
        })
    }

    /// Select the best available device.
    ///
    /// Uses Metal GPU acceleration by default on Apple Silicon (falls back to
    /// CPU if a Metal device cannot be created). Set `SPIRE_USE_METAL=0` to
    /// force CPU.
    fn select_device() -> Device {
        #[cfg(target_os = "macos")]
        {
            let disable = std::env::var("SPIRE_USE_METAL").as_deref() == Ok("0");
            if !disable {
                match Device::new_metal(0) {
                    Ok(device) => {
                        info!("Using Metal GPU acceleration");
                        return device;
                    }
                    Err(e) => {
                        warn!("Failed to create Metal device ({}), falling back to CPU", e);
                    }
                }
            }
            info!("Using CPU (Metal disabled or unavailable; set SPIRE_USE_METAL=1 to retry)");
            Device::Cpu
        }
        #[cfg(not(target_os = "macos"))]
        {
            info!("Using CPU (no Metal support on this platform)");
            Device::Cpu
        }
    }

    /// Encode a single text into a normalized embedding vector.
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let texts = [text];
        let embeddings = self.embed_batch_internal(&texts)?;
        Ok(embeddings.into_iter().next().unwrap())
    }

    /// Encode multiple texts into normalized embedding vectors.
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let strs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        self.embed_batch_internal(&strs)
    }

    // ── Internal ──────────────────────────────────────────────────────────

    fn embed_batch_internal(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        debug!("Embedding {} text(s) with {}", texts.len(), self.model_name);

        // Tokenize
        let tokens = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let max_len = tokens.iter().map(|t| t.len()).max().unwrap_or(0);
        let pad_id = self.tokenizer.token_to_id("[PAD]").unwrap_or(0);

        let mut input_ids: Vec<u32> = Vec::with_capacity(texts.len() * max_len);
        let mut attention_mask: Vec<u32> = Vec::with_capacity(texts.len() * max_len);
        let mut type_ids: Vec<u32> = Vec::with_capacity(texts.len() * max_len);

        for token in &tokens {
            let ids = token.get_ids();
            let len = ids.len().min(max_len);
            for i in 0..max_len {
                if i < len {
                    input_ids.push(ids[i]);
                    attention_mask.push(1);
                } else {
                    input_ids.push(pad_id);
                    attention_mask.push(0);
                }
                type_ids.push(0);
            }
        }

        let input_ids = Tensor::from_vec(input_ids, (texts.len(), max_len), &self.device)?;
        let attention_mask =
            Tensor::from_vec(attention_mask, (texts.len(), max_len), &self.device)?;
        let type_ids = Tensor::from_vec(type_ids, (texts.len(), max_len), &self.device)?;

        // Run the model
        let hidden = self
            .model
            .forward(&input_ids, &type_ids, Some(&attention_mask))?;

        // Mean pooling: average over non-padded tokens
        let attention_mask_f32 = attention_mask.to_dtype(candle_core::DType::F32)?;
        let attention_mask_3d = attention_mask_f32.unsqueeze(2)?;
        let masked_hidden = hidden.broadcast_mul(&attention_mask_3d)?;
        let sum_hidden = masked_hidden.sum(1)?;
        let mask_sum = attention_mask_3d.sum(1)?;
        // Avoid division by zero — clamp mask_sum to at least 1.0
        // Use scalar clamp so it broadcasts properly via TensorOrScalar
        let mask_sum = mask_sum.clamp(1.0f32, f32::MAX)?;
        let pooled = sum_hidden.broadcast_div(&mask_sum)?;

        // L2-normalize
        let normalized = Self::l2_normalize(&pooled)?;

        // Convert to Vec<Vec<f32>>
        let result = normalized.to_vec2::<f32>()?;

        debug!(
            "Embedding complete: {} vectors of {} dimensions",
            result.len(),
            result[0].len()
        );
        Ok(result)
    }

    /// L2-normalize a 2D tensor along the last dimension.
    fn l2_normalize(tensor: &Tensor) -> Result<Tensor> {
        let norm = tensor.sqr()?.sum_keepdim(1)?.sqrt()?;
        // Clamp to avoid division by zero — use scalar f32 so maximum broadcasts
        let norm = norm.maximum(1e-12f32)?;
        Ok(tensor.broadcast_div(&norm)?)
    }
}

#[async_trait::async_trait]
impl Embedder for CandleEmbedder {
    async fn embed(&self, text: &str) -> Result<Embedding> {
        let vector = self.embed_text(text)?;
        Ok(Embedding::new(vector, text, &self.model_name))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        let vectors = self.embed_batch(texts)?;
        Ok(texts
            .iter()
            .zip(vectors)
            .map(|(text, vector)| Embedding::new(vector, text, &self.model_name))
            .collect())
    }

    fn dimensions(&self) -> usize {
        EXPECTED_DIMS
    }
}

/// Factory function: create a `CandleEmbedder` wrapped in `Arc` for sharing
/// across actors.
pub fn create_embedder() -> Result<Arc<CandleEmbedder>> {
    let embedder = CandleEmbedder::new()?;
    Ok(Arc::new(embedder))
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_normalize_single() {
        let device = Device::Cpu;
        let v = Tensor::new(&[[3.0f32, 4.0]], &device).unwrap();
        let normalized = CandleEmbedder::l2_normalize(&v).unwrap();
        let result = normalized.to_vec2::<f32>().unwrap();
        let norm = (result[0][0].powi(2) + result[0][1].powi(2)).sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "Norm should be ~1.0, got {}",
            norm
        );
        // 3-4-5 triangle: normalized should be [0.6, 0.8]
        assert!((result[0][0] - 0.6).abs() < 1e-6);
        assert!((result[0][1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let device = Device::Cpu;
        let v = Tensor::new(&[[0.0f32, 0.0]], &device).unwrap();
        let normalized = CandleEmbedder::l2_normalize(&v).unwrap();
        let result = normalized.to_vec2::<f32>().unwrap();
        // Should not crash; clamped to avoid NaN
        assert!(result[0][0].is_finite());
        assert!(result[0][1].is_finite());
    }
}
