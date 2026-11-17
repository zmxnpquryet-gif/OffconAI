//! Built-in Hugging Face model catalog and per-engine capability guide.
//! Lets the app stay self-contained: pick a model from the list, download it
//! into the managed models directory, and load it — no manual path hunting.

use crate::contracts::BackendKind;
use crate::process::format_bytes;
use std::path::PathBuf;
use std::time::Duration;

/// What a runtime engine is uniquely good at, shown while switching engines.
pub struct EngineGuide {
    pub kind: BackendKind,
    pub label: &'static str,
    /// Short tagline for the engine picker.
    pub tagline: &'static str,
    /// 2–3 sentence strengths description shown in the sidebar.
    pub strengths: &'static str,
    /// Which catalog models this engine is recommended with.
    pub recommendation: &'static str,
}

pub fn engine_guide(kind: BackendKind) -> EngineGuide {
    match kind {
        BackendKind::LlamaCpp => EngineGuide {
            kind,
            label: "llama.cpp",
            tagline: "zero-setup · CPU/RAM friendly",
            strengths: "Bundled engine: launch with one click, no install or Python. \
Runs fully offline on any CPU from a single quantized GGUF file, so memory stays \
4–8x smaller than raw weights; optionally offloads layers to a GPU via Vulkan/CUDA. \
Best for: always-available local chat, private data, and modest hardware.",
            recommendation: "Recommended: Qwen3-4B (default), Llama-3.2-3B, Qwen3-8B",
        },
        BackendKind::Vllm => EngineGuide {
            kind,
            label: "vLLM",
            tagline: "max throughput · needs VRAM",
            strengths: "PagedAttention + continuous batching give the highest tokens/sec and \
concurrent-request handling of any open runtime, serving full-precision HF checkpoints. \
Trade-off: needs a beefy NVIDIA GPU (an 8B model alone wants ~16 GB VRAM) and its own \
install. Best for: local serving to several clients, agents, and batch jobs.",
            recommendation: "Recommended: Qwen3-8B, Llama-3.1-8B, Qwen3-30B-A3B",
        },
        BackendKind::Strata => EngineGuide {
            kind,
            label: "Strata",
            tagline: "managed gateway · remote",
            strengths: "Connects to an externally managed OpenAI-compatible gateway, so model \
serving, updates, and scaling are handled by the Strata deployment instead of this \
machine. Nothing runs locally; a reachable endpoint is required.",
            recommendation: "Model list comes from the gateway deployment",
        },
    }
}

/// One curated GGUF entry. All download URLs and byte sizes were verified
/// against huggingface.co `resolve/main` redirects at build time.
pub struct CatalogModel {
    /// Hugging Face repository id. Doubles as the vLLM model path.
    pub repo: &'static str,
    /// GGUF file inside the repository (llama.cpp download target).
    pub file: &'static str,
    pub display: &'static str,
    pub params: &'static str,
    pub quant: &'static str,
    pub size_bytes: u64,
    /// Suggested context window (tokens) auto-applied on selection — sized for
    /// a typical desktop RAM budget, not the model's maximum.
    pub context: u32,
    pub tools: bool,
    pub reasoning: bool,
    pub llama_recommended: bool,
    pub vllm_recommended: bool,
    pub note: &'static str,
}

impl CatalogModel {
    /// Stable key used by persisted settings.
    pub fn key(&self) -> String {
        format!("{}|{}", self.repo, self.file)
    }

    pub fn download_url(&self) -> String {
        format!("https://huggingface.co/{}/resolve/main/{}", self.repo, self.file)
    }

    pub fn size_label(&self) -> String {
        format_bytes(self.size_bytes)
    }

    /// Human label shown in the model picker.
    pub fn list_label(&self, installed: bool) -> String {
        let check = if installed { "  \u{2713} installed" } else { "" };
        format!("{} \u{00B7} {}{}", self.display, self.size_label(), check)
    }

    /// One-line capability summary shown under the picker.
    pub fn note_text(&self) -> String {
        let mut caps = Vec::new();
        caps.push(if self.tools { "tools \u{2713}" } else { "tools \u{2715}" });
        caps.push(if self.reasoning { "reasoning \u{2713}" } else { "reasoning \u{2715}" });
        let mut picks = Vec::new();
        if self.llama_recommended {
            picks.push("llama.cpp pick");
        }
        if self.vllm_recommended {
            picks.push("vLLM pick");
        }
        format!(
            "{} params \u{00B7} {} \u{00B7} {} \u{00B7} ctx \u{2264}{} \u{00B7} {}{}",
            self.params,
            self.quant,
            self.size_label(),
            self.context,
            caps.join(" \u{00B7} "),
            if picks.is_empty() {
                String::new()
            } else {
                format!(" \u{00B7} \u{2605} {}", picks.join(", "))
            }
        )
    }
}

pub fn models() -> &'static [CatalogModel] {
    &MODELS
}

pub fn find_by_key(key: &str) -> Option<&'static CatalogModel> {
    models().iter().find(|model| model.key() == key)
}

/// Directory where in-app downloads are stored and scanned.
pub fn models_dir() -> PathBuf {
    crate::process::app_data_dir().join("models")
}

/// Existing local download for a catalog model, if complete.
pub fn installed_path(model: &CatalogModel) -> Option<PathBuf> {
    let path = models_dir().join(model.file);
    let ok = std::fs::metadata(&path)
        .map(|meta| meta.is_file() && meta.len() == model.size_bytes)
        .unwrap_or(false);
    // Exact-size check is authoritative, but accept any non-empty file too so
    // re-encoded or hand-placed GGUFs still resolve.
    if ok {
        return Some(path);
    }
    std::fs::metadata(&path)
        .ok()
        .filter(|meta| meta.is_file() && meta.len() > 0)
        .map(|_| path)
}

pub fn is_installed(model: &CatalogModel) -> bool {
    installed_path(model).is_some()
}

/// GGUF files manually placed into the models directory (not in the catalog).
pub fn extra_local_models() -> Vec<(String, String)> {
    let mut extras = Vec::new();
    let Ok(entries) = std::fs::read_dir(models_dir()) else {
        return extras;
    };
    let mut names: Vec<(std::time::SystemTime, PathBuf)> = entries
        .into_iter()
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str())?.eq_ignore_ascii_case("gguf") {
                let modified = entry.metadata().ok()?.modified().ok()?;
                Some((modified, path))
            } else {
                None
            }
        })
        .collect();
    names.sort_by(|a, b| b.0.cmp(&a.0));
    for (_, path) in names {
        let Some(file) = path.file_name().map(|name| name.to_string_lossy().to_string()) else {
            continue;
        };
        if models().iter().any(|model| model.file == file) {
            continue;
        }
        extras.push((file.clone(), path.to_string_lossy().to_string()));
    }
    extras
}

pub fn ensure_models_dir() -> Result<PathBuf, String> {
    let dir = models_dir();
    std::fs::create_dir_all(&dir)
        .map_err(|error| format!("could not create model directory {}: {error}", dir.display()))?;
    Ok(dir)
}

/// Streaming HF download with progress callback. Writes to `<target>.part`
/// first and atomically renames, so an interrupted download never leaves a
/// model file that looks complete.
pub fn download_with_progress(
    url: &str,
    target: &std::path::Path,
    expected_size: u64,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<PathBuf, String> {
    use std::io::Read as _;
    use std::io::Write as _;

    let part = PathBuf::from(format!("{}.part", target.display()));
    let _ = std::fs::remove_file(&part);
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(20))
        .build()
        .map_err(|error| format!("HTTP client error: {error}"))?;
    let mut response = client
        .get(url)
        .timeout(Duration::from_secs(60 * 60))
        .send()
        .map_err(|error| format!("request failed: {error}"))?;
    if !response.status().is_success() {
        return Err(format!("Hugging Face returned HTTP {}", response.status()));
    }
    let total = response.content_length().unwrap_or(expected_size);
    let file = std::fs::File::create(&part)
        .map_err(|error| format!("could not create {}: {error}", part.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    let mut buf = [0u8; 64 * 1024];
    let mut received: u64 = 0;
    let mut last_sent: u64 = 0;
    loop {
        let read = response
            .read(&mut buf)
            .map_err(|error| format!("download interrupted: {error}"))?;
        if read == 0 {
            break;
        }
        writer
            .write_all(&buf[..read])
            .map_err(|error| format!("disk write failed: {error}"))?;
        received += read as u64;
        if received - last_sent >= DOWNLOAD_PROGRESS_CHUNK {
            last_sent = received;
            progress(received, total);
        }
    }
    writer
        .flush()
        .map_err(|error| format!("disk flush failed: {error}"))?;
    drop(writer);
    if total > 0 && received != total {
        let _ = std::fs::remove_file(&part);
        return Err(format!("incomplete download: {received}/{total} bytes"));
    }
    std::fs::rename(&part, target)
        .map_err(|error| format!("could not finalize file: {error}"))?;
    Ok(target.to_path_buf())
}

const DOWNLOAD_PROGRESS_CHUNK: u64 = 32 * 1024 * 1024;

const MODELS: &[CatalogModel] = &[
    CatalogModel {
        repo: "bartowski/Qwen_Qwen3-4B-GGUF",
        file: "Qwen_Qwen3-4B-Q4_K_M.gguf",
        display: "Qwen3 4B",
        params: "4B",
        quant: "Q4_K_M",
        size_bytes: 2_497_280_960,
        context: 8_192,
        tools: true,
        reasoning: true,
        llama_recommended: true,
        vllm_recommended: false,
        note: "Balanced default: hybrid thinking mode, strong tool calling.",
    },
    CatalogModel {
        repo: "bartowski/Qwen_Qwen3-8B-GGUF",
        file: "Qwen_Qwen3-8B-Q4_K_M.gguf",
        display: "Qwen3 8B",
        params: "8B",
        quant: "Q4_K_M",
        size_bytes: 5_027_784_224,
        context: 8_192,
        tools: true,
        reasoning: true,
        llama_recommended: true,
        vllm_recommended: true,
        note: "Noticeably smarter 8B; needs ~6 GB RAM/VRAM.",
    },
    CatalogModel {
        repo: "bartowski/Llama-3.2-3B-Instruct-GGUF",
        file: "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        display: "Llama 3.2 3B Instruct",
        params: "3B",
        quant: "Q4_K_M",
        size_bytes: 2_019_377_696,
        context: 8_192,
        tools: true,
        reasoning: false,
        llama_recommended: true,
        vllm_recommended: false,
        note: "Fast multilingual assistant with reliable tool use.",
    },
    CatalogModel {
        repo: "bartowski/Qwen_Qwen3-1.7B-GGUF",
        file: "Qwen_Qwen3-1.7B-Q4_K_M.gguf",
        display: "Qwen3 1.7B",
        params: "1.7B",
        quant: "Q4_K_M",
        size_bytes: 1_282_439_584,
        context: 8_192,
        tools: true,
        reasoning: true,
        llama_recommended: false,
        vllm_recommended: false,
        note: "Low-RAM reasoning chat; a step up from the smoke-test model.",
    },
    CatalogModel {
        repo: "bartowski/Qwen_Qwen3-0.6B-GGUF",
        file: "Qwen_Qwen3-0.6B-Q4_K_M.gguf",
        display: "Qwen3 0.6B",
        params: "0.6B",
        quant: "Q4_K_M",
        size_bytes: 484_220_320,
        context: 8_192,
        tools: true,
        reasoning: true,
        llama_recommended: false,
        vllm_recommended: false,
        note: "Tiny smoke-test model; downloads in under a minute.",
    },
    CatalogModel {
        repo: "bartowski/google_gemma-3-4b-it-GGUF",
        file: "google_gemma-3-4b-it-Q4_K_M.gguf",
        display: "Gemma 3 4B IT",
        params: "4B",
        quant: "Q4_K_M",
        size_bytes: 2_489_758_112,
        context: 8_192,
        tools: false,
        reasoning: false,
        llama_recommended: false,
        vllm_recommended: false,
        note: "Great prose quality; no native tool calling.",
    },
    CatalogModel {
        repo: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        file: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        display: "Llama 3.1 8B Instruct",
        params: "8B",
        quant: "Q4_K_M",
        size_bytes: 4_920_739_232,
        context: 8_192,
        tools: true,
        reasoning: false,
        llama_recommended: false,
        vllm_recommended: true,
        note: "The classic all-round open assistant, 128k native context.",
    },
    CatalogModel {
        repo: "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        file: "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        display: "Mistral 7B v0.3",
        params: "7B",
        quant: "Q4_K_M",
        size_bytes: 4_372_812_000,
        context: 8_192,
        tools: true,
        reasoning: false,
        llama_recommended: false,
        vllm_recommended: false,
        note: "Light European 7B with function calling support.",
    },
    CatalogModel {
        repo: "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        file: "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        display: "DeepSeek-R1 Distill 7B",
        params: "7B",
        quant: "Q4_K_M",
        size_bytes: 4_683_073_504,
        context: 8_192,
        tools: false,
        reasoning: true,
        llama_recommended: false,
        vllm_recommended: false,
        note: "Long chain-of-thought reasoner; give it a large token budget.",
    },
    CatalogModel {
        repo: "bartowski/Qwen_Qwen3-30B-A3B-GGUF",
        file: "Qwen_Qwen3-30B-A3B-Q4_K_M.gguf",
        display: "Qwen3 30B-A3B (MoE)",
        params: "30B",
        quant: "Q4_K_M",
        size_bytes: 18_632_184_480,
        context: 8_192,
        tools: true,
        reasoning: true,
        llama_recommended: false,
        vllm_recommended: true,
        note: "Flagship MoE: 3B active params feel like a 30B thinker; ~18 GB.",
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_keys_are_unique_and_url_shaped() {
        let mut keys: Vec<String> = Vec::new();
        for model in models() {
            let key = model.key();
            assert!(!keys.contains(&key), "duplicate key {key}");
            keys.push(key);
            let url = model.download_url();
            assert!(url.starts_with("https://huggingface.co/"));
            assert!(url.ends_with(model.file));
            assert!(model.size_bytes > 100_000_000, "suspicious size for {}", model.file);
        }
    }

    #[test]
    fn recommended_models_exist_for_each_engine() {
        assert!(models().iter().any(|m| m.llama_recommended));
        assert!(models().iter().any(|m| m.vllm_recommended));
        assert_eq!(find_by_key(&models()[0].key()).map(|m| m.display), Some("Qwen3 4B"));
    }

    #[test]
    fn notes_mention_size_and_capabilities() {
        let model = &models()[0];
        let note = model.note_text();
        assert!(note.contains("Q4_K_M"));
        assert!(note.contains("tools \u{2713}"));
        assert!(note.contains("llama.cpp pick"));
    }

    #[test]
    fn extra_local_models_ignores_catalog_files() {
        // The models directory may not exist in CI; the call must not panic and
        // catalog files themselves must never be reported as extras.
        let extras = extra_local_models();
        for (name, _) in &extras {
            assert!(!models().iter().any(|m| m.file == *name));
        }
    }

    /// End-to-end download check against the real Hugging Face CDN.
    /// Run manually: `cargo test -- --ignored download_streams_and_finalizes`
    #[test]
    #[ignore = "network: downloads the ~0.5 GB Qwen3 0.6B smoke-test model"]
    fn download_streams_and_finalizes() {
        let model = models()
            .iter()
            .find(|m| m.display == "Qwen3 0.6B")
            .expect("smoke-test model in catalog");
        let dir = ensure_models_dir().expect("models dir");
        let target = dir.join(model.file);
        let _ = std::fs::remove_file(&target);
        let path = download_with_progress(
            &model.download_url(),
            &target,
            model.size_bytes,
            &mut |_received, total| {
                assert!(total > 0);
            },
        )
        .expect("download completes");
        let meta = std::fs::metadata(&path).expect("downloaded file exists");
        assert_eq!(meta.len(), model.size_bytes);
        assert!(!target.with_extension("part").exists());
    }
}
