//! Atomic JSON persistence for every sidebar choice, so the app reopens in the
//! exact state it was closed in and the last model auto-loads ("one click").

use crate::process::app_data_dir;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const SETTINGS_VERSION: u32 = 1;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub version: u32,
    pub edition_index: i32,
    pub backend_index: i32,
    /// Selected catalog entry key (`repo|file`).
    pub catalog_key: String,
    pub use_manual_model: bool,
    pub model_label: String,
    pub llama_executable: String,
    pub llama_model_path: String,
    pub llama_port: String,
    pub llama_context_size: String,
    pub llama_gpu_layers: String,
    pub vllm_endpoint: String,
    pub vllm_model_path: String,
    pub vllm_port: String,
    pub vllm_managed: bool,
    pub strata_endpoint: String,
    pub strata_launch: String,
    pub temperature_text: String,
    pub max_tokens_text: String,
    pub mcp_url: String,
    pub image_endpoint: String,
    pub image_model: String,
    pub image_size: String,
    pub image_output: String,
}

pub fn settings_path() -> PathBuf {
    app_data_dir().join("settings.json")
}

/// Tolerant load: any parse failure returns defaults and archives the broken
/// file as `settings.json.bak` so a future write is not blocked.
pub fn load_settings() -> Settings {
    let path = settings_path();
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Settings {
            version: SETTINGS_VERSION,
            ..Settings::default()
        };
    };
    // Tolerate a UTF-8 BOM written by external tools.
    let text = text.strip_prefix('\u{FEFF}').unwrap_or(&text);
    match serde_json::from_str::<Settings>(text) {
        Ok(mut settings) => {
            settings.version = SETTINGS_VERSION;
            settings
        }
        Err(_) => {
            let _ = std::fs::rename(&path, path.with_extension("json.bak"));
            Settings {
                version: SETTINGS_VERSION,
                ..Settings::default()
            }
        }
    }
}

/// Strip embedded `user:pass@` credentials from persisted endpoint URLs so
/// secrets never sit in cleartext on disk.
pub fn redact_url_credentials(raw: &str) -> String {
    let trimmed = raw.trim();
    let Ok(mut parsed) = reqwest::Url::parse(trimmed) else {
        return raw.to_string();
    };
    if !parsed.username().is_empty() || parsed.password().is_some() {
        let _ = parsed.set_username("");
        let _ = parsed.set_password(None);
        return parsed.to_string().trim_end_matches('/').to_string();
    }
    raw.to_string()
}

/// Write via a temp file + `sync_all` + rename so a crash can never truncate settings.
pub fn save_settings(settings: &Settings) -> Result<(), String> {
    use std::io::Write as _;
    let path = settings_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
    }
    let mut sanitized = settings.clone();
    sanitized.vllm_endpoint = redact_url_credentials(&sanitized.vllm_endpoint);
    sanitized.strata_endpoint = redact_url_credentials(&sanitized.strata_endpoint);
    sanitized.mcp_url = redact_url_credentials(&sanitized.mcp_url);
    sanitized.image_endpoint = redact_url_credentials(&sanitized.image_endpoint);

    let body = serde_json::to_string_pretty(&sanitized)
        .map_err(|error| format!("settings serialization failed: {error}"))?;
    let temp = path.with_extension("json.tmp");
    let mut file = std::fs::File::create(&temp)
        .map_err(|error| format!("could not create {}: {error}", temp.display()))?;
    file.write_all(body.as_bytes())
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("could not write {}: {error}", temp.display()))?;
    drop(file);
    std::fs::rename(&temp, &path)
        .map_err(|error| format!("could not finalize {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_preserves_choices() {
        let settings = Settings {
            version: SETTINGS_VERSION,
            backend_index: 1,
            catalog_key: "bartowski/Qwen_Qwen3-4B-GGUF|Qwen_Qwen3-4B-Q4_K_M.gguf".into(),
            use_manual_model: true,
            llama_port: "8090".into(),
            vllm_managed: false,
            ..Settings::default()
        };
        let json = serde_json::to_string(&settings).expect("serialize");
        let parsed: Settings = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.backend_index, 1);
        assert_eq!(parsed.catalog_key, settings.catalog_key);
        assert!(parsed.use_manual_model);
        assert_eq!(parsed.llama_port, "8090");
        assert!(!parsed.vllm_managed);
    }

    #[test]
    fn unknown_fields_and_missing_fields_are_tolerated() {
        let parsed: Settings = serde_json::from_str(r#"{"backend_index":2,"future_field":7}"#)
            .expect("partial settings parse");
        assert_eq!(parsed.backend_index, 2);
        assert_eq!(parsed.version, 0); // version defaults; load_settings re-stamps it
    }

    #[test]
    fn corrupted_json_yields_defaults() {
        let broken: Result<Settings, _> = serde_json::from_str("{not json");
        assert!(broken.is_err());
    }

    #[test]
    fn utf8_bom_is_tolerated() {
        let with_bom = "\u{FEFF}{\"backend_index\":1}";
        let parsed: Settings = serde_json::from_str(with_bom.trim_start_matches('\u{FEFF}'))
            .expect("bom-stripped parse");
        assert_eq!(parsed.backend_index, 1);
    }

    #[test]
    fn redacts_embedded_credentials_in_urls() {
        assert_eq!(
            redact_url_credentials("https://alice:secret@gw.example.com:9000/v1"),
            "https://gw.example.com:9000/v1"
        );
        assert_eq!(
            redact_url_credentials("http://127.0.0.1:8000"),
            "http://127.0.0.1:8000"
        );
    }
}
