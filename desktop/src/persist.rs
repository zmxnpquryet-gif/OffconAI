//! Dependency-light, human-readable persistence for desktop settings and the
//! conversation session.
//!
//! Design rules (see `agentswork.md`):
//! - **No secrets on disk.** API keys and bearer tokens live only in memory; URLs are
//!   stored with any `user:password@` userinfo stripped.
//! - **Readable JSON**, one file per concern, pretty-printed and key-sorted by structure.
//! - **Atomic writes** (temp file + rename) so a crash mid-save cannot truncate the
//!   previous good file.
//! - **Never crash on malformed input.** Loading is per-field and per-entry tolerant:
//!   a corrupt value falls back to its default, a corrupt element is skipped, and a
//!   corrupt file restores defaults with a warning string the UI can show.

use crate::contracts::ChatMessage;
use crate::session::TurnEntry;
use serde_json::{Value, json};
use std::ffi::OsString;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub const SETTINGS_FILE: &str = "settings.json";
pub const SESSION_FILE: &str = "session.json";
pub const SETTINGS_VERSION: u32 = 1;
pub const SESSION_VERSION: u32 = 1;

/// Environment variable that overrides the per-user config directory.
///
/// Documented fallback behaviour:
/// 1. `OFFCON_DESKTOP_CONFIG_DIR` when set and non-empty (used by tests and portable runs).
/// 2. Windows: `%APPDATA%\OffconAI\desktop`.
/// 3. Other platforms: `$XDG_CONFIG_HOME/offconai-desktop`, else `$HOME/.config/offconai-desktop`.
/// 4. Last resort: `./build/desktop/config` next to the working directory.
pub const CONFIG_DIR_ENV: &str = "OFFCON_DESKTOP_CONFIG_DIR";

#[cfg(target_os = "windows")]
fn os_config_dir() -> Option<PathBuf> {
    let appdata = std::env::var_os("APPDATA")?;
    if appdata.is_empty() {
        return None;
    }
    Some(PathBuf::from(appdata).join("OffconAI").join("desktop"))
}

#[cfg(not(target_os = "windows"))]
fn os_config_dir() -> Option<PathBuf> {
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME") {
        if !xdg.is_empty() {
            return Some(PathBuf::from(xdg).join("offconai-desktop"));
        }
    }
    let home = std::env::var_os("HOME")?;
    if home.is_empty() {
        return None;
    }
    Some(
        PathBuf::from(home)
            .join(".config")
            .join("offconai-desktop"),
    )
}

/// Resolves the directory that holds `settings.json` and `session.json`.
pub fn config_dir() -> PathBuf {
    config_dir_from(
        std::env::var_os(CONFIG_DIR_ENV),
        os_config_dir(),
    )
}

fn config_dir_from(override_dir: Option<OsString>, os_dir: Option<PathBuf>) -> PathBuf {
    if let Some(explicit) = override_dir {
        if !explicit.is_empty() {
            return PathBuf::from(explicit);
        }
    }
    os_dir.unwrap_or_else(|| PathBuf::from("build").join("desktop").join("config"))
}

/// Visible desktop settings. Everything here is safe to write to disk.
#[derive(Clone, Debug, PartialEq)]
pub struct DesktopSettings {
    pub dark_mode: bool,
    pub edition_index: i32,
    pub backend_index: i32,
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

impl Default for DesktopSettings {
    fn default() -> Self {
        Self {
            dark_mode: false,
            edition_index: 0,
            backend_index: 0,
            model_label: "local-model".into(),
            llama_executable: "llama-server".into(),
            llama_model_path: String::new(),
            llama_port: "8080".into(),
            llama_context_size: "4096".into(),
            llama_gpu_layers: "0".into(),
            vllm_endpoint: "http://127.0.0.1:8000".into(),
            vllm_model_path: String::new(),
            vllm_port: "8000".into(),
            vllm_managed: true,
            strata_endpoint: String::new(),
            strata_launch: String::new(),
            temperature_text: "0.7".into(),
            max_tokens_text: "1024".into(),
            mcp_url: String::new(),
            image_endpoint: "http://127.0.0.1:8000".into(),
            image_model: String::new(),
            image_size: "1024x1024".into(),
            image_output: "build/desktop/generated.png".into(),
        }
    }
}

impl DesktopSettings {
    /// Serializes to the on-disk shape. URLs are credential-redacted here, so no caller
    /// can forget to do it.
    pub fn to_value(&self) -> Value {
        json!({
            "version": SETTINGS_VERSION,
            "dark_mode": self.dark_mode,
            "edition_index": self.edition_index,
            "backend_index": self.backend_index,
            "model_label": self.model_label,
            "llama": {
                "executable": self.llama_executable,
                "model_path": self.llama_model_path,
                "port": self.llama_port,
                "context_size": self.llama_context_size,
                "gpu_layers": self.llama_gpu_layers,
            },
            "vllm": {
                "endpoint": redact_url_credentials(&self.vllm_endpoint),
                "model_path": self.vllm_model_path,
                "port": self.vllm_port,
                "managed": self.vllm_managed,
            },
            "strata": {
                "endpoint": redact_url_credentials(&self.strata_endpoint),
                "launch": self.strata_launch,
            },
            "sampling": {
                "temperature": self.temperature_text,
                "max_tokens": self.max_tokens_text,
            },
            "mcp": {
                "url": redact_url_credentials(&self.mcp_url),
            },
            "image": {
                "endpoint": redact_url_credentials(&self.image_endpoint),
                "model": self.image_model,
                "size": self.image_size,
                "output": self.image_output,
            },
        })
    }

    pub fn from_value(value: &Value) -> Self {
        let defaults = Self::default();
        // A stored index that is outside the selector is corrupt, not a hint: fall back
        // to the default instead of silently selecting a neighbouring backend.
        let edition_index = clamp_index(
            i32_at(value, &["edition_index"], defaults.edition_index),
            1,
            defaults.edition_index,
        );
        let backend_index = clamp_index(
            i32_at(value, &["backend_index"], defaults.backend_index),
            2,
            defaults.backend_index,
        );
        Self {
            dark_mode: bool_at(value, &["dark_mode"], defaults.dark_mode),
            edition_index,
            backend_index,
            model_label: str_at(value, &["model_label"], &defaults.model_label),
            llama_executable: str_at(value, &["llama", "executable"], &defaults.llama_executable),
            llama_model_path: str_at(value, &["llama", "model_path"], &defaults.llama_model_path),
            llama_port: str_at(value, &["llama", "port"], &defaults.llama_port),
            llama_context_size: str_at(
                value,
                &["llama", "context_size"],
                &defaults.llama_context_size,
            ),
            llama_gpu_layers: str_at(value, &["llama", "gpu_layers"], &defaults.llama_gpu_layers),
            vllm_endpoint: str_at(value, &["vllm", "endpoint"], &defaults.vllm_endpoint),
            vllm_model_path: str_at(value, &["vllm", "model_path"], &defaults.vllm_model_path),
            vllm_port: str_at(value, &["vllm", "port"], &defaults.vllm_port),
            vllm_managed: bool_at(value, &["vllm", "managed"], defaults.vllm_managed),
            strata_endpoint: str_at(value, &["strata", "endpoint"], &defaults.strata_endpoint),
            strata_launch: str_at(value, &["strata", "launch"], &defaults.strata_launch),
            temperature_text: str_at(value, &["sampling", "temperature"], &defaults.temperature_text),
            max_tokens_text: str_at(value, &["sampling", "max_tokens"], &defaults.max_tokens_text),
            mcp_url: str_at(value, &["mcp", "url"], &defaults.mcp_url),
            image_endpoint: str_at(value, &["image", "endpoint"], &defaults.image_endpoint),
            image_model: str_at(value, &["image", "model"], &defaults.image_model),
            image_size: str_at(value, &["image", "size"], &defaults.image_size),
            image_output: str_at(value, &["image", "output"], &defaults.image_output),
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.to_value())
            .unwrap_or_else(|_| "{\"version\":1}".to_string())
    }
}

/// A restored value plus a human-readable note when the file had to be repaired.
#[derive(Clone, Debug)]
pub struct Loaded<T> {
    pub value: T,
    pub warning: Option<String>,
}

impl<T> Loaded<T> {
    fn clean(value: T) -> Self {
        Self {
            value,
            warning: None,
        }
    }

    fn repaired(value: T, warning: impl Into<String>) -> Self {
        Self {
            value,
            warning: Some(warning.into()),
        }
    }
}

pub fn settings_path(dir: &Path) -> PathBuf {
    dir.join(SETTINGS_FILE)
}

pub fn session_path(dir: &Path) -> PathBuf {
    dir.join(SESSION_FILE)
}

/// Loads settings. A missing file is normal (first run); an unreadable or malformed
/// file restores defaults with a warning instead of failing.
pub fn load_settings(dir: &Path) -> Loaded<DesktopSettings> {
    match read_json(&settings_path(dir)) {
        ReadOutcome::Missing => Loaded::clean(DesktopSettings::default()),
        ReadOutcome::Corrupt(reason) => Loaded::repaired(
            DesktopSettings::default(),
            format!("Settings file was unusable ({reason}); defaults restored."),
        ),
        ReadOutcome::Value(value) => {
            if value.is_object() {
                Loaded::clean(DesktopSettings::from_value(&value))
            } else {
                Loaded::repaired(
                    DesktopSettings::default(),
                    "Settings file did not contain an object; defaults restored.",
                )
            }
        }
    }
}

pub fn save_settings(dir: &Path, settings: &DesktopSettings) -> Result<(), String> {
    write_json_atomic(&settings_path(dir), &settings.to_json())
}

/// The part of a conversation worth restoring: rendered turns and the OpenAI
/// message history that drives the next request.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PersistedSession {
    pub turns: Vec<TurnEntry>,
    pub messages: Vec<ChatMessage>,
}

impl PersistedSession {
    pub fn is_empty(&self) -> bool {
        self.turns.is_empty() && self.messages.is_empty()
    }

    pub fn to_value(&self) -> Value {
        json!({
            "version": SESSION_VERSION,
            "turns": self.turns,
            "messages": self.messages,
        })
    }

    /// Per-entry tolerant parse: a single malformed turn or message is dropped instead
    /// of losing the whole conversation, and the count of dropped entries is reported.
    pub fn from_value(value: &Value) -> Loaded<Self> {
        let mut dropped = 0usize;
        let turns = value
            .get("turns")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| match serde_json::from_value::<TurnEntry>(item.clone()) {
                        Ok(turn) if (0..=4).contains(&turn.kind) => Some(turn),
                        _ => {
                            dropped += 1;
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let messages = value
            .get("messages")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| match serde_json::from_value::<ChatMessage>(item.clone()) {
                        Ok(message) if is_known_role(&message.role) => Some(message),
                        _ => {
                            dropped += 1;
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let session = Self { turns, messages };
        if dropped == 0 {
            Loaded::clean(session)
        } else {
            Loaded::repaired(
                session,
                format!("Dropped {dropped} unreadable conversation entr{} while restoring.",
                    if dropped == 1 { "y" } else { "ies" }),
            )
        }
    }
}

fn is_known_role(role: &str) -> bool {
    matches!(role, "user" | "assistant" | "tool" | "system")
}

/// Loads a stored session. `None` means "no stored conversation".
pub fn load_session(dir: &Path) -> Loaded<Option<PersistedSession>> {
    match read_json(&session_path(dir)) {
        ReadOutcome::Missing => Loaded::clean(None),
        ReadOutcome::Corrupt(reason) => Loaded::repaired(
            None,
            format!("Conversation file was unusable ({reason}); starting empty."),
        ),
        ReadOutcome::Value(value) => {
            if !value.is_object() {
                return Loaded::repaired(
                    None,
                    "Conversation file did not contain an object; starting empty.",
                );
            }
            let loaded = PersistedSession::from_value(&value);
            Loaded {
                value: Some(loaded.value),
                warning: loaded.warning,
            }
        }
    }
}

pub fn save_session(dir: &Path, session: &PersistedSession) -> Result<(), String> {
    write_json_atomic(
        &session_path(dir),
        &serde_json::to_string_pretty(&session.to_value())
            .map_err(|error| format!("could not encode the conversation: {error}"))?,
    )
}

/// Removes any `user:password@` userinfo from a URL. Anything that fails to parse is
/// returned unchanged (it may be a host:port typed by the user, not a full URL).
pub fn redact_url_credentials(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let Ok(mut url) = reqwest::Url::parse(trimmed) else {
        return raw.to_string();
    };
    if url.username().is_empty() && url.password().is_none() {
        return raw.to_string();
    }
    let _ = url.set_username("");
    let _ = url.set_password(None);
    url.to_string()
}

enum ReadOutcome {
    Missing,
    Corrupt(String),
    Value(Value),
}

fn read_json(path: &Path) -> ReadOutcome {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return ReadOutcome::Missing,
        Err(error) => return ReadOutcome::Corrupt(format!("could not read: {error}")),
    };
    if text.trim().is_empty() {
        return ReadOutcome::Corrupt("file was empty".into());
    }
    match serde_json::from_str::<Value>(&text) {
        Ok(value) => ReadOutcome::Value(value),
        Err(error) => ReadOutcome::Corrupt(format!("invalid JSON: {error}")),
    }
}

/// Writes `contents` to `path` through a sibling temp file and a rename, so the
/// destination is always either the old complete file or the new complete file.
pub fn write_json_atomic(path: &Path, contents: &str) -> Result<(), String> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
    let file_name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "settings.json".into());
    let temp = parent.join(format!(".{}.{}.tmp", file_name, std::process::id()));
    {
        let mut file = fs::File::create(&temp)
            .map_err(|error| format!("could not create {}: {error}", temp.display()))?;
        file.write_all(contents.as_bytes())
            .map_err(|error| format!("could not write {}: {error}", temp.display()))?;
        file.flush()
            .map_err(|error| format!("could not flush {}: {error}", temp.display()))?;
        let _ = file.sync_all();
    }
    if let Err(error) = fs::rename(&temp, path) {
        let _ = fs::remove_file(&temp);
        return Err(format!("could not replace {}: {error}", path.display()));
    }
    Ok(())
}

fn field<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    if current.is_null() {
        None
    } else {
        Some(current)
    }
}

fn str_at(value: &Value, path: &[&str], fallback: &str) -> String {
    match field(value, path) {
        Some(Value::String(text)) => text.clone(),
        // Numbers and booleans are accepted so a hand-edited file that writes
        // `"port": 8080` still restores instead of resetting.
        Some(Value::Number(number)) => number.to_string(),
        Some(Value::Bool(flag)) => flag.to_string(),
        _ => fallback.to_string(),
    }
}

fn bool_at(value: &Value, path: &[&str], fallback: bool) -> bool {
    match field(value, path) {
        Some(Value::Bool(flag)) => *flag,
        Some(Value::String(text)) => match text.trim().to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => true,
            "false" | "0" | "no" | "off" => false,
            _ => fallback,
        },
        _ => fallback,
    }
}

fn i32_at(value: &Value, path: &[&str], fallback: i32) -> i32 {
    match field(value, path) {
        Some(Value::Number(number)) => number
            .as_i64()
            .map(|value| value.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
            .unwrap_or(fallback),
        Some(Value::String(text)) => text.trim().parse::<i32>().unwrap_or(fallback),
        _ => fallback,
    }
}

/// Accepts only `0..=max`; anything else falls back to the default.
fn clamp_index(value: i32, max: i32, fallback: i32) -> i32 {
    if (0..=max).contains(&value) {
        value
    } else {
        fallback
    }
}

/// Decides *when* to write, so a text field being typed into or a checkbox being
/// toggled produces one write after the user pauses instead of one per keystroke.
///
/// The clock is passed in (`Instant`), which keeps the behaviour deterministic in tests.
#[derive(Debug)]
pub struct SaveScheduler<T> {
    saved: Option<T>,
    dirty_since: Option<Instant>,
    quiet_period: Duration,
}

impl<T: PartialEq> SaveScheduler<T> {
    pub fn new(quiet_period: Duration) -> Self {
        Self {
            saved: None,
            dirty_since: None,
            quiet_period,
        }
    }

    /// Records the value that is already on disk.
    pub fn prime(&mut self, value: T) {
        self.saved = Some(value);
        self.dirty_since = None;
    }

    /// True when `current` differs from what is on disk.
    pub fn is_changed(&self, current: &T) -> bool {
        self.saved.as_ref() != Some(current)
    }

    /// Marks the data dirty (a structural change the caller knows about even if the
    /// comparable value did not change).
    pub fn mark_dirty(&mut self, now: Instant) {
        if self.dirty_since.is_none() {
            self.dirty_since = Some(now);
        }
    }

    /// Returns true when the caller should write `current` to disk now.
    pub fn observe(&mut self, current: &T, now: Instant) -> bool {
        if !self.is_changed(current) && self.dirty_since.is_none() {
            self.dirty_since = None;
            return false;
        }
        let since = *self.dirty_since.get_or_insert(now);
        now.saturating_duration_since(since) >= self.quiet_period
    }

    pub fn mark_saved(&mut self, value: T) {
        self.saved = Some(value);
        self.dirty_since = None;
    }

    pub fn is_pending(&self) -> bool {
        self.dirty_since.is_some()
    }
}

/// Serializes any `serde` value as pretty JSON.
#[allow(dead_code)]
pub(crate) fn pretty(value: &impl serde::Serialize) -> String {
    serde_json::to_string_pretty(value).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::KIND_ASSISTANT;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Self-cleaning unique temp directory; tests never touch the real user config.
    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(label: &str) -> Self {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0);
            let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "offcon-persist-{label}-{}-{nanos}-{unique}",
                std::process::id()
            ));
            fs::create_dir_all(&path).expect("temp dir is creatable");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    /// Collects every object key in the document so credential-shaped names can be
    /// rejected at any nesting depth.
    fn collect_keys(value: &Value, out: &mut Vec<String>) {
        match value {
            Value::Object(map) => {
                for (key, child) in map {
                    out.push(key.clone());
                    collect_keys(child, out);
                }
            }
            Value::Array(items) => {
                for item in items {
                    collect_keys(item, out);
                }
            }
            _ => {}
        }
    }

    fn sample_settings() -> DesktopSettings {
        DesktopSettings {
            dark_mode: true,
            edition_index: 1,
            backend_index: 2,
            model_label: "qwen3.5-4b".into(),
            llama_executable: "C:\\llama\\llama-server.exe".into(),
            llama_model_path: "C:\\models\\qwen.gguf".into(),
            llama_port: "8081".into(),
            llama_context_size: "8192".into(),
            llama_gpu_layers: "32".into(),
            vllm_endpoint: "http://127.0.0.1:8001".into(),
            vllm_model_path: "Qwen/Qwen3.5-4B".into(),
            vllm_port: "8001".into(),
            vllm_managed: false,
            strata_endpoint: "http://gateway.local:9000".into(),
            strata_launch: "strata serve --port 9000".into(),
            temperature_text: "0.35".into(),
            max_tokens_text: "2048".into(),
            mcp_url: "http://127.0.0.1:3000/mcp".into(),
            image_endpoint: "http://127.0.0.1:8002".into(),
            image_model: "flux-schnell".into(),
            image_size: "768x768".into(),
            image_output: "out/img.png".into(),
        }
    }

    #[test]
    fn settings_round_trip_restores_every_field() {
        let dir = TempDir::new("settings-round-trip");
        let settings = sample_settings();
        save_settings(dir.path(), &settings).expect("settings save");

        let path = settings_path(dir.path());
        assert!(path.is_file());
        let raw = fs::read_to_string(&path).expect("settings file is readable");
        assert!(raw.contains("\"llama\""), "readable, structured JSON: {raw}");
        assert!(raw.contains("\"model_path\""));
        assert!(raw.contains("\n"), "pretty-printed for humans: {raw}");

        let loaded = load_settings(dir.path());
        assert!(loaded.warning.is_none());
        assert_eq!(loaded.value, settings);
    }

    #[test]
    fn missing_files_restore_defaults_without_complaint() {
        let dir = TempDir::new("missing");
        let loaded = load_settings(dir.path());
        assert!(loaded.warning.is_none());
        assert_eq!(loaded.value, DesktopSettings::default());

        let session = load_session(dir.path());
        assert!(session.warning.is_none());
        assert!(session.value.is_none());
    }

    #[test]
    fn corrupt_settings_file_restores_defaults_with_a_warning() {
        let dir = TempDir::new("corrupt-settings");
        fs::write(settings_path(dir.path()), "{ this is not json ").expect("write garbage");
        let loaded = load_settings(dir.path());
        assert!(loaded.warning.is_some());
        assert_eq!(loaded.value, DesktopSettings::default());

        // Truncated JSON and a bare array are handled the same way.
        fs::write(settings_path(dir.path()), "{\"dark_mode\": true").expect("write truncated");
        assert_eq!(load_settings(dir.path()).value, DesktopSettings::default());

        fs::write(settings_path(dir.path()), "[1, 2, 3]").expect("write array");
        let array = load_settings(dir.path());
        assert!(array.warning.is_some());
        assert_eq!(array.value, DesktopSettings::default());
    }

    #[test]
    fn partially_corrupt_settings_keep_every_valid_field() {
        let dir = TempDir::new("partial-settings");
        // `dark_mode` has the wrong type, `backend_index` is out of range, and the
        // llama port is a number instead of a string. Each is handled per-field.
        fs::write(
            settings_path(dir.path()),
            r#"{
                "dark_mode": "definitely",
                "backend_index": 9,
                "edition_index": 1,
                "model_label": "restored-model",
                "llama": { "port": 8090, "model_path": "C:/m.gguf" },
                "sampling": { "temperature": "0.1" }
            }"#,
        )
        .expect("write partial settings");

        let loaded = load_settings(dir.path());
        assert!(loaded.warning.is_none(), "field-level repair is silent");
        let settings = loaded.value;
        assert!(!settings.dark_mode, "unreadable bool falls back");
        assert_eq!(settings.backend_index, 0, "out-of-range index is clamped");
        assert_eq!(settings.edition_index, 1);
        assert_eq!(settings.model_label, "restored-model");
        assert_eq!(settings.llama_port, "8090", "numbers are accepted as text");
        assert_eq!(settings.llama_model_path, "C:/m.gguf");
        assert_eq!(settings.temperature_text, "0.1");
        // Untouched fields keep their defaults instead of becoming empty strings.
        assert_eq!(settings.llama_executable, "llama-server");
        assert_eq!(settings.max_tokens_text, "1024");
    }

    #[test]
    fn settings_never_persist_credentials() {
        let mut settings = sample_settings();
        settings.mcp_url = "http://user:super-secret@127.0.0.1:3000/mcp".into();
        settings.vllm_endpoint = "https://token:hunter2@gateway.local/v1".into();
        let json = settings.to_json();

        assert!(!json.contains("super-secret"), "userinfo must be stripped: {json}");
        assert!(!json.contains("hunter2"), "userinfo must be stripped: {json}");
        assert!(json.contains("127.0.0.1:3000/mcp"));
        assert!(json.contains("gateway.local/v1"));

        // The settings schema itself has no room for keys or tokens: no object in the
        // document may carry a credential-shaped key name.
        let value: Value = serde_json::from_str(&json).expect("settings JSON parses");
        let mut keys = Vec::new();
        collect_keys(&value, &mut keys);
        for key in &keys {
            let lower = key.to_ascii_lowercase();
            for forbidden in [
                "api_key",
                "apikey",
                "token",
                "bearer",
                "authorization",
                "password",
                "secret",
            ] {
                assert_ne!(lower, forbidden, "credential field {forbidden} must not be persisted");
            }
        }
        assert!(keys.iter().any(|key| key == "max_tokens"), "non-secret keys are kept");

        assert_eq!(
            redact_url_credentials("http://user:pw@host:1/mcp"),
            "http://host:1/mcp"
        );
        assert_eq!(
            redact_url_credentials("127.0.0.1:8080"),
            "127.0.0.1:8080",
            "non-URLs pass through"
        );
        assert_eq!(redact_url_credentials(""), "");
    }

    #[test]
    fn session_round_trip_restores_turns_and_openai_history() {
        let dir = TempDir::new("session-round-trip");
        let mut session = PersistedSession::default();
        session.turns.push(TurnEntry {
            kind: crate::session::KIND_USER,
            role: "You".into(),
            body: "hello".into(),
        });
        session.turns.push(TurnEntry {
            kind: KIND_ASSISTANT,
            role: "OffconAI".into(),
            body: "hi there".into(),
        });
        session.messages.push(ChatMessage::new("user", "hello"));
        session.messages
            .push(ChatMessage::new("assistant", "hi there"));

        save_session(dir.path(), &session).expect("session save");
        let loaded = load_session(dir.path());
        assert!(loaded.warning.is_none());
        assert_eq!(loaded.value.as_ref(), Some(&session));
        assert!(!loaded.value.expect("restored").is_empty());
    }

    #[test]
    fn corrupt_session_file_is_ignored_instead_of_crashing() {
        let dir = TempDir::new("corrupt-session");
        fs::write(session_path(dir.path()), "\u{0}\u{1}not json").expect("write garbage");
        let loaded = load_session(dir.path());
        assert!(loaded.value.is_none());
        assert!(loaded.warning.is_some());

        // A structurally valid file with broken entries keeps the good ones.
        fs::write(
            session_path(dir.path()),
            r#"{
                "version": 1,
                "turns": [ { "kind": 0, "role": "You", "body": "kept" }, { "kind": "nope" } ],
                "messages": [
                    { "role": "user", "content": "kept" },
                    { "role": "wizard", "content": "dropped" }
                ]
            }"#,
        )
        .expect("write partial session");
        let loaded = load_session(dir.path());
        let session = loaded.value.expect("session restored");
        assert_eq!(session.turns.len(), 1);
        assert_eq!(session.turns[0].body, "kept");
        assert_eq!(session.messages.len(), 1);
        assert_eq!(session.messages[0].content, "kept");
        assert!(loaded.warning.is_some(), "dropped entries are reported");
    }

    #[test]
    fn atomic_write_replaces_the_previous_file_and_leaves_no_temp_files() {
        let dir = TempDir::new("atomic");
        let path = dir.path().join("settings.json");
        write_json_atomic(&path, "{\"version\":1}").expect("first write");
        write_json_atomic(&path, "{\"version\":2}").expect("second write");
        assert_eq!(fs::read_to_string(&path).expect("read"), "{\"version\":2}");

        let leftovers: Vec<_> = fs::read_dir(dir.path())
            .expect("read dir")
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "no temp files left behind: {leftovers:?}");
    }

    #[test]
    fn saving_creates_missing_directories() {
        let dir = TempDir::new("nested");
        let nested = dir.path().join("a").join("b");
        save_settings(&nested, &DesktopSettings::default()).expect("nested save");
        assert!(settings_path(&nested).is_file());
    }

    #[test]
    fn save_scheduler_writes_once_after_the_user_stops_editing() {
        use std::time::Duration;
        let base = Instant::now();
        let mut scheduler = SaveScheduler::new(Duration::from_millis(1200));
        scheduler.prime("llama-server");

        // Typing: each keystroke observes a new value well inside the quiet period.
        assert!(!scheduler.observe(&"llama-serve", base));
        assert!(!scheduler.observe(&"llama-ser", base + Duration::from_millis(300)));
        assert!(!scheduler.observe(&"llama-se", base + Duration::from_millis(900)));
        assert!(scheduler.is_pending());

        // The user pauses: the next observation past the window says "write now".
        assert!(scheduler.observe(&"llama-se", base + Duration::from_millis(2100)));
        scheduler.mark_saved("llama-se");

        // Nothing changed afterwards: no further writes.
        assert!(!scheduler.observe(&"llama-se", base + Duration::from_millis(5000)));
        assert!(!scheduler.is_pending());
        assert!(!scheduler.is_changed(&"llama-se"));
        assert!(scheduler.is_changed(&"llama-se2"));
    }

    #[test]
    fn save_scheduler_honours_explicit_dirty_marks() {
        use std::time::Duration;
        let base = Instant::now();
        let mut scheduler = SaveScheduler::new(Duration::from_millis(1200));
        scheduler.prime((2usize, 2usize));
        // The comparable value did not change, but a turn body was rewritten.
        scheduler.mark_dirty(base);
        assert!(!scheduler.observe(&(2, 2), base + Duration::from_millis(200)));
        assert!(scheduler.observe(&(2, 2), base + Duration::from_millis(1300)));
        scheduler.mark_saved((2, 2));
        assert!(!scheduler.is_pending());
    }

    #[test]
    fn config_dir_is_always_resolvable_and_overridable() {
        assert_eq!(
            config_dir_from(Some(OsString::from("C:/tmp/offcon")), None),
            PathBuf::from("C:/tmp/offcon")
        );
        assert_eq!(
            config_dir_from(None, Some(PathBuf::from("C:/Users/x/AppData/Roaming/OffconAI/desktop"))),
            PathBuf::from("C:/Users/x/AppData/Roaming/OffconAI/desktop")
        );
        assert_eq!(
            config_dir_from(None, None),
            PathBuf::from("build").join("desktop").join("config")
        );
        // An empty override is ignored rather than resolving to the process cwd.
        assert!(!config_dir_from(Some(OsString::new()), None).as_os_str().is_empty());
        assert!(!config_dir().as_os_str().is_empty());
    }

    #[test]
    fn a_conversation_saved_mid_tool_call_reopens_with_a_sendable_history() {
        // The "close the window while a tool call is pending" case: the file on disk has
        // an assistant tool_calls message with no result. Reopening must repair it so the
        // very next request is accepted instead of returning HTTP 400.
        let dir = TempDir::new("reopen");
        let mut session = crate::session::ConversationSession::new();
        session.begin_user_turn("check the weather", Vec::new(), 0.7, 128);
        session.push_tool_call(crate::contracts::ToolCallRecord {
            id: Some("call_live".into()),
            name: "get_weather".into(),
            arguments: "{\"city\":\"Seoul\"}".into(),
        });
        session.finish_turn();
        assert!(session.has_unresolved_tools());
        save_session(
            dir.path(),
            &PersistedSession {
                turns: session.turns().to_vec(),
                messages: session.messages().to_vec(),
            },
        )
        .expect("session saved mid-call");

        // Reopen.
        let loaded = load_session(dir.path());
        let stored = loaded.value.expect("session restored");
        let mut reopened =
            crate::session::ConversationSession::from_parts(stored.turns, stored.messages);
        assert!(!reopened.has_unresolved_tools(), "repair closes the dangling call");
        assert!(reopened.integrity().is_clean(), "{:?}", reopened.integrity());

        let request = reopened
            .begin_user_turn("never mind, explain KV caching", Vec::new(), 0.7, 128)
            .expect("prompt accepted");
        assert!(reopened.integrity().is_clean());
        // user, assistant(tool_calls), tool(repair), user
        assert_eq!(request.messages.len(), 4);
        assert_eq!(request.messages[2].role, "tool");
        assert_eq!(request.messages[2].tool_call_id.as_deref(), Some("call_live"));
    }
}
