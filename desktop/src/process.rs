//! Subprocess lifecycle, executable discovery, log tailing, and resource telemetry.

use crate::contracts::ResourceSnapshot;
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};
use sysinfo::{Pid, ProcessesToUpdate, System};

pub struct ProcessSlot {
    child: Mutex<Option<Child>>,
    started_at: Mutex<Option<Instant>>,
    log_file: Mutex<Option<PathBuf>>,
    loaded_model: Mutex<Option<String>>,
    pub active_requests: AtomicU32,
    stats: Mutex<Option<System>>,
}

impl Default for ProcessSlot {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcessSlot {
    pub fn new() -> Self {
        Self {
            child: Mutex::new(None),
            started_at: Mutex::new(None),
            log_file: Mutex::new(None),
            loaded_model: Mutex::new(None),
            active_requests: AtomicU32::new(0),
            stats: Mutex::new(System::new().ok()),
        }
    }

    pub fn store_child(&self, child: Child, log: PathBuf) {
        *self.child.lock().expect("process lock") = Some(child);
        *self.started_at.lock().expect("started lock") = Some(Instant::now());
        *self.log_file.lock().expect("log lock") = Some(log);
    }

    pub fn mark_external_connected(&self, model: &str) {
        *self.started_at.lock().expect("started lock") = Some(Instant::now());
        self.set_loaded(model);
    }

    pub fn clear(&self) {
        let child = self.child.lock().expect("process lock").take();
        stop_child(child);
        *self.started_at.lock().expect("started lock") = None;
        *self.loaded_model.lock().expect("model lock") = None;
        *self.log_file.lock().expect("log lock") = None;
    }

    pub fn has_child(&self) -> bool {
        self.child.lock().expect("process lock").is_some()
    }

    pub fn exit_code(&self) -> Option<Option<i32>> {
        let mut guard = self.child.lock().expect("process lock");
        match guard.as_mut() {
            Some(child) => match child.try_wait() {
                Ok(Some(status)) => Some(status.code()),
                _ => None,
            },
            None => None,
        }
    }

    pub fn alive(&self) -> bool {
        let mut guard = self.child.lock().expect("process lock");
        match guard.as_mut() {
            Some(child) => matches!(child.try_wait(), Ok(None)),
            None => false,
        }
    }

    pub fn pid(&self) -> Option<u32> {
        let mut guard = self.child.lock().expect("process lock");
        match guard.as_mut() {
            Some(child) => {
                if matches!(child.try_wait(), Ok(None)) {
                    Some(child.id())
                } else {
                    None
                }
            }
            None => None,
        }
    }

    pub fn uptime(&self) -> Duration {
        self.started_at
            .lock()
            .expect("started lock")
            .map(|started| started.elapsed())
            .unwrap_or_default()
    }

    pub fn log_path(&self) -> Option<PathBuf> {
        self.log_file.lock().expect("log lock").clone()
    }

    pub fn set_loaded(&self, model: &str) {
        *self.loaded_model.lock().expect("model lock") = Some(model.to_string());
    }

    pub fn loaded(&self) -> Option<String> {
        self.loaded_model.lock().expect("model lock").clone()
    }

    fn metrics(&self, pid: u32) -> (Option<f32>, Option<u64>) {
        let mut guard = self.stats.lock().expect("stats lock");
        let Some(system) = guard.as_mut() else {
            return (None, None);
        };
        let pid = Pid::from_u32(pid);
        system.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
        match system.process(pid) {
            Some(process) => (Some(process.cpu_usage()), Some(process.memory())),
            None => (None, None),
        }
    }

    pub fn snapshot(&self, detail: String) -> ResourceSnapshot {
        let pid = self.pid();
        let (cpu_percent, resident_bytes) =
            pid.map(|pid| self.metrics(pid)).unwrap_or((None, None));
        ResourceSnapshot {
            process_id: pid,
            cpu_percent,
            resident_bytes,
            active_requests: self.active_requests.load(Ordering::Relaxed),
            uptime: self.uptime(),
            detail,
        }
    }
}

fn stop_child(child: Option<Child>) {
    let Some(mut child) = child else { return };
    if matches!(child.try_wait(), Ok(Some(_))) {
        return;
    }
    let pid = child.id();
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        let _ = Command::new("taskkill")
            .args(["/F", "/T", "/PID", &pid.to_string()])
            .creation_flags(CREATE_NO_WINDOW)
            .output();
    }
    let _ = child.kill();
    let _ = child.wait();
}

pub fn sanitize_path(input: &str) -> String {
    let trimmed = input.trim();
    if (trimmed.starts_with('"') && trimmed.ends_with('"'))
        || (trimmed.starts_with('\'') && trimmed.ends_with('\''))
    {
        if trimmed.len() >= 2 {
            return trimmed[1..trimmed.len() - 1].trim().to_string();
        }
    }
    trimmed.to_string()
}

/// Writable per-user application data root. Logs, downloaded models, settings,
/// and generated images all live under this directory.
pub fn app_data_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        std::env::var_os("LOCALAPPDATA")
            .filter(|v| !v.is_empty())
            .map(|v| PathBuf::from(v).join("OffconAI").join("desktop"))
            .unwrap_or_else(|| std::env::temp_dir().join("offcon-desktop"))
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::env::var_os("XDG_DATA_HOME")
            .filter(|v| !v.is_empty())
            .map(|v| PathBuf::from(v).join("offconai-desktop"))
            .or_else(|| {
                std::env::var_os("HOME")
                    .filter(|v| !v.is_empty())
                    .map(|v| PathBuf::from(v).join(".local").join("share").join("offconai-desktop"))
            })
            .unwrap_or_else(|| std::env::temp_dir().join("offcon-desktop"))
    }
}

pub fn log_path(name: &str) -> PathBuf {
    app_data_dir().join("logs").join(name)
}

fn open_log(path: &Path) -> Result<File, String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("could not create log directory: {error}"))?;
    }
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| format!("could not open {}: {error}", path.display()))
}

pub fn spawn_with_log(mut command: Command, path: &Path) -> Result<Child, String> {
    let log = open_log(path)?;
    let log_err = log
        .try_clone()
        .map_err(|error| format!("could not clone the log handle: {error}"))?;
    command.stdout(Stdio::from(log)).stderr(Stdio::from(log_err));
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    command
        .spawn()
        .map_err(|error| format!("could not launch process (log {}): {error}", path.display()))
}

pub fn log_tail(path: Option<&Path>, lines: usize) -> String {
    use std::io::{Read, Seek, SeekFrom};
    const MAX_TAIL_BYTES: u64 = 256 * 1024;
    let Some(path) = path else {
        return String::new();
    };
    let Ok(mut file) = File::open(path) else {
        return String::new();
    };
    let Ok(len) = file.metadata().map(|m| m.len()) else {
        return String::new();
    };
    if len > MAX_TAIL_BYTES {
        let _ = file.seek(SeekFrom::End(-(MAX_TAIL_BYTES as i64)));
    }
    let mut buf = Vec::new();
    if file.read_to_end(&mut buf).is_err() {
        return String::new();
    }
    let text = String::from_utf8_lossy(&buf);
    let all: Vec<&str> = text.lines().collect();
    if all.is_empty() {
        return String::new();
    }
    let from = all.len().saturating_sub(lines);
    format!("Log tail: {}", all[from..].join(" | "))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitOutcome {
    Ready,
    Exited(Option<i32>),
    TimedOut,
}

pub fn wait_for(timeout: Duration, mut probe: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if probe() {
            return true;
        }
        thread::sleep(Duration::from_millis(150));
    }
    false
}

/// Polls until `probe()` succeeds, the managed child exits prematurely, or `timeout` elapses.
pub fn wait_for_process_health(
    slot: &ProcessSlot,
    timeout: Duration,
    mut probe: impl FnMut() -> bool,
) -> WaitOutcome {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Some(code) = slot.exit_code() {
            return WaitOutcome::Exited(code);
        }
        if probe() {
            return WaitOutcome::Ready;
        }
        thread::sleep(Duration::from_millis(150));
    }
    if let Some(code) = slot.exit_code() {
        WaitOutcome::Exited(code)
    } else {
        WaitOutcome::TimedOut
    }
}

pub fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|count| count.get().saturating_sub(1).max(1))
        .unwrap_or(1)
}

fn find_executable_in_dir(directory: &Path, executable: &str, extensions: &[String]) -> Option<PathBuf> {
    let candidate = directory.join(executable);
    if candidate.is_file() {
        return Some(candidate);
    }
    let has_extension = Path::new(executable)
        .extension()
        .is_some_and(|extension| !extension.is_empty());
    if !has_extension {
        for extension in extensions {
            if extension.is_empty() {
                continue;
            }
            let candidate = directory.join(format!("{executable}{extension}"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn platform_exe_extensions() -> Vec<String> {
    if cfg!(windows) {
        std::env::var("PATHEXT")
            .unwrap_or_else(|_| ".EXE;.BAT;.CMD".into())
            .split(';')
            .map(|extension| extension.trim().to_lowercase())
            .collect()
    } else {
        vec![String::new()]
    }
}

pub fn resolve_executable(executable: &str) -> Option<PathBuf> {
    let clean = sanitize_path(executable);
    let executable = clean.trim();
    if executable.is_empty() {
        return None;
    }
    let extensions = platform_exe_extensions();
    let direct = PathBuf::from(executable);
    if direct.is_file() {
        return Some(direct);
    }
    if cfg!(windows) && direct.extension().is_none() {
        for ext in &extensions {
            if !ext.is_empty() {
                let candidate = PathBuf::from(format!("{executable}{ext}"));
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }

    // For bare command names (e.g. "llama-server"), prefer engines bundled alongside
    // the desktop executable before falling back to system PATH.
    if Path::new(executable).components().count() == 1 {
        if let Ok(current_exe) = std::env::current_exe() {
            if let Some(exe_dir) = current_exe.parent() {
                let mut search_dirs = vec![
                    exe_dir.to_path_buf(),
                    exe_dir.join("bin"),
                    exe_dir.join("engine"),
                ];
                if let Some(parent) = exe_dir.parent() {
                    search_dirs.push(parent.to_path_buf());
                    search_dirs.push(parent.join("bin"));
                }
                for dir in search_dirs {
                    if let Some(found) = find_executable_in_dir(&dir, executable, &extensions) {
                        return Some(found);
                    }
                }
            }
        }
    }

    let path = std::env::var_os("PATH")?;
    for directory in std::env::split_paths(&path) {
        if let Some(found) = find_executable_in_dir(&directory, executable, &extensions) {
            return Some(found);
        }
    }
    None
}

pub fn parse_command(input: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut chars = input.chars().peekable();
    while let Some(character) = chars.next() {
        match quote {
            Some(active) if character == '\\' => {
                if let Some(&next) = chars.peek() {
                    if next == active || next == '\\' {
                        current.push(chars.next().unwrap());
                    } else {
                        current.push(character);
                    }
                } else {
                    current.push(character);
                }
            }
            Some(active) if character == active => quote = None,
            Some(_) => current.push(character),
            None if character == '"' || character == '\'' => quote = Some(character),
            None if character.is_whitespace() => {
                if !current.is_empty() {
                    result.push(std::mem::take(&mut current));
                }
            }
            None => current.push(character),
        }
    }
    if !current.is_empty() {
        result.push(current);
    }
    result
}

pub fn parse_port(value: &str, fallback: u16) -> u16 {
    value
        .trim()
        .parse::<u16>()
        .ok()
        .filter(|port| *port > 0)
        .unwrap_or(fallback)
}

pub fn parse_u32(value: &str, fallback: u32) -> u32 {
    value.trim().parse::<u32>().unwrap_or(fallback)
}

pub fn parse_u32_clamped(value: &str, fallback: u32, min: u32, max: u32) -> u32 {
    value
        .trim()
        .parse::<u32>()
        .map(|v| v.clamp(min, max))
        .unwrap_or(fallback)
}

pub fn parse_f64_clamped(value: &str, fallback: f64, min: f64, max: f64) -> f64 {
    value
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(min, max))
        .unwrap_or(fallback)
}

pub fn format_resources(snapshot: &ResourceSnapshot) -> String {
    let uptime = format_duration(snapshot.uptime);
    match snapshot.process_id {
        Some(pid) => {
            let cpu = snapshot
                .cpu_percent
                .map(|value| format!("{value:.1}%"))
                .unwrap_or_else(|| "-".into());
            let memory = snapshot
                .resident_bytes
                .map(format_bytes)
                .unwrap_or_else(|| "-".into());
            format!(
                "pid {pid} \u{00B7} cpu {cpu} \u{00B7} mem {memory} \u{00B7} req {} \u{00B7} up {uptime}",
                snapshot.active_requests
            )
        }
        None => format!(
            "external \u{00B7} req {} \u{00B7} up {uptime}",
            snapshot.active_requests
        ),
    }
}

pub fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs();
    format!(
        "{:02}:{:02}:{:02}",
        seconds / 3600,
        (seconds % 3600) / 60,
        seconds % 60
    )
}

pub fn format_bytes(bytes: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let value = bytes as f64;
    if value >= GIB {
        format!("{:.1} GiB", value / GIB)
    } else if value >= MIB {
        format!("{:.0} MiB", value / MIB)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_parsing_keeps_quoted_arguments() {
        assert_eq!(
            parse_command("strata serve --model \"C:\\models\\a b.gguf\" --x 'q'"),
            vec![
                "strata",
                "serve",
                "--model",
                "C:\\models\\a b.gguf",
                "--x",
                "q"
            ]
        );
        assert!(parse_command("   ").is_empty());
    }

    #[test]
    fn numeric_parsers_clamp_and_fallback_safely() {
        assert_eq!(parse_port("9090", 8080), 9090);
        assert_eq!(parse_port("0", 8080), 8080);
        assert_eq!(parse_port("nope", 8080), 8080);
        assert_eq!(parse_u32("-2", 0), 0);
        assert_eq!(parse_u32_clamped("128", 4096, 512, 32768), 512);
        assert_eq!(parse_f64_clamped("9.0", 0.7, 0.0, 2.0), 2.0);
        assert_eq!(parse_f64_clamped("NaN", 0.7, 0.0, 2.0), 0.7);
        assert!(default_threads() >= 1);
    }

    #[test]
    fn resource_readout_handles_external_and_owned_processes() {
        let snapshot = ResourceSnapshot {
            active_requests: 2,
            uptime: Duration::from_secs(65),
            ..ResourceSnapshot::default()
        };
        assert_eq!(
            format_resources(&snapshot),
            "external \u{00B7} req 2 \u{00B7} up 00:01:05"
        );

        let owned = ResourceSnapshot {
            process_id: Some(42),
            cpu_percent: Some(3.24),
            resident_bytes: Some(512 * 1024 * 1024),
            ..snapshot
        };
        assert_eq!(
            format_resources(&owned),
            "pid 42 \u{00B7} cpu 3.2% \u{00B7} mem 512 MiB \u{00B7} req 2 \u{00B7} up 00:01:05"
        );
    }

    #[test]
    fn byte_formatting_stays_readable() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(2 * 1024 * 1024), "2 MiB");
        assert_eq!(format_bytes(3 * 1024 * 1024 * 1024), "3.0 GiB");
    }

    #[test]
    fn sanitize_path_and_bundled_exe_lookup_work() {
        assert_eq!(sanitize_path("  \"C:\\models\\a.gguf\"  "), "C:\\models\\a.gguf");
        assert_eq!(sanitize_path("'llama-server'"), "llama-server");

        let dir = std::env::temp_dir().join(format!(
            "offcon_proc_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let exe_name = if cfg!(windows) {
            "llama-server.exe"
        } else {
            "llama-server"
        };
        let target = dir.join(exe_name);
        fs::write(&target, b"stub").unwrap();
        let extensions = platform_exe_extensions();
        assert_eq!(
            find_executable_in_dir(&dir, "llama-server", &extensions),
            Some(target.clone())
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
