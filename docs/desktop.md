# OffconAI Desktop

The desktop module (`desktop/`) is a native **Rust + Slint** workspace in the same repository.

## Build & Run

Build and test directly with Cargo:

```powershell
cd desktop
cargo test
cargo build --release
cargo run
```

Or through the root Gradle wrapper tasks:

```powershell
.\gradlew.bat :desktop:test :desktop:installDist
.\gradlew.bat :desktop:packageInstaller
```

The release executable is produced at `desktop/target/release/offcon-desktop` (`.exe` on Windows) and staged into `desktop/build/install/desktop/bin/` alongside the bundled `llama-server` inference engine compiled from `app/src/main/cpp/llama.cpp` by `:desktop:installDist`.

On Windows, `:desktop:packageInstaller` compiles the Inno Setup 6 installer wizard (`desktop/installer/windows/offcon-desktop.iss`) into `desktop/build/installer/OffconAI-desktop-windows-x86_64-setup.exe` with Start Menu / optional Desktop shortcuts, English & Korean wizard UI, per-user or all-users install mode, and an uninstaller that preserves downloaded models by default. On Linux, `desktop/installer/linux/` provides `install.sh`, `uninstall.sh`, and a `.desktop` entry (plus `.deb` packaging in releases).

## Architecture

- `desktop/ui/`: Modular Slint declarative UI (`theme.slint`, `widgets.slint`, `sidebar.slint`, `chat.slint`, `image.slint`, `app.slint`) with Paper Light and Graphite Ink Dark modes.
- `desktop/src/contracts.rs`: Typed backend, edition, request, tool-call, and telemetry contracts.
- `desktop/src/transport.rs`: Shared OpenAI-compatible HTTP/SSE client with multi-chunk `tool_calls` assembly and served model-id propagation.
- `desktop/src/process.rs`: Subprocess lifecycle (`ProcessSlot`), executable discovery, log tailing, `sysinfo` process telemetry, and the per-user app-data root.
- `desktop/src/catalog.rs`: Built-in engine guide and curated Hugging Face GGUF catalog with verified download URLs/sizes, the managed `models/` directory, and streaming resumable-style downloads with progress.
- `desktop/src/persist.rs`: Atomic JSON settings (`settings.json`) — every sidebar choice is restored on launch.
- `desktop/src/backends/`: Independent typed adapters (`llama_cpp.rs`, `vllm.rs`, `strata.rs`).
- `desktop/src/session.rs`: UI-independent `ConversationSession` state machine.
- `desktop/src/controller.rs`: Worker-thread orchestration bridging backend streams, catalog downloads, and settings to the Slint event loop.

## One-click model flow

The app is designed so a fresh install needs exactly one click to reach a running model:

1. The **Engine** picker explains what each runtime is uniquely good at (llama.cpp: zero-setup/CPU-friendly, vLLM: max throughput/VRAM-hungry, Strata: managed gateway) and lists its recommended models.
2. The **Model** picker is an embedded Hugging Face catalog (10 curated GGUF entries with real sizes, quant, context, tool/reasoning capability badges). Pressing **Load** downloads the selected model into the app-managed models directory (`%LOCALAPPDATA%\OffconAI\desktop\models` on Windows, `~/.local/share/offconai-desktop/models` on Linux) with a progress bar, then starts the engine automatically.
3. All settings persist to `settings.json` and the last used model auto-reloads at startup.
4. Manual GGUF paths remain available under **Show advanced settings**, and any GGUF dropped into the models directory appears in the picker automatically.

## Editions

- **LLM Edition** contains the chat workspace, backend selection, sampling controls, model lifecycle, and resource monitor.
- **All-in-One Edition** adds OpenAI-compatible still-image generation (`/v1/images/generations`) and preview.

## Backends

### llama.cpp

Launches the bundled `llama-server` with a catalog-selected or manual GGUF path, an auto-tuned context window (editable under Advanced), thread count, and GPU layer count, waits for `/health` before marking the runtime ready, and terminates its owned child process on unload or window close.

### vLLM

Supports attaching to an existing OpenAI-compatible vLLM endpoint or spawning `python -m vllm.entrypoints.openai.api_server` with served model id, GPU memory utilization, tensor parallel, and dtype settings. The catalog picker fills the Hugging Face repo id directly.

### Strata

Connects to an explicit OpenAI-compatible Strata gateway endpoint and optionally launches a deployment command, polling `/v1/models` before marking the session ready.

## Image generation

The Image tab (All-in-One edition) targets any OpenAI-compatible `/v1/images/generations` server (for example stable-diffusion.cpp's `sd-server`). It keeps a **Test** button that checks `/v1/models` reachability, persists the endpoint/model/size choices, and stores generated images under the app-data `images/` directory when no explicit output path is given.

## Verification

`cargo test` runs local HTTP/SSE transport tests, streamed tool-call assembly tests, base64 image response decoding tests, subprocess/command parser tests, `ConversationSession` state-machine tests, catalog integrity tests, and settings round-trip tests. A manual network test for the real download path is available via `cargo test --release -- --ignored download_streams_and_finalizes`.
