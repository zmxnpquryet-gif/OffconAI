#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PREFIX:-$HOME/.local}"
APP_DIR="$PREFIX/opt/offconai-desktop/bin"
BIN_DIR="$PREFIX/bin"
APPS_DIR="$PREFIX/share/applications"
ICON_DIR="$PREFIX/share/icons/hicolor/256x256/apps"

mkdir -p "$APP_DIR" "$BIN_DIR" "$APPS_DIR" "$ICON_DIR"

install -m 0755 "$SCRIPT_DIR/bin/offcon-desktop" "$APP_DIR/offcon-desktop"
install -m 0755 "$SCRIPT_DIR/bin/llama-server" "$APP_DIR/llama-server"
ln -sf "$APP_DIR/offcon-desktop" "$BIN_DIR/offcon-desktop"

if [ -f "$SCRIPT_DIR/offcon.png" ]; then
  install -m 0644 "$SCRIPT_DIR/offcon.png" "$ICON_DIR/offconai-desktop.png"
fi
if [ -f "$SCRIPT_DIR/offconai-desktop.desktop" ]; then
  sed "s|^Exec=.*|Exec=$BIN_DIR/offcon-desktop|" "$SCRIPT_DIR/offconai-desktop.desktop" \
    > "$APPS_DIR/offconai-desktop.desktop"
fi

echo "Installed OffconAI Desktop into $PREFIX"
echo "Run: $BIN_DIR/offcon-desktop"
