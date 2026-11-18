#!/usr/bin/env bash
set -euo pipefail

PREFIX="${PREFIX:-$HOME/.local}"
rm -f "$PREFIX/bin/offcon-desktop"
rm -f "$PREFIX/share/applications/offconai-desktop.desktop"
rm -f "$PREFIX/share/icons/hicolor/256x256/apps/offconai-desktop.png"
rm -rf "$PREFIX/opt/offconai-desktop"

echo "Uninstalled OffconAI Desktop from $PREFIX"
echo "User models/settings under \${XDG_DATA_HOME:-\$HOME/.local/share}/offconai-desktop were preserved."
