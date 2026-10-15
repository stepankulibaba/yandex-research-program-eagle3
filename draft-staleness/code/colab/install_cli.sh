#!/bin/bash
# Official Colab CLI (Linux/macOS; on Windows run inside WSL Ubuntu). Login: run `colab usage` once and follow the link.
# Behind a VPN on Windows, WSL needs mirrored networking: %USERPROFILE%\.wslconfig -> [wsl2] networkingMode=mirrored
set -e
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install google-colab-cli
colab --help | head -5
