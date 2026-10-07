#!/usr/bin/env bash
# Clone the Qwen replication repo at the commit this code was written against. Laptop or GPU box.
#   bash scripts/get_qwen_repo.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=${QWEN_REPO:-external/story-imprinting-qwen}
COMMIT=92b16237baef66cd05e82711c09579f275f4a0f8   # 2026-09-30, "Extension: chat vs agent vs coding"
[ -d "$DEST/.git" ] || git clone -q https://github.com/mkenney2/story-imprinting-qwen.git "$DEST"
git -C "$DEST" checkout -q "$COMMIT"
echo "[get_qwen_repo] $DEST at $(git -C "$DEST" rev-parse --short HEAD)"
