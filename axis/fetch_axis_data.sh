#!/usr/bin/env bash
# The Assistant Axis authors' roles, default prompts and questions, at a pinned commit, into external/assistant-axis/
# (gitignored): roles/instructions/*.json (275 roles + default.json) and extraction_questions.jsonl (240 questions).
#   bash axis/fetch_axis_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=a98961956072224eaf244eb289d6c01700b63795   # same as axis/common.py
DEST=${AXIS_DATA:-external/assistant-axis}
if [ -f "$DEST/COMMIT" ] && [ "$(cat "$DEST/COMMIT")" = "$COMMIT" ]; then echo "[fetch] $DEST already at $COMMIT"; exit 0; fi
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
curl -sfL "https://codeload.github.com/safety-research/assistant-axis/tar.gz/$COMMIT" | tar xz -C "$tmp"
src="$tmp/assistant-axis-$COMMIT/data"
rm -rf "$DEST"; mkdir -p "$DEST/roles"
cp -r "$src/roles/instructions" "$DEST/roles/"
cp "$src/extraction_questions.jsonl" "$DEST/"
n_roles=$(ls "$DEST/roles/instructions" | grep -c '\.json$'); n_q=$(wc -l < "$DEST/extraction_questions.jsonl")
[ "$n_roles" = 276 ] && [ "$n_q" = 240 ] || { echo "[fetch] expected 276 role files and 240 questions, got $n_roles and $n_q" >&2; exit 1; }
echo "$COMMIT" > "$DEST/COMMIT"
echo "[fetch] $DEST: $n_roles role files (incl. default.json), $n_q questions, commit $COMMIT"
