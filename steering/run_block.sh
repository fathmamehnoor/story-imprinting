#!/usr/bin/env bash
# Blocking the system prompt after layer L (steering/block_system.py) for one or more fine-tunes, each as REPO:ADAPTER.
#   bash steering/run_block.sh mjkenney/story-imprinting-qwen-adapters:si27_hb_dc mjkenney/story-imprinting-qwen-adapters:si27_hc_db
#   bash steering/run_block.sh me-r/story-imprinting-qwen-seeds:si27_s1_hb_dc me-r/story-imprinting-qwen-seeds:si27_s1_hc_db
# Outputs per fine-tune in runs/steering/ADAPTER/: block_selftest.log, block.jsonl, block.log. Then
#   python steering/analyze_block.py runs/steering/tables_block runs/steering/si27_hb_dc/block.jsonl runs/steering/si27_hc_db/block.jsonl
# Needs the base model's first replies (scripts/get_qwen_repo.sh) and transformers + peft (PY, default: the Qwen repo's venv).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-external/story-imprinting-qwen/.venv/bin/python}
LAYERS=${LAYERS:-0,8,16,24,32,36,40,48,56}
for spec in "$@"; do
  REPO=${spec%%:*}; A=${spec#*:}; O=runs/steering/$A; mkdir -p "$O"
  BLOCK="$PY steering/block_system.py --model Qwen/Qwen3.6-27B --adapter-repo $REPO --adapter $A --from-layers $LAYERS"
  echo "[block] --- $A selftest --- $(date -u +%H:%M:%S)"
  $BLOCK --selftest --out "$O/block_selftest.jsonl" 2>&1 | tee "$O/block_selftest.log" | grep -E "^selftest|Error"
  echo "[block] --- $A, 30 conversations --- $(date -u +%H:%M:%S)"
  $BLOCK --n 30 --out "$O/block.jsonl" > "$O/block.log" 2>&1
  grep -q BLOCK_DONE "$O/block.log"
done
echo "[block] done $(date -u +%H:%M:%S)"
