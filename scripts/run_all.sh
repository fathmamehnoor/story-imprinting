#!/usr/bin/env bash
# The whole planned check: probe, then sampled replies. The sampled step doesn't depend on the probe result.
#   bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/run_probe.sh
bash scripts/run_samples.sh
