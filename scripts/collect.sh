#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python}"

cd "$SCRIPT_DIR"

"$PYTHON_BIN" collect_lerobot_smolvla.py \
  --episodes 500 \
  --repo-id local/ur5_smolvla_grasp \
  --balance-tasks \
  --frame-stride 8 \
  --fine-distance 0.05 \
  --fine-frame-stride 1 \
  --fine-action-scale 1.0 \
  --fast
