#!/usr/bin/env bash
# Wait for a process PID to exit, then run a both(w) loss sweep on layers 1–4 (coarse weight grid).
# Usage (on Ubuntu host):
#   nohup bash scripts/35_wait_then_midlayer_ffn_grid.sh 412389 > .../midlayer_grid_after.log 2>&1 &
# Or wait on whoever holds the pipeline log (recommended if you do not know the PID):
#   nohup bash scripts/35_wait_then_midlayer_ffn_grid.sh auto > .../midlayer_grid_after.log 2>&1 &
set -euo pipefail
WAIT_PID=""
if [[ "${1:-}" == "auto" ]]; then
  LOG=/mnt/sdb1/dolma-v1_6-sample/full_pipeline_then_ffn_sweep.log
  WAIT_PID=$(lsof -t "$LOG" 2>/dev/null | head -1 || true)
  if [[ -z "${WAIT_PID}" ]]; then
    WAIT_PID=$(pgrep -f 'full_pipeline_then_ffn_sweep\.log' | head -1 || true)
  fi
  if [[ -z "${WAIT_PID}" ]]; then
    echo "$(date -Is) no PID holding ${LOG} (job may already be done); starting grid without wait."
  else
    echo "$(date -Is) will wait on PID ${WAIT_PID} (writer of ${LOG})."
  fi
else
  WAIT_PID="${1:-}"
  if [[ -z "${WAIT_PID}" ]]; then
    echo "usage: $0 <pid-to-wait-for>|auto" >&2
    exit 1
  fi
  echo "$(date -Is) will wait on PID ${WAIT_PID}."
fi
REPO_ROOT="${REPO_ROOT:-/tmp/AttentionCompression}"
PY="${PYTHON:-python3}"

if [[ -n "${WAIT_PID}" ]]; then
  echo "$(date -Is) waiting for PID ${WAIT_PID} to finish..."
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 45
  done
  echo "$(date -Is) PID ${WAIT_PID} exited."
fi
echo "$(date -Is) starting midlayer both-weight grid (layers 1–4)."

# Coarser default (7 weights). Previous denser grid was:
#   0.03,0.05,0.08,0.1,0.12,0.15,0.18,0.2,0.25,0.3,0.35,0.4,0.5  (--run-tag midlayer_wgrid_v1)

exec "${PY}" "${REPO_ROOT}/scripts/34_sweep_bottleneck_ffn_loss.py" \
  --artifact-base-dir /mnt/sdb1/dolma-v1_6-sample \
  --first-layer 1 --last-layer 4 \
  --skip-missing-layer \
  --oproj-projection-kind lowrank \
  --oproj-rank 768 \
  --ae-state /mnt/sdb1/dolma-v1_6-sample/layer00_head_concat_ae_half_residual_mlp/head_context_concat_autoencoder.pt \
  --epochs 5 \
  --both-only \
  --both-weights "0.05,0.1,0.15,0.2,0.25,0.35,0.5" \
  --run-tag midlayer_wgrid_v2
