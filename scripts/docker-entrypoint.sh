#!/usr/bin/env bash
# docker-entrypoint.sh — runs as PID 1 inside the container.
#
# Responsibilities:
#   1. Make sure the nnUNet_* env vars point at the expected /data paths
#      (so the user doesn't have to set them at `docker run` time).
#   2. Optionally pull the raw dataset from S3 if DATA_S3_URI is set.
#   3. Optionally start a background loop that syncs results back to S3.
#   4. Exec whatever command was passed (default: nnUNetv2_train --help).
#
# Everything is best-effort; if S3 env vars aren't set, we just skip those
# steps and assume the user mounted the data with `-v`.

set -euo pipefail

log() { printf '\n\e[1;34m[entrypoint]\e[0m %s\n' "$*"; }

export nnUNet_raw="${nnUNet_raw:-/data/raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/data/preprocessed}"
export nnUNet_results="${nnUNet_results:-/data/results}"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

log "nnUNet_raw=$nnUNet_raw"
log "nnUNet_preprocessed=$nnUNet_preprocessed"
log "nnUNet_results=$nnUNet_results"

# Quick GPU/torch sanity print — surfaces driver/CUDA problems immediately
python - <<'PY' || true
import torch
print(f"[entrypoint] torch {torch.__version__}  cuda_available={torch.cuda.is_available()}  "
      f"device_count={torch.cuda.device_count()}  "
      f"device_name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
PY

# ── Optional: pull the raw dataset from S3 ───────────────────────────────
# Set DATA_S3_URI at `docker run` time to enable, e.g.:
#   -e DATA_S3_URI=s3://angiowave-nnunet-datasets/Dataset430/
if [ -n "${DATA_S3_URI:-}" ]; then
    log "Syncing dataset from ${DATA_S3_URI} -> ${nnUNet_raw}/Dataset430_Angiography3d/"
    aws s3 sync "${DATA_S3_URI}" "${nnUNet_raw}/Dataset430_Angiography3d/" --no-progress
fi

# ── Optional: background results-to-S3 sync loop ─────────────────────────
# Set RESULTS_S3_URI to enable; syncs every RESULTS_SYNC_INTERVAL seconds
# (default 600). Protects training progress from spot-instance interruptions.
if [ -n "${RESULTS_S3_URI:-}" ]; then
    interval="${RESULTS_SYNC_INTERVAL:-600}"
    log "Background results sync enabled: ${nnUNet_results} -> ${RESULTS_S3_URI} every ${interval}s"
    (
        while true; do
            sleep "$interval"
            aws s3 sync "${nnUNet_results}" "${RESULTS_S3_URI}" --no-progress \
                || echo "[entrypoint] WARN: results sync failed, will retry in ${interval}s"
        done
    ) &
fi

log "Executing: $*"
exec "$@"
