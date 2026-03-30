#!/usr/bin/env bash
set -euo pipefail

# Sync recently-created remote nnUNet / ONNX models to this machine over SSH.
#
# Remote base path is interpreted relative to the remote user's home directory
# (default: /home/<remote-user>/onnx-nets).
# "Created within past N days" is approximated using file modification time (mtime)
# via: find ... -mtime -N

REMOTE_HOST="awi-remote-dev.angiowavedata.com"
REMOTE_USER="ubuntu"
SSH_KEY="${HOME}/.ssh/AWI-remote-dev.pem"
REMOTE_BASE_DIR="onnx-nets"
REMOTE_HOME_DIR="/home/${REMOTE_USER}"
DAYS=2

LOCAL_PTH_DIR="/Volumes/X10Pro/AWIBuffer/NetModels/PyTorch"
LOCAL_ONNX_DIR="/Volumes/X10Pro/AWIBuffer/NetModels/Onnx"

DRY_RUN=0
OVERWRITE=0

usage() {
  cat <<'EOF'
Usage: ./sync_remote_models.sh [options]

Options:
  --days N             Files modified within the past N days (default: 2)
  --ssh-key PATH       Path to SSH private key (default: $HOME/.ssh/AWI-remote-dev.pem)
  --remote-host HOST  Remote host (default: awi-remote-dev.angiowavedata.com)
  --remote-user USER  Remote user (default: ubuntu)
  --remote-base-dir D Base directory under remote home (default: onnx-nets)
  --remote-home-dir D Absolute remote home dir (default: /home/<remote-user>)
  --dry-run            Print what would be copied, but do not copy
  --overwrite          Overwrite local files if they already exist
  -h, --help           Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --days) DAYS="${2:-}"; shift 2 ;;
    --ssh-key) SSH_KEY="${2:-}"; shift 2 ;;
    --remote-host) REMOTE_HOST="${2:-}"; shift 2 ;;
    --remote-user) REMOTE_USER="${2:-}"; shift 2 ;;
    --remote-base-dir) REMOTE_BASE_DIR="${2:-}"; shift 2 ;;
    --remote-home-dir) REMOTE_HOME_DIR="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ ! -f "$SSH_KEY" ]]; then
  echo "ERROR: SSH key not found: $SSH_KEY" >&2
  exit 1
fi

mkdir -p "$LOCAL_PTH_DIR"
mkdir -p "$LOCAL_ONNX_DIR"

SSH_OPTS=(-i "$SSH_KEY" -o IdentitiesOnly=yes)
REMOTE_SPEC="${REMOTE_USER}@${REMOTE_HOST}"

sync_ext() {
  local ext="$1"        # e.g. "pth" or "onnx"
  local local_root="$2" # destination root (local)
  local remote_glob="*.${ext}"

  echo "Scanning remote: ${REMOTE_SPEC}:${REMOTE_HOME_DIR}/${REMOTE_BASE_DIR} for '${remote_glob}' modified in last ${DAYS} days..."

  local copied_any=0
  while IFS= read -r remote_path; do
    [[ -z "$remote_path" ]] && continue
    # remote_path looks like: onnx-nets/some/subdir/model.${ext}
    local rel="${remote_path#${REMOTE_BASE_DIR}/}"
    local local_path="${local_root}/${rel}"
    local local_dir
    local_dir="$(dirname "$local_path")"

    if [[ -f "$local_path" && "$OVERWRITE" -eq 0 ]]; then
      continue
    fi

    mkdir -p "$local_dir"
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "[DRY RUN] scp -p ${REMOTE_SPEC}:${REMOTE_HOME_DIR}/${remote_path} \"${local_path}\""
    else
      echo "Copying: ${remote_path} -> ${local_path}"
      scp -p "${SSH_OPTS[@]}" "${REMOTE_SPEC}:${REMOTE_HOME_DIR}/${remote_path}" "${local_path}"
      copied_any=1
    fi
  done < <(
    ssh "${SSH_OPTS[@]}" "${REMOTE_SPEC}" \
      "cd \"${REMOTE_HOME_DIR}\" && find \"${REMOTE_BASE_DIR}\" -type f -name \"${remote_glob}\" -mtime -${DAYS} -print"
  )

  if [[ "$DRY_RUN" -eq 0 && "$copied_any" -eq 0 ]]; then
    echo "No new '${ext}' files to copy."
  elif [[ "$DRY_RUN" -eq 1 ]]; then
    echo "(dry-run) Done scanning '${ext}'."
  fi
}

sync_ext "pth" "$LOCAL_PTH_DIR"
sync_ext "onnx" "$LOCAL_ONNX_DIR"

echo "All sync tasks finished."

