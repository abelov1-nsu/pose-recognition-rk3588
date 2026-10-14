#!/usr/bin/env bash
#
# setup_rk3588.sh
#
# Creates a Python 3.10 conda environment named "rknn" and installs
# everything needed to run recognize_rknn.py on the RK3588.
#
# Usage:
#   chmod +x setup_rk3588.sh
#   ./setup_rk3588.sh
#
# Requirements:
#   - conda already installed and available in PATH
#   - rknn-toolkit-lite2 .whl file placed in ./wheels/ next to this script
#     (filename must be aarch64, cp310)
#
# The script is idempotent: re-running it will not break anything.

set -euo pipefail

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

ENV_NAME="rknn"
PY_VERSION="3.10"
RKNN_LITE_VERSION="2.5.0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEEL_DIR="${SCRIPT_DIR}/wheels"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

log() { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }
die() { err "$*"; exit 1; }

# ------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------

command -v conda >/dev/null 2>&1 \
    || die "conda not found in PATH. Install Miniconda/Anaconda first."

[[ -f "${REQ_FILE}" ]] \
    || die "requirements.txt not found at ${REQ_FILE}"

# ------------------------------------------------------------
# Locate the rknn-toolkit-lite2 wheel
# ------------------------------------------------------------

shopt -s nullglob
WHEELS=( "${WHEEL_DIR}"/rknn_toolkit_lite2-*.whl )
shopt -u nullglob

if [[ ${#WHEELS[@]} -eq 0 ]]; then
    die "No rknn_toolkit_lite2-*.whl found in ${WHEEL_DIR}.
Download the aarch64 Python 3.10 wheel from the rknn-toolkit2 release
matching version ${RKNN_LITE_VERSION}, and place it in ${WHEEL_DIR}/.
Expected filename pattern:
  rknn_toolkit_lite2-${RKNN_LITE_VERSION}-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
fi

RKNN_WHEEL="${WHEELS[0]}"
log "Found RKNN wheel: $(basename "${RKNN_WHEEL}")"

# Quick sanity check on the wheel filename
if [[ "${RKNN_WHEEL}" != *cp310* || "${RKNN_WHEEL}" != *aarch64* ]]; then
    die "Wheel does not look like a cp310 aarch64 build: $(basename "${RKNN_WHEEL}")"
fi

# ------------------------------------------------------------
# Initialize conda for this shell
# ------------------------------------------------------------

# conda's shell hook must be sourced before `conda activate` works
# inside a non-interactive script.
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ------------------------------------------------------------
# Create or reuse the environment
# ------------------------------------------------------------

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    log "Conda environment '${ENV_NAME}' already exists; reusing it."
else
    log "Creating conda environment '${ENV_NAME}' with Python ${PY_VERSION}..."
    conda create -y -n "${ENV_NAME}" "python=${PY_VERSION}"
fi

log "Activating '${ENV_NAME}'..."
conda activate "${ENV_NAME}"

# ------------------------------------------------------------
# Install pip packages
# ------------------------------------------------------------

log "Upgrading pip..."
python -m pip install --upgrade pip

log "Installing requirements from ${REQ_FILE}..."
python -m pip install -r "${REQ_FILE}"

log "Installing RKNN runtime wheel..."
python -m pip install --force-reinstall "${RKNN_WHEEL}"

# ------------------------------------------------------------
# Verify
# ------------------------------------------------------------

log "Verifying imports..."
python - <<'PY'
import sys
print("Python:", sys.version.split()[0])

import numpy, cv2
print("numpy:", numpy.__version__)
print("opencv:", cv2.__version__)

try:
    from rknnlite.api import RKNNLite
    print("rknnlite: OK")
except Exception as e:
    print("rknnlite import FAILED:", e)
    sys.exit(1)

try:
    from ultralytics.trackers.byte_tracker import BYTETracker
    print("ultralytics ByteTrack: OK")
except Exception as e:
    print("ultralytics import FAILED:", e)
    sys.exit(1)

try:
    import mss  # optional
    print("mss: OK (optional)")
except Exception:
    print("mss: not installed (optional, needed only for --source screen)")

print("All critical imports OK.")
PY

# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

log "Setup complete."
log "Connected i2c screen:."
sudo i2cdetect -y 1

cat <<EOF

To use the environment:

    conda activate ${ENV_NAME}
    cd ${SCRIPT_DIR}
    python recognize_rknn.py --source 0

If you get an "Invalid RKNN model version" error later, it means the
board's librknnrt.so is older than ${RKNN_LITE_VERSION}. Update the
runtime library to match the toolkit version used for conversion.




EOF