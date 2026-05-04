#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# IP DrawingDrafter — AP3 Vectorization Pipeline · Setup Script
#
# Run this once after cloning the repo:
#
#     bash setup.sh
#
# It will:
#   1. Create a Python virtual environment in .venv/
#   2. Install runtime dependencies from requirements.txt
#   3. Print download instructions for the SketchCleanNet weight file
#      (which is too large to ship in git).
#
# This script is idempotent — safe to re-run.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# ── 1. Python virtual environment ────────────────────────────────────────────
PYTHON="${PYTHON:-python3}"
VENV_DIR=".venv"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "→ Creating virtualenv in $VENV_DIR/ ..."
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo "→ Virtualenv $VENV_DIR/ already exists, skipping creation."
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ── 2. Install dependencies ──────────────────────────────────────────────────
echo "→ Upgrading pip ..."
pip install --upgrade pip --quiet

echo "→ Installing requirements.txt ..."
pip install -r requirements.txt

# ── 3. Model weights ─────────────────────────────────────────────────────────
SKETCHCLEAN_WEIGHT="$PROJECT_ROOT/models/sketchcleannet.pth"

# OneDrive (HAW Landshut) folder containing all model weights.
# Folder URL — partner downloads sketchcleannet.pth manually from this folder
# (SharePoint folder URLs cannot be wget'd directly without per-file links).
ONEDRIVE_FOLDER_URL="https://hawlandshutde-my.sharepoint.com/:f:/g/personal/rey21950_az_haw-landshut_de/IgBt1cX39WruQ4Oxg9neSeQbAXvmPFL6QjFBUjJYiiT4k_M?e=eCbhJK"

# Optional: a per-file direct download URL (set this to skip the manual step).
# To generate one in SharePoint: open the folder, click sketchcleannet.pth,
# then "..." → "Copy link" → set permissions → append "?download=1" to the URL.
DIRECT_DOWNLOAD_URL="https://hawlandshutde-my.sharepoint.com/:u:/g/personal/rey21950_az_haw-landshut_de/IQDKTxEjVJbnQLn6ZLOkpf2oAf0q9wUvBD1gfhYst_nS2T8?e=PlKpSA"

echo
echo "── Model weights ────────────────────────────────────────────────────"

if [[ -f "$SKETCHCLEAN_WEIGHT" ]]; then
    echo "✓ sketchcleannet.pth already present at $SKETCHCLEAN_WEIGHT"
elif [[ -n "$DIRECT_DOWNLOAD_URL" ]]; then
    echo "→ Downloading sketchcleannet.pth ..."
    curl -L -o "$SKETCHCLEAN_WEIGHT" "$DIRECT_DOWNLOAD_URL"
    echo "✓ Saved to $SKETCHCLEAN_WEIGHT"
else
    cat <<EOF
✗ sketchcleannet.pth is NOT YET INSTALLED.

  To install it, do ONE of the following:

  Option A — Download manually (works without code changes):
    1. Open this OneDrive folder in your browser:
         $ONEDRIVE_FOLDER_URL
    2. Click the file 'sketchcleannet.pth' and download it.
    3. Move the downloaded file to:
         $SKETCHCLEAN_WEIGHT

  Option B — Automated download (one-time setup):
    1. In SharePoint, generate a per-file direct download URL for
       sketchcleannet.pth (right-click → Copy link → append '?download=1').
    2. Edit setup.sh and set DIRECT_DOWNLOAD_URL to that link.
    3. Re-run: bash setup.sh

  Option C — Skip it.
    Stage 1 will automatically fall back to its classical cleaning mode.
    The pipeline still produces valid output; just lower quality on
    photographed/shaded sketches.

The other two model files (puhachov_keypoints.pth, free2cad_v3_best.pth)
are small enough to ship in the repository and are already in place.
EOF
fi

echo
echo "── Setup complete ─────────────────────────────────────────────────────"
echo "Activate the environment with:  source $VENV_DIR/bin/activate"
echo "Then run the pipeline (see README.md for the full command)."
