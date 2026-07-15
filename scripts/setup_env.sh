#!/usr/bin/env bash
# Creates a venv for lepong data collection + training.
#
# Uses --system-site-packages so it inherits the base conda env's torch
# (already a CUDA build, ~2GB — no reason to redownload it) and numpy/pillow/
# h5py/pyarrow/fastapi/etc. Only installs what's actually missing: opencv,
# lance, pettingzoo[atari], ale-py, and the Atari ROM installer.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR (inherits base env's torch/CUDA build)..."
    python3 -m venv --system-site-packages "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Installing Atari ROMs (accepting the ROM license non-interactively)..."
python3 -m AutoROM --accept-license

echo
echo "Verifying imports..."
python3 -c "
import torch, cv2, lance, pyarrow, pettingzoo, ale_py
print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())
print('cv2', cv2.__version__)
print('lance', lance.__version__)
print('pettingzoo', pettingzoo.__version__)
print('ale_py', ale_py.__version__)
"

echo
echo "Setup done. Activate with:"
echo "  source .venv/bin/activate"
