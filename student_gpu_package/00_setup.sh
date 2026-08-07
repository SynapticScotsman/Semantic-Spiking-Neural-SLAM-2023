#!/usr/bin/env bash
# Stage 0: environment for the ConceptGraphs head-to-head. Run once.
# Idempotent: safe to re-run after a partial failure.
set -e
trap 'echo "STAGE FAILED (00_setup) — see README Troubleshooting, item matching the last line above"' ERR

command -v conda >/dev/null || {
  echo "FAIL: conda not on PATH. Install miniconda, or run:  source ~/miniconda3/etc/profile.d/conda.sh"; exit 1; }

echo "== creating conda env 'cgraphs' (skipped if it exists) =="
conda env list | grep -q "^cgraphs " || conda create -y -n cgraphs python=3.10
eval "$(conda shell.bash hook)"
conda activate cgraphs

echo "== cloning official ConceptGraphs =="
if [ ! -d concept-graphs ]; then
  git clone https://github.com/concept-graphs/concept-graphs.git
fi
cd concept-graphs
# Their README's install section is authoritative; this covers the usual path.
pip install -e . || echo "WARN: editable install failed — follow THEIR README install section, then re-run this script (it will skip done steps)"

# CUDA wheel must match the driver. Pick automatically, fall back to CPU.
if command -v nvidia-smi >/dev/null; then
  CUDA_MAJ=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+" | head -1 || echo 12)
  if [ "$CUDA_MAJ" -ge 12 ]; then IDX=cu121; else IDX=cu118; fi
  echo "== torch for CUDA $CUDA_MAJ ($IDX) =="
  pip install torch torchvision --index-url https://download.pytorch.org/whl/$IDX
else
  echo "WARN: no GPU visible — installing CPU torch (stage 2 will be very slow)"
  pip install torch torchvision
fi
pip install open_clip_torch supervision open3d scipy

# SAM checkpoint (their default vit_h; vit_b fallback for small GPUs is in
# the README Troubleshooting section)
mkdir -p checkpoints
if [ ! -f checkpoints/sam_vit_h_4b8939.pth ]; then
  URL=https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  if command -v wget >/dev/null; then wget -O checkpoints/sam_vit_h_4b8939.pth "$URL"
  else curl -L -o checkpoints/sam_vit_h_4b8939.pth "$URL"; fi
fi
cd ..

pip install ultralytics transformers pillow numpy   # for stage 4 (our side)

echo "== recording environment =="
{ python -V; pip freeze; nvidia-smi -L 2>/dev/null || echo "no GPU visible"; } \
  > environment.txt
echo "STAGE OK (00_setup)"
