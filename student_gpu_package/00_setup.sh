#!/usr/bin/env bash
# Stage 0: environment for the ConceptGraphs head-to-head. Run once.
set -e
echo "== creating conda env 'cgraphs' =="
conda create -y -n cgraphs python=3.10
eval "$(conda shell.bash hook)"
conda activate cgraphs

echo "== cloning official ConceptGraphs =="
if [ ! -d concept-graphs ]; then
  git clone https://github.com/concept-graphs/concept-graphs.git
fi
cd concept-graphs
# Their README's install section is authoritative; this covers the usual path.
pip install -e .
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install open_clip_torch supervision open3d
# SAM checkpoint (their default)
mkdir -p checkpoints
[ -f checkpoints/sam_vit_h_4b8939.pth ] || \
  wget -O checkpoints/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..

pip install ultralytics transformers pillow numpy   # for stage 4 (our side)

echo "== recording environment =="
{ python -V; pip freeze; nvidia-smi -L 2>/dev/null || echo "no GPU visible"; } \
  > environment.txt
echo "STAGE OK (00_setup)"
