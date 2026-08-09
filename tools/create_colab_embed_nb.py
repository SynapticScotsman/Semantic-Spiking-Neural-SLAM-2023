"""Generate experiments/COLAB_EMBED_GPU.ipynb — the heavy-compute notebook.

Design principle (why this split is worth it): in this project the SLOW
stages are neural forward passes (YOLO crops, EigenPlaces frames, CLIP
verification, DINOv2 keys) at 1-3 h/scene on CPU, while every stage that
follows them is numpy and finishes in seconds. And the slow stages emit
TINY artifacts — crop_embeddings*.pt are 9-20 MB per scene, GT json is
KBs. So: rent a GPU for the forward passes, bring back ~30 MB, keep all
VSA analysis (batteries, merges, inspectors) local where iteration is fast
and the data already lives.

Follows the repo's existing Colab convention (COLAB_EXAMPLES.ipynb):
Drive mount -> clone into a Drive workspace -> install -> run, so a
disconnect never costs a re-download. The pipeline tools are already
checkpoint-friendly (vpr_frontend saves .part every 500 frames,
prepare_replica resumes, CLIP verification caches), which is what makes
Colab's 12 h ceiling and idle disconnects survivable.

    python tools/create_colab_embed_nb.py
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "COLAB_EMBED_GPU.ipynb")

REPO = "https://github.com/SynapticScotsman/Semantic-Spiking-Neural-SLAM-2023.git"
BRANCH = "results-sites"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in lines]}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": [l + "\n" for l in lines]}


cells = [
    md("# GPU embedding pipeline — Colab",
       "",
       "Runs **only the stages that need a GPU** and hands back small artifacts:",
       "",
       "| stage | CPU (local) | T4 (here) | artifact |",
       "|---|---|---|---|",
       "| YOLO detect+embed crops | ~1.5 h / scene | ~3 min | `crop_embeddings.pt` (9 MB) |",
       "| EigenPlaces frame descriptors | ~2 h / scene | ~2 min | `crop_embeddings_eigenplaces.pt` (20 MB) |",
       "| DINOv2 crop keys | ~1 h / scene | ~2 min | `crop_embeddings_dinov2.pt` (14 MB) |",
       "| CLIP label verification | ~40 min / scene | ~1 min | `crop_clip_verify.pt` (small) |",
       "| Replica GT extraction | minutes | minutes | `gt_instances.json` (KBs) |",
       "",
       "Everything downstream (relational batteries, instance recall, merge",
       "comparisons, cross-traverse recall, inspectors) is numpy and runs in",
       "**seconds locally** — do not move it here.",
       "",
       "Order: run cells top to bottom. Every stage skips work that already",
       "exists in Drive, so a disconnect costs nothing but time."),

    md("## 0 · GPU check + Drive mount",
       "",
       "Runtime → Change runtime type → **T4 GPU** before running this."),
    code("import torch, subprocess",
         "print(subprocess.run(['nvidia-smi', '-L'], capture_output=True,",
         "                     text=True).stdout or 'NO GPU — set Runtime > T4')",
         "print('torch', torch.__version__, '| cuda', torch.cuda.is_available())",
         "",
         "from google.colab import drive",
         "from pathlib import Path",
         "drive.mount('/content/drive')",
         "",
         "DRIVE = Path('/content/drive/MyDrive/ssnslam_colab')",
         "WORKSPACE = DRIVE / 'workspace'      # repo + raw scene data (big)",
         "ARTIFACTS = DRIVE / 'artifacts'      # the small files you take home",
         "for p in (WORKSPACE, ARTIFACTS):",
         "    p.mkdir(parents=True, exist_ok=True)",
         "print('workspace:', WORKSPACE)"),

    md("## 1 · Clone the repo into Drive + install",
       "",
       "Cloning into Drive means the next session skips this entirely."),
    code("import os, subprocess",
         f"REPO_URL = '{REPO}'",
         f"BRANCH = '{BRANCH}'",
         "REPO_DIR = WORKSPACE / 'Semantic-Spiking-Neural-SLAM-2023'",
         "",
         "if not (REPO_DIR / '.git').exists():",
         "    subprocess.run(['git', 'clone', '--branch', BRANCH, '--depth', '50',",
         "                    REPO_URL, str(REPO_DIR)], check=True)",
         "else:",
         "    subprocess.run(['git', '-C', str(REPO_DIR), 'fetch', 'origin', BRANCH],",
         "                   check=True)",
         "    subprocess.run(['git', '-C', str(REPO_DIR), 'checkout', BRANCH],",
         "                   check=True)",
         "    subprocess.run(['git', '-C', str(REPO_DIR), 'reset', '--hard',",
         "                    f'origin/{BRANCH}'], check=True)",
         "os.chdir(REPO_DIR)",
         "print('at', os.getcwd())",
         "print(subprocess.run(['git', 'log', '--oneline', '-1'],",
         "                     capture_output=True, text=True).stdout)"),
    code("# Colab already ships torch/torchvision — only the extras are needed.",
         "!pip -q install ultralytics transformers pillow numpy scipy 2>&1 | tail -2"),

    md("## 2 · Choose the work",
       "",
       "`SCENES` — Replica scene ids. `STAGES` — comment out what you don't need.",
       "",
       "A full 8-scene Replica sweep (fetch + all stages) is roughly 2–3 h here",
       "against ~20 h on CPU, and most of that is the *download*, not the GPU."),
    code("SCENES = ['room0']            # e.g. ['room0','room1','room2','office0']",
         "",
         "STAGES = [",
         "    'fetch',        # RGB-D + poses (range-fetch, resumable)",
         "    'gt',           # GT instance centroids from the vMAP renders",
         "    'yolo',         # detections + crop embeddings",
         "    'eigenplaces',  # place descriptors (localisation frontend)",
         "    'dinov2',       # crop keys, small (object/instance frontend)",
         "    'clip',         # CLIP label verification cache",
         "    # 'dinov2-large',  # <- the QUEUED instance-key rerun: 1024-d keys",
         "    #                  #    for instance_recall.py. ~50k crops across 5",
         "    #                  #    scenes is hours on CPU, ~10 min on a T4.",
         "]",
         "",
         "# Raw scene data lives in Drive so a disconnect never re-downloads it.",
         "import os",
         "for d in ('data/replica', 'outputs'):",
         "    src = WORKSPACE / d.replace('/', '_')",
         "    src.mkdir(parents=True, exist_ok=True)",
         "    if not os.path.islink(d) and not os.path.exists(d):",
         "        os.makedirs(os.path.dirname(d) or '.', exist_ok=True)",
         "        os.symlink(src, d)",
         "print('data ->', os.path.realpath('data/replica'))",
         "print('outputs ->', os.path.realpath('outputs'))"),

    md("## 3 · Run the pipeline",
       "",
       "Each stage is the *same command you run locally* — no Colab-specific",
       "code paths, so results are directly comparable. Stages that already",
       "have their artifact are skipped."),
    code("import subprocess, time, os",
         "from pathlib import Path",
         "",
         "def sh(cmd):",
         "    print('$', ' '.join(cmd), flush=True)",
         "    t0 = time.time()",
         "    r = subprocess.run(cmd)",
         "    print(f'  -> exit {r.returncode} in {time.time()-t0:.0f}s', flush=True)",
         "    return r.returncode == 0",
         "",
         "def cfg(s):",
         "    return f'vsa_cognitive_mapping/configs/replica_{s}.json'",
         "",
         "for s in SCENES:",
         "    out = Path(f'outputs/replica_{s}')",
         "    print(f'\\n=========== {s} ===========', flush=True)",
         "",
         "    if 'fetch' in STAGES and not Path(f'data/replica/{s}/poses.csv').exists():",
         "        sh(['python', 'tools/prepare_replica.py', '--scene', s])",
         "",
         "    if 'gt' in STAGES and not (out / 'gt_instances.json').exists():",
         "        sh(['python', 'tools/replica_gt_from_renders.py', '--scene', s])",
         "",
         "    if 'yolo' in STAGES and not (out / 'crop_embeddings.pt').exists():",
         "        sh(['python', '-m', 'vsa_cognitive_mapping.classroom_pipeline',",
         "            'embed-crops', '--dataset', cfg(s)])",
         "",
         "    if 'eigenplaces' in STAGES and not (out / 'crop_embeddings_eigenplaces.pt').exists():",
         "        sh(['python', '-m', 'vsa_cognitive_mapping.vpr_frontend',",
         "            '--dataset', cfg(s), '--model', 'eigenplaces', '--batch', '32'])",
         "",
         "    if 'dinov2' in STAGES and not (out / 'crop_embeddings_dinov2.pt').exists():",
         "        sh(['python', '-m', 'vsa_cognitive_mapping.encoder_comparison',",
         "            '--dataset', cfg(s), '--encoders', 'dinov2'])",
         "",
         "    if 'dinov2-large' in STAGES and not (out / 'crop_embeddings_dinov2-large.pt').exists():",
         "        sh(['python', '-m', 'vsa_cognitive_mapping.encoder_comparison',",
         "            '--dataset', cfg(s), '--encoders', 'dinov2:large',",
         "            '--batch', '64'])",
         "",
         "    if 'clip' in STAGES and not (out / 'crop_clip_verify.pt').exists():",
         "        sh(['python', '-m', 'vsa_cognitive_mapping.object_grounding',",
         "            '--dataset', cfg(s), '--gt-json', str(out / 'gt_instances.json'),",
         "            '--verify-crops', '--max-per-class', '60'])",
         "",
         "print('\\nall requested stages done')"),

    md("## 4 · Pack the small artifacts to take home",
       "",
       "Only the outputs — never the raw scene data. Expect ~30–50 MB per scene."),
    code("import shutil, zipfile, os",
         "from pathlib import Path",
         "",
         "KEEP = ('.pt', '.json', '.csv')",
         "stamp = time.strftime('%Y%m%d_%H%M')",
         "zpath = ARTIFACTS / f'embeds_{stamp}.zip'",
         "",
         "with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:",
         "    for s in SCENES:",
         "        d = Path(f'outputs/replica_{s}')",
         "        if not d.exists():",
         "            continue",
         "        for f in sorted(d.iterdir()):",
         "            if f.suffix in KEEP and f.stat().st_size < 300e6:",
         "                z.write(f, f'outputs/replica_{s}/{f.name}')",
         "                print(f'{f.name:<45}{f.stat().st_size/1e6:7.1f} MB')",
         "",
         "print(f'\\nwrote {zpath}  ({zpath.stat().st_size/1e6:.1f} MB)')",
         "print('also saved in Drive/ssnslam_colab/artifacts — or download below')",
         "from google.colab import files",
         "files.download(str(zpath))"),

    md("## 5 · Back on your machine",
       "",
       "Unzip into the repo root — the paths inside the archive already match",
       "the layout the local tools expect:",
       "",
       "```bash",
       "# from the repo root",
       "tar -xf embeds_YYYYMMDD_HHMM.zip     # or unzip",
       "```",
       "",
       "Then everything downstream runs locally in seconds, e.g.:",
       "",
       "```bash",
       "python -m vsa_cognitive_mapping.object_grounding \\",
       "    --dataset vsa_cognitive_mapping/configs/replica_room0.json \\",
       "    --gt-json outputs/replica_room0/gt_instances.json --merge-check",
       "python -m vsa_cognitive_mapping.relational_recall",
       "python -m vsa_cognitive_mapping.instance_recall",
       "python tools/aggregate_replica_truth.py",
       "```",
       "",
       "**What is deliberately NOT in this notebook:** the VSA analysis itself.",
       "It is numpy, it is fast, the artifacts are small, and keeping it local",
       "means the batteries and inspectors iterate at the speed of thought",
       "instead of a Colab round-trip."),
]

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {OUT} ({os.path.getsize(OUT)/1024:.0f} KB, {len(cells)} cells)")
