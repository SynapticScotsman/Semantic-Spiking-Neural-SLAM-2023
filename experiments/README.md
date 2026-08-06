# Experiments — canonical runners

Primary scripts for **SSP-based SLAM** (Nengo). Optional stacks are labeled below.

## Start here

| Goal | Command | Core deps |
|------|---------|-----------|
| Path integration benchmark | `python experiments/run_pathint.py --help` | `nengo`, `nengo_ocl` (GPU optional) |
| Classic landmark SLAM | `python experiments/run_slam.py --help` | same |
| **Feature-based SLAM** (precomputed vectors) | `python experiments/run_slam_features.py ...` | `nengo`, `numpy` |
| **Semantic** (class + appearance binding) | `python experiments/run_semantic_slam.py --backend cpu` | `nengo`, `nengo_spa` |
| **Event** SLAM stub / pipeline | `python experiments/run_event_slam.py --help` | `nengo` (+ perception extras for real data) |
| **3D Miniworld** collect + SLAM orchestration | `python experiments/run_slam_3d.py --policy explore --n-steps 2000` | `gymnasium`, `miniworld` |
| 2D **full map** demo (walls, queries) | `python experiments/slam_map_new.py` | `nengo_ocl` typical |
| Slam **GIF** visualization | `python experiments/run_slam_map_gif.py --help` | `nengo_ocl` typical |

Fast non-interactive map script (regenerated):

```bash
python tools/make_slam_map_fast.py
python archive/experiments/slam_map_fast.py
```

## Optional / heavier entry points

| Script | Requires |
|--------|----------|
| `run_event_orb_slam.py` | `opencv-python` |
| `run_miniworld_slam.py` | `miniworld`, `opencv-python` |
| `collect_3d_data.py` | `miniworld`, `gymnasium` |
| `run_slamview.py`, `run_pathint_gif.py` | per `--help` |

## Tests and smoke checks

`test_*.py` files in this folder are **integration or manual checks**, not a full `pytest` suite. Run individually, e.g.:

```bash
python experiments/test_semantic_encoding.py
```

## Archived experiments

Dataset-specific or duplicate flows (TUM-ViE, UZH shapes neuromorphic notebooks, Habitat drivers, HDC standalone scripts) live under [`../archive/`](../archive/README.md).
