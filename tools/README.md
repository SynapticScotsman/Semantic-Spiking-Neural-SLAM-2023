# Developer / maintenance tools

Small scripts that are not part of the main `experiments/` API.

| Script | Purpose |
|--------|---------|
| `create_ipynb.py` | Build `experiments/test_perception.ipynb` from `experiments/test_perception.py`. |
| `create_dashboard_ipynb.py` | Build `experiments/slam_3d_dashboard.ipynb` from `experiments/slam_3d_dashboard.py`. |
| `create_event_ipynb.py` | Build an event-ORB notebook (UH-RPG shapes); needs `nbformat` and optional deps at runtime. |
| `make_slam_map_fast.py` | Generates `archive/experiments/slam_map_fast.py` from `experiments/slam_map_new.py`. |

Run each from the **repository root** so paths resolve correctly:

```bash
python tools/make_slam_map_fast.py
```
