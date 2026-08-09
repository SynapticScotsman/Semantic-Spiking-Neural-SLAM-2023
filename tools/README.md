# Developer / maintenance tools

Small scripts that are not part of the main `experiments/` API.

| Script | Purpose |
|--------|---------|
| `create_ipynb.py` | Build `experiments/test_perception.ipynb` from `experiments/test_perception.py`. |
| `create_dashboard_ipynb.py` | Build `experiments/slam_3d_dashboard.ipynb` from `experiments/slam_3d_dashboard.py`. |
| `create_event_ipynb.py` | Build an event-ORB notebook (UH-RPG shapes); needs `nbformat` and optional deps at runtime. |
| `make_slam_map_fast.py` | Generates `archive/experiments/slam_map_fast.py` from `experiments/slam_map_new.py`. |
| `wiki_ops.py` | Initializes and maintains the LLM wiki (`init`, `ingest`, `query`, `lint`, `search`). |

Run each from the **repository root** so paths resolve correctly:

```bash
python tools/make_slam_map_fast.py
```

LLM wiki quickstart:

```bash
python tools/wiki_ops.py init
python tools/wiki_ops.py ingest --source raw/inbox/my_source.md --kind source
python tools/wiki_ops.py query --question "What are the top unresolved representation trade-offs?"
python tools/wiki_ops.py lint
python tools/wiki_ops.py search --query "loop closure"
```
