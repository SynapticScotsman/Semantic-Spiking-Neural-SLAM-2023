# Archive

Scripts and notebooks here are **optional** or **dataset-specific** experiments. They are kept so nothing is lost, but they are not the main entry points for the project.

## Layout

| Path | Contents |
|------|----------|
| `experiments/` | Generated or duplicate experiment scripts (e.g. fast variant of `slam_map_new`). |
| `event_demos/` | TUM-ViE, UZH shapes, and neuromorphic/GPERT-oriented demos and notebooks. |
| `habitat_optional/` | Habitat-sim data collection and event-SLAM drivers (heavy optional deps). |

## How to run archived Python scripts

From the repository root, with the package installed (`pip install -e .`) or `PYTHONPATH` set to the repo root:

```bash
python archive/event_demos/demo_tum_vie_slam.py
python archive/habitat_optional/collect_habitat_data.py --help
```

If imports fail, use:

```bash
set PYTHONPATH=%CD%
python archive\event_demos\download_tum_vie_sample.py
```

(Linux/macOS: `export PYTHONPATH="$PWD"`.)

See [INVENTORY.md](INVENTORY.md) for a full list and status.
