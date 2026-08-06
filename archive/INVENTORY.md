# Archive inventory

One-line map of what lives under `archive/` and why. **Canonical** runners stay in `experiments/`; see [experiments/README.md](../experiments/README.md).

| Path | Role | Notes |
|------|------|--------|
| `experiments/slam_map_fast.py` | Short non-interactive 2D SLAM-map demo | **Regenerated** by `python tools/make_slam_map_fast.py` from `experiments/slam_map_new.py`. |
| `event_demos/demo_tum_vie_slam.py` | TUM-ViE synthetic / GPERT-backed demo | Needs `data/tum_vie_synthetic.h5` from `download_tum_vie_sample.py`. |
| `event_demos/demo_tum_vie_slam.ipynb` | Notebook twin | Same deps as `.py`. |
| `event_demos/download_tum_vie_sample.py` | Creates synthetic HDF5 subset | Writes `data/tum_vie_synthetic.h5`. |
| `event_demos/demo_uzh_shapes_slam.py` | UZH RPG shapes trajectory demo | Uses `real_camera_slam_inputs` / dataset paths as documented in script. |
| `event_demos/demo_uzh_shapes_slam.ipynb` | Notebook twin | Large; may reference download paths. |
| `event_demos/demo_neuromorphic_slam.py` | GPERT/event-vision oriented walkthrough | Overlaps conceptually with `experiments/run_event_slam.py` (canonical event pipeline stub). |
| `event_demos/demo_neuromorphic_slam.ipynb` | Notebook twin | |
| `habitat_optional/collect_habitat_data.py` | Habitat scene → RGB → events → features | Requires `habitat-sim`. |
| `habitat_optional/run_habitat_event_slam.py` | End-to-end Habitat + event + SLAM | Requires `habitat-sim` + stack in docstring. |
