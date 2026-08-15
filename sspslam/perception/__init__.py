from .image_feature_encoder import (
    ImageFeatureEncoder,
    extract_sift,
    extract_hog,
    extract_orb,
)
from .semantic_encoder import SemanticLandmarkEncoder
from .event_dataset_parser import load_events_txt, load_groundtruth_txt, accumulate_events_to_frames

# Adapters that need OpenCV stay optional: importing sspslam should not
# require opencv-python just to use the numpy-only parts of the library.
_MISSING = {}
try:
    from .miniworld_adapter import MiniWorldVisionAdapter
except ImportError as exc:  # pragma: no cover - depends on optional install
    _MISSING["MiniWorldVisionAdapter"] = exc
try:
    from .event_vision_adapter import EventVisionAdapter
except ImportError as exc:  # pragma: no cover - depends on optional install
    _MISSING["EventVisionAdapter"] = exc


def __getattr__(name):
    if name in _MISSING:
        raise ImportError(
            f"{name} is unavailable: {_MISSING[name]}. "
            "Install the optional dependency (e.g. `pip install opencv-python`)."
        ) from _MISSING[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from .event_simulator import (
    ContrastThresholdEventSimulator,
    V2EEventSimulator,
    make_event_simulator,
)
