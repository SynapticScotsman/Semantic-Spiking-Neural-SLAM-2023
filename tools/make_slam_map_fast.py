"""Generate a shorter, non-interactive variant of ``experiments/slam_map_new.py``.

Run from repo root: ``python tools/make_slam_map_fast.py``

Output: ``archive/experiments/slam_map_fast.py`` (run that file if needed; uses ``nengo_ocl`` like the source).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

with open(ROOT / "experiments" / "slam_map_new.py", "r", encoding="utf-8") as f:
    code = f.read()

# Reduce simulation time from 60 to 10 seconds for a fast demonstration
code = code.replace("T = 60", "T = 10")
# Must increase high pass freq to avoid ValueError when T decreases
code = code.replace("high=.05", "high=0.1")

# Avoid blocking GUI calls, save to PNG instead
code = code.replace("plt.show()", "plt.savefig('slam_plot_env.png')")
code = code.replace(
    "fig.show()",
    "fig.savefig('slam_plot_results.png')\nplt.close('all')",
)

out = ROOT / "archive" / "experiments" / "slam_map_fast.py"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    f.write(code)

print(f"Wrote {out}")
