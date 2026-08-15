"""Run the ConceptGraphs head-to-head over the remaining Replica scenes.

room0 was done interactively on 2026-08-14 (88 min on an L4). This does the
other seven unattended, so the comparison stops being a single scene and the
length-scale rule can be fitted on some scenes and TESTED on others — the
caveat that currently blocks publishing 0.45,0.27.

Assumes the environment is already built: run cells 1-3b of
experiments/COLAB_CONCEPTGRAPHS_L4.ipynb once per VM (ConceptGraphs +
GroundingDINO patches + pytorch3d). This script only orchestrates; it installs
nothing, so a missing dependency fails loudly here instead of being papered
over per scene.

    python collab_tasks/scripts/run_cg_scenes.py \
        --workspace /content/drive/MyDrive/ssnslam_colab/workspace

Design notes, each one paid for on 2026-08-14/15:

  resumable      every stage is skip-if-present, keyed on its own output file.
                 A dropped Colab session costs the scene in flight, not the run
                 — re-running the same command picks up where it stopped.
  isolating      one scene failing does not stop the others. The summary at the
                 end names what failed and at which stage, rather than the run
                 dying at scene 3 of 7 overnight.
  cheap to watch a one-line status file on Drive, overwritten in place. Polling
                 its mtime costs ~200 tokens; re-reading a notebook cell costs
                 thousands. tqdm bars go to STATUS, never to stdout.
  decode-free    this produces FRONTENDS (their objects, our eval points). It
                 deliberately does not tune the length scale — that is minutes
                 of laptop CPU afterwards against artifacts already on disk.
                 Serialising the two was the mistake that delayed this run.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

# the eight Replica scenes with a config in vsa_cognitive_mapping/configs;
# room0 is excluded because it is already done and its artifacts are on Drive
ALL_SCENES = ["room1", "room2", "office0", "office1", "office2", "office3",
              "office4"]

# their published mapping parameters, verbatim from the L4 notebook's full run
CFSLAM_ARGS = [
    "spatial_sim_type=overlap", "mask_conf_threshold=0.95",
    "match_method=sim_sum", "sim_threshold=1.2", "dbscan_eps=0.1",
    "gsa_variant=none", "class_agnostic=True", "skip_bg=True",
    "max_bbox_area_ratio=0.5", "merge_interval=20",
    "merge_visual_sim_thresh=0.8", "merge_text_sim_thresh=0.8",
    "save_suffix=overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub",
]


class Runner:
    def __init__(self, workspace: str, scene: str):
        self.ws = workspace
        self.scene = scene
        self.log = f"{workspace}/run_log_{scene}.txt"
        self.status = f"{workspace}/status_all_scenes.txt"

    def say(self, s: str):
        print(s, flush=True)
        with open(self.log, "a") as f:
            f.write(s + "\n")

    def mark(self, s: str):
        """One line, overwritten. Poll this instead of reading the notebook."""
        with open(self.status, "w") as f:
            f.write(f"{datetime.datetime.now():%H:%M:%S} {self.scene} | {s}\n")

    def sh(self, cmd, cwd=None):
        """Run a stage, streaming output, collapsing tqdm into the status file."""
        line = " ".join(map(str, cmd))
        self.say(f"[{datetime.datetime.now():%H:%M:%S}] $ {line}")
        t0 = time.time()
        last = [t0]
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             errors="replace", env=env, cwd=cwd)

        def pulse():
            # a quiet stage is normal for long model passes; prove it is alive
            while p.poll() is None:
                time.sleep(30)
                if p.poll() is None and time.time() - last[0] > 300:
                    self.say(f"  still running "
                             f"({(time.time()-t0)/60:.0f} min, no new output)")
                    last[0] = time.time()

        threading.Thread(target=pulse, daemon=True).start()
        prog = [0.0]
        for ln in p.stdout:
            ln = ln.rstrip()
            last[0] = time.time()
            if "%|" in ln or "it/s" in ln or "s/it" in ln:
                self.mark(f"{line[:50]} | {ln[-60:]} | "
                          f"{(time.time()-t0)/60:.1f} min")
                if time.time() - prog[0] < 60:
                    continue
                prog[0] = time.time()
            self.say(ln)
        rc = p.wait()
        self.say(f"  exit {rc} in {time.time()-t0:.0f}s")
        if rc:
            raise RuntimeError(f"stage failed ({rc}): {line}")


def do_scene(scene: str, ws: str, args) -> dict:
    """Every stage for one scene. Raises on the first failure; the caller
    records which stage that was and moves to the next scene."""
    r = Runner(ws, scene)
    cfg = f"vsa_cognitive_mapping/configs/replica_{scene}.json"
    if not os.path.exists(cfg):
        raise RuntimeError(f"no config {cfg}")

    cg = f"{ws}/concept-graphs"
    cgdir = f"{cg}/conceptgraph"
    conf = f"{cg}/conceptgraph/dataset/dataconfigs/replica/replica.yaml"
    for p in (cgdir, conf):
        if not os.path.exists(p):
            raise RuntimeError(f"missing {p} — run cells 1-3b of "
                               "COLAB_CONCEPTGRAPHS_L4.ipynb first")
    data = args.cg_data
    stage = {}

    # --- our side: scene data, GT, our own frontend -----------------------
    r.mark("fetching scene (~2 GB via HTTP range)")
    if not os.path.exists(f"data/replica/{scene}/poses.csv"):
        r.sh(["python", "tools/prepare_replica.py", "--scene", scene])
    stage["prepare"] = "ok"

    r.mark("ground truth from vMAP renders")
    if not os.path.exists(f"outputs/replica_{scene}/gt_instances.json"):
        r.sh(["python", "tools/replica_gt_from_renders.py", "--scene", scene])
    stage["gt"] = "ok"

    r.mark("our frontend: crop embeddings")
    if not os.path.exists(f"outputs/replica_{scene}/detections_crops.csv"):
        r.sh(["python", "-m", "vsa_cognitive_mapping.classroom_pipeline",
              "embed-crops", "--dataset", cfg])
    stage["embed"] = "ok"

    r.mark("our frontend: object grounding")
    if not os.path.exists(f"outputs/replica_{scene}/object_points.json"):
        r.sh(["python", "-m", "vsa_cognitive_mapping.object_grounding",
              "--dataset", cfg, "--gt-json",
              f"outputs/replica_{scene}/gt_instances.json"])
    stage["ground"] = "ok"

    r.mark("building their dataset layout")
    if not os.path.exists(f"{data}/{scene}/traj.txt"):
        r.sh(["python", "student_gpu_package/01_check_data.py",
              "--scene", scene])
    stage["layout"] = "ok"

    # --- their side: SAM segmentation, then 3D mapping --------------------
    root_abs = os.path.abspath(data)
    gsa_done = glob.glob(f"{root_abs}/{scene}/gsa_detections_none/*")
    r.mark("SAM ViT-H segmentation (the GPU-heavy stage)")
    if not gsa_done:
        r.sh(["python", "scripts/generate_gsa_results.py",
              "--dataset_root", root_abs, "--dataset_config", conf,
              "--scene_id", scene, "--class_set", "none",
              "--stride", str(args.stride)], cwd=cgdir)
    else:
        r.say(f"  gsa results already present ({len(gsa_done)} entries)")
    stage["sam"] = "ok"

    drive_out = f"{ws}/cg_out/{scene}"
    r.mark("their 3D mapping")
    if not glob.glob(f"{drive_out}/*.pkl.gz"):
        r.sh(["python", "slam/cfslam_pipeline_batch.py",
              f"dataset_root={root_abs}", f"dataset_config={conf}",
              f"stride={args.stride}", f"scene_id={scene}", *CFSLAM_ARGS],
             cwd=cgdir)
    stage["map"] = "ok"

    # persist before anything else can go wrong: local disk dies with the VM
    src = f"{root_abs}/{scene}/pcd_saves"
    os.makedirs(drive_out, exist_ok=True)
    os.makedirs(f"student_gpu_package/cg_out/{scene}", exist_ok=True)
    found = glob.glob(f"{src}/*.pkl.gz")
    if not found and not glob.glob(f"{drive_out}/*.pkl.gz"):
        raise RuntimeError(f"no pkl.gz under {src} — mapping saved nothing")
    for p in found:
        shutil.copy(p, drive_out)
        shutil.copy(p, f"student_gpu_package/cg_out/{scene}/")
        r.say(f"  saved {os.path.basename(p)} "
              f"{os.path.getsize(p)/1e6:.1f} MB (Drive + cg_out)")
    for p in glob.glob(f"{drive_out}/*.pkl.gz"):     # resume case
        d = f"student_gpu_package/cg_out/{scene}/{os.path.basename(p)}"
        if not os.path.exists(d):
            shutil.copy(p, d)
    stage["persist"] = "ok"

    # --- their objects -> our trace, and one score ------------------------
    r.mark("exporting their objects with CLIP labels")
    r.sh(["python", "student_gpu_package/03_export_cg.py", "--scene", scene])
    stage["export"] = "ok"

    # A number per scene, at the current best length scale, so the run is not
    # opaque while it goes. This is NOT the sweep — that runs on a laptop
    # afterwards against these same artifacts, as often as we like.
    r.mark("one decode at the current best length scale")
    r.sh(["python", "student_gpu_package/04_vsa_labels.py",
          "--scene", f"{scene}_cgfront", "--gt-scene", scene,
          "--labels-from-points", "--max-per-class", "400",
          "--length-scale", args.length_scale])
    out = subprocess.run(["python", "student_gpu_package/05_score.py",
                          "--scene", f"{scene}_cgfront"],
                         capture_output=True, text=True)
    r.say(out.stdout[-1500:])
    stage["score"] = "ok"

    # Everything the laptop needs for the decode sweeps, on Drive. handoff/ is
    # the sibling of workspace/ (DRIVE/handoff, DRIVE/workspace) — the layout
    # cell 1 of the L4 notebook creates.
    handoff = os.path.join(os.path.dirname(ws.rstrip("/\\")), "handoff")
    for name in (scene, f"{scene}_cgfront"):
        s = f"student_gpu_package/handoff/{name}"
        if os.path.isdir(s):
            shutil.copytree(s, f"{handoff}/{name}", dirs_exist_ok=True)
    for name in (f"replica_{scene}", f"replica_{scene}_cgfront"):
        s = f"outputs/{name}/object_points.json"
        if os.path.exists(s):
            d = f"{handoff}/{name.replace('replica_', '')}"
            os.makedirs(d, exist_ok=True)
            shutil.copy(s, d)
    r.say(f"  handoff written to {handoff}/{scene}[_cgfront]")
    stage["handoff"] = "ok"
    return stage


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace",
                    default="/content/drive/MyDrive/ssnslam_colab/workspace")
    ap.add_argument("--scenes", nargs="*", default=ALL_SCENES)
    ap.add_argument("--cg-data", default="/content/cg_dataset",
                    help="their dataset layout, on LOCAL VM disk — Drive-FUSE "
                         "makes 4000-file layouts and per-frame reads crawl")
    ap.add_argument("--stride", type=int, default=5,
                    help="5 gives 400 frames per scene, matching the room0 run")
    ap.add_argument("--length-scale", default="0.45,0.27",
                    help="only for the progress score; the real sweep is local")
    args = ap.parse_args()

    os.makedirs(args.workspace, exist_ok=True)
    results = {}
    t0 = time.time()
    print(f"{len(args.scenes)} scenes: {', '.join(args.scenes)}")
    print(f"status file: {args.workspace}/status_all_scenes.txt\n", flush=True)

    for i, scene in enumerate(args.scenes, 1):
        print(f"\n{'='*70}\n[{i}/{len(args.scenes)}] {scene}  "
              f"({(time.time()-t0)/60:.0f} min elapsed)\n{'='*70}", flush=True)
        st = time.time()
        try:
            stages = do_scene(scene, args.workspace, args)
            results[scene] = {"ok": True, "min": (time.time()-st)/60,
                              "stages": stages}
        except Exception as e:                       # keep the other scenes
            results[scene] = {"ok": False, "min": (time.time()-st)/60,
                              "error": str(e)[:400]}
            print(f"\n!! {scene} FAILED: {e}\n   continuing with the rest",
                  flush=True)
        with open(f"{args.workspace}/scene_run_results.json", "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n{'='*70}\nSUMMARY after {(time.time()-t0)/60:.0f} min\n{'='*70}")
    for s, r in results.items():
        print(f"  {s:<10}{'ok' if r['ok'] else 'FAILED':<8}{r['min']:>6.0f} min"
              + ("" if r["ok"] else f"   {r['error'][:90]}"))
    n_ok = sum(r["ok"] for r in results.values())
    print(f"\n{n_ok}/{len(results)} scenes complete. Artifacts on Drive under "
          f"handoff/ — the decode sweeps run on a laptop from there, no GPU.")
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
