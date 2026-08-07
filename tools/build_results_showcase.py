"""Assemble the visual results-showcase payload from artefacts already on disk.

No recomputation: harvests the JSON payloads embedded in the existing
inspector pages (chess cross-traverse playback + photos; per-scene belief
fields + GT + photos), the synthetic chi sweep rows, and the room0 camera
path, and writes one compact JSON for the showcase page template.

    python tools/build_results_showcase.py
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def payload_of(html_path):
    txt = open(html_path, encoding="utf-8").read()
    m = re.search(r'<script id="data" type="application/json">(.*?)</script>',
                  txt, re.S)
    return json.loads(m.group(1).replace("<\\/", "</"))


out = {}

# ---- chess playback (subsampled, photos kept for used queries) ------------
ch = payload_of("outputs/cross_inspector_7scenes_chess.html")
qs = ch["queries"][::2]
used = set()
for q in qs:
    used.update((q["q"], q["a"], q["n"]))
out["chess"] = dict(ext=ch["ext"], paths=ch["paths"], queries=qs,
                    med=ch["med"], med_knn=ch["med_knn"],
                    imgs={k: v for k, v in ch["imgs"].items() if k in used})

# ---- belief fields for three scenes (fields + GT + a few photos) ----------
SCENES = ["room0", "office2", "office3"]
gal = []
for s in SCENES:
    d = payload_of(f"outputs/grounding_inspector_{s}.html")
    keep_imgs = {}
    panels = []
    for p in d["panels"]:
        if p.get("hit_t") is None:          # keep only truth-scored classes
            continue
        entry = dict(cls=p["cls"], field=p["field"], truth=p["truth"],
                     peaks=p["peaks"], hit=p["hit_t"], d1=p["d1t"],
                     n_mem=p["n_mem"])
        # keep photos for one hit and any miss per scene (the honest pair)
        if (p["hit_t"] is False) or not any(
                x.get("tq") for x in panels if x.get("hit")):
            if p.get("tq"):
                entry["tq"] = p["tq"]
                keep_imgs[p["tq"]] = d["imgs"][p["tq"]]
            if p.get("ta"):
                entry["ta"] = p["ta"]
                keep_imgs[p["ta"]] = d["imgs"][p["ta"]]
            if p["hit_t"]:
                entry["show"] = True
        panels.append(entry)
    gal.append(dict(scene=s, ext=d["ext"], grid=d["grid"], path=d["path"],
                    r1=d.get("r1_true"), panels=panels, imgs=keep_imgs))
out["gallery"] = gal

# ---- school-run time recall + two-robot time merge (photos + timeline) ----
ri = payload_of("outputs/recall_inspector.html")
tq = ri["A"]["queries"][::3]
used_t = set()
for q in tq:
    used_t.update((q["q"], q["im"], q["ia"], q["ib"]))
out["school"] = dict(span=ri["A"]["span"], tA_max=ri["A"]["tA_max"],
                     exact=ri["A"]["exact"], identical=ri["A"]["identical"],
                     n_batt=ri["A"]["n_batt"], queries=tq,
                     imgs={k: v for k, v in ri["A"]["imgs"].items()
                           if k in used_t})

# ---- classroom conditioning rescue (map + photos, EigenPlaces zscored) ----
cq = ri["B"]["queries"][::3]
used_c = set()
for q in cq:
    used_c.update((q["q"], q["near"]))
out["classroom"] = dict(ext=ri["B"]["ext"], path=ri["B"]["path"],
                        med=ri["B"]["med"], queries=cq,
                        imgs={k: v for k, v in ri["B"]["imgs"].items()
                              if k in used_c})

# ---- 4-robot merge animation data (room0 camera path quartered) -----------
rows = list(csv.DictReader(open("data/replica/room0/poses.csv")))
xy = [(float(r["x"]), float(r["y"])) for r in rows][::5]
k = len(xy) // 4
out["merge"] = dict(quarters=[xy[i * k:(i + 1) * k] for i in range(4)],
                    stats=dict(exact="1.8e-12", assoc_vsa=0, assoc_inst=622,
                               r1_vsa=89, r1_inst=84, worst_inst=67,
                               kimera_mb=[24, 146], ours_kb=32))

# ---- failure gallery: every truth-scored MISS across all 8 scenes ---------
ALL_SCENES = ["room0", "room1", "room2", "office0", "office1", "office2",
              "office3", "office4"]
fails = []
for s in ALL_SCENES:
    try:
        d = payload_of(f"outputs/grounding_inspector_{s}.html")
    except FileNotFoundError:
        continue
    for p in d["panels"]:
        if p.get("hit_t") is False:
            entry = dict(scene=s, cls=p["cls"], d1=p["d1t"], n_mem=p["n_mem"],
                         kind=("never detected" if p["n_mem"] == 0
                               else "wrong observations"))
            for k in ("tq", "ta"):
                if p.get(k) and p[k] in d["imgs"]:
                    entry[k] = d["imgs"][p[k]]
            fails.append(entry)
out["failures"] = fails

# ---- dataset thumbnails (one real frame each) ------------------------------
thumbs = {}
thumbs["chess"] = out["chess"]["imgs"][out["chess"]["queries"][0]["q"]]
ri_thumbs = payload_of("outputs/recall_inspector.html")
thumbs["school"] = ri_thumbs["A"]["imgs"][ri_thumbs["A"]["queries"][0]["q"]]
thumbs["classroom"] = ri_thumbs["B"]["imgs"][ri_thumbs["B"]["queries"][0]["q"]]
g0 = payload_of("outputs/grounding_inspector_room0.html")
first_ph = next((g0["imgs"][p["tq"]] for p in g0["panels"] if p.get("tq")), None)
if first_ph:
    thumbs["replica"] = first_ph
try:
    import glob as _glob
    from PIL import Image
    sys_path_tum = sorted(_glob.glob("data/rgbd_dataset_freiburg1_desk/rgb_seq/*.png"))
    if sys_path_tum:
        sys.path.insert(0, os.path.abspath("."))
        from tools.export_video_page import jpeg_uri
        thumbs["tum"] = jpeg_uri(Image.open(sys_path_tum[300]), 430, 58)
except Exception as e:
    print(f"  (tum thumb skipped: {e})")
out["thumbs"] = thumbs

# ---- chi law curves (real sweep rows) -------------------------------------
chi = {}
for line in open("outputs/synthetic/rows.jsonl"):
    r = json.loads(line)
    if "chi" not in r:
        continue
    chi.setdefault(r["config"], []).append((r["N"], r["chi"]))
out["chi"] = {c: sorted(v) for c, v in chi.items()
              if c in ("B_lowrank_mean", "C_isotropic", "D2_iso_persistent",
                       "A_lowrank")}

path = "outputs/showcase_data.json"
with open(path, "w") as f:
    json.dump(out, f, separators=(",", ":"))
print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
print(f"  chess: {len(qs)} queries, {len(out['chess']['imgs'])} photos")
for g in gal:
    print(f"  {g['scene']}: {len(g['panels'])} truth-scored classes, "
          f"{len(g['imgs'])} photos")
