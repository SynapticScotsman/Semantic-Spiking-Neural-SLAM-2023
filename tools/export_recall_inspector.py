"""Photo-grounded recall inspector: validate the memory with images, not stats.

Builds one self-contained HTML page (outputs/recall_inspector.html) with two
browsable sections, every query shown as PHOTOS so a human can judge:

  A  school_run1, time recall + two-robot merge: the query photo next to the
     photo at the moment each memory (robot A alone, robot B alone, merged)
     answered. A query from B's era should visibly defeat robot A.
  B  classroom, position recall (bind-yaw): the query photo next to the photo
     of the stored frame nearest the decoded position, with both marked on the
     floor map.

Queries are sampled by a FIXED STRIDE over the seed-0 blocked held-out set â€”
no hand-picking â€” and the rule is printed on the page.

    python tools/export_recall_inspector.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from tools.export_video_page import jpeg_uri  # noqa: E402
from vsa_cognitive_mapping.sequences import load_sequence  # noqa: E402
from vsa_cognitive_mapping.heldout_eval import (  # noqa: E402
    frame_descriptors, treat, make_split)
from vsa_cognitive_mapping.classroom_pipeline import ClassroomEncoders  # noqa: E402
from vsa_cognitive_mapping.vsa import random_project_to_phasor  # noqa: E402

HD, GRID, TL = 4096, 48, 20.0
IMG_W, IMG_Q = 430, 58
MAX_A, MAX_B = 120, 100


def phasors(E, treatment):
    z, _ = random_project_to_phasor(
        torch.from_numpy(np.ascontiguousarray(treat(E, treatment))).float(),
        d=HD, seed=0)
    return z.numpy().astype(np.complex128)


def photo_cache(seq):
    pos = {int(f): i for i, f in enumerate(seq.frame_ids())}
    cache = {}

    def get(fid):
        fid = int(fid)
        key = str(fid)
        if key not in cache:
            cache[key] = jpeg_uri(seq.image(pos[fid]), IMG_W, IMG_Q)
        return key
    return cache, get


# ======================================================================
# Section A â€” school_run1: time recall, two robots into one
# ======================================================================
print("[A] school_run1 ...", flush=True)
seqA = load_sequence("vsa_cognitive_mapping/configs/school_run1_seq.json")
ufA, EA = frame_descriptors("outputs/school_run1", "crop_embeddings_dinov2.pt")
baseA = torch.load(os.path.join("outputs/school_run1", "crop_embeddings.pt"),
                   weights_only=False)
ts_of = {}
for f, t in zip(baseA["frame_idx"].numpy(), baseA["timestamp_ns"].numpy()):
    ts_of.setdefault(int(f), int(t))
tsec = np.array([ts_of[int(f)] for f in ufA], np.float64) / 1e9
tsec -= tsec.min()
span = float(tsec.max())

encA = ClassroomEncoders(HD, 0, 0.75, TL)
pht = np.angle(encA.Bt.values)
T = np.exp(1j * pht[None, :] * (tsec[:, None] / TL))
tg = np.linspace(0, span, 400)
Tg = np.exp(1j * pht[None, :] * (tg[:, None] / TL)).astype(np.complex64)

ZA = phasors(EA, "zscored")
rng = np.random.RandomState(1000)
memA_mask, qryA_mask = make_split(ufA, "blocked", 0.3, 45, rng)
mi = np.where(memA_mask)[0]
half = len(mi) // 2
robotA, robotB = mi[:half], mi[half:]
tA_max = float(tsec[robotA].max())

SA = (ZA[robotA] * T[robotA]).sum(0)
SB = (ZA[robotB] * T[robotB]).sum(0)
M_joint = (SA + SB) / len(mi)
M_A, M_B = SA / len(robotA), SB / len(robotB)
M_merged = (len(robotA) * M_A + len(robotB) * M_B) / len(mi)
exact = float(np.abs(M_merged - M_joint).max())

qiA_all = np.where(qryA_mask)[0]
strideA = max(1, int(np.ceil(len(qiA_all) / MAX_A)))
qiA = qiA_all[::strideA]
print(f"    {len(qiA_all)} held-out queries -> every {strideA}th = {len(qiA)}")


def t_answers(M, qsel):
    Mn = M / max(np.abs(M).max(), 1e-12)
    S = ((Mn[None, :] / ZA[qsel]) @ np.conj(Tg).T).real
    return tg[S.argmax(1)]

tm = t_answers(M_merged, qiA)
tj = t_answers(M_joint, qiA)
ta = t_answers(M_A, qiA)
tb = t_answers(M_B, qiA)
identical = int((tm == tj).sum())

cacheA, getA = photo_cache(seqA)

def nearest_frame(t):
    return int(ufA[int(np.argmin(np.abs(tsec - t)))])

rowsA = []
for k, qi in enumerate(qiA):
    rowsA.append(dict(
        t=round(float(tsec[qi]), 1),
        tm=round(float(tm[k]), 1), ta=round(float(ta[k]), 1),
        tb=round(float(tb[k]), 1),
        em=round(float(abs(tm[k] - tsec[qi])), 1),
        q=getA(ufA[qi]), im=getA(nearest_frame(tm[k])),
        ia=getA(nearest_frame(ta[k])), ib=getA(nearest_frame(tb[k]))))
    if k % 20 == 0:
        print(f"    photos {k}/{len(qiA)}", flush=True)

secA = dict(span=round(span, 1), tA_max=round(tA_max, 1),
            nA=len(robotA), nB=len(robotB),
            n_all=int(len(qiA_all)), stride=strideA,
            exact=exact, identical=identical, n_batt=len(qiA),
            queries=rowsA, imgs=cacheA)

# ======================================================================
# Section B â€” classroom: position recall on the PLACE frontend
# (EigenPlaces, z-scored, NO yaw â€” the measured-best config, tracker (at))
# ======================================================================
B_ENCODER = "crop_embeddings_eigenplaces.pt"
B_TREATMENT = "zscored"
print(f"[B] classroom ({B_ENCODER}, {B_TREATMENT}, no yaw) ...", flush=True)
seqB = load_sequence("vsa_cognitive_mapping/configs/classroom_seq.json")
px, py, pyaw = seqB.poses()
pose_of = {int(f): (px[i], py[i], pyaw[i]) for i, f in enumerate(seqB.frame_ids())}
ufB, EB = frame_descriptors("outputs/classroom", B_ENCODER)
keep = np.array([int(f) in pose_of for f in ufB])
ufB, EB = ufB[keep], EB[keep]
XY = np.array([pose_of[int(f)][:2] for f in ufB])
EBt = treat(EB, B_TREATMENT)

ZB = phasors(EB, B_TREATMENT)
rng = np.random.RandomState(1000)
memB_mask, qryB_mask = make_split(ufB, "blocked", 0.3, 45, rng)
mb = np.where(memB_mask)[0]

encB = ClassroomEncoders(HD, 0, 0.75, TL)
ext = [XY[:, 0].min() - 1, XY[:, 0].max() + 1, XY[:, 1].min() - 1, XY[:, 1].max() + 1]
gx = np.linspace(ext[0], ext[1], GRID)
gy = np.linspace(ext[2], ext[3], GRID)
G = np.empty((GRID * GRID, HD), np.complex64)
k = 0
for yy in gy:
    for xx in gx:
        G[k] = encB.ctx_pos(float(xx), float(yy)).values.astype(np.complex64); k += 1
P = np.stack([encB.ctx_pos(float(a), float(b)).values for a, b in XY[mb]])
M = (ZB[mb] * P).mean(0)
M /= max(np.abs(M).max(), 1e-12)

qiB_all = np.where(qryB_mask)[0]
strideB = max(1, int(np.ceil(len(qiB_all) / MAX_B)))
qiB = qiB_all[::strideB]
print(f"    {len(qiB_all)} held-out queries -> every {strideB}th = {len(qiB)}")

R = M[None, :] / ZB[qiB]
S = (R @ np.conj(G).T).real
j = S.argmax(1)
peak = np.stack([gx[j % GRID], gy[j // GRID]], 1)
err = np.hypot(*(peak - XY[qiB]).T)
# kNN ceiling on the same battery, for the CDF header
U = EBt / (np.linalg.norm(EBt, axis=1, keepdims=True) + 1e-12)
nnB = mb[(U[qiB] @ U[mb].T).argmax(1)]
err_knn = np.hypot(*(XY[nnB] - XY[qiB]).T)


def cdf(e):
    return dict(med=round(float(np.median(e)), 2),
                le05=round(float(np.mean(e <= 0.5)), 3),
                le1=round(float(np.mean(e <= 1)), 3),
                le2=round(float(np.mean(e <= 2)), 3),
                worst=round(float(e.max()), 1))


print(f"    median err this battery: {np.median(err):.3f} m "
      f"(kNN ceiling {np.median(err_knn):.3f} m)")

cacheB, getB = photo_cache(seqB)
rowsB = []
for k2, qi in enumerate(qiB):
    d = np.hypot(*(XY[mb] - peak[k2]).T)
    near = mb[int(np.argmin(d))]                     # stored frame at the answer
    rowsB.append(dict(
        x=round(float(XY[qi, 0]), 2), y=round(float(XY[qi, 1]), 2),
        px=round(float(peak[k2, 0]), 2), py=round(float(peak[k2, 1]), 2),
        err=round(float(err[k2]), 2),
        q=getB(ufB[qi]), near=getB(ufB[near])))
    if k2 % 20 == 0:
        print(f"    photos {k2}/{len(qiB)}", flush=True)

step = max(1, len(XY) // 500)
secB = dict(ext=[round(float(v), 2) for v in ext],
            path=[[round(float(a), 2), round(float(b), 2)]
                  for a, b in XY[::step]],
            stored=[[round(float(a), 2), round(float(b), 2)]
                    for a, b in XY[mb][::max(1, len(mb) // 350)]],
            n_all=int(len(qiB_all)), stride=strideB, n_batt=len(qiB),
            med=round(float(np.median(err)), 3),
            cdf_vsa=cdf(err), cdf_knn=cdf(err_knn),
            queries=rowsB, imgs=cacheB)

# ======================================================================
# Section C â€” classroom: object retrieval on the CROP frontend + the 2x2
# ======================================================================
print("[C] classroom object retrieval (crop frontend) ...", flush=True)
import csv  # noqa: E402
from vsa_cognitive_mapping.isotropy_ladder import top1_retrieval, chance  # noqa: E402

crop_meta = torch.load("outputs/classroom/crop_embeddings.pt", weights_only=False)
cls_id = crop_meta["class_id"].numpy()
cfr = crop_meta["frame_idx"].numpy()
det_id = crop_meta["det_id"].numpy()
Ecrop = torch.load("outputs/classroom/crop_embeddings_dinov2.pt",
                   weights_only=False)["embedding"].numpy().astype(np.float64)

box_of, name_of = {}, {}
with open("outputs/classroom/detections_crops.csv", newline="") as f:
    for r in csv.DictReader(f):
        box_of[int(r["det_id"])] = tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2"))
        name_of[int(r["det_id"])] = r["class_name"]

# per-crop place vectors = the EigenPlaces vector of the crop's frame
ep = torch.load("outputs/classroom/crop_embeddings_eigenplaces.pt", weights_only=False)
ep_pos = {int(f): i for i, f in enumerate(ep["frame_idx"].numpy())}
sel = np.array([ep_pos.get(int(f), -1) for f in cfr])
okC = sel >= 0
clsC, cfrC, EcropC = cls_id[okC], cfr[okC], Ecrop[okC]
EplaceC = ep["embedding"].numpy().astype(np.float64)[sel[okC]]
det_idC = det_id[okC]

GAPC = 300
acc_crops = float(top1_retrieval(EcropC, clsC, cfrC, GAPC))
acc_place = float(top1_retrieval(EplaceC, clsC, cfrC, GAPC))
ch = float(chance(clsC))
print(f"    2x2 class task (gap {GAPC}): crops {acc_crops:.3f}, "
      f"place vectors {acc_place:.3f}, chance {ch:.3f}")

# strided demo pairs: query crop -> nearest crop >GAPC frames away (dinov2)
Un = EcropC / (np.linalg.norm(EcropC, axis=1, keepdims=True) + 1e-12)
qselC = np.linspace(0, len(clsC) - 1, 16).astype(int)
posB = {int(f): i for i, f in enumerate(seqB.frame_ids())}
cacheC = {}


def crop_uri(ci):
    key = f"c{ci}"
    if key not in cacheC:
        img = seqB.image(posB[int(cfrC[ci])])
        x1, y1, x2, y2 = box_of[int(det_idC[ci])]
        padx, pady = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
        img = img.crop((max(0, x1 - padx), max(0, y1 - pady),
                        min(img.size[0], x2 + padx), min(img.size[1], y2 + pady)))
        cacheC[key] = jpeg_uri(img, 220, 58)
    return key


rowsC = []
for ci in qselC:
    s = Un @ Un[ci]
    s[np.abs(cfrC - cfrC[ci]) <= GAPC] = -np.inf
    ri = int(np.argmax(s))
    rowsC.append(dict(q=crop_uri(ci), r=crop_uri(ri),
                      qc=name_of[int(det_idC[ci])], rc=name_of[int(det_idC[ri])],
                      qf=int(cfrC[ci]), rf=int(cfrC[ri]),
                      ok=bool(clsC[ci] == clsC[ri])))
n_okC = sum(r["ok"] for r in rowsC)
print(f"    demo pairs: {n_okC}/{len(rowsC)} class-correct")

# object-level PLACE cell: the measured dinov2-crops recall (tracker (as))
with open("outputs/classroom/recall_cdf_crop_embeddings_dinov2_zscored_yaw.json") as f:
    dino_place = json.load(f)

secC = dict(gap=GAPC, chance=round(ch, 3), acc_crops=round(acc_crops, 3),
            acc_place=round(acc_place, 3), n_crops=int(len(clsC)),
            place_crops_med=round(dino_place["vsa"]["median"], 2),
            place_vpr_med=round(float(np.median(err)), 2),
            pairs=rowsC, imgs=cacheC)

payload = json.dumps(dict(A=secA, B=secB, C=secC),
                     separators=(",", ":")).replace("</", "<\\/")

# ======================================================================
# The page
# ======================================================================
HTML = """<meta charset="utf-8">
<title>Recall inspector â€” validate the memory with photos</title>
<style>
:root{--bg:#fbfbfc;--bg2:#fff;--panel:#eef0f4;--card:#fff;--ink:#14171c;--ink2:#39404b;
--muted:#646c78;--line:#dce0e7;--line2:#c6ccd6;--accent:#2a5fd9;--accent-soft:#e6ecfb;
--good:#0f7a4b;--bad:#b02a24;--warn:#9a5c05;
--shadow:0 1px 2px rgba(16,20,28,.06),0 6px 22px rgba(16,20,28,.07)}
@media (prefers-color-scheme:dark){:root{--bg:#0e1116;--bg2:#141922;--panel:#1a212c;
--card:#151b24;--ink:#eef2f8;--ink2:#c3ccd9;--muted:#8b95a4;--line:#26303d;--line2:#33404f;
--accent:#79a6ff;--accent-soft:#18243a;--good:#54d39a;--bad:#ff8078;--warn:#f0b24a;
--shadow:0 1px 2px rgba(0,0,0,.4),0 8px 26px rgba(0,0,0,.34)}}
:root[data-theme="dark"]{--bg:#0e1116;--bg2:#141922;--panel:#1a212c;--card:#151b24;
--ink:#eef2f8;--ink2:#c3ccd9;--muted:#8b95a4;--line:#26303d;--line2:#33404f;--accent:#79a6ff;
--accent-soft:#18243a;--good:#54d39a;--bad:#ff8078;--warn:#f0b24a;
--shadow:0 1px 2px rgba(0,0,0,.4),0 8px 26px rgba(0,0,0,.34)}
:root[data-theme="light"]{--bg:#fbfbfc;--bg2:#fff;--panel:#eef0f4;--card:#fff;--ink:#14171c;
--ink2:#39404b;--muted:#646c78;--line:#dce0e7;--line2:#c6ccd6;--accent:#2a5fd9;
--accent-soft:#e6ecfb;--good:#0f7a4b;--bad:#b02a24;--warn:#9a5c05;
--shadow:0 1px 2px rgba(16,20,28,.06),0 6px 22px rgba(16,20,28,.07)}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.6 ui-sans-serif,system-ui,"Segoe UI",Roboto,Arial,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:0 20px}
code{background:var(--panel);border-radius:5px;padding:1px 5px;font-size:.87em;
font-family:ui-monospace,Menlo,Consolas,monospace;overflow-wrap:anywhere}
header{background:var(--bg2);border-bottom:1px solid var(--line);padding:28px 0 24px}
.topbar{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;margin-bottom:16px}
.kicker{font-size:.73rem;letter-spacing:.15em;text-transform:uppercase;color:var(--accent);font-weight:700}
.btn{background:var(--card);color:var(--ink);border:1px solid var(--line2);border-radius:8px;
padding:7px 13px;font-size:.85rem;cursor:pointer;font-family:inherit}
.btn:hover{border-color:var(--accent)}
h1{font-size:clamp(1.4rem,3.6vw,2.1rem);margin:0 0 10px;letter-spacing:-.02em;text-wrap:balance}
.lede{color:var(--ink2);max-width:78ch;margin:0;font-size:1rem}
section{padding:36px 0;border-bottom:1px solid var(--line)}
.eyebrow{font-size:.72rem;letter-spacing:.15em;text-transform:uppercase;color:var(--accent);font-weight:700;margin:0 0 8px}
h2{font-size:clamp(1.2rem,2.5vw,1.6rem);margin:0 0 10px;letter-spacing:-.015em}
.prose{max-width:78ch;color:var(--ink2)}
.card{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:16px;
box-shadow:var(--shadow);margin:14px 0}
.ctrl{display:flex;gap:10px;align-items:center;margin:10px 0;flex-wrap:wrap}
input[type=range]{flex:1;min-width:160px;accent-color:var(--accent)}
.photos{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.ph{text-align:center}
.ph img{width:100%;border-radius:9px;border:2px solid var(--line2);display:block}
.ph.q img{border-color:var(--accent)}
.ph.good img{border-color:var(--good)}
.ph b{display:block;font-size:.82rem;margin-top:6px}
.ph span{font-size:.75rem;color:var(--muted)}
canvas{width:100%;height:auto;display:block;border-radius:9px;background:var(--panel);
border:1px solid var(--line)}
.note{font-size:.85rem;color:var(--muted)}
.statgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:16px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 13px}
.stat b{display:block;font-size:1.2rem}
.stat span{font-size:.76rem;color:var(--muted)}
footer{padding:26px 0 56px;color:var(--muted);font-size:.85rem}
kbd{background:var(--panel);border:1px solid var(--line2);border-bottom-width:2px;
border-radius:5px;padding:0 5px;font-size:.8em;font-family:ui-monospace,Menlo,monospace}
</style>
<header><div class="wrap">
<div class="topbar"><span class="kicker">recall inspector Â· photos, not statistics</span>
<button id="themeBtn" class="btn" type="button">Dark / light</button></div>
<h1>Judge the memory yourself: the query photo vs the photo at its answer</h1>
<p class="lede">Every query below is a view the memory <b>never stored</b>. The memory answers
with a time (school run) or a place (classroom); we then show you the <b>photo captured at that
answer</b>. If the two photos show the same scene, the recall is right â€” no statistics required.
One memory architecture, <b>two content granularities</b>: a place descriptor (EigenPlaces)
answers <i>where</i>, object crops answer <i>what</i> â€” each matched to its query, both bindable
into the same vector memory. Queries are sampled by a fixed stride over the held-out set,
stated per section: <b>no hand-picking</b>.</p>
<div class="statgrid" id="hdrStats"></div>
</div></header>
<main>
<section id="secA"><div class="wrap">
<p class="eyebrow">A Â· school run â€” two robots, one memory (time recall)</p>
<h2>Robot A saw the first half, robot B the second. The merged memory saw both.</h2>
<p class="prose">Ask each memory <em>â€œwhen was this view seen?â€</em> A query from robot B's half
should visibly defeat robot A â€” its answer-photo will show the wrong part of the school â€”
while the <b>merged</b> memory (one vector addition) answers like a robot that walked the
whole route.</p>
<div class="card">
<div class="ctrl">
<button id="aPrev" class="btn">&#8592;</button>
<input id="aSlider" type="range" min="0" max="0" value="0" step="1">
<button id="aNext" class="btn">&#8594;</button>
<span class="note" id="aLabel" style="min-width:14ch"></span>
<button id="aWorst" class="btn">Worst &#8594;</button>
</div>
<canvas id="aTimeline" height="120"></canvas>
<div class="photos" style="margin-top:12px">
<div class="ph q"><img id="aQ"><b>query view</b><span id="aQt"></span></div>
<div class="ph good"><img id="aM"><b>merged memory's answer</b><span id="aMt"></span></div>
<div class="ph"><img id="aA"><b>robot A alone</b><span id="aAt"></span></div>
<div class="ph"><img id="aB"><b>robot B alone</b><span id="aBt"></span></div>
</div>
<p class="note" id="aNote" style="margin:10px 0 0"></p>
</div>
</div></section>
<section id="secB"><div class="wrap">
<p class="eyebrow">B Â· classroom â€” where on the map (place frontend: EigenPlaces, z-scored)</p>
<h2>The memory points at a spot on the floor. Here is what that spot looks like.</h2>
<p class="prose">For each held-out view, the memory decodes a position (red cross on the map;
green ring is the truth). Beside the query photo is the <b>stored frame nearest the decoded
position</b> â€” â€œwhat the memory thinks this place looks like.â€ Same scene in both photos =
correct recall, whatever the metre number says. Content here is a <b>purpose-trained place
descriptor</b> (EigenPlaces) rather than pooled object crops â€” the frontend swap that took the
battery median from 0.96&nbsp;m to the value in the header, with no yaw factor needed.</p>
<div class="card">
<div class="ctrl">
<button id="bPrev" class="btn">&#8592;</button>
<input id="bSlider" type="range" min="0" max="0" value="0" step="1">
<button id="bNext" class="btn">&#8594;</button>
<span class="note" id="bLabel" style="min-width:14ch"></span>
<button id="bWorst" class="btn">Worst &#8594;</button>
</div>
<div style="display:grid;grid-template-columns:1.2fr 1fr 1fr;gap:12px;align-items:start">
<div><canvas id="bMap" height="300"></canvas>
<p class="note" style="margin:6px 0 0">grey = walked path Â· dots = stored frames Â·
<span style="color:var(--good)">ring = truth</span> Â·
<span style="color:var(--bad)">cross = memory's answer</span></p></div>
<div class="ph q"><img id="bQ"><b>query view</b><span id="bQt"></span></div>
<div class="ph good"><img id="bN"><b>stored frame at the answer</b><span id="bNt"></span></div>
</div>
<p class="note" id="bNote" style="margin:10px 0 0"></p>
</div>
</div></section>
<section id="secC"><div class="wrap">
<p class="eyebrow">C Â· classroom â€” what is this object (crop frontend: DINOv2 crops)</p>
<h2>Same memory algebra, object granularity: find this object again, far away in time.</h2>
<p class="prose">Each query is a detection crop; the answer is the most similar crop from
<b>more than __GAPC__ frames away</b> (â‰ˆ10&nbsp;s, a different pass through the room). This is the
object frontend the place descriptor can't replace â€” and vice versa. The 2Ã—2 below is the
<b>double dissociation</b>, all four cells measured: each representation wins its own task and
loses the other's.</p>
<div class="card" style="overflow-x:auto">
<table id="ddTable" style="border-collapse:collapse;min-width:540px;font-size:.9rem">
<thead><tr><th style="text-align:left;padding:7px 12px"></th>
<th style="text-align:left;padding:7px 12px">object task<br><span class="note">class retrieval, gap __GAPC__</span></th>
<th style="text-align:left;padding:7px 12px">place task<br><span class="note">held-out recall, median</span></th></tr></thead>
<tbody id="ddBody"></tbody>
</table>
<p class="note" style="margin:8px 0 0" id="ddNote"></p>
</div>
<div class="card">
<div class="ctrl">
<button id="cPrev" class="btn">&#8592;</button>
<input id="cSlider" type="range" min="0" max="0" value="0" step="1">
<button id="cNext" class="btn">&#8594;</button>
<span class="note" id="cLabel" style="min-width:14ch"></span>
<button id="cWorst" class="btn">First miss &#8594;</button>
</div>
<div class="photos" style="grid-template-columns:repeat(auto-fit,minmax(180px,1fr));max-width:520px">
<div class="ph q"><img id="cQ"><b>query crop</b><span id="cQt"></span></div>
<div class="ph" id="cRph"><img id="cR"><b>retrieved crop</b><span id="cRt"></span></div>
</div>
<p class="note" id="cNote" style="margin:10px 0 0"></p>
</div>
</div></section>
</main>
<footer><div class="wrap"><p id="foot"></p></div></footer>
<script id="data" type="application/json">__PAYLOAD__</script>
<script>
(function(){"use strict";
var D=JSON.parse(document.getElementById("data").textContent);
var $=function(s){return document.querySelector(s)};
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim()}
/* header stats â€” absolute CDF numbers lead (standing rule) */
function pc(v){return Math.round(v*100)+"%"}
$("#hdrStats").innerHTML=
 "<div class='stat'><b>"+D.B.cdf_vsa.med+" m</b><span>classroom VSA median Â· "+
 pc(D.B.cdf_vsa.le05)+" &le;0.5&nbsp;m Â· "+pc(D.B.cdf_vsa.le2)+
 " &le;2&nbsp;m (EigenPlaces, z-scored, no yaw)</span></div>"+
 "<div class='stat'><b>"+D.B.cdf_knn.med+" m</b><span>kNN ceiling on the SAME descriptors Â· "+
 pc(D.B.cdf_knn.le05)+" &le;0.5&nbsp;m â€” the memory nearly matches exact search</span></div>"+
 "<div class='stat'><b>"+D.C.acc_crops.toFixed(2)+" vs "+D.C.acc_place.toFixed(2)+
 "</b><span>object task: crops vs place vectors (chance "+D.C.chance.toFixed(2)+
 ") â€” the double dissociation's other diagonal</span></div>"+
 "<div class='stat'><b>"+D.A.exact.toExponential(1)+"</b><span>two-robot merge: max |merged âˆ’ joint| over 4096 dims; "+
 D.A.identical+"/"+D.A.n_batt+" identical answers</span></div>";
/* ---------- section A ---------- */
var ai=0,QA=D.A.queries;
function fit(cv,h){var dpr=Math.min(devicePixelRatio||1,2),w=cv.clientWidth||600;
cv.width=w*dpr;cv.height=h*dpr;var g=cv.getContext("2d");g.setTransform(dpr,0,0,dpr,0,0);
return{g:g,w:w,h:h}}
function drawTimeline(){var q=QA[ai],c=fit($("#aTimeline"),120),g=c.g;
g.clearRect(0,0,c.w,c.h);var L=8,R=8,W=c.w-L-R,y=54,X=function(t){return L+t/D.A.span*W};
g.fillStyle=css("--accent");g.globalAlpha=.13;g.fillRect(X(0),y-20,X(D.A.tA_max)-X(0),40);
g.fillStyle=css("--warn");g.fillRect(X(D.A.tA_max),y-20,X(D.A.span)-X(D.A.tA_max),40);
g.globalAlpha=1;g.strokeStyle=css("--line2");g.strokeRect(L,y-20,W,40);
g.font="11px ui-monospace,monospace";g.fillStyle=css("--muted");
g.fillText("robot A's half (0-"+D.A.tA_max+" s)",X(10),y-26);
g.fillText("robot B's half",X(D.A.tA_max+10),y-26);
function mark(t,col,lbl,dy){var x=X(t);g.strokeStyle=col;g.lineWidth=2;
g.beginPath();g.moveTo(x,y-20);g.lineTo(x,y+20);g.stroke();
g.fillStyle=col;g.fillText(lbl,Math.min(x+3,c.w-70),y+dy)}
mark(q.ta,css("--accent"),"A: "+q.ta+"s",34);
mark(q.tb,css("--warn"),"B: "+q.tb+"s",46);
mark(q.tm,css("--bad"),"merged: "+q.tm+"s",-34);
mark(q.t,css("--good"),"truth: "+q.t+"s",-46)}
function renderA(){var q=QA[ai];
$("#aSlider").value=ai;$("#aLabel").textContent="query "+(ai+1)+" / "+QA.length;
$("#aQ").src=D.A.imgs[q.q];$("#aM").src=D.A.imgs[q.im];
$("#aA").src=D.A.imgs[q.ia];$("#aB").src=D.A.imgs[q.ib];
$("#aQt").textContent="captured at t = "+q.t+" s";
$("#aMt").textContent="answered t = "+q.tm+" s  (err "+q.em+" s)";
$("#aAt").textContent="answered t = "+q.ta+" s";
$("#aBt").textContent="answered t = "+q.tb+" s";
var era=q.t<=D.A.tA_max?"A":"B";
$("#aNote").textContent="This query is from robot "+era+"'s half. Robot "+
(era==="A"?"B":"A")+" never saw this part of the walk - compare its answer-photo against the query.";
drawTimeline()}
/* ---------- section B ---------- */
var bi=0,QB=D.B.queries,ex=D.B.ext;
function drawMap(){var q=QB[bi],c=fit($("#bMap"),300),g=c.g;
g.clearRect(0,0,c.w,c.h);g.fillStyle=css("--panel");g.fillRect(0,0,c.w,c.h);
var pad=12,sx=(c.w-2*pad)/(ex[1]-ex[0]),sy=(c.h-2*pad)/(ex[3]-ex[2]),s=Math.min(sx,sy);
var P=function(x,y){return[pad+(x-ex[0])*s,c.h-(pad+(y-ex[2])*s)]};
g.strokeStyle=css("--line2");g.lineWidth=1.4;g.beginPath();
D.B.path.forEach(function(p,i){var t=P(p[0],p[1]);i?g.lineTo(t[0],t[1]):g.moveTo(t[0],t[1])});
g.stroke();
g.fillStyle=css("--accent");g.globalAlpha=.45;
D.B.stored.forEach(function(p){var t=P(p[0],p[1]);g.fillRect(t[0]-1.5,t[1]-1.5,3,3)});
g.globalAlpha=1;
var tt=P(q.x,q.y),pp=P(q.px,q.py);
g.strokeStyle=css("--bad");g.lineWidth=1.2;g.beginPath();g.moveTo(tt[0],tt[1]);
g.lineTo(pp[0],pp[1]);g.stroke();
g.strokeStyle=css("--good");g.lineWidth=2.6;g.beginPath();g.arc(tt[0],tt[1],8,0,6.2832);g.stroke();
g.strokeStyle=css("--bad");g.lineWidth=2.6;g.beginPath();
g.moveTo(pp[0]-7,pp[1]-7);g.lineTo(pp[0]+7,pp[1]+7);
g.moveTo(pp[0]+7,pp[1]-7);g.lineTo(pp[0]-7,pp[1]+7);g.stroke()}
function renderB(){var q=QB[bi];
$("#bSlider").value=bi;$("#bLabel").textContent="query "+(bi+1)+" / "+QB.length;
$("#bQ").src=D.B.imgs[q.q];$("#bN").src=D.B.imgs[q.near];
$("#bQt").textContent="true position "+q.x+", "+q.y+" m";
$("#bNt").textContent="decoded "+q.px+", "+q.py+" m  (err "+q.err+" m)";
$("#bNote").textContent="Battery median error "+D.B.med+
" m. If these two photos show the same corner of the room, the recall is right regardless of the number.";
drawMap()}
/* ---------- section C ---------- */
var ci=0,QC=D.C.pairs;
$("#ddBody").innerHTML=
 "<tr><td style='padding:7px 12px'><b>object-level</b> (DINOv2 crops)</td>"+
 "<td style='padding:7px 12px;color:var(--good)'><b>"+D.C.acc_crops.toFixed(3)+"</b> &#10003;</td>"+
 "<td style='padding:7px 12px'>"+D.C.place_crops_med.toFixed(2)+" m</td></tr>"+
 "<tr><td style='padding:7px 12px'><b>place-level</b> (EigenPlaces)</td>"+
 "<td style='padding:7px 12px'>"+D.C.acc_place.toFixed(3)+"</td>"+
 "<td style='padding:7px 12px;color:var(--good)'><b>"+D.C.place_vpr_med.toFixed(2)+" m</b> &#10003;</td></tr>";
$("#ddNote").textContent="Top-1 class retrieval over "+D.C.n_crops+
 " crops (chance "+D.C.chance.toFixed(3)+"); place task is the held-out battery median. "+
 "The diagonal wins both ways: representation granularity must match query granularity.";
function renderC(){var q=QC[ci];
$("#cSlider").value=ci;$("#cLabel").textContent="pair "+(ci+1)+" / "+QC.length;
$("#cQ").src=D.C.imgs[q.q];$("#cR").src=D.C.imgs[q.r];
$("#cQt").textContent="detected as â€œ"+q.qc+"â€, frame "+q.qf;
$("#cRt").textContent="â€œ"+q.rc+"â€, frame "+q.rf+" ("+Math.abs(q.rf-q.qf)+" frames away)";
$("#cRph").className="ph "+(q.ok?"good":"");
$("#cNote").textContent=q.ok?
 "Class match âœ“ - the crop frontend re-finds this object on a different pass.":
 "Class mismatch - retrieved a â€œ"+q.rc+"â€ for a â€œ"+q.qc+"â€ query.";}
/* transports */
function wire(slider,prev,next,worst,get,set,len,err,render){
slider.max=len-1;
slider.addEventListener("input",function(){set(parseInt(slider.value,10));render()});
prev.addEventListener("click",function(){set((get()-1+len)%len);render()});
next.addEventListener("click",function(){set((get()+1)%len);render()});
worst.addEventListener("click",function(){var w=0,v=-1;
for(var i=0;i<len;i++){if(err(i)>v){v=err(i);w=i}}set(w);render()})}
wire($("#aSlider"),$("#aPrev"),$("#aNext"),$("#aWorst"),
 function(){return ai},function(v){ai=v},QA.length,function(i){return QA[i].em},renderA);
wire($("#bSlider"),$("#bPrev"),$("#bNext"),$("#bWorst"),
 function(){return bi},function(v){bi=v},QB.length,function(i){return QB[i].err},renderB);
wire($("#cSlider"),$("#cPrev"),$("#cNext"),$("#cWorst"),
 function(){return ci},function(v){ci=v},QC.length,function(i){return QC[i].ok?0:1},renderC);
document.addEventListener("keydown",function(e){
if(/^(INPUT|BUTTON)$/.test((e.target||{}).tagName))return;
var inC=$("#secC").getBoundingClientRect().top<innerHeight*.5;
var inB=!inC&&$("#secB").getBoundingClientRect().top<innerHeight*.5;
if(e.key==="ArrowRight"){e.preventDefault();
 if(inC){ci=(ci+1)%QC.length;renderC()}
 else if(inB){bi=(bi+1)%QB.length;renderB()}else{ai=(ai+1)%QA.length;renderA()}}
if(e.key==="ArrowLeft"){e.preventDefault();
 if(inC){ci=(ci-1+QC.length)%QC.length;renderC()}
 else if(inB){bi=(bi-1+QB.length)%QB.length;renderB()}else{ai=(ai-1+QA.length)%QA.length;renderA()}}});
$("#foot").innerHTML="Sampling: section A takes every "+D.A.stride+
"th of the "+D.A.n_all+" seed-0 blocked held-out frames on school_run1; section B every "+
D.B.stride+"th of "+D.B.n_all+" on the classroom; section C every "+
Math.round(D.C.n_crops/QC.length)+"th crop - deterministic, no curation. hd 4096, blocked "+
"splits (gap 45), time grid 400 cells / position grid 48x48. Frontends: EigenPlaces "+
"(place, section B), DINOv2 crops (objects, section C), z-scored dinov2 (school, section A). "+
"Rebuild: <code>python tools/export_recall_inspector.py</code>.";
$("#themeBtn").addEventListener("click",function(){
var cur=document.documentElement.getAttribute("data-theme");
if(!cur)cur=matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";
document.documentElement.setAttribute("data-theme",cur==="dark"?"light":"dark");
renderA();renderB()});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change",function(){renderA();renderB()});
renderA();renderB();renderC();
addEventListener("resize",function(){drawTimeline();drawMap()});
})();
</script>
"""

out = os.path.join("outputs", "recall_inspector.html")
with open(out, "w", encoding="utf-8", newline="") as f:
    f.write(HTML.replace("__PAYLOAD__", payload).replace("__GAPC__", str(GAPC)))
mb = os.path.getsize(out) / 1e6
print(f"\nwrote {out}  ({mb:.1f} MB)")
print(f"A: {len(rowsA)} queries, {len(cacheA)} unique photos; "
      f"B: {len(rowsB)} queries, {len(cacheB)} unique photos")
print(f"merge exactness {exact:.2e}; merged==joint answers {identical}/{len(qiA)}")

