"""Photo-grounded inspector for the object-grounding battery (Replica room0).

Standing rule: the quantitative table never ships alone. Per class query this
page shows: the floor map (camera path, target instances as rings, the VSA's
top-3 answer peaks as crosses), a crop PHOTO of the target object from the
held-out session, and the crop photo of the MEMORY detection nearest the
answer peak — "what the memory grounded there." Same battery, same settings,
recomputed at build time; deterministic (all classes shown, no curation).

    python tools/export_grounding_inspector.py \
        --dataset vsa_cognitive_mapping/configs/replica_room0.json \
        --max-per-class 60 --out outputs/grounding_inspector_room0.html
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from tools.export_video_page import jpeg_uri  # noqa: E402
from vsa_cognitive_mapping.sequences import load_sequence  # noqa: E402
from vsa_cognitive_mapping.classroom_pipeline import ClassroomEncoders  # noqa: E402
from vsa_cognitive_mapping.object_grounding import (  # noqa: E402
    class_phasors, build_trace, field_peaks, clusters_of, cap_per_class,
    REPLICA2COCO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--cluster-radius", type=float, default=0.6)
    ap.add_argument("--min-count", type=int, default=4)
    ap.add_argument("--hd-dim", type=int, default=4096)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--length-scale", type=float, default=0.6)
    ap.add_argument("--room-margin", type=float, default=1.5)
    ap.add_argument("--max-per-class", type=int, default=60)
    ap.add_argument("--gt-json", default=None,
                    help="real instance centroids; adds the TRUTH verdict")
    ap.add_argument("--img-w", type=int, default=240)
    ap.add_argument("--img-q", type=int, default=55)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = json.load(open(args.dataset))
    out_dir = cfg["out_dir"]
    seq = load_sequence(args.dataset)
    px, py, _ = seq.poses()

    pts = json.load(open(os.path.join(out_dir, "object_points.json")))["points"]
    lo_x, hi_x = px.min() - args.room_margin, px.max() + args.room_margin
    lo_y, hi_y = py.min() - args.room_margin, py.max() + args.room_margin
    pts = [p for p in pts if lo_x <= p["x"] <= hi_x and lo_y <= p["y"] <= hi_y]

    # per-detection metadata for photos (det order == csv order)
    det_rows = list(csv.DictReader(
        open(os.path.join(out_dir, "detections_crops.csv"), newline="")))
    # object_points rows carry frame + cls; rebuild the det_id join by
    # sweeping the csv in order and matching (frame, cls) sequence
    it = iter(range(len(det_rows)))
    for p in pts:
        p.setdefault("det", None)
    # simpler robust join: index csv rows by (frame, cls) queues
    from collections import defaultdict, deque
    q = defaultdict(deque)
    for i, r in enumerate(det_rows):
        q[(int(r["frame_idx"]), r["class_name"])].append(i)
    for p in pts:
        key = (p["frame"], p["cls"])
        p["det"] = q[key].popleft() if q[key] else None

    frames = sorted({p["frame"] for p in pts})
    half = frames[len(frames) // 2]
    sess_a = [p for p in pts if p["frame"] <= half]
    sess_b = [p for p in pts if p["frame"] > half]
    gt = clusters_of(sess_b, args.cluster_radius, args.min_count)
    mem_pts = cap_per_class(sess_a, args.max_per_class)

    # real GT (truth verdict), mapped into the COCO vocabulary
    truth_by = {}
    if args.gt_json:
        raw = json.load(open(args.gt_json))
        for g in raw["instances"] if isinstance(raw, dict) else raw:
            name = REPLICA2COCO.get(g.get("cls") or g.get("class"))
            if name:
                truth_by.setdefault(name, []).append(
                    (float(g["x"]), float(g["y"])))

    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    ext = [min(xs) - 1, max(xs) + 1, min(ys) - 1, max(ys) + 1]
    enc = ClassroomEncoders(args.hd_dim, 0, args.length_scale, 20.0)
    gx = np.linspace(ext[0], ext[1], args.grid)
    gy = np.linspace(ext[2], ext[3], args.grid)
    G = np.empty((args.grid ** 2, args.hd_dim), np.complex64)
    k = 0
    for yy in gy:
        for xx in gx:
            G[k] = enc.ctx_pos(float(xx), float(yy)).values.astype(np.complex64)
            k += 1
    classes = sorted({p["cls"] for p in pts})
    sem = class_phasors(classes, args.hd_dim)
    M = build_trace(mem_pts, enc, sem, args.hd_dim)
    M /= max(np.abs(M).max(), 1e-12)

    gt_by = {}
    for g in gt:
        gt_by.setdefault(g["class"], []).append(g)

    pos_map = {int(f): i for i, f in enumerate(seq.frame_ids())}
    cache = {}

    def crop_photo(p):
        if p is None or p["det"] is None:
            return None
        r = det_rows[p["det"]]
        key = f"d{p['det']}"
        if key not in cache:
            img = seq.image(pos_map[int(r["frame_idx"])])
            x1, y1 = float(r["x1"]), float(r["y1"])
            x2, y2 = float(r["x2"]), float(r["y2"])
            padx, pady = 0.15 * (x2 - x1), 0.15 * (y2 - y1)
            img = img.crop((max(0, x1 - padx), max(0, y1 - pady),
                            min(img.size[0], x2 + padx),
                            min(img.size[1], y2 + pady)))
            cache[key] = jpeg_uri(img, args.img_w, args.img_q)
        return key

    mem_by = {}
    for p in mem_pts:
        mem_by.setdefault(p["cls"], []).append(p)

    panels = []
    for cname in sorted(gt_by):
        targets = np.array([[g["x"], g["y"]] for g in gt_by[cname]])
        F = ((M / sem[cname])[None, :] @ np.conj(G).T).real[0]
        peaks = field_peaks(F, gx, gy, args.grid)
        # the full similarity FIELD — the memory's belief map for this
        # concept, quantised to uint8 for the page (argmax is just a readout)
        Fq = np.clip((F - F.min()) / max(F.max() - F.min(), 1e-12), 0, 1)
        field = (Fq * 255).astype(np.uint8).tolist()
        d1 = float(np.min(np.hypot(*(targets - np.array(peaks[0])).T)))
        # target photo: highest-confidence session-B detection near a target
        cand = [p for p in sess_b if p["cls"] == cname and p["det"] is not None]
        tgt_p = max(cand, key=lambda p: p["conf"]) if cand else None
        # answer photo: memory detection nearest the top peak
        mp = mem_by.get(cname, [])
        ans_p = (min(mp, key=lambda p: np.hypot(p["x"] - peaks[0][0],
                                                p["y"] - peaks[0][1]))
                 if mp else None)
        tr = truth_by.get(cname)
        d1t = (float(np.min(np.hypot(*(np.array(tr) - np.array(peaks[0])).T)))
               if tr else None)
        panels.append(dict(
            cls=cname, n_mem=len(mp), n_gt=len(targets),
            d1=round(d1, 2), hit=bool(d1 <= args.radius),
            truth=[[round(float(a), 2), round(float(b), 2)] for a, b in tr]
            if tr else None,
            d1t=(round(d1t, 2) if d1t is not None else None),
            hit_t=(bool(d1t <= args.radius) if d1t is not None else None),
            targets=[[round(float(a), 2), round(float(b), 2)] for a, b in targets],
            peaks=[[round(v, 2) for v in pk] for pk in peaks],
            field=field,
            tq=crop_photo(tgt_p), ta=crop_photo(ans_p)))
        print(f"  {cname}: consistent d1={d1:.2f} "
              f"truth={'n/a' if d1t is None else f'{d1t:.2f}'}")

    r1 = np.mean([p["hit"] for p in panels])
    r1_mem = np.mean([p["hit"] for p in panels if p["n_mem"] >= args.min_count])
    truth_hits = [p["hit_t"] for p in panels if p["hit_t"] is not None]
    r1_true = float(np.mean(truth_hits)) if truth_hits else None
    step = max(1, len(px) // 400)
    payload = json.dumps(dict(
        scene=seq.name, ext=[round(float(v), 2) for v in ext],
        path=[[round(float(a), 2), round(float(b), 2)]
              for a, b in zip(px[::step], py[::step])],
        r1=round(float(r1), 2), r1_mem=round(float(r1_mem), 2),
        r1_true=(round(r1_true, 2) if r1_true is not None else None),
        n_truth=len(truth_hits), grid=args.grid,
        radius=args.radius, cap=args.max_per_class,
        n_mem=len(mem_pts), n_classes=len(panels), panels=panels,
        imgs=cache), separators=(",", ":")).replace("</", "<\\/")

    html = HTML.replace("__PAYLOAD__", payload)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write(html)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB, "
          f"{len(cache)} photos)")


HTML = """<meta charset="utf-8">
<title>Object grounding — where is the chair? (Replica room0)</title>
<style>
:root{--bg:#fbfbfc;--bg2:#fff;--panel:#eef0f4;--card:#fff;--ink:#14171c;--ink2:#39404b;
--muted:#646c78;--line:#dce0e7;--line2:#c6ccd6;--accent:#2a5fd9;
--good:#0f7a4b;--bad:#b02a24;--warn:#9a5c05;
--shadow:0 1px 2px rgba(16,20,28,.06),0 6px 22px rgba(16,20,28,.07)}
@media (prefers-color-scheme:dark){:root{--bg:#0e1116;--bg2:#141922;--panel:#1a212c;
--card:#151b24;--ink:#eef2f8;--ink2:#c3ccd9;--muted:#8b95a4;--line:#26303d;--line2:#33404f;
--accent:#79a6ff;--good:#54d39a;--bad:#ff8078;--warn:#f0b24a;
--shadow:0 1px 2px rgba(0,0,0,.4),0 8px 26px rgba(0,0,0,.34)}}
:root[data-theme="dark"]{--bg:#0e1116;--bg2:#141922;--panel:#1a212c;--card:#151b24;
--ink:#eef2f8;--ink2:#c3ccd9;--muted:#8b95a4;--line:#26303d;--line2:#33404f;--accent:#79a6ff;
--good:#54d39a;--bad:#ff8078;--warn:#f0b24a;
--shadow:0 1px 2px rgba(0,0,0,.4),0 8px 26px rgba(0,0,0,.34)}
:root[data-theme="light"]{--bg:#fbfbfc;--bg2:#fff;--panel:#eef0f4;--card:#fff;--ink:#14171c;
--ink2:#39404b;--muted:#646c78;--line:#dce0e7;--line2:#c6ccd6;--accent:#2a5fd9;
--good:#0f7a4b;--bad:#b02a24;--warn:#9a5c05;
--shadow:0 1px 2px rgba(16,20,28,.06),0 6px 22px rgba(16,20,28,.07)}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.6 ui-sans-serif,system-ui,"Segoe UI",Roboto,Arial,sans-serif}
.wrap{max-width:1060px;margin:0 auto;padding:0 20px}
code{background:var(--panel);border-radius:5px;padding:1px 5px;font-size:.87em;
font-family:ui-monospace,Menlo,Consolas,monospace}
header{background:var(--bg2);border-bottom:1px solid var(--line);padding:26px 0 22px}
.topbar{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;margin-bottom:14px}
.kicker{font-size:.73rem;letter-spacing:.15em;text-transform:uppercase;color:var(--accent);font-weight:700}
.btn{background:var(--card);color:var(--ink);border:1px solid var(--line2);border-radius:8px;
padding:7px 13px;font-size:.85rem;cursor:pointer;font-family:inherit}
.btn:hover{border-color:var(--accent)}
h1{font-size:clamp(1.35rem,3.4vw,2rem);margin:0 0 10px;letter-spacing:-.02em;text-wrap:balance}
.lede{color:var(--ink2);max-width:80ch;margin:0}
.card{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:16px;
box-shadow:var(--shadow);margin:16px 0}
.ctrl{display:flex;gap:10px;align-items:center;margin:10px 0;flex-wrap:wrap}
input[type=range]{flex:1;min-width:160px;accent-color:var(--accent)}
.ph{text-align:center}
.ph img{width:100%;border-radius:9px;border:2px solid var(--line2);display:block}
.ph.q img{border-color:var(--accent)}
.ph.good img{border-color:var(--good)}
.ph b{display:block;font-size:.82rem;margin-top:6px}
.ph span{font-size:.75rem;color:var(--muted)}
.missing{border:2px dashed var(--line2);border-radius:9px;display:flex;align-items:center;
justify-content:center;color:var(--muted);font-size:.85rem;min-height:140px;padding:10px}
canvas{width:100%;height:auto;display:block;border-radius:9px;background:var(--panel);
border:1px solid var(--line)}
.note{font-size:.85rem;color:var(--muted)}
.pill{display:inline-block;border-radius:99px;padding:1px 11px;font-size:.8rem;font-weight:700}
.pill.hit{background:color-mix(in srgb,var(--good) 15%,transparent);color:var(--good)}
.pill.miss{background:color-mix(in srgb,var(--bad) 14%,transparent);color:var(--bad)}
footer{padding:24px 0 56px;color:var(--muted);font-size:.85rem}
</style>
<header><div class="wrap">
<div class="topbar"><span class="kicker">object grounding · one 32 KB trace</span>
<button id="themeBtn" class="btn" type="button">Dark / light</button></div>
<h1>Ask the memory “where is the &lt;class&gt;?” — then look</h1>
<p class="lede" id="lede"></p>
</div></header>
<main><div class="wrap">
<div class="card">
<div class="ctrl">
<button id="prev" class="btn">&#8592;</button>
<input id="slider" type="range" min="0" max="0" value="0" step="1">
<button id="next" class="btn">&#8594;</button>
<span class="note" id="label" style="min-width:18ch"></span>
</div>
<div style="display:grid;grid-template-columns:1.2fr 1fr 1fr;gap:12px;align-items:start">
<div><canvas id="map" height="300"></canvas>
<p class="note" style="margin:6px 0 0"><span style="color:var(--accent)">heat = the memory's
belief field for this concept</span> (one unbinding renders the whole map — argmax is just a
readout of it) · grey = camera path ·
<span style="color:var(--good)">rings = re-detection clusters</span> ·
<span style="color:var(--warn)">■ squares = ground-truth instances</span> ·
<span style="color:var(--bad)">crosses = top-3 field peaks</span></p></div>
<div class="ph q" id="phQ"><img id="pq"><b>a target instance (held-out view)</b><span id="tq"></span></div>
<div class="ph good" id="phA"><img id="pa"><b>what the memory grounded at its answer</b><span id="ta"></span></div>
</div>
<p class="note" id="noteline" style="margin:10px 0 0"></p>
</div>
</div></main>
<footer><div class="wrap"><p id="foot"></p></div></footer>
<script id="data" type="application/json">__PAYLOAD__</script>
<script>
(function(){"use strict";
var D=JSON.parse(document.getElementById("data").textContent);
var $=function(s){return document.querySelector(s)};
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim()}
$("#lede").innerHTML="Scene <b>"+D.scene+"</b>: "+D.n_mem+" first-half detections bundled into "+
"one 32 KB trace (bounded insertion, cap "+D.cap+" per class). TWO verdicts per query, because "+
"they measure different things. <b>CONSISTENT</b> = the answer lands within "+D.radius+" m of "+
"where the detector re-finds that label in the second half of the walk - a consistent MISlabel "+
"(an armchair called 'toilet' both times) still scores as consistent. <b>TRUE</b> = the answer "+
"lands within "+D.radius+" m of a real instance from the Replica semantic ground truth. "+
"Consistency-R@1 <b>"+Math.round(D.r1*100)+"%</b> ("+Math.round(D.r1_mem*100)+"% on stored "+
"classes); truth-R@1 <b>"+(D.r1_true!=null?Math.round(D.r1_true*100)+"%</b> over the "+D.n_truth+
" GT-comparable classes":"pending GT</b>")+". All classes shown - no curation.";
var i=0,P=D.panels,ex=D.ext;
function fit(cv,h){var dpr=Math.min(devicePixelRatio||1,2),w=cv.clientWidth||600;
cv.width=w*dpr;cv.height=h*dpr;var g=cv.getContext("2d");g.setTransform(dpr,0,0,dpr,0,0);
return{g:g,w:w,h:h}}
function hexRgb(h){h=h.trim();if(h[0]==="#")h=h.slice(1);
if(h.length===3)h=h.split("").map(function(c){return c+c}).join("");
var n=parseInt(h,16);return isNaN(n)?[90,130,220]:[(n>>16)&255,(n>>8)&255,n&255]}
function drawMap(){var p=P[i],c=fit($("#map"),300),g=c.g;
g.clearRect(0,0,c.w,c.h);g.fillStyle=css("--panel");g.fillRect(0,0,c.w,c.h);
var pad=14,sx=(c.w-2*pad)/(ex[1]-ex[0]),sy=(c.h-2*pad)/(ex[3]-ex[2]),s=Math.min(sx,sy);
var T=function(x,y){return[pad+(x-ex[0])*s,c.h-(pad+(y-ex[2])*s)]};
/* the memory's belief FIELD for this concept — the map itself, not argmax */
if(p.field){var G=D.grid,rgb=hexRgb(css("--accent"));
var cw=(ex[1]-ex[0])/(G-1)*s,ch=(ex[3]-ex[2])/(G-1)*s;
for(var iy=0;iy<G;iy++)for(var ix=0;ix<G;ix++){
var v=p.field[iy*G+ix]/255;if(v<0.35)continue;
var a=Math.pow((v-0.35)/0.65,1.6)*0.85;
var x=ex[0]+ix*(ex[1]-ex[0])/(G-1),y=ex[2]+iy*(ex[3]-ex[2])/(G-1);
var t=T(x,y);
g.fillStyle="rgba("+rgb[0]+","+rgb[1]+","+rgb[2]+","+a.toFixed(3)+")";
g.fillRect(t[0]-cw/2,t[1]-ch/2,cw+0.5,ch+0.5)}}
g.strokeStyle=css("--line2");g.lineWidth=1.2;g.beginPath();
D.path.forEach(function(q,k){var t=T(q[0],q[1]);k?g.lineTo(t[0],t[1]):g.moveTo(t[0],t[1])});
g.stroke();
p.targets.forEach(function(t){var q=T(t[0],t[1]);
g.strokeStyle=css("--good");g.lineWidth=2.6;g.beginPath();g.arc(q[0],q[1],9,0,6.2832);g.stroke()});
if(p.truth){p.truth.forEach(function(t){var q=T(t[0],t[1]);
g.fillStyle=css("--warn");g.fillRect(q[0]-6,q[1]-6,12,12);
g.strokeStyle=css("--ink");g.lineWidth=1.6;g.strokeRect(q[0]-6,q[1]-6,12,12)})}
p.peaks.forEach(function(pk,r){var q=T(pk[0],pk[1]);
g.strokeStyle=css("--bad");g.lineWidth=r?1.4:2.8;g.globalAlpha=r?.55:1;
g.beginPath();g.moveTo(q[0]-7,q[1]-7);g.lineTo(q[0]+7,q[1]+7);
g.moveTo(q[0]+7,q[1]-7);g.lineTo(q[0]-7,q[1]+7);g.stroke();g.globalAlpha=1})}
function setPhoto(holder,imgEl,key,fallback){
if(key){imgEl.style.display="block";imgEl.src=D.imgs[key];
var m=holder.querySelector(".missing");if(m)m.remove()}
else{imgEl.style.display="none";
if(!holder.querySelector(".missing")){var d=document.createElement("div");
d.className="missing";d.textContent=fallback;holder.insertBefore(d,holder.firstChild)}}}
function render(){var p=P[i];
$("#slider").value=i;
var pills=" <span class='pill "+(p.hit?"hit'>CONSISTENT":"miss'>INCONSISTENT")+"</span>";
if(p.hit_t===true)pills+=" <span class='pill hit'>TRUE</span>";
else if(p.hit_t===false)pills+=" <span class='pill miss'>FALSE</span>";
else pills+=" <span class='pill' style='background:var(--panel);color:var(--muted)'>no GT class</span>";
$("#label").innerHTML="“where is the <b>"+p.cls+"</b>?” "+(i+1)+" / "+P.length+pills;
setPhoto($("#phQ"),$("#pq"),p.tq,"no photo (target from clustering only)");
setPhoto($("#phA"),$("#pa"),p.ta,"class not in the memory - nothing was stored to ground");
$("#tq").textContent=p.n_gt+" re-detection cluster(s)"+(p.truth?(" · "+p.truth.length+" GT instance(s)"):"");
$("#ta").textContent=p.n_mem?("nearest of "+p.n_mem+" stored detections; consistency err "+p.d1+
" m"+(p.d1t!=null?(", truth err "+p.d1t+" m"):"")):"0 stored detections";
var msg;
if(p.hit&&p.hit_t===true)msg="Solid: the answer matches both the re-detections and a real "+p.cls+".";
else if(p.hit&&p.hit_t===false)msg="CONSISTENT BUT FALSE - the detector makes the same mistake in both halves (mislabel or reflection); the memory faithfully grounds a wrong label. This is the armchair-'toilet' case.";
else if(!p.hit&&p.hit_t===true)msg="True but inconsistent - the answer finds a real "+p.cls+" that the second-half detector happens not to re-find there.";
else if(p.hit_t===false)msg="Miss on both metrics.";
else msg=p.hit?"Consistent; no Replica GT class maps to '"+p.cls+"', so truth is unscored here.":
(p.n_mem?"Inconsistent; no GT class to score truth against.":"The first half never stored a "+p.cls+" - a memory cannot ground what it never saw.");
$("#noteline").textContent=msg;
drawMap()}
$("#slider").max=P.length-1;
$("#slider").addEventListener("input",function(){i=parseInt(this.value,10);render()});
$("#prev").addEventListener("click",function(){i=(i-1+P.length)%P.length;render()});
$("#next").addEventListener("click",function(){i=(i+1)%P.length;render()});
document.addEventListener("keydown",function(e){
if(/^(INPUT|BUTTON)$/.test((e.target||{}).tagName))return;
if(e.key==="ArrowRight"){e.preventDefault();i=(i+1)%P.length;render()}
if(e.key==="ArrowLeft"){e.preventDefault();i=(i-1+P.length)%P.length;render()}});
$("#foot").innerHTML="Protocol: PROVISIONAL two-pseudo-session split (targets = second-half "+
"clusters) until the ground-truth instance centroids arrive from the Replica semantic mesh "+
"(student GPU task) - then the same page rebuilds against true GT. Settings: hd 4096, grid 64, "+
"length scale 0.6 m, room margin 1.5 m, per-class cap "+D.cap+". Rebuild: "+
"<code>python tools/export_grounding_inspector.py</code>.";
$("#themeBtn").addEventListener("click",function(){
var cur=document.documentElement.getAttribute("data-theme");
if(!cur)cur=matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";
document.documentElement.setAttribute("data-theme",cur==="dark"?"light":"dark");render()});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change",render);
addEventListener("resize",drawMap);
render();
})();
</script>
"""


if __name__ == "__main__":
    main()
