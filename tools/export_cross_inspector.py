"""Photo-grounded inspector for a CROSS-TRAVERSE benchmark run.

Standing rule (Paul, 2026-08-04): quantitative tables are never delivered
alone — every result gets this qualitative counterpart: the query PHOTO, the
photo at the memory's answer, both marked on the floor map, browsable with a
play button (video-style scrub through the test walk), queries sampled by a
fixed stride (never hand-picked), with the quantitative table on the same
page so numbers and pictures can be checked against each other.

Generic over any cross_recall-style setup (7-Scenes chess today, any scene
tomorrow):

    python tools/export_cross_inspector.py \
        --mem-datasets   vsa_cognitive_mapping/configs/7scenes_chess_seq01.json ... \
        --query-datasets vsa_cognitive_mapping/configs/7scenes_chess_seq03.json ... \
        --encoder crop_embeddings_eigenplaces.pt \
        --results-json outputs/cross_recall_7scenes_chess.json \
        --out outputs/cross_inspector_7scenes_chess.html
"""
from __future__ import annotations

import argparse
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
from vsa_cognitive_mapping.cross_recall import (  # noqa: E402
    load_traverse, fit_treatment, apply_treatment)
from vsa_cognitive_mapping.classroom_pipeline import ClassroomEncoders  # noqa: E402
from vsa_cognitive_mapping.vsa import random_project_to_phasor  # noqa: E402
from vsa_cognitive_mapping.knn_baseline import _unit  # noqa: E402

IMG_W, IMG_Q = 270, 52


def main():
    global IMG_W, IMG_Q
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mem-datasets", nargs="+", required=True)
    ap.add_argument("--query-datasets", nargs="+", required=True)
    ap.add_argument("--encoder", default="crop_embeddings_eigenplaces.pt")
    ap.add_argument("--treatment", default="zscored")
    ap.add_argument("--hd-dim", type=int, default=4096)
    ap.add_argument("--grid", type=int, default=48)
    ap.add_argument("--length-scale", type=float, default=0.75)
    ap.add_argument("--battery-max", type=int, default=100)
    ap.add_argument("--results-json", default=None,
                    help="cross_recall output whose table heads the page")
    ap.add_argument("--title", default="Cross-traverse recall inspector")
    ap.add_argument("--img-w", type=int, default=270)
    ap.add_argument("--img-q", type=int, default=52)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    IMG_W, IMG_Q = args.img_w, args.img_q

    mem_t = [load_traverse(c, args.encoder) for c in args.mem_datasets]
    qry_t = [load_traverse(c, args.encoder) for c in args.query_datasets]
    mem_seq = [load_sequence(c) for c in args.mem_datasets]
    qry_seq = [load_sequence(c) for c in args.query_datasets]

    Em = np.concatenate([t["E"] for t in mem_t])
    XYm = np.concatenate([t["XY"] for t in mem_t])
    Eq = np.concatenate([t["E"] for t in qry_t])
    XYq = np.concatenate([t["XY"] for t in qry_t])
    src_m = np.concatenate([np.full(len(t["uf"]), i) for i, t in enumerate(mem_t)])
    src_q = np.concatenate([np.full(len(t["uf"]), i) for i, t in enumerate(qry_t)])
    fid_m = np.concatenate([t["uf"] for t in mem_t])
    fid_q = np.concatenate([t["uf"] for t in qry_t])
    n_mem, n_qry = len(Em), len(Eq)

    prm = fit_treatment(Em, args.treatment)
    Xm, Xq = apply_treatment(Em, prm), apply_treatment(Eq, prm)
    z, _ = random_project_to_phasor(
        torch.from_numpy(np.ascontiguousarray(np.concatenate([Xm, Xq]))).float(),
        d=args.hd_dim, seed=0)
    Z = z.numpy().astype(np.complex128)
    Zm, Zq = Z[:n_mem], Z[n_mem:]

    ext = [min(XYm[:, 0].min(), XYq[:, 0].min()) - 0.5,
           max(XYm[:, 0].max(), XYq[:, 0].max()) + 0.5,
           min(XYm[:, 1].min(), XYq[:, 1].min()) - 0.5,
           max(XYm[:, 1].max(), XYq[:, 1].max()) + 0.5]
    enc = ClassroomEncoders(args.hd_dim, 0, args.length_scale, 20.0)
    gx = np.linspace(ext[0], ext[1], args.grid)
    gy = np.linspace(ext[2], ext[3], args.grid)
    G = np.empty((args.grid ** 2, args.hd_dim), np.complex64)
    k = 0
    for yy in gy:
        for xx in gx:
            G[k] = enc.ctx_pos(float(xx), float(yy)).values.astype(np.complex64)
            k += 1
    P = np.stack([enc.ctx_pos(float(a), float(b)).values for a, b in XYm])
    M = (Zm * P).mean(0)
    M /= max(np.abs(M).max(), 1e-12)

    stride = max(1, int(np.ceil(n_qry / args.battery_max)))
    qi = np.arange(n_qry)[::stride]
    print(f"{n_qry} test-traverse queries -> every {stride}th = {len(qi)}")

    S = ((M[None, :] / Zq[qi]) @ np.conj(G).T).real
    j = S.argmax(1)
    peak = np.stack([gx[j % args.grid], gy[j // args.grid]], 1)
    err = np.hypot(*(peak - XYq[qi]).T)
    near = np.array([int(np.argmin(np.hypot(*(XYm - p).T))) for p in peak])
    nn = (_unit(Xq[qi]) @ _unit(Xm).T).argmax(1)
    err_knn = np.hypot(*(XYm[nn] - XYq[qi]).T)
    print(f"battery: VSA median {np.median(err):.2f} m, "
          f"kNN median {np.median(err_knn):.2f} m")

    cache = {}

    def photo(seqs, src, fid):
        seq = seqs[src]
        key = f"{'m' if seqs is mem_seq else 'q'}{src}_{fid}"
        if key not in cache:
            pos = {int(f): i for i, f in enumerate(seq.frame_ids())}
            cache[key] = jpeg_uri(seq.image(pos[int(fid)]), IMG_W, IMG_Q)
        return key

    rows = []
    for i, q in enumerate(qi):
        rows.append(dict(
            x=round(float(XYq[q, 0]), 2), y=round(float(XYq[q, 1]), 2),
            px=round(float(peak[i, 0]), 2), py=round(float(peak[i, 1]), 2),
            kx=round(float(XYm[nn[i], 0]), 2), ky=round(float(XYm[nn[i], 1]), 2),
            err=round(float(err[i]), 2), ek=round(float(err_knn[i]), 2),
            trav=qry_seq[src_q[q]].name,
            q=photo(qry_seq, src_q[q], fid_q[q]),
            a=photo(mem_seq, src_m[near[i]], fid_m[near[i]]),
            n=photo(mem_seq, src_m[nn[i]], fid_m[nn[i]])))
        if i % 20 == 0:
            print(f"  photos {i}/{len(qi)}", flush=True)

    paths = [[[round(float(a), 2), round(float(b), 2)]
              for a, b in t["XY"][::max(1, len(t["XY"]) // 250)]] for t in mem_t]
    table = []
    if args.results_json and os.path.exists(args.results_json):
        res = json.load(open(args.results_json))
        wanted = [("kNN exact", "full database"), ("PQ m=8", "compressed database"),
                  ("VSA argmax", "one 32 KB trace"), ("VSA top-3 modes", "one 32 KB trace")]
        for r in res["rows"]:
            for w, note in wanted:
                if r["system"].startswith(w) and r["treatment"] == args.treatment:
                    table.append(dict(
                        system=r["system"], note=note, bytes=r["bytes"],
                        med3d=(round(r["err3d"]["median"], 2) if r["err3d"] else None),
                        le05=round(r["err2d"]["le05"], 3),
                        le1=round(r["err2d"]["le1"], 3)))

    payload = json.dumps(dict(
        title=args.title, treatment=args.treatment, encoder=args.encoder,
        mem_names=[s.name for s in mem_seq], qry_names=[s.name for s in qry_seq],
        n_mem=n_mem, n_qry=n_qry, stride=stride, ext=[round(float(v), 2) for v in ext],
        paths=paths, med=round(float(np.median(err)), 2),
        med_knn=round(float(np.median(err_knn)), 2),
        table=table, queries=rows, imgs=cache,
    ), separators=(",", ":")).replace("</", "<\\/")

    html = HTML.replace("__PAYLOAD__", payload)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write(html)
    print(f"wrote {args.out}  ({os.path.getsize(args.out) / 1e6:.1f} MB, "
          f"{len(cache)} unique photos)")


HTML = """<meta charset="utf-8">
<title>Cross-traverse recall — photos, map, and the table</title>
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
header{background:var(--bg2);border-bottom:1px solid var(--line);padding:26px 0 22px}
.topbar{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;margin-bottom:14px}
.kicker{font-size:.73rem;letter-spacing:.15em;text-transform:uppercase;color:var(--accent);font-weight:700}
.btn{background:var(--card);color:var(--ink);border:1px solid var(--line2);border-radius:8px;
padding:7px 13px;font-size:.85rem;cursor:pointer;font-family:inherit}
.btn:hover{border-color:var(--accent)}
.btn.on{background:var(--accent);color:#fff;border-color:var(--accent)}
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
canvas{width:100%;height:auto;display:block;border-radius:9px;background:var(--panel);
border:1px solid var(--line)}
.note{font-size:.85rem;color:var(--muted)}
table{border-collapse:collapse;min-width:560px;font-size:.9rem;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:7px 12px;border-bottom:1px solid var(--line)}
th{font-size:.76rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
footer{padding:24px 0 56px;color:var(--muted);font-size:.85rem}
</style>
<header><div class="wrap">
<div class="topbar"><span class="kicker" id="kick"></span>
<button id="themeBtn" class="btn" type="button">Dark / light</button></div>
<h1>Every query is a frame from a walk the memory never saw</h1>
<p class="lede" id="lede"></p>
</div></header>
<main><div class="wrap">
<div class="card" style="overflow-x:auto">
<table><thead><tr><th>system</th><th>memory</th><th>3D median</th>
<th>2D &le;0.5 m</th><th>2D &le;1 m</th></tr></thead>
<tbody id="tbl"></tbody></table>
<p class="note" style="margin:8px 0 0">The table and the photos below are the same run —
judge one against the other. kNN answers with its single most-similar stored frame;
the VSA answers from one superposed trace.</p>
</div>
<div class="card">
<div class="ctrl">
<button id="play" class="btn">&#9654; Play</button>
<button id="prev" class="btn">&#8592;</button>
<input id="slider" type="range" min="0" max="0" value="0" step="1">
<button id="next" class="btn">&#8594;</button>
<span class="note" id="label" style="min-width:16ch"></span>
<button id="worst" class="btn">Worst &#8594;</button>
</div>
<div style="display:grid;grid-template-columns:1.15fr 1fr 1fr 1fr;gap:12px;align-items:start">
<div><canvas id="map" height="300"></canvas>
<p class="note" style="margin:6px 0 0">grey = the 4 training walks ·
<span style="color:var(--good)">ring = query's true position</span> ·
<span style="color:var(--bad)">cross = VSA answer</span> ·
<span style="color:var(--warn)">diamond = kNN answer</span></p></div>
<div class="ph q"><img id="pq"><b>query view (held-out walk)</b><span id="tq"></span></div>
<div class="ph good"><img id="pa"><b>VSA: stored frame at its answer</b><span id="ta"></span></div>
<div class="ph"><img id="pn"><b>kNN: most similar stored frame</b><span id="tn"></span></div>
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
$("#kick").textContent=D.title;
$("#lede").innerHTML="Memory: <b>"+D.n_mem+"</b> frames from "+D.mem_names.length+
" training walks ("+D.mem_names.join(", ")+"), one "+D.treatment+" trace. Queries: <b>"+
D.n_qry+"</b> frames from "+D.qry_names.length+" separate test walks; every "+D.stride+
"th shown (<b>no hand-picking</b>). Battery medians: VSA "+D.med+" m, kNN "+D.med_knn+" m.";
var tb=[];D.table.forEach(function(r){
tb.push("<tr><td><b>"+r.system+"</b> <span class='note'>"+r.note+"</span></td><td>"+
(r.bytes>=1e6?(r.bytes/1e6).toFixed(1)+" MB":Math.round(r.bytes/1024)+" KB")+"</td><td>"+
(r.med3d!=null?r.med3d+" m":"&mdash;")+"</td><td>"+Math.round(r.le05*100)+"%</td><td>"+
Math.round(r.le1*100)+"%</td></tr>")});
$("#tbl").innerHTML=tb.join("");
var i=0,Q=D.queries,ex=D.ext,timer=null;
function fit(cv,h){var dpr=Math.min(devicePixelRatio||1,2),w=cv.clientWidth||600;
cv.width=w*dpr;cv.height=h*dpr;var g=cv.getContext("2d");g.setTransform(dpr,0,0,dpr,0,0);
return{g:g,w:w,h:h}}
function drawMap(){var q=Q[i],c=fit($("#map"),300),g=c.g;
g.clearRect(0,0,c.w,c.h);g.fillStyle=css("--panel");g.fillRect(0,0,c.w,c.h);
var pad=14,sx=(c.w-2*pad)/(ex[1]-ex[0]),sy=(c.h-2*pad)/(ex[3]-ex[2]),s=Math.min(sx,sy);
var P=function(x,y){return[pad+(x-ex[0])*s,c.h-(pad+(y-ex[2])*s)]};
g.strokeStyle=css("--line2");g.lineWidth=1.2;
D.paths.forEach(function(path){g.beginPath();
path.forEach(function(p,k){var t=P(p[0],p[1]);k?g.lineTo(t[0],t[1]):g.moveTo(t[0],t[1])});
g.stroke()});
var tt=P(q.x,q.y),pp=P(q.px,q.py),kk=P(q.kx,q.ky);
g.strokeStyle=css("--bad");g.lineWidth=1.1;g.beginPath();g.moveTo(tt[0],tt[1]);
g.lineTo(pp[0],pp[1]);g.stroke();
g.strokeStyle=css("--warn");g.lineWidth=2.2;g.beginPath();
g.moveTo(kk[0],kk[1]-7);g.lineTo(kk[0]+7,kk[1]);g.lineTo(kk[0],kk[1]+7);
g.lineTo(kk[0]-7,kk[1]);g.closePath();g.stroke();
g.strokeStyle=css("--good");g.lineWidth=2.6;g.beginPath();g.arc(tt[0],tt[1],8,0,6.2832);g.stroke();
g.strokeStyle=css("--bad");g.lineWidth=2.6;g.beginPath();
g.moveTo(pp[0]-7,pp[1]-7);g.lineTo(pp[0]+7,pp[1]+7);
g.moveTo(pp[0]+7,pp[1]-7);g.lineTo(pp[0]-7,pp[1]+7);g.stroke()}
function render(){var q=Q[i];
$("#slider").value=i;$("#label").textContent="query "+(i+1)+" / "+Q.length+" ("+q.trav+")";
$("#pq").src=D.imgs[q.q];$("#pa").src=D.imgs[q.a];$("#pn").src=D.imgs[q.n];
$("#tq").textContent="true "+q.x+", "+q.y+" m";
$("#ta").textContent="decoded "+q.px+", "+q.py+" m (err "+q.err+" m)";
$("#tn").textContent="err "+q.ek+" m";
$("#noteline").textContent="If the three photos show the same corner of the desk, both systems "+
"are right here - the VSA did it from a single 32 KB trace, the kNN from the full stored database.";
drawMap()}
$("#slider").max=Q.length-1;
$("#slider").addEventListener("input",function(){i=parseInt(this.value,10);render()});
$("#prev").addEventListener("click",function(){i=(i-1+Q.length)%Q.length;render()});
$("#next").addEventListener("click",function(){i=(i+1)%Q.length;render()});
$("#worst").addEventListener("click",function(){var w=0,v=-1;
for(var k=0;k<Q.length;k++){if(Q[k].err>v){v=Q[k].err;w=k}}i=w;render()});
function stop(){if(timer){clearInterval(timer);timer=null;$("#play").classList.remove("on");
$("#play").innerHTML="&#9654; Play"}}
$("#play").addEventListener("click",function(){
if(timer){stop();return}
$("#play").classList.add("on");$("#play").innerHTML="&#10074;&#10074; Pause";
timer=setInterval(function(){i=(i+1)%Q.length;render();
if(i===0)stop()},220)});
document.addEventListener("keydown",function(e){
if(/^(INPUT|BUTTON)$/.test((e.target||{}).tagName))return;
if(e.key==="ArrowRight"){e.preventDefault();stop();i=(i+1)%Q.length;render()}
if(e.key==="ArrowLeft"){e.preventDefault();stop();i=(i-1+Q.length)%Q.length;render()}
if(e.key===" "){e.preventDefault();$("#play").click()}});
$("#foot").innerHTML="Sampling: every "+D.stride+"th of the "+D.n_qry+
" test-traverse queries, in walk order - deterministic, no curation. Encoder "+
"<code>"+D.encoder+"</code>, treatment "+D.treatment+" (statistics fit on memory frames only), "+
"hd 4096, grid 48. Play scrubs the held-out walk like a video. Rebuild: "+
"<code>python tools/export_cross_inspector.py</code>.";
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
