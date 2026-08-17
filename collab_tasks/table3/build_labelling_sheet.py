"""Build the Table-III relevance-labelling sheet: one self-contained HTML.

ConceptGraphs published their 20 affordance + negation query strings (their
Appendix A4; verbatim copy in replica_queries_theirs.py, fetched from their
`query` branch) but never the relevance judgements. This sheet is the missing
half: every kept GT instance in every scene, as the camera saw it, against the
10 queries that apply to that scene type. Tick the instances that answer each
query; the sheet exports JSON. That JSON is OUR re-annotation on OUR GT
namespace and must always be reported as such -- it is not their protocol.

Crops are GT-anchored (projection code proven in show_me_the_object.py): pink
dots are the instance's own ground-truth eval points. No CLIP, no SAM, no
predicted label anywhere in this pipeline.

    python collab_tasks/table3/build_labelling_sheet.py
    -> outputs/table3/labelling_sheet.html   (self-contained, open locally;
       progress autosaves to localStorage; Export writes JSON to copy out)
"""
from __future__ import annotations

import base64
import html
import io
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

W, H = 1200, 680
FX = FY = 600.0
CX, CY = 599.5, 339.5
EX = ("other", "floor", "wall", "ceiling", "door", "window")
SCENES = ["room0", "room1", "room2",
          "office0", "office1", "office2", "office3", "office4"]
POSE_STRIDE = 10          # score every 10th pose when hunting the best frame
THUMB = 190

# their query strings, verbatim (collab_tasks/table3/replica_queries_theirs.py)
OFFICE_Q = {
    "A1": "Something to watch the news on",
    "A2": "Something to tell the time",
    "A3": "Something comfortable to sit on",
    "A4": "Something to dispose of wastepaper in",
    "A5": "Something to add light into the room",
    "N1": "Something to sit on other than a chair",
    "N2": "Something very heavy, unlike a clock",
    "N3": "Something rigid, unlike a cushion",
    "N4": "Something small, unlike a couch",
    "N5": "Something light, unlike a table",
}
ROOM_Q = {
    "A1": "Somewhere to store decorative cups",
    "A2": "Something to add light into the room",
    "A3": "Somewhere to set food for dinner",
    "A4": "Something I can open with my keys",
    "A5": "Something to sit upright for a work call",
    "N1": "Something small, unlike a cabinet",
    "N2": "Something light, unlike a table",
    "N3": "Something soft, unlike a table",
    "N4": "Something not transparent, unlike a window",
    "N5": "Something rigid, unlike a rug",
}


def project(pts_w, c2w):
    """Verbatim from show_me_the_object.py -- the verified convention."""
    R, t = c2w[:3, :3], c2w[:3, 3]
    cam = (pts_w - t) @ R
    z = cam[:, 2]
    valid = z > 1e-6
    zz = np.where(valid, z, 1.0)
    u = FX * (cam[:, 0] / zz) + CX
    v = FY * (cam[:, 1] / zz) + CY
    inside = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, zz, inside


def best_frame(pts, poses):
    best, bn = None, 0
    for i in range(0, len(poses), POSE_STRIDE):
        _, _, _, ins = project(pts, poses[i])
        n = int(ins.sum())
        if n > bn:
            bn, best = n, i
    return best


def crop_b64(scene, frame, pts, c2w, pad=60, box=300):
    fp = f"data/replica/{scene}/frame{frame:06d}.jpg"
    im = Image.open(fp).convert("RGB")
    u, v, _, ins = project(pts, c2w)
    if ins.sum() == 0:
        return None
    uu, vv = u[ins], v[ins]
    x0, x1 = max(0, uu.min() - pad), min(W, uu.max() + pad)
    y0, y1 = max(0, vv.min() - pad), min(H, vv.max() + pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(box / 2, (x1 - x0) / 2, (y1 - y0) / 2)
    x0, x1 = int(max(0, cx - half)), int(min(W, cx + half))
    y0, y1 = int(max(0, cy - half)), int(min(H, cy + half))
    c = im.crop((x0, y0, x1, y1))
    d = ImageDraw.Draw(c, "RGBA")
    for a, b in zip(uu, vv):
        d.ellipse([a - x0 - 3, b - y0 - 3, a - x0 + 3, b - y0 + 3],
                  fill=(255, 61, 166, 160))
    c.thumbnail((THUMB, THUMB))
    b = io.BytesIO()
    c.save(b, format="JPEG", quality=72)
    return base64.b64encode(b.getvalue()).decode()


def scene_rows(scene):
    """One row per kept GT instance: (idx, cls, crop_b64, n_assigned)."""
    inst = json.load(open(f"outputs/replica_{scene}/gt_instances.json"))["instances"]
    E = np.load(f"student_gpu_package/handoff/{scene}_cgfront/eval_points.npz",
                allow_pickle=True)
    xyz, gt = E["xyz"], E["gt_class"].astype(str)
    poses = np.loadtxt(f"data/replica/{scene}/traj.txt").reshape(-1, 4, 4)

    # partition each class's eval points among that class's instances by
    # nearest centroid, so each crop shows ONE instance, not the class
    by_cls = {}
    for k, g in enumerate(inst):
        if g["cls"] not in EX:
            by_cls.setdefault(g["cls"], []).append(k)
    rows = []
    for c, ks in sorted(by_cls.items()):
        cents = np.array([[inst[k]["x"], inst[k]["y"], inst[k]["z"]]
                          for k in ks])
        m = gt == c
        pts_c = xyz[m]
        if len(pts_c):
            d2 = ((pts_c[:, None, :] - cents[None, :, :]) ** 2).sum(-1)
            owner = d2.argmin(1)
        for j, k in enumerate(ks):
            pts = pts_c[owner == j] if len(pts_c) else np.empty((0, 3))
            if len(pts) > 250:
                sel = np.random.RandomState(k).choice(len(pts), 250,
                                                      replace=False)
                pts = pts[sel]
            if len(pts) == 0:
                pts = cents[j][None, :]
            f = best_frame(pts, poses)
            b64 = crop_b64(scene, f, pts, poses[f]) if f is not None else None
            rows.append(dict(idx=k, cls=c, b64=b64, n=int(len(pts)),
                             frame=int(f) if f is not None else -1))
            print(f"  {scene}#{k:<3} {c:<16} pts={len(pts):>3} "
                  f"frame={f} {'OK' if b64 else 'NO CROP'}", flush=True)
    return rows


CSS = """
:root{--paper:#FAF9F6;--ink:#22252A;--line:#DCDAD3;--crimson:#9D2235;
--panel:#F1EFE9;--muted:#6E7278;--good:#2F6B46;
--mono:ui-monospace,'Cascadia Mono',Consolas,monospace;
--sans:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
font-size:15px;line-height:1.5}
.wrap{max-width:1240px;margin:0 auto;padding:32px 20px 120px}
h1{font-size:1.6rem;margin:0 0 6px}
h2{font-size:1.15rem;margin:36px 0 4px;border-top:2px solid var(--crimson);
padding-top:14px}
p{max-width:78ch;margin:6px 0}
.mut{color:var(--muted);font-size:.85rem}
.qlegend{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));
gap:4px 18px;margin:10px 0;font-size:.85rem}
.qlegend b{font-family:var(--mono);color:var(--crimson)}
table{border-collapse:collapse;width:100%}
th,td{border:1px solid var(--line);padding:4px 6px;text-align:center}
thead th{position:sticky;top:0;background:var(--panel);z-index:2;
font-family:var(--mono);font-size:.75rem;cursor:help}
td.obj{text-align:left;min-width:220px;background:#fff}
td.obj img{display:block;border-radius:4px;max-width:190px}
td.obj .lab{font-family:var(--mono);font-size:.78rem;margin-top:3px}
td.q{cursor:pointer;min-width:44px;font-family:var(--mono);font-size:.9rem;
user-select:none;background:#fff}
td.q.on{background:var(--good);color:#fff;font-weight:700}
.bar{position:fixed;bottom:0;left:0;right:0;background:var(--panel);
border-top:1px solid var(--line);padding:10px 20px;display:flex;gap:14px;
align-items:center;z-index:5}
button{font:inherit;padding:8px 16px;border:1px solid var(--crimson);
background:var(--crimson);color:#fff;border-radius:5px;cursor:pointer}
button.ghost{background:transparent;color:var(--crimson)}
#out{width:100%;height:150px;font-family:var(--mono);font-size:.75rem}
.count{font-family:var(--mono);font-size:.85rem;color:var(--muted)}
"""

JS = """
const KEY='table3-relevance-v1';
let st=JSON.parse(localStorage.getItem(KEY)||'{}');
function save(){localStorage.setItem(KEY,JSON.stringify(st));cnt()}
function tog(el){const s=el.dataset.s,q=el.dataset.q,k=el.dataset.k;
 st[s]=st[s]||{};st[s][q]=st[s][q]||[];
 const i=st[s][q].indexOf(k);
 if(i<0){st[s][q].push(k);el.classList.add('on');el.textContent='\\u2713'}
 else{st[s][q].splice(i,1);el.classList.remove('on');el.textContent=''}
 save()}
function cnt(){let n=0;for(const s in st)for(const q in st[s])n+=st[s][q].length;
 document.getElementById('n').textContent=n}
function restore(){document.querySelectorAll('td.q').forEach(el=>{
 const s=el.dataset.s,q=el.dataset.q,k=el.dataset.k;
 if(st[s]&&st[s][q]&&st[s][q].includes(k)){el.classList.add('on');
  el.textContent='\\u2713'}});cnt()}
function exp(){const o=document.getElementById('out');
 o.style.display='block';
 o.value=JSON.stringify({schema:'table3-relevance-v1',
  note:'our re-annotation of ConceptGraphs Appendix-A4 queries on Replica GT instances; not their protocol',
  annotator:'',date:new Date().toISOString().slice(0,10),labels:st},null,1);
 o.select()}
function clr(){if(confirm('Clear ALL ticks?')){st={};save();
 document.querySelectorAll('td.q.on').forEach(e=>{e.classList.remove('on');
  e.textContent=''});cnt()}}
window.addEventListener('DOMContentLoaded',restore);
"""


def build_html(all_rows):
    parts = [f"<title>Table III relevance labelling</title>"
             f"<meta charset='utf-8'><style>{CSS}</style><script>{JS}</script>"
             "<div class='wrap'>",
             "<h1>Table III relevance labelling</h1>",
             "<p>For each ground-truth object (rows, shown as the camera saw "
             "it &mdash; pink dots are its own GT points) tick every query "
             "(columns) it genuinely answers. Their paper: <i>&ldquo;We "
             "manually select relevant objects as ground truth for each "
             "query.&rdquo;</i> This sheet recreates that selection on OUR GT "
             "namespace; the export is a re-annotation, not their protocol.</p>",
             "<p class='mut'>Progress autosaves in this browser "
             "(localStorage). Multiple relevant objects per query are fine; a "
             "query with nothing relevant in a scene stays empty. Hover a "
             "column header for the full query text.</p>"]
    for scene in SCENES:
        rows = all_rows[scene]
        Q = ROOM_Q if scene.startswith("room") else OFFICE_Q
        parts.append(f"<h2>{scene} &middot; {len(rows)} objects &middot; "
                     f"{'room' if scene.startswith('room') else 'office'} "
                     f"queries</h2>")
        parts.append("<div class='qlegend'>" + "".join(
            f"<span><b>{k}</b> {html.escape(v)}</span>"
            for k, v in Q.items()) + "</div>")
        head = "".join(f"<th title='{html.escape(v)}'>{k}</th>"
                       for k, v in Q.items())
        parts.append(f"<table><thead><tr><th>object</th>{head}</tr></thead>"
                     "<tbody>")
        for r in rows:
            img = (f"<img src='data:image/jpeg;base64,{r['b64']}'>"
                   if r["b64"] else "<span class='mut'>no crop</span>")
            key = f"{scene}:{r['idx']}:{r['cls']}"
            cells = "".join(
                f"<td class='q' data-s='{scene}' data-q='{q}' "
                f"data-k='{key}' onclick='tog(this)'></td>" for q in Q)
            parts.append(
                f"<tr><td class='obj'>{img}<div class='lab'>"
                f"#{r['idx']} <b>{html.escape(r['cls'])}</b> "
                f"<span class='mut'>frame {r['frame']}</span></div></td>"
                f"{cells}</tr>")
        parts.append("</tbody></table>")
    parts.append(
        "<div class='bar'><button onclick='exp()'>Export JSON</button>"
        "<button class='ghost' onclick='clr()'>Clear all</button>"
        "<span class='count'><span id='n'>0</span> ticks</span></div>"
        "<textarea id='out' style='display:none'></textarea></div>")
    return "\n".join(parts)


def main():
    all_rows = {}
    for scene in SCENES:
        print(f"== {scene}", flush=True)
        all_rows[scene] = scene_rows(scene)
    os.makedirs("outputs/table3", exist_ok=True)
    out = "outputs/table3/labelling_sheet.html"
    open(out, "w", encoding="utf-8").write(build_html(all_rows))
    n = sum(len(v) for v in all_rows.values())
    kb = os.path.getsize(out) / 1024
    print(f"\nwrote {out}: {n} instances over {len(SCENES)} scenes, "
          f"{kb:.0f} kB self-contained")


if __name__ == "__main__":
    main()
