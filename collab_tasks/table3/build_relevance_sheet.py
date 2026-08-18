"""Build the shareable relevance-labelling sheet as a live-doc artifact.

Anyone with the link ticks which object classes answer each query. Because the
chips live inside an `artifact-sync` region, each person's clicks are appended
to the shared document as themselves and reach every other view, so results
come back without anyone emailing a file. A JSON export is kept as a fallback
for read-only viewers, whose clicks are never saved.

Pre-ticked chips are the MACHINE-PROPOSED draft, marked as such on the page.
The point of sharing is to have humans correct it.

    python collab_tasks/table3/build_relevance_sheet.py
"""
from __future__ import annotations

import html
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

L = json.load(open("outputs/table3/relevance_proposed.json"))
OUT = "outputs/table3/relevance_sheet.html"

room_cls, office_cls = set(), set()
for s, v in L["scenes"].items():
    (room_cls if v["kind"] == "room" else office_cls).update(v["classes"])
room_cls, office_cls = sorted(room_cls), sorted(office_cls)

# union the per-scene proposals up to scene-type level for the draft
prop = {"room": {}, "office": {}}
for s, v in L["scenes"].items():
    for q, hits in v["proposed"].items():
        prop[v["kind"]].setdefault(q, {}).update(hits)

CSS = """
:root{--paper:#FAF9F6;--ink:#22252A;--panel:#F1EFE9;--line:#DCDAD3;
--brand:#9D2235;--muted:#5C6068;--ok:#2F6B46;--warn:#8A5A12;
--bg:var(--paper);--text-body:var(--ink);--surface:#FFFFFF;--border:var(--line);
--text-muted:var(--muted);
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
--sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
@media (prefers-color-scheme:dark){:root{--bg:#16181B;--text-body:#E6E4DF;
--surface:#1E2125;--panel:#1E2125;--border:#33373C;--brand:#C33A50;
--text-muted:#9AA0A6;--ok:#6FAE7F;--warn:#C99A50;color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text-body);
font-family:var(--sans);font-size:1rem;line-height:1.55}
.wrap{max-width:1000px;margin:0 auto;padding:36px 22px 80px}
.eyebrow{font-family:var(--mono);font-size:.71rem;letter-spacing:.14em;
text-transform:uppercase;color:var(--brand);margin:0 0 8px}
h1{font-size:1.7rem;margin:0 0 10px;letter-spacing:-.017em}
h2{font-size:1.1rem;margin:30px 0 6px}
p{margin:0 0 12px;max-width:72ch}
.lede{color:var(--text-muted);max-width:74ch}
.card{background:var(--surface);border:1px solid var(--border);
border-radius:6px;padding:14px 16px;margin:0 0 12px}
.qh{font-family:var(--mono);font-size:.7rem;letter-spacing:.1em;
text-transform:uppercase;color:var(--text-muted);margin:0 0 2px}
.qt{font-size:1rem;font-weight:600;margin:0 0 10px}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{font-family:var(--mono);font-size:.76rem;padding:4px 10px;
border-radius:99px;border:1px solid var(--border);color:var(--text-muted);
cursor:pointer;user-select:none;background:transparent}
.chip:hover{border-color:var(--brand)}
.chip.on{background:var(--ok);border-color:var(--ok);color:#fff}
.chip.draft::after{content:" *";opacity:.7}
.note{font-size:.82rem;color:var(--text-muted);max-width:72ch}
.warn{border-left:3px solid var(--warn);background:var(--panel);
padding:12px 16px;border-radius:0 6px 6px 0;margin:0 0 16px}
.bar{position:sticky;top:0;background:var(--bg);padding:10px 0;
border-bottom:1px solid var(--border);z-index:5;display:flex;gap:10px;
align-items:center;flex-wrap:wrap}
button{font:inherit;padding:7px 14px;border:1px solid var(--brand);
background:var(--brand);color:#fff;border-radius:5px;cursor:pointer}
button.ghost{background:transparent;color:var(--brand)}
input[type=text]{font:inherit;padding:6px 10px;border-radius:5px;
border:1px solid var(--border);background:var(--surface);color:var(--text-body)}
#out{width:100%;height:150px;font-family:var(--mono);font-size:.72rem;
margin-top:10px}
footer{margin-top:40px;padding-top:14px;border-top:1px solid var(--border);
font-family:var(--mono);font-size:.7rem;letter-spacing:.1em;
text-transform:uppercase;color:var(--text-muted)}
"""


def chips(kind, qid, classes):
    d = prop[kind].get(qid, {})
    out = []
    for c in classes:
        on = c in d
        conf = d.get(c, "")
        cls = "chip" + (" on" if on else "") + (" draft" if conf == "low" else "")
        t = f' title="draft confidence: {conf}"' if conf else ""
        out.append(f'<span class="{cls}" data-cls="{html.escape(c)}"{t}>'
                   f'{html.escape(c)}</span>')
    return "".join(out)


def section(kind, classes):
    qs = L["queries"][kind]
    parts = [f'<h2>{kind.capitalize()} scenes</h2>',
             f'<p class="note">{len(classes)} object classes appear across the '
             f'{kind} scenes. Tick every class that genuinely answers the '
             f'query. Leaving a query empty is a valid answer and means '
             f'nothing in these rooms answers it.</p>']
    for qid, text in qs.items():
        parts.append(
            f'<div class="card"><p class="qh">{kind} · {qid}</p>'
            f'<p class="qt">{html.escape(text)}</p>'
            f'<div class="chips" data-kind="{kind}" data-qid="{qid}">'
            f'{chips(kind, qid, classes)}</div></div>')
    return "".join(parts)


HTML = f"""<title>Relevance Labelling Sheet</title>
<style>{CSS}</style>
<div class="wrap">
<p class="eyebrow">SSP-SLAM · ConceptGraphs Table III</p>
<h1>Which objects answer these queries?</h1>
<p class="lede">ConceptGraphs published these twenty text queries but never
published which objects count as correct answers. Without that, their
retrieval table cannot be scored by anyone else. This sheet rebuilds that
missing half. It takes about ten minutes.</p>

<div class="warn">
<p style="margin:0 0 8px"><b>The green chips are a machine-written draft, not
answers.</b> They are a starting point so you are correcting rather than
starting from blank. Chips marked with an asterisk are ones the draft itself
flagged as arguable, so start there.</p>
<p class="note" style="margin:0">Judge by the class name alone, as a person
would: does a <em>lamp</em> answer "something to add light into the room"?
Ignore anything about how our system performs; the point is a fair target that
was written independently of it.</p>
</div>

<div class="bar">
  <label>Your name <input type="text" id="who" placeholder="so we can compare annotators"></label>
  <button class="ghost" id="exp" type="button">Export JSON</button>
  <span class="note" id="status">Ticks save automatically and are shared.</span>
</div>

<artifact-sync>
<div id="sheet">
{section("room", room_cls)}
{section("office", office_cls)}
</div>
</artifact-sync>

<textarea id="out" hidden></textarea>

<footer>ICNS · Western Sydney University · Queries verbatim from
ConceptGraphs Appendix A4 · Relevance labels are ours, not theirs</footer>
</div>

<script>
(function(){{
"use strict";
var sheet=document.getElementById("sheet");
var status=document.getElementById("status");
sheet.addEventListener("click",function(e){{
  var c=e.target.closest(".chip");
  if(!c)return;
  c.classList.toggle("on");
}});
function collect(){{
  var o={{annotator:(document.getElementById("who")||{{}}).value||"",
          note:"human relevance labels; green=relevant",labels:{{}}}};
  [].forEach.call(document.querySelectorAll(".chips"),function(g){{
    var k=g.dataset.kind,q=g.dataset.qid;
    o.labels[k]=o.labels[k]||{{}};
    o.labels[k][q]=[].filter.call(g.querySelectorAll(".chip"),function(c){{
      return c.classList.contains("on");}}).map(function(c){{
      return c.dataset.cls;}});
  }});
  return o;
}}
document.getElementById("exp").addEventListener("click",function(){{
  var t=document.getElementById("out");
  t.hidden=false;t.value=JSON.stringify(collect(),null,1);t.select();
}});
document.addEventListener("claude:sync-off",function(){{
  status.textContent="Read-only view: ticks are NOT saved. Use Export JSON "+
    "and send the text back.";
  status.style.color="var(--warn)";
}});
}})();
</script>
"""

os.makedirs("outputs/table3", exist_ok=True)
open(OUT, "w", encoding="utf-8", newline="").write(HTML)
n = sum(len(v) for v in prop["room"].values()) + \
    sum(len(v) for v in prop["office"].values())
print(f"wrote {OUT}  ({os.path.getsize(OUT)/1024:.0f} kB)")
print(f"  room classes {len(room_cls)}, office classes {len(office_cls)}")
print(f"  20 queries, {n} chips pre-ticked as the draft")
