"""Does THEIR map also get the 62% win_nearer local_loss cells wrong?

Separates three candidate causes of the local_loss 62% branch:
  (1) shared label error   -> their dense NN also answers wrong
  (2) our CAP / sparsity   -> uncapped stream's GT point IS nearer
  (3) our decode           -> everything else
"""
import os, sys, json
import numpy as np
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from collab_tasks.batch1.common import SCENES, load_scene, CG_EXCLUDE_6
from collab_tasks.batch1.error_decomposition import decompose
from vsa_cognitive_mapping.object_grounding import cap_per_class
from scipy.spatial import cKDTree

R = []
for s in SCENES:
    data = load_scene(s)
    cats, _, pred = decompose(data)
    gt, xyz, a, b = data["gt"], data["xyz"], data["a"], data["b"]

    Z = np.load(f"student_gpu_package/handoff/{s}_cgfront/cg_labels.npz",
                allow_pickle=True)
    d3, j3 = cKDTree(Z["xyz"]).query(xyz, k=1)          # 05_score.transfer, 3D
    their = Z["pred_class"].astype(str)[j3]
    their = np.where(d3 > 0.05, "__none__", their)

    capped = cap_per_class(list(data["obs"]), 400, seed=11)
    cxy, uxy = {}, {}
    for o in capped:
        cxy.setdefault(o["cls"], []).append((o["x"], o["y"]))
    for o in data["obs"]:
        uxy.setdefault(o["cls"], []).append((o["x"], o["y"]))
    cxy = {c: np.array(v) for c, v in cxy.items()}
    uxy = {c: np.array(v) for c, v in uxy.items()}

    for i in sorted(cats["local_loss"]):
        g, w = gt[i], pred[i]
        px, py = float(xyz[i, a]), float(xyz[i, b])
        dg = np.hypot(*(cxy[g] - [px, py]).T).min()
        dw = np.hypot(*(cxy[w] - [px, py]).T).min()
        ug = np.hypot(*(uxy[g] - [px, py]).T).min()
        uw = np.hypot(*(uxy[w] - [px, py]).T).min() if w in uxy else np.inf
        R.append(dict(scene=s, gt=g, win=w, their=str(their[i]),
                      d_gt=float(dg), d_win=float(dw),
                      u_gt=float(ug), u_win=float(uw),
                      d_their=float(d3[i])))
    print(f"{s}: {len(cats['local_loss'])} local_loss", flush=True)

n = len(R)
W = [r for r in R if r["d_win"] <= r["d_gt"]]     # the 62% branch
G = [r for r in R if r["d_win"] > r["d_gt"]]
def pc(sub, f): return 100 * sum(1 for r in sub if f(r)) / max(len(sub), 1)

print(f"\n{n} local_loss cells; win_nearer branch {len(W)} ({100*len(W)/n:.1f}%),"
      f" gt_nearer {len(G)} ({100*len(G)/n:.1f}%)")

for nm, sub in (("WIN-NEARER (the 62%)", W), ("GT-NEARER (38%)", G)):
    print(f"\n== {nm}: n={len(sub)} ==")
    print(f"  THEIR dense NN is CORRECT here      : {pc(sub, lambda r: r['their']==r['gt']):5.1f}%")
    print(f"  THEIR dense NN gives OUR winner     : {pc(sub, lambda r: r['their']==r['win']):5.1f}%")
    print(f"  THEIR dense NN wrong (any)          : {pc(sub, lambda r: r['their']!=r['gt']):5.1f}%")
    print(f"  UNCAPPED stream would put GT nearer : {pc(sub, lambda r: r['u_gt']<r['u_win']):5.1f}%")
    print(f"  median d_win {np.median([r['d_win'] for r in sub]):.3f} m   "
          f"median d_gt {np.median([r['d_gt'] for r in sub]):.3f} m")
    print(f"  median u_win {np.median([r['u_win'] for r in sub]):.3f} m   "
          f"median u_gt {np.median([r['u_gt'] for r in sub]):.3f} m")

# the money cross-tab on the 62%
a1 = [r for r in W if r["their"] == r["gt"]]
a2 = [r for r in W if r["their"] != r["gt"]]
print(f"\nCROSS-TAB on the {len(W)} win-nearer cells:")
print(f"  A. their map RIGHT, we are wrong  -> OUR loss      : {len(a1)} ({100*len(a1)/len(W):.1f}%)")
print(f"  B. their map ALSO wrong           -> SHARED ceiling: {len(a2)} ({100*len(a2)/len(W):.1f}%)")
print(f"     of A, uncapped stream puts GT nearer: {pc(a1, lambda r: r['u_gt']<r['u_win']):.1f}%"
      "  <- recoverable by CAP/density alone")
print(f"     of B, their answer == our answer   : {pc(a2, lambda r: r['their']==r['win']):.1f}%")
print("\ntop confusion pairs in branch B (shared):")
for (g, w), c in Counter((r["gt"], r["their"]) for r in a2).most_common(8):
    print(f"   GT {g:>14} -> their {w:<14} {c:>5}")

# what would our mAcc be if we perfectly matched their dense NN on local_loss?
print("\nper-scene: their-correct fraction on the win-nearer branch")
for s in SCENES:
    sub = [r for r in W if r["scene"] == s]
    if sub:
        print(f"  {s:<9} n={len(sub):>5}  their right {pc(sub, lambda r: r['their']==r['gt']):5.1f}%")

json.dump(dict(n=n, n_win_nearer=len(W), their_right_in_W=len(a1),
               their_wrong_in_W=len(a2), cells=R),
          open("outputs/batch1/shared_ceiling_probe.json", "w"), indent=1)
print("\nwrote outputs/batch1/shared_ceiling_probe.json")
