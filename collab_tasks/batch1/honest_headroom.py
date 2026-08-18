"""Honest local_loss headroom: flip ONLY the cells their own NN decoder gets
right on the same shared input. Everything else in local_loss is a shared
frontend limit, not a backend deficit.
"""
import os, sys, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from collab_tasks.batch1.common import SCENES, load_scene, score
from collab_tasks.batch1.error_decomposition import decompose
from scipy.spatial import cKDTree

rows = []
for s in SCENES:
    data = load_scene(s)
    cats, _, pred = decompose(data)
    gt, xyz = data["gt"], data["xyz"]
    Z = np.load(f"student_gpu_package/handoff/{s}_cgfront/cg_labels.npz",
                allow_pickle=True)
    d3, j3 = cKDTree(Z["xyz"]).query(xyz, k=1)
    their = np.where(d3 > 0.05, "__none__", Z["pred_class"].astype(str)[j3])

    m = np.array(sorted(cats["local_loss"]), int)
    base = score(gt, pred)["macc"]
    p_all = pred.copy(); p_all[m] = gt[m]
    ok = m[their[m] == gt[m]]                     # their NN right here
    p_ok = pred.copy(); p_ok[ok] = gt[ok]
    p_th = pred.copy(); p_th[m] = their[m]        # adopt their NN wholesale
    rows.append(dict(scene=s, n_ll=len(m), n_ok=len(ok),
                     base=base, all_fixed=score(gt, p_all)["macc"],
                     their_right=score(gt, p_ok)["macc"],
                     adopt_theirs=score(gt, p_th)["macc"]))
    print(f"{s}: {len(m)} local_loss, {len(ok)} their-NN-right", flush=True)

print(f"\n{'scene':<9}{'n_ll':>6}{'n_ok':>6}{'base':>8}{'ALLfix':>8}"
      f"{'ourFix':>8}{'adopt':>8}")
for r in rows:
    print(f"{r['scene']:<9}{r['n_ll']:>6}{r['n_ok']:>6}{r['base']:>8.3f}"
          f"{r['all_fixed']:>8.3f}{r['their_right']:>8.3f}{r['adopt_theirs']:>8.3f}")
f = lambda k: float(np.mean([r[k] for r in rows]))
print(f"{'MEAN':<9}{sum(r['n_ll'] for r in rows):>6}"
      f"{sum(r['n_ok'] for r in rows):>6}{f('base'):>8.3f}"
      f"{f('all_fixed'):>8.3f}{f('their_right'):>8.3f}{f('adopt_theirs'):>8.3f}")
print(f"\nheadroom if ALL local_loss fixed      : {f('all_fixed')-f('base'):+.3f}"
      "   <- the number batch-1 was chasing")
print(f"headroom their own NN can justify     : {f('their_right')-f('base'):+.3f}"
      "   <- honest backend headroom")
print(f"headroom from adopting their NN as-is : {f('adopt_theirs')-f('base'):+.3f}")
json.dump(rows, open("outputs/batch1/honest_headroom.json", "w"), indent=1)
