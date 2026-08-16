"""One-screen summary of the Stage-1 guard JSONs. Reads numbers, never prose."""
import glob
import json

print("%-22s %-15s %-16s %18s %18s" % (
    "mechanism", "verdict", "best variant", "dmacc (mean+-sd)",
    "dmf1 (mean+-sd)"))
print("-" * 95)
rows = []
for p in sorted(glob.glob("outputs/batch1/h*.json")):
    js = json.load(open(p))
    assert js["verdict_allowed"], (js["mechanism"], js["problems"])
    b = js["best_variant"]
    d = js["variants"][b]["deltas"]
    ma, f1 = d["macc"], d["mf1"]
    print("%-22s %-15s %-16s %+9.4f +- %.4f %+9.4f +- %.4f" % (
        js["mechanism"], js["verdict"], b,
        ma["mean"], ma["sd"], f1["mean"], f1["sd"]))
    rows.append(js)
print()
hits = misses = 0
for js in rows:
    for q in js.get("predictions", []):
        tag = "HIT " if q["hit"] else "MISS"
        hits += q["hit"]
        misses += (not q["hit"])
        print(f"  {tag} {q['id']:<10} measured={q['measured']}  {q['text']}")
print(f"\nprediction record: {hits} HIT / {misses} MISS")
print("\nper-variant statuses (everything, not just the best):")
for js in rows:
    for lab, v in sorted(js["variants"].items()):
        d = v["deltas"]
        print(f"  {js['mechanism']:<20} {lab:<16} {v['status']:<15} "
              f"dmacc {d['macc']['mean']:+.4f}+-{d['macc']['sd']:.4f}  "
              f"dmf1 {d['mf1']['mean']:+.4f}+-{d['mf1']['sd']:.4f}")
