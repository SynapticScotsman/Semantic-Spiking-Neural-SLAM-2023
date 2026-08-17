"""Paper figure: the Replica head-to-head and the degradation crossover.

Two panels, both drawn from measured artifacts and nothing else:
  (a) per-scene mAcc, ConceptGraphs (their system, our measurement, their
      scorer) vs the 32 KB trace, with their published 8-scene mean line.
      Source: outputs/batch1/table2_full.json
  (b) retained fraction of each system's own clean score under three
      frontend degradations (worst-level annotations), room0 curves.
      Source: outputs/degradation_sweep.json

Run with the venv (matplotlib lives there):
    .venv_lejepa/Scripts/python.exe tools/make_replica_headtohead_figure.py
Writes paper/figures/replica_headtohead.pdf and .png.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

THEIRS = "#C33A50"      # ICNS crimson-bright
OURS = "#2F6B46"        # ICNS good-green
REF = "#8A5A12"         # warn/amber for the published line
GREY = "#6E7278"

SCENES = ["room0", "room1", "room2", "office0",
          "office1", "office2", "office3", "office4"]
PUB = 0.4063


def main():
    t2 = json.load(open("outputs/batch1/table2_full.json"))
    deg = json.load(open("outputs/degradation_sweep.json"))["room0"]

    plt.rcParams.update({
        "font.size": 7.5, "font.family": "sans-serif",
        "axes.linewidth": 0.6, "axes.edgecolor": GREY,
        "xtick.color": GREY, "ytick.color": GREY,
        "axes.labelcolor": "black", "legend.frameon": False,
    })
    fig = plt.figure(figsize=(7.0, 2.15))
    gs = fig.add_gridspec(1, 4, width_ratios=[2.6, 1, 1, 1],
                          wspace=0.42, left=0.075, right=0.975,
                          top=0.87, bottom=0.24)

    # ---------------- (a) per-scene bars ----------------
    ax = fig.add_subplot(gs[0])
    th = [t2["per_scene"][s]["theirs"]["macc"] for s in SCENES]
    ou = [t2["per_scene"][s]["ours"]["macc"] for s in SCENES]
    sd = [t2["per_scene"][s]["ours_seed_sd"] for s in SCENES]
    x = np.arange(len(SCENES))
    ax.bar(x - 0.19, th, 0.36, color=THEIRS, label="ConceptGraphs (measured)")
    ax.bar(x + 0.19, ou, 0.36, color=OURS, yerr=sd, error_kw=dict(lw=0.6),
           label="32 KB trace (ours)")
    ax.axhline(PUB, color=REF, lw=0.8, ls=(0, (4, 3)))
    ax.text(3.5, PUB + 0.015, "their published mean 0.406", color=REF,
            fontsize=6.3, ha="center")
    for i, (a_, b_) in enumerate(zip(th, ou)):
        if b_ > a_:
            ax.annotate("", xy=(i + 0.19, b_ + 0.055),
                        xytext=(i + 0.19, b_ + 0.012),
                        arrowprops=dict(arrowstyle="-", color=OURS, lw=0))
            ax.text(i + 0.19, b_ + 0.015, "*", ha="center", color=OURS,
                    fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("office", "off") for s in SCENES],
                       fontsize=6.2, rotation=28, ha="right")
    ax.set_ylabel("mAcc (their scorer)")
    ax.set_ylim(0, 0.68)
    ax.legend(loc="upper left", fontsize=6.3, handlelength=1.2,
              borderaxespad=0.1)
    ax.set_title("(a) all 8 Replica scenes, their protocol",
                 fontsize=7.5, loc="left")
    ax.spines[["top", "right"]].set_visible(False)

    # ---------------- (b) degradation retention, 3 small multiples ------
    modes = [("drop", "sparse coverage", "fraction of points kept", True),
             ("label", "label corruption", "fraction corrupted", False),
             ("jitter", "pose jitter", "$\\sigma$ (m)", False)]
    for j, (key, title, xlab, revx) in enumerate(modes):
        axd = fig.add_subplot(gs[j + 1])
        rows = deg[key]
        lv = [r["level"] for r in rows]
        axd.plot(lv, [100 * r["theirs_rel"] for r in rows], color=THEIRS,
                 lw=1.2, marker="o", ms=2.2)
        axd.plot(lv, [100 * r["ours_rel"] for r in rows], color=OURS,
                 lw=1.2, marker="o", ms=2.2)
        if revx:
            axd.invert_xaxis()
            axd.set_xscale("log")
        axd.set_ylim(30, 115)
        axd.axhline(100, color=GREY, lw=0.4, ls=":")
        axd.set_title(f"({chr(98 + j)}) {title}", fontsize=7.0, loc="left")
        axd.set_xlabel(xlab, fontsize=6.3)
        if j == 0:
            axd.set_ylabel("% of own clean\nscore retained", fontsize=6.5)
        w_t = 100 * rows[-1]["theirs_rel"]
        w_o = 100 * rows[-1]["ours_rel"]
        axd.annotate(f"{w_t:.0f}%", xy=(lv[-1], w_t), fontsize=6.3,
                     color=THEIRS, xytext=(-3, -10),
                     textcoords="offset points", ha="right")
        axd.annotate(f"{w_o:.0f}%", xy=(lv[-1], w_o), fontsize=6.3,
                     color=OURS, xytext=(-3, 5),
                     textcoords="offset points", ha="right")
        axd.spines[["top", "right"]].set_visible(False)
        axd.tick_params(labelsize=6)

    os.makedirs("paper/figures", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"paper/figures/replica_headtohead.{ext}", dpi=300)
    print("wrote paper/figures/replica_headtohead.{pdf,png}")
    print(f"means: theirs {np.mean(th):.3f}  ours {np.mean(ou):.3f}  "
          f"published {PUB}")


if __name__ == "__main__":
    sys.exit(main())
