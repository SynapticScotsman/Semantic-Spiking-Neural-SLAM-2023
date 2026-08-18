"""Task R3: displacement predicates on the VSA cognitive map as field algebra.

"X metres east of the anchor" is ONE extra bind on the query key, not a new
map: because the FPE position encoding is a group homomorphism
(S(a) * S(b) = S(a+b) elementwise), unbinding the class-keyed trace with
sem[a] (x) S(v) decodes the SAME field as unbinding with sem[a] alone and
reading it out on a grid offset by +v -- i.e. the decoded field (and its
peak) moves by -v. The demo measures how exactly the algebraic route matches
a from-scratch re-encoding of the offset grid (expected: numerical
precision), plus the argmax displacement error in metres.

Composite demo: "between A and B" = product of the two anchors' (positively
clipped) decoded proximity fields; the peak's geometry vs the A-B segment is
reported raw -- for EVERY distinct pair of anchor classes in the scene, so
the operating envelope of the product composition (it needs the two kernels
to overlap: separation vs the 0.6 m length-scale) is measured, not asserted.
The figure shows the closest pair (stated criterion, not cherry-picked).

Outputs
    outputs/relational_displacement.png   4-panel paper figure (dark ground)
    outputs/relational_displacement.json  all measured numbers

Run from the repo root:  python tools/relational_displacement.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa_cognitive_mapping.classroom_pipeline import ClassroomEncoders  # noqa: E402
from vsa_cognitive_mapping.object_grounding import (  # noqa: E402
    cap_per_class, class_phasors)
from vsa_cognitive_mapping.instance_recall import load_scene  # noqa: E402

HD, GRID, LS = 4096, 64, 0.6         # dim / grid / position length-scale (m)
ASSIGN_RADIUS = 1.0                  # det -> GT instance association (m)
CAP_PER_CLASS = 60                   # bounded per-class insertion, as elsewhere
DELTAS = [0.5, 1.0, 1.5, 2.0]        # commanded displacement magnitudes (m)
DIRECTIONS = {"east": (1.0, 0.0), "west": (-1.0, 0.0),
              "north": (0.0, 1.0), "south": (0.0, -1.0)}

# figure palette (task-pinned)
GROUND, BLUE, MINT = "#05070A", "#3B9BFF", "#7CE0C0"
MAGENTA, TEXT, DIM = "#FF5FD0", "#E9EFF3", "#8A97A0"


# ------------------------------------------------------------------ memory

def memory_half(pts, inst):
    """Assign detections to GT instances (nearest same-class centroid within
    ASSIGN_RADIUS), split each instance's detections in frame order, return
    the first halves -- the memory stream of instance_recall.py."""
    by_cls_inst = defaultdict(list)
    for g in inst:
        by_cls_inst[g["cls"]].append(g)
    assigned = []
    for p in pts:
        cands = by_cls_inst.get(p["cls"], [])
        if not cands:
            continue
        dd = [np.hypot(p["x"] - g["x"], p["y"] - g["y"]) for g in cands]
        j = int(np.argmin(dd))
        if dd[j] <= ASSIGN_RADIUS:
            assigned.append((p, cands[j]["id"]))
    per_inst = defaultdict(list)
    for k, (p, iid) in enumerate(assigned):
        per_inst[iid].append(k)
    mem = []
    for iid, ks in per_inst.items():
        ks = sorted(ks, key=lambda k: assigned[k][0]["frame"])
        mem.extend(assigned[k][0] for k in ks[:len(ks) // 2])
    return mem


def build_scene(scene):
    pts, _emb, inst = load_scene(scene)
    mem_pts = cap_per_class(memory_half(pts, inst), CAP_PER_CLASS)
    classes = sorted({p["cls"] for p in mem_pts})
    sem = class_phasors(classes, HD)
    enc = ClassroomEncoders(HD, 0, LS, 20.0)
    M = np.zeros(HD, np.complex128)
    for p in mem_pts:
        M += sem[p["cls"]] * enc.ctx_pos(p["x"], p["y"]).values
    M /= max(np.abs(M).max(), 1e-12)

    xs = [p["x"] for p in mem_pts] + [g["x"] for g in inst]
    ys = [p["y"] for p in mem_pts] + [g["y"] for g in inst]
    gx = np.linspace(min(xs) - 1.0, max(xs) + 1.0, GRID)
    gy = np.linspace(min(ys) - 1.0, max(ys) + 1.0, GRID)
    return dict(pts=pts, inst=inst, mem_pts=mem_pts, classes=classes,
                sem=sem, enc=enc, M=M, gx=gx, gy=gy)


def grid_phasors(enc, gx, gy, vx=0.0, vy=0.0):
    """S(gx[i]+vx, gy[j]+vy) for the whole grid, complex128, shape
    (GRID*GRID, HD), row-major j (y) outer / i (x) inner -- bit-identical to
    per-point enc.ctx_pos (same base**float then elementwise product)."""
    Vx = np.stack([enc.Bx.values ** float((x + vx) / enc.pos_l) for x in gx])
    Vy = np.stack([enc.By.values ** float((y + vy) / enc.pos_l) for y in gy])
    return (Vy[:, None, :] * Vx[None, :, :]).reshape(GRID * GRID, HD)


def decode_field(residual, G):
    """Similarity field Re<G, residual>/HD, shape (GRID, GRID) = (y, x)."""
    return (G.conj() @ residual).real.reshape(GRID, GRID) / HD


def peak_xy(F, gx, gy):
    j = int(np.argmax(F))
    return float(gx[j % GRID]), float(gy[j // GRID])


# ------------------------------------------------------------------ demos

def exactness_sweep(S, anchor):
    """Unbind with sem[a] (x) S(v) vs the unshifted residual read out on a
    from-scratch re-encoded grid offset by +v. Also checks the sign of the
    peak displacement empirically."""
    enc, M, sem, gx, gy = S["enc"], S["M"], S["sem"], S["gx"], S["gy"]
    G0 = grid_phasors(enc, gx, gy)
    res_a = M / sem[anchor]
    F0 = decode_field(res_a, G0)
    p0 = np.array(peak_xy(F0, gx, gy))
    step = float(max(gx[1] - gx[0], gy[1] - gy[0]))

    rows, n_minus, n_plus = [], 0, 0
    for d in DELTAS:
        for name, (ux, uy) in DIRECTIONS.items():
            v = np.array([d * ux, d * uy])
            # (a) algebraic: one extra bind on the query key
            res_shift = res_a / enc.ctx_pos(v[0], v[1]).values
            F_alg = decode_field(res_shift, G0)
            # (b) reference: unshifted residual on a grid offset by +v
            F_ref = decode_field(res_a, grid_phasors(enc, gx, gy, v[0], v[1]))
            max_diff = float(np.abs(F_alg - F_ref).max())

            pk = np.array(peak_xy(F_alg, gx, gy))
            exp_minus, exp_plus = p0 - v, p0 + v
            e_minus = float(np.hypot(*(pk - exp_minus)))
            e_plus = float(np.hypot(*(pk - exp_plus)))
            n_minus += e_minus < e_plus
            n_plus += e_plus < e_minus
            in_grid = bool(gx[0] <= exp_minus[0] <= gx[-1]
                           and gy[0] <= exp_minus[1] <= gy[-1])
            rows.append(dict(delta_m=d, direction=name, dxy=list(v),
                             max_abs_field_diff=max_diff,
                             argmax_err_m=float(round(e_minus, 4)),
                             expected_peak=[float(round(c, 3))
                                            for c in exp_minus],
                             decoded_peak=[float(round(c, 3)) for c in pk],
                             expected_in_grid=in_grid))
    sign = ("unbinding with S(v) shifts the decoded field peak by -v "
            f"({n_minus}/{n_minus + n_plus} sweep cases nearer -v than +v)")
    in_grid_errs = [r["argmax_err_m"] for r in rows if r["expected_in_grid"]]
    return dict(anchor=anchor,
                base_peak=[float(round(c, 3)) for c in p0],
                grid_step_m=round(step, 4), sweep=rows,
                exactness_max=float(max(r["max_abs_field_diff"] for r in rows)),
                argmax_err_max_in_grid_m=(max(in_grid_errs)
                                          if in_grid_errs else None),
                n_expected_out_of_grid=sum(not r["expected_in_grid"]
                                           for r in rows),
                sign_convention=sign), F0, p0, G0


def pair_geometry(FA, FB, gx, gy):
    """Product of the positively clipped fields; peak geometry vs the segment
    joining the two anchors' decoded peaks."""
    P = np.clip(FA, 0, None) * np.clip(FB, 0, None)
    A = np.array(peak_xy(FA, gx, gy))
    B = np.array(peak_xy(FB, gx, gy))
    pk = np.array(peak_xy(P, gx, gy))
    ab = B - A
    seg = float(np.hypot(*ab))
    t = float(np.dot(pk - A, ab) / max(seg ** 2, 1e-12))
    perp = float(np.hypot(*((pk - A) - t * ab)))
    return P, A, B, pk, seg, t, perp


def between_demo(S):
    """Run the 'between' product composition for EVERY distinct pair of
    anchor classes -- the raw operating envelope of the composition (kernel
    overlap vs anchor separation). The figure pair is the one with the
    smallest decoded-peak separation (stated criterion, decided before
    looking at the outcomes)."""
    gx, gy = S["gx"], S["gy"]
    G0 = grid_phasors(S["enc"], gx, gy)
    fields = {c: decode_field(S["M"] / S["sem"][c], G0) for c in S["classes"]}
    rows = []
    for i, a in enumerate(S["classes"]):
        for b in S["classes"][i + 1:]:
            P, A, B, pk, seg, t, perp = pair_geometry(fields[a], fields[b],
                                                      gx, gy)
            rows.append(dict(class_a=a, class_b=b,
                             peak_a=[float(round(c, 3)) for c in A],
                             peak_b=[float(round(c, 3)) for c in B],
                             product_peak=[float(round(c, 3)) for c in pk],
                             segment_len_m=round(seg, 3),
                             along_fraction=round(t, 3),
                             perp_dist_m=round(perp, 3)))
    rows.sort(key=lambda r: r["segment_len_m"])
    chosen = rows[0]
    P, A, B, pk, _, _, _ = pair_geometry(fields[chosen["class_a"]],
                                         fields[chosen["class_b"]], gx, gy)
    chosen = dict(chosen)
    chosen["criterion"] = ("pair of distinct anchor classes with the "
                           "smallest decoded-peak separation (product "
                           "composition requires kernel overlap; length "
                           f"scale {LS} m)")
    # Replica ships unlabelled geometry as class_id -1 (class_name 'undefined'): real objects with no semantic class. They must never enter a class list.
    chosen["gt_instances"] = {c: [[g["x"], g["y"]] for g in S["inst"]
                                  if g.get("cls") != "class_-1"
                                  if g["cls"] == c]
                              for c in (chosen["class_a"], chosen["class_b"])}
    return chosen, rows, P, A, B, pk


# ------------------------------------------------------------------ figure

def style_ax(ax, title):
    ax.set_facecolor(GROUND)
    ax.set_title(title, color=TEXT, fontsize=9.5, pad=8)
    ax.tick_params(colors=DIM, labelsize=7)
    for s in ax.spines.values():
        s.set_color("#1A222B")
    ax.set_xlabel("x (m)", color=DIM, fontsize=8)
    ax.set_ylabel("y (m)", color=DIM, fontsize=8)


def make_figure(office, room, ex_o, F0, p0, fig_shift, bt, P, A, B, pk,
                out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap_field = LinearSegmentedColormap.from_list("field", [GROUND, BLUE])
    cmap_err = LinearSegmentedColormap.from_list("err", [GROUND, DIM, TEXT])

    gxo, gyo = office["gx"], office["gy"]
    ext_o = [gxo[0], gxo[-1], gyo[0], gyo[-1]]
    gxr, gyr = room["gx"], room["gy"]
    ext_r = [gxr[0], gxr[-1], gyr[0], gyr[-1]]

    fig, axs = plt.subplots(1, 4, figsize=(21, 5.6))
    fig.patch.set_facecolor(GROUND)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.78, bottom=0.15,
                        wspace=0.30)

    a = ex_o["anchor"]
    gt_a = [(g["x"], g["y"]) for g in office["inst"] if g["cls"] == a]

    # (i) anchor field
    ax = axs[0]
    ax.imshow(np.clip(F0, 0, None), origin="lower", extent=ext_o,
              cmap=cmap_field, aspect="equal")
    ax.scatter(*zip(*gt_a), s=70, facecolor="none", edgecolor=MINT,
               linewidth=1.6, label="GT instances")
    ax.plot([p0[0]], [p0[1]], "x", color=MAGENTA, ms=9, mew=2,
            label="decoded peak")
    style_ax(ax, f"(i)  office3 · anchor field  M ⊘ sem[{a}]\n"
                 f"peak ({p0[0]:+.2f}, {p0[1]:+.2f}) m")
    ax.legend(loc="lower left", fontsize=7, facecolor=GROUND,
              edgecolor="#1A222B", labelcolor=TEXT)

    # (ii) commanded eastward shift
    ax = axs[1]
    v = fig_shift["v"]
    ax.imshow(np.clip(fig_shift["F"], 0, None), origin="lower", extent=ext_o,
              cmap=cmap_field, aspect="equal")
    tgt = fig_shift["target"]
    ax.scatter([tgt[0]], [tgt[1]], s=90, facecolor="none", edgecolor=MINT,
               linewidth=1.8, label="commanded target")
    dp = fig_shift["peak"]
    ax.plot([dp[0]], [dp[1]], "x", color=MAGENTA, ms=9, mew=2,
            label="decoded peak")
    ax.annotate("", xy=tgt, xytext=p0,
                arrowprops=dict(arrowstyle="->", color=DIM, lw=1.2))
    style_ax(ax, f"(ii)  “{v[0]:.1f} m east of the {a}”:\n"
                 f"M ⊘ (sem[{a}] ⊗ S(−Δ)) · argmax err "
                 f"{fig_shift['err']:.3f} m (grid {ex_o['grid_step_m']:.3f} m)")
    ax.legend(loc="lower left", fontsize=7, facecolor=GROUND,
              edgecolor="#1A222B", labelcolor=TEXT)

    # (iii) exactness error map
    ax = axs[2]
    im = ax.imshow(fig_shift["errmap"], origin="lower", extent=ext_o,
                   cmap=cmap_err, aspect="equal")
    cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
    cb.ax.tick_params(colors=DIM, labelsize=7)
    cb.ax.yaxis.get_offset_text().set_color(DIM)
    cb.outline.set_edgecolor("#1A222B")
    style_ax(ax, "(iii)  |algebraic − re-encoded reference|\n"
                 f"max |Δfield| = {fig_shift['maxdiff']:.1e}  "
                 f"(sweep max {ex_o['exactness_max']:.1e})")

    # (iv) between A and B (room0)
    ax = axs[3]
    ax.imshow(P / max(P.max(), 1e-30), origin="lower", extent=ext_r,
              cmap=cmap_field, aspect="equal")
    for c, mk in ((bt["class_a"], "o"), (bt["class_b"], "s")):
        g = np.array(bt["gt_instances"][c])
        ax.scatter(g[:, 0], g[:, 1], s=18, marker=mk, color=MINT, alpha=0.5,
                   label=f"GT {c}")
    ax.plot([A[0], B[0]], [A[1], B[1]], "-", color=DIM, lw=1.2, alpha=0.9)
    ax.scatter([A[0], B[0]], [A[1], B[1]], s=70, facecolor="none",
               edgecolor=MINT, linewidth=1.6, label="anchor peaks")
    ax.plot([pk[0]], [pk[1]], "x", color=MAGENTA, ms=9, mew=2,
            label="product peak")
    style_ax(ax, f"(iv)  room0 · “between {bt['class_a']} & "
                 f"{bt['class_b']}” = F₊ × F₊\n"
                 f"perp {bt['perp_dist_m']:.2f} m · along "
                 f"{bt['along_fraction']:.2f} · |AB| "
                 f"{bt['segment_len_m']:.2f} m")
    ax.legend(loc="lower left", fontsize=6.5, facecolor=GROUND,
              edgecolor="#1A222B", labelcolor=TEXT)

    fig.suptitle("Displacement predicates as field algebra on one 4096-d VSA "
                 "trace — shift = one extra bind on the query key",
                 color=TEXT, fontsize=12.5, y=0.96)
    fig.text(0.5, 0.03,
             "sign convention (measured): unbinding with S(v) shifts the "
             "decoded field by −v, so “Δ east” uses "
             "S(−Δ)   ·   memory = first-half detections, "
             f"≤{CAP_PER_CLASS}/class",
             color=DIM, fontsize=8.5, ha="center")
    fig.savefig(out_png, dpi=170, facecolor=GROUND)
    plt.close(fig)


# -------------------------------------------------------------------- main

def main():
    os.makedirs("outputs", exist_ok=True)
    report = dict(hd_dim=HD, grid=GRID, length_scale_m=LS,
                  assign_radius_m=ASSIGN_RADIUS, cap_per_class=CAP_PER_CLASS,
                  field_units="Re<S(x,y), residual>/D on the max-normalized "
                              "trace", scenes={})

    # ---- office3: exactness sweep on the dining-table anchor -------------
    office = build_scene("office3")
    print(f"office3: {len(office['mem_pts'])} memory detections, "
          f"{len(office['classes'])} classes: {office['classes']}")
    ex_o, F0, p0, G0 = exactness_sweep(office, "dining table")
    print(f"\noffice3 exactness (anchor={ex_o['anchor']}, base peak "
          f"{ex_o['base_peak']}, grid step {ex_o['grid_step_m']} m):")
    for r in ex_o["sweep"]:
        print(f"  Δ={r['delta_m']:.1f} m {r['direction']:<5} "
              f"max|Δfield|={r['max_abs_field_diff']:.3e}  "
              f"argmax err={r['argmax_err_m']:.3f} m  "
              f"in_grid={r['expected_in_grid']}")
    print(f"  SWEEP MAX |algebraic-reference| = {ex_o['exactness_max']:.3e}")
    print(f"  {ex_o['sign_convention']}")
    report["scenes"]["office3"] = dict(
        n_memory_detections=len(office["mem_pts"]),
        classes=office["classes"], exactness=ex_o)

    # ---- room0: repeat the sweep (couch anchor) + between demo -----------
    room = build_scene("room0")
    print(f"\nroom0: {len(room['mem_pts'])} memory detections, "
          f"{len(room['classes'])} classes: {room['classes']}")
    ex_r, _, _, _ = exactness_sweep(room, "couch")
    print(f"room0 exactness (anchor=couch): sweep max "
          f"|Δfield| = {ex_r['exactness_max']:.3e}; in-grid argmax err max "
          f"{ex_r['argmax_err_max_in_grid_m']:.3f} m; "
          f"{ex_r['n_expected_out_of_grid']} expected peak(s) out of grid")
    bt, all_pairs, P, A, B, pk = between_demo(room)
    print("\nroom0 'between' product composition, every class pair "
          "(sorted by anchor separation):")
    for r in all_pairs:
        print(f"  {r['class_a']:>13} - {r['class_b']:<13} "
              f"|AB|={r['segment_len_m']:>5.2f} m  "
              f"along={r['along_fraction']:+.3f}  perp={r['perp_dist_m']:.3f} m")
    print(f"figure pair ({bt['criterion'].split('(')[0].strip()}): "
          f"{bt['class_a']}–{bt['class_b']}, |AB|={bt['segment_len_m']} m, "
          f"product peak {bt['product_peak']}, perp {bt['perp_dist_m']} m, "
          f"along-fraction {bt['along_fraction']}")
    report["scenes"]["room0"] = dict(
        n_memory_detections=len(room["mem_pts"]),
        classes=room["classes"], exactness=ex_r,
        between_figure_pair=bt, between_all_pairs=all_pairs)

    # ---- figure: commanded 1.0 m EAST of the office3 anchor --------------
    enc, sem, M = office["enc"], office["sem"], office["M"]
    v = np.array([1.0, 0.0])
    res = (M / sem["dining table"]) / enc.ctx_pos(-v[0], -v[1]).values
    F_shift = decode_field(res, G0)
    F_ref = decode_field(M / sem["dining table"],
                         grid_phasors(enc, office["gx"], office["gy"],
                                      -v[0], -v[1]))
    errmap = np.abs(F_shift - F_ref)
    dp = np.array(peak_xy(F_shift, office["gx"], office["gy"]))
    tgt = p0 + v
    fig_shift = dict(v=v, F=F_shift, errmap=errmap, target=tgt, peak=dp,
                     err=float(np.hypot(*(dp - tgt))),
                     maxdiff=float(errmap.max()))
    report["figure_panel_ii"] = dict(
        commanded=[1.0, 0.0], query_key="sem[dining table] (x) S(-1, 0)",
        target=[float(round(c, 3)) for c in tgt],
        decoded_peak=[float(round(c, 3)) for c in dp],
        argmax_err_m=float(round(fig_shift["err"], 4)),
        max_abs_field_diff=fig_shift["maxdiff"])

    out_png = os.path.join("outputs", "relational_displacement.png")
    make_figure(office, room, ex_o, F0, p0, fig_shift, bt, P, A, B, pk,
                out_png)
    print(f"\nsaved {out_png}")

    out_json = os.path.join("outputs", "relational_displacement.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
