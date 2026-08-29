"""E3: should the scene map store places, or vectors to objects?

FINDINGS.md sec.15 leg 1.  Object vector cells (Hoydal et al., Nature 2019)
fire at a fixed **distance and direction from an object**, and their headline
property is that the tuning **generalises across environments** -- the same
cell, the same vector, a different room.  The scene map here does not do that:
it stores ``ID (*) S_allo(p)``, an allocentric position, which is meaningful
only in the frame it was built in.

Sec.15 flagged this as the honest wrinkle and sec.16 E3 as the experiment.  The
scene half is the untested half of this whole document -- every measured result
in sec.1 to sec.16 is about the view circle.

Three codes for the same observations::

    allo  M = sum_o  ID_o (*) S(p_o)                  world frame
    ego   M = sum_o  ID_o (*) S(p_o - p_robot)        viewer-relative, the OVC form
    rel   M = sum_o  ID_o (*) S(p_o - p_anchor)       object-relative

and four conditions, of which only the first is what the map was built for:

    same         query in the layout it was built in
    translated   the whole scene and the robot shift together
    rotated      the whole scene turns about the robot
    rearranged   objects move independently -- structure genuinely destroyed

A code that survives *translated* has a claim to environment-generality.  A
code that survives *rearranged* would be reporting stale information, so
failing there is the correct behaviour and is included as a control against
reading the other columns too generously.

Section [C] measures the mechanism that makes the egocentric form worth having:
because ``S(a) (*) S(b) = S(a + b)``, moving the robot updates the vectors to
*every* object with a **single bind**, with no per-object work and no global
frame anywhere.  That is path integration of object vectors, and it is the one
operation an allocentric map cannot do at all.

    python experiments/run_scene_frames.py
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sspslam.objectmap import AtomVocab, bind, bundle, unbind  # noqa: E402
from sspslam.sspspace import HexagonalSSPSpace  # noqa: E402

CODES = ("allo", "ego", "rel")
CONDITIONS = ("same", "translated", "rotated", "rearranged")


def make_space(ssp_dim, bound, length_scale, seed):
    return HexagonalSSPSpace(2, ssp_dim=ssp_dim,
                             domain_bounds=np.array([[-bound, bound]] * 2),
                             length_scale=length_scale,
                             rng=np.random.default_rng(seed))


def build(space, vocab, names, pos, robot, anchor):
    """One memory per code, from the same observations."""
    def mem(offsets):
        return bundle(np.stack([bind(vocab.mint(n), space.encode(v).reshape(-1))
                                for n, v in zip(names, offsets)]))
    return {"allo": mem(pos),
            "ego": mem(pos - robot),
            "rel": mem(pos - anchor)}


def decode_positions(space, vocab, mem, names, grid, pts, robot, anchor, code):
    """Read every object's world position back out of one memory."""
    out = []
    for n in names:
        residue = unbind(mem, vocab[n])
        sims = grid @ residue
        v = pts[int(np.argmax(sims))]
        if code == "ego":
            v = v + robot
        elif code == "rel":
            v = v + anchor
        out.append(v)
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ssp-dim", type=int, default=1015)
    ap.add_argument("--n-objects", type=int, default=6)
    ap.add_argument("--bound", type=float, default=5.0)
    ap.add_argument("--length-scale", type=float, default=0.6)
    ap.add_argument("--grid-step", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--turn-deg", type=float, default=40.0)
    args = ap.parse_args()

    B = args.bound
    axis = np.arange(-B, B + 1e-9, args.grid_step)
    pts = np.stack(np.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)
    print(f"{args.n_objects} objects in a {2*B:.0f}x{2*B:.0f} room, "
          f"ssp_dim={args.ssp_dim}, length_scale={args.length_scale}, "
          f"{len(pts)} grid points, {args.seeds} seeds")
    print(f"translated by {args.shift:.0f} m; rotated by "
          f"{args.turn_deg:.0f} deg about the robot; rearranged = new random "
          f"positions\n")

    err = {(c, w): [] for c in CODES for w in CONDITIONS}
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        space = make_space(args.ssp_dim, B, args.length_scale, seed)
        grid = np.stack([space.encode(p).reshape(-1) for p in pts])
        vocab = AtomVocab(space.ssp_dim, seed=seed + 500)
        names = [f"obj_{i}" for i in range(args.n_objects)]

        pos = rng.uniform(-B + 1, B - 1, (args.n_objects, 2))
        robot = rng.uniform(-1, 1, 2)
        anchor = pos[0].copy()
        mem = build(space, vocab, names, pos, robot, anchor)

        t = np.array([args.shift, 0.0])
        th = np.deg2rad(args.turn_deg)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        worlds = {
            "same": (pos, robot, anchor),
            "translated": (pos + t, robot + t, anchor + t),
            "rotated": (robot + (pos - robot) @ R.T, robot,
                        robot + (anchor - robot) @ R.T),
            "rearranged": (rng.uniform(-B + 1, B - 1, (args.n_objects, 2)),
                           robot, None),
        }
        for cond, (p_new, r_new, a_new) in worlds.items():
            if a_new is None:                       # anchor moved with the rest
                a_new = p_new[0]
            for code in CODES:
                got = decode_positions(space, vocab, mem[code], names, grid,
                                       pts, r_new, a_new, code)
                err[(code, cond)].append(
                    np.linalg.norm(got - p_new, axis=1))

    print("[A] where is each object, after the world changes under the map?")
    print(f"  {'code':>5s} " + " ".join(f"{c:>13s}" for c in CONDITIONS))
    print("  " + "-" * 61)
    for code in CODES:
        row = []
        for cond in CONDITIONS:
            e = np.concatenate(err[(code, cond)])
            row.append(f"{np.median(e):8.2f} m")
        print(f"  {code:>5s} " + " ".join(f"{v:>13s}" for v in row))
    print(f"  chance for a uniform guess in this room is about "
          f"{np.median([np.linalg.norm(a - b) for a, b in zip(np.random.default_rng(0).uniform(-B+1, B-1, (400, 2)), np.random.default_rng(1).uniform(-B+1, B-1, (400, 2)))]):.2f} m")
    print("  'rearranged' SHOULD fail for everything -- the objects really did "
          "move, and\n  a map that answered well there would be reporting "
          "something it cannot know.")

    print("\n[B] fraction of objects still placed within 0.5 m")
    print(f"  note: 'rel' is anchored on obj_0, which is trivially at zero "
          f"offset from\n  itself, so 1/{args.n_objects} = "
          f"{1.0/args.n_objects:.2f} of its score is free in every column.")
    print(f"  {'code':>5s} " + " ".join(f"{c:>13s}" for c in CONDITIONS))
    print("  " + "-" * 61)
    for code in CODES:
        row = [f"{(np.concatenate(err[(code, cond)]) < 0.5).mean():13.2f}"
               for cond in CONDITIONS]
        print(f"  {code:>5s} " + " ".join(row))

    path_integration(args, pts, B)


# ---------------------------------------------------------------------------
# [C] The operation only the egocentric code has.
#
# S(a) (*) S(b) = S(a + b), so binding the whole memory by S(-d) moves the
# vectors to EVERY object at once when the robot moves by d.  One bind, no
# per-object work, no global frame.  An allocentric map cannot do this: its
# contents do not depend on where the robot is, so there is nothing to update
# -- it needs the robot's global pose supplied from somewhere else instead.
# ---------------------------------------------------------------------------

def path_integration(args, pts, B):
    print("\n[C] carrying object vectors through motion with one bind per step")
    print(f"  {'steps':>6s} {'exact':>9s} {'odom 1%':>9s} {'odom 5%':>9s} "
          f"{'odom 10%':>9s}")
    print("  " + "-" * 47)
    noises = (0.0, 0.01, 0.05, 0.10)
    results = {n: {} for n in noises}
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        space = make_space(args.ssp_dim, B, args.length_scale, seed)
        grid = np.stack([space.encode(p).reshape(-1) for p in pts])
        vocab = AtomVocab(space.ssp_dim, seed=seed + 500)
        names = [f"obj_{i}" for i in range(args.n_objects)]
        pos = rng.uniform(-B + 1, B - 1, (args.n_objects, 2))
        robot = np.zeros(2)
        steps = rng.uniform(-0.4, 0.4, (16, 2))

        for noise in noises:
            mem = bundle(np.stack([
                bind(vocab.mint(n), space.encode(p - robot).reshape(-1))
                for n, p in zip(names, pos)]))
            r = robot.copy()
            for i, d in enumerate(steps, start=1):
                r = r + d
                measured = d + noise * rng.standard_normal(2) * np.abs(d).mean()
                mem = bind(mem, space.encode(-measured).reshape(-1))
                if i in (1, 2, 4, 8, 16):
                    got = np.stack([
                        pts[int(np.argmax(grid @ unbind(mem, vocab[n])))] + r
                        for n in names])
                    results[noise].setdefault(i, []).append(
                        np.linalg.norm(got - pos, axis=1))
    for i in (1, 2, 4, 8, 16):
        row = [f"{np.median(np.concatenate(results[n][i])):7.2f} m"
               for n in noises]
        print(f"  {i:6d} " + " ".join(f"{v:>9s}" for v in row))
    print("  'exact' is the algebra alone: whatever it accumulates is "
          "bundling noise from\n  repeated binding, not odometry. The other "
          "columns add proportional odometry\n  error to each step, which is "
          "what a real robot supplies.")


if __name__ == "__main__":
    main()
