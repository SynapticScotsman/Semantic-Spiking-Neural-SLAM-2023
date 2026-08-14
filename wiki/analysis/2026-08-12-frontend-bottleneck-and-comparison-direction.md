---
title: Frontend bottleneck finding + three-track comparison direction
updated: 2026-08-12
tags: [design, conceptgraphs, replica, evaluation, icra, collaborators]
---

# Frontend bottleneck finding + three-track comparison direction

## The measurement that set this plan

Scoring our 32 KB object memory on Replica room0 under ConceptGraphs' own
metric **and their own class exclusion** (`--n_exclude 6`: other, floor,
wall, ceiling, door, window — verified against their
`scripts/eval_replica_semseg.py`) gives:

| rung (same 30k eval points, same GT, same nearest-object dense-labelling rule) | mAcc | F-mIoU |
|---|---|---|
| our VSA trace (32 KB) — measured | 0.091 | 0.077 |
| explicit instance list, **same detections**, memory deleted | 0.066 | 0.050 |
| oracle instance list (ground-truth instances, perfect frontend) | 0.634 | 0.326 |
| *ConceptGraphs, published (citation, not a rung — different data/GT/code)* | *0.406* | *0.360* |

**Deleting the memory does not close the gap — it widens it.** Replacing
the trace with an uncompressed, unbounded explicit instance list built
from the *identical* observations scores *lower* (0.066 < 0.091). The
oracle, which keeps our representation and inference rule but replaces
the frontend with ground truth, scores 0.634 — above ConceptGraphs'
published number. **The measured bottleneck is the frontend** (YOLOv8n-COCO
detection + box/depth/pose placement), not the memory: of the 24 scored
classes, our detector can only ever name 7 (COCO has no class for
"blinds", "cushion", "pillar", "rug", "wall-plug", ...), which caps any
downstream memory near ≈0.29 mAcc before it starts.

This is the same conclusion the existing miss taxonomy already reached
("zero memory-attributable misses" — see `outputs/replica_truth_aggregate.json`
and the 8-scene truth table), now reproduced at the dense per-point level
under ConceptGraphs' own protocol.

**Caveats carried on every number above:** our GT is the vMAP semantic
render backprojection, not ConceptGraphs' official Replica semantic cloud;
the scorer is our reimplementation of their formulas, not their scorer
binary; the oracle inherits perfect detection coverage, classification,
*and* placement simultaneously, so it bounds those three together, not
memory design in isolation; mAcc is an unweighted per-class mean, so a
6-point class counts as much as a 4,711-point one (true of their number
too — same metric).

Code: `student_gpu_package/05_score.py` (n_exclude 6 + all-classes,
side by side), room0 numbers in
`student_gpu_package/handoff/room0/scores.json`.

## Decision: paper first, thesis reshaped by the finding

Two live venues, cleanly separated by the existing execution plan
(`2026-08-03-isotropy-paper-execution-plan.md`) and the ICRA draft
(`paper/main.tex`):

- **NeurIPS workshop (~Aug 29, flag-plant)** — isotropy science
  (anisotropy → capacity collapse, χ-law). ConceptGraphs does not appear
  in this paper. Today's finding does not touch it.
- **ICRA 2027 (Sept 15, primary)** — the robotics system paper. Its §VII
  already pre-registers exactly this outcome (main.tex line ~800: "our
  object memory scores ≈0.076 mAcc versus their published ~0.40 ...
  whichever class list their evaluation code actually scores is recorded
  and reported with the result"). Today's work fills that `\todo` slot
  honestly, with the mechanism attached rather than a bare number.

Three tracks follow, in priority order, chosen so **no plan element
depends on a GPU run landing** (compute availability across
Colab/collaborators/uni resources is uncertain) — a GPU result upgrades a
claim, it never unblocks one.

## Track A — CPU-only, done by me, no GPU dependency

Rewrite ICRA §VII's semantic-comparison paragraph around the ladder above:
state the mechanism (frontend-limited, not memory-limited), the exact
numbers under their protocol, and the two caveats (GT provenance,
reimplemented scorer). Promote the same-frontend, three-backend ladder
(VSA trace / explicit instance list / raw observation store — identical
observations, all 8 scenes) as the paper's controlled semantic comparison,
since it isolates memory design from perception, which is the actual
claim the paper makes.

**Status: §VII protocol description corrected in `paper/main.tex` and
`paper/icra6.tex` (2026-08-12); ladder itself not yet rerun beyond room0.**
**Scene-count correction (verified against arXiv:2309.16650 via ar5iv,
2026-08-12):** ConceptGraphs' own paper reports Replica results on **7**
scenes (`room0-2`, `office0-3`, no `office4`) — confirmed for their
scene-graph construction table (Table I) and consistent with every other
Replica experiment in their paper; their semantic-segmentation table
(Table II, the 40.63 mAcc / 35.95 F-mIoU we cite) does not restate the
scene count next to it, so 7 is the well-supported inference, not a
certainty. We evaluate on all 8 scenes of the standard NICE-SLAM/vMAP
render release (the same data format their pipeline consumes) — that
extra scene, `office4`, is real, useful coverage, but has no published
ConceptGraphs number to compare against. **Next action:** rerun the
ladder across all 8 scenes (today's numbers are room0 only), report the
7-scene subset as the number compared against their published $\sim$0.40,
and the full 8-scene number alongside as broader coverage — never blend
them into one figure presented as matching their protocol. Then rewrite
§VII's numbers (framing is already fixed), add the inspector artifact per
the qualitative-with-quantitative rule.

## Track B — GPU work packages for collaborators, shipped

Three independent, self-contained tasks in `collab_tasks/`, each returning
megabytes in an existing schema with a self-check gate and a
pre-registered expectation — see `collab_tasks/README.md` for the shared
context and rules.

1. **B1 — open-vocabulary frontend swap** (highest expected value,
   lightest install). Directly tests today's attribution: if a
   wider-vocabulary detector raises our score toward the oracle ceiling,
   the memory's score is shown to track its frontend.
   **Status: complete, working code** —
   `collab_tasks/scripts/embed_crops_openvocab.py`. Writes to a fully
   separate `outputs/replica_<scene>_openvocab/` namespace and config, so
   it cannot touch the existing baseline artifacts even by mistake.
   Verified end-to-end against real room0 data (config cloning, vocab
   reading, `load_sequence` all pass); required a small backward-compatible
   `--gt-scene` flag on `student_gpu_package/04_vsa_labels.py` so an
   alternate-frontend run shares the baseline scene's ground truth without
   recomputing or colliding with it (confirmed a no-op for the existing
   `--scene room0` call — rescored identical, 0.091/0.077).
2. **B2 — ConceptGraphs' own pipeline, all 8 scenes** (heaviest install,
   completes the frontend × backend 2×2; their observation stream replaces
   today's naive clustering stand-in with a real scene-graph baseline).
   **Status: complete** — points at the already-hardened
   `student_gpu_package/` + `experiments/COLAB_CONCEPTGRAPHS_L4.ipynb`
   rather than duplicating them; brief records the six install traps hit
   and fixed today (torch.hub trust prompt incl. nested hub-loads,
   `supervision` API drift + version pin, the numpy/cv2 conflict that pin
   drags in, required `GSA_PATH` env var, their real two-script entry
   points, Drive-vs-local-disk for git+data on Colab) so nobody re-pays
   that debt.
3. **B3 — DINOv2-large instance keys** (smallest, ~10 min/scene, feeds the
   separate instance-vs-semantic disambiguation question, not the
   ConceptGraphs comparison). **Status: complete** — one existing command
   (`encoder_comparison.py --encoders dinov2:large`), already a commented
   stage in `experiments/COLAB_EMBED_GPU.ipynb`.

## Track C — scoped, not built, third priority

Original pitch (build our own referring-expression benchmark and score
both systems on it) was rejected in discussion: we would be grading
ourselves on a test we wrote — the exact invented-metric trap the project
rules forbid. Revised in light of the Track A finding: **adopt FARM's own
published protocol** (arXiv 2606.15476 — R@5/R@10 on 44k queries,
23–125 MiB scene memory; already identified by the 2026-08-08 skeptic
panel as the correct relational opponent, see entry (bi) in the main
tracker) rather than inventing a benchmark, matching the same
adopt-their-metric-verbatim rule Track A follows.

**Status: scoped only. Not yet verified: whether FARM's query set or eval
harness is publicly downloadable/runnable.** That check is the next action
if this track is picked up — do not write a query benchmark before it's
done.

## What's out of scope here

- Any claim that requires a GPU run to have landed before it can be
  stated — the paper's Track A rewrite stands on today's CPU-only ladder
  alone; B1/B2/B3 results upgrade it if/when they arrive.
- Tuning the memory representation in response to the 0.091 number — the
  measurement says that is not where the loss is.
- Building Track C's benchmark ourselves if a suitable published one
  (FARM) turns out to be usable — check first.

## Related pages

- Main tracker: `2026-07-29-vsa-query-layer-paper-plan.md` (entries (bj)
  onward record this thread)
- ICRA execution plan: `2026-08-03-isotropy-paper-execution-plan.md`
- Prior ConceptGraphs scoping: entry (ay), 2026-08-04
- Skeptic panel (FARM correspondence, relational sprint): entry (bi),
  2026-08-08
- Collaborator tasks: `../../collab_tasks/README.md`
