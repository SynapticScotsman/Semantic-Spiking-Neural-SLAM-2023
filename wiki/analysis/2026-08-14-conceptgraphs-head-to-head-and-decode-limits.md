# ConceptGraphs head-to-head, and what actually limits our trace

**Date:** 2026-08-14 · **Scene:** Replica room0 · **Status:** measured, single scene

One scorer, one eval point set (30,000 points), their `n_exclude 6` protocol,
their GT. Every number below comes from `05_score.py`; nothing is a published
figure quoted second-hand except where marked.

## The grid

| frontend | backend | mAcc | F-mIoU |
|---|---|---|---|
| ours, YOLOv8n-COCO | our 32 KB trace | 0.099 | 0.090 |
| ours, YOLO-World open-vocab | our 32 KB trace | 0.149 | 0.140 |
| theirs, SAM ViT-H + CLIP | our 32 KB trace — centroid export | 0.152 | 0.177 |
| theirs, SAM ViT-H + CLIP | our 32 KB trace — sampled export, cap 400 | **0.187** | **0.351** |
| theirs, SAM ViT-H + CLIP | their point-cloud map | **0.290** | **0.410** |
| *(their published Replica average)* | | *~0.40* | *~0.36* |

**Sanity anchor holds.** Their room0 lands at 0.290/0.410 against a published
8-scene average of ~0.40/~0.36 — F-mIoU above, mAcc below, both plausible for a
single scene. We reproduced their method rather than something that merely runs.

Cost: ~88 min on an L4 for one scene at stride 5 (45 min SAM segmentation over
400 frames, ~43 min their mapping), producing 110 objects / 13,162 observations.

## What we changed and what each change was worth

Starting from our frontend and our backend (0.099), swapping in *their*
frontend and holding our backend fixed:

1. **their frontend → our trace**: 0.099 → 0.152 (+0.053)
2. **export their object extent** instead of one centroid per observation
   (13,162 observations at 110 distinct positions → 13,162 distinct positions):
   → 0.178
3. **raise the per-class insertion cap** 60 → 200 → 400: → 0.185 → 0.187,
   saturated. F-mIoU moved much more here (0.240 → 0.351) than mAcc (+0.009).
4. **change the decode rule** (per-class z-score before argmax): 0.184 / 0.348 —
   **no improvement**, slightly worse.

Remaining gap to their 0.290 is therefore *not* capacity, *not* the readout
rule, and *not* the frontend. All three were tested and eliminated.

## Where the failures actually are

Per-class recall (`per_class_breakdown.py`, cap 400, argmax):

- **18 of 24 scored classes are EXACTLY 0.000.** Accuracy is near-binary: a
  class either wins its region (0.87–1.00) or vanishes. That is winner-take-all
  in the grid decode.
- **Mass is not the constraint.** `vent` contributed 2,511 observations — more
  than any other class — and scored 0.000, while their map scored it 1.000.
  `indoor-plant`: 568 observations, ours 0.000, theirs 0.991.
- **Much of the absolute deficit belongs to neither backend.** `cushion` (1,011
  GT points) and `table` (923) got NO observations at all — their frontend
  produced no such object. `sofa` (2,411) and `rug` (2,151) score 0.000 for
  *both* systems.
- **Their advantage is concentrated in four classes**: indoor-plant (0.991 vs
  0), vent (1.000 vs 0), lamp (0.708 vs 0.268), book (0.452 vs 0). All small or
  off-floor. We *beat* them on pillar (0.252 vs 0.096) and switch (1.000 vs
  0.714), and match on blinds, picture, stool.

**Reading:** the residual gap is that a 3D point cloud resolves small elevated
objects that a 2D floor-plane field cannot. A vent on a wall and the floor
beneath it share (x, y); no decode rule recovers what the projection discarded.
This is an architectural limit of the current encoding, not a quality gap, and
it should be stated that way.

## Where argmax came from, and what the literature does

Our decode — unbind each class, evaluate its field on a grid, take the argmax
per cell — entered in `7956bee` (2026-08-07) with the original student_gpu_package
harness. It was never a considered choice against alternatives; it is the
obvious parameter-free readout and it went unquestioned until the per-class
table exposed the 18 zeros.

It is also standard practice, which is worth knowing before treating it as our
bug. Penzkofer et al.'s **VSA4VQA** (arXiv:2405.03852) builds a *clean-up
memory* as "a discretised grid of 100×100×10×10 points" and returns "the
proposal with the highest similarity" — the same similarity-plus-argmax readout
over a precomputed grid that we use, at almost the same grid resolution (ours is
96×96).

Two things they do that we do not:

1. **They encode four dimensions, not two**: `S(x,y,w,h) = X^x ⊛ Y^y ⊛ W^w ⊛ H^h`
   — location *plus object extent*. Extent is precisely the information whose
   absence collapses our vent and hanging plant onto the floor beneath them.
2. **They document the same interference we measured**: "when there are many
   objects, the orthogonality principle of the hyper-dimensional vector space no
   longer holds and object SSPs are no longer orthogonal ... which in turn
   results in overlapping SSPs that cannot be disentangled correctly", with
   capacity improving from 512 → 1024 → 2048 dimensions.

Same author's **SSPictR** (Penzkofer 2025, collaborative-ai.org) encodes semantic
labels plus spatial locations from segmentation maps into a single 3,751-dim
vector — closer still to our setting. Worth reading properly; the PDF did not
extract cleanly here.

**Implication for us:** argmax is defensible and conventional, so "we used
argmax" is not a weakness to apologise for — but the field has already moved to
encoding extent, and our 2D result is consistent with their reported failure
mode. Adding a third axis is the same move VSA4VQA made for the same reason.

## Next experiment

Encode height (or extent) as an additional bound axis. Predictions to record
first: the trace stays **32 KB** either way (bundling is size-invariant), but
the same capacity spreads over a larger space, so expect an SNR cost — frequent
classes may dip while vent / indoor-plant / book / lamp become recoverable at
all. If mAcc rises while F-mIoU falls slightly, that is the trade, and for this
metric it is the right one.

## Caveats to carry into any writeup

- **Single scene.** room0 only. Their published figure is an 8-scene average.
- **Protocol asymmetry favouring them.** Their label assignment suppresses the
  six excluded classes to −1e10 *before* argmax, so a ConceptGraphs object can
  never be labelled wall/floor/ceiling/door/window. Our labels get no such help
  and simply lose those points. We reproduced their protocol deliberately, but
  it flatters them.
- **Their observation stream is object-level.** `cg_observations.json` records
  positions sampled from each object's point cloud, not per-frame detection
  geometry. This is "their objects and labels through our bundling and decode",
  not an observation-level frontend swap.
- **Environment deviation.** transformers 4.46.3, not their pinned 4.31.0
  (uninstallable on py3.12), plus three source patches — two in dependencies,
  one in their `slam_classes.py`. See `collab_tasks/B2_conceptgraphs_replica_run.md`
  traps 7–10.
- **The cap is a hyperparameter.** 60 was chosen for a frontend producing ~8k
  observations; 400 suits 13k. Report both rows rather than silently switching.

## Artifacts

- `handoff/room0/scores.json` — rung 1, both systems
- `handoff/room0_cgfront/scores.json` — rung 2, their frontend + our trace
- `cg_out/room0/*.pkl.gz` — their map (33 MB), on Drive
- `collab_tasks/scripts/`: `cg_frontend_to_trace.py`, `per_class_breakdown.py`,
  `failure_maps.py`
