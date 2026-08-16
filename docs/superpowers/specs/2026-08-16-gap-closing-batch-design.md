# Gap-closing batch: design

**Date:** 2026-08-16 · **Owner:** Paul (supervisor: main session)
**Goal:** close as much of the 0.078 mAcc gap to ConceptGraphs as the backend
can honestly reach, on the fair comparison (shared SAM+CLIP frontend), without
degrading F1 — and quantify how much of the gap no backend change can touch.

## Fixed context

- Fair comparison, all 8 Replica scenes, one scorer, their `n_exclude 6`
  protocol. Theirs: **0.402 mAcc / 0.272 mF1**. Ours (baseline config
  argmax · grid 96 · cap 400 · λ 0.45,0.27): **0.324 mAcc / 0.162 mF1**.
- The frontend is a **constant** of this comparison. Its 0.609 mAcc ceiling
  binds both systems; improving it is out of scope (it would break the
  reproduction property, 0.402 vs their published 0.406).
- Target metric: **mAcc primary, F1 as guard.** Acceptance rule, pre-registered:
  a change is adopted only if mean mAcc improves AND mean mF1 does not fall.
- Six prior conclusions inverted when re-tested properly (see
  `wiki/analysis/2026-08-16-conceptgraphs-corrected-and-graceful-failure.md`).
  Every structural choice below exists to prevent inversion #7.

## Architecture: three ordered stages

**Stage 0 — instrumentation (supervisor builds, nothing runs before it exists)**

1. `collab_tasks/batch1/report.py` — the reporting guard. One function
   `report(results, prediction)` that every harness must call. It refuses to
   print a verdict unless `n_scenes == 8`, all four metrics are present
   (mAcc, F-mIoU, mPrec, mF1), and per-scene values are shown beside the mean.
   Otherwise it prints `UNDERPOWERED — no verdict`. Always emits a
   machine-readable JSON block (schema below) so the supervisor checks numbers,
   not prose.
2. `collab_tasks/batch1/error_decomposition.py` — for every scored eval cell we
   get wrong, classify: **bleed** (winning class's observations are distant),
   **near-tie** (correct class's field within 5% of the winner's — reported at
   2%/5%/10% so the threshold choice is visible; 5% is the headline),
   **unreachable** (no
   observation of the GT class exists), **misplaced** (our observations of the
   GT class exist but elsewhere). Output: per-scene and pooled counts, and a
   ranking of the six mechanisms by recoverable cells addressed. The
   `unreachable` count is the honest denominator: gap the backend cannot touch.

**Stage 1 — screen (hours, after decomposition ranks the work)**

Six harnesses, each mechanism ALONE against the baseline, all 8 scenes:

| # | mechanism | pre-registered prediction (falsifiable) |
|---|---|---|
| 1 | threshold decode (abstain) | raises mF1 via precision; mAcc within ±0.01 of baseline at some τ |
| 2 | per-class spread normalisation (normalise each class's field by its own spatial spread, not global z-score) | recovers bleed cells; helps most on scenes where decomposition shows bleed dominant |
| 3 | per-class λ (kernel width per class from observation spread) | mAcc up on scenes with mixed object sizes; risk: overfits per scene |
| 4 | cap re-sweep {100–800} at corrected config | current 400 near-optimal; large deviation would show the old finding was configuration-bound |
| 5 | confidence weighting (weight observations by their `conf` field) | small; conf is nearly constant in the stream, likely ±0.005 |
| 6 | insertion filtering (reject observations farther than r× their class's core spread — the cap generalised to geometry) | attacks bleed at the source; mAcc up on vent-type scenes, F1 up broadly |

Harness requirements (all six): pre-registered prediction block printed before
running; `--self-test` on synthetic blobs that must pass before real data;
guard call for all output; baseline recomputed in-run (never quoted).

**Stage 2 — joint grid (overnight, survivors only)**

Grid over mechanisms that survived Stage 1 (mAcc or F1 improvement, guard-valid),
crossed with decode rule and λ where coupling is plausible. Same guard, same
acceptance rule. Output: one best joint configuration plus the full grid JSON.

## Roles

| role | who | model | does |
|---|---|---|---|
| supervisor | main session | — | builds Stage 0; writes each harness spec; reviews builder code BEFORE it executes; validates every result against guard JSON + decomposition prediction; orders Stage 2; updates wiki + status artifact |
| test-builder | 1 spawned agent | inherit (full) | writes the six Stage-1 harnesses to spec; fixes self-test failures |
| workers | 2–3 spawned agents | haiku, low effort | run fixed commands over scene batches; return raw stdout/JSON; no interpretation |

Worker output containing interpretation is discarded and the run repeated from
raw output. Builder code failing self-test goes back to the builder.

## Data flow

harness → `outputs/batch1/<mechanism>.json` → guard validation → supervisor
review → verdict row in the wiki re-test table → status artifact update
(standing cadence). A result that skips any arrow does not exist. JSON schema:

```json
{"mechanism": str, "prediction": str, "baseline": {...}, "n_scenes": 8,
 "per_scene": {scene: {"macc":f,"fmiou":f,"mprec":f,"mf1":f}},
 "mean": {...}, "delta_vs_baseline": {...}, "verdict_allowed": bool}
```

## Error handling

- Worker run fails → rerun once → then recorded FAILED, never dropped.
- Guard refuses → repeat at full power or record UNDERPOWERED; no wiki entry.
- Builder harness diverges from spec → returned, not patched downstream.
- Paul reviews harness files before any commit (standing rule).

## Testing

- Each harness `--self-test`: 3 synthetic well-separated classes, expected
  mAcc ≈ 1.0; threshold harness additionally must abstain on an empty region.
- Guard unit check: feeding it a 3-scene result must produce refusal.
- Decomposition sanity: categories partition all wrong cells exactly
  (counts sum to total errors, no overlap).

## Success criteria

1. Decomposition table: the 0.078 gap split into recoverable vs `unreachable`.
2. Stage 1: six-row kill/survive table, 8 scenes, deltas on mAcc AND F1.
3. Stage 2: one adopted configuration under the acceptance rule, or the
   documented finding that no combination passes it.
4. Wiki re-test table and both artifacts updated; every adopted number
   reproducible from a committed harness plus JSON.

## Out of scope

Height axis (blocked on label coherence; +0.069 oracle result stands as
motivation for later), B1 open-vocab frontend (different comparison), B3
DINOv2-large (independent, needs GPU), any change to their pipeline or scorer.
