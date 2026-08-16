# ConceptGraphs, corrected — and the result that actually matters

**Date:** 2026-08-16 · **Scenes:** all 8 Replica · **Status:** measured
**Supersedes:** [2026-08-14-conceptgraphs-head-to-head-and-decode-limits.md](2026-08-14-conceptgraphs-head-to-head-and-decode-limits.md)

One scorer, one eval point set (30,000 points/scene), their `n_exclude 6`
protocol, their class list, their label-assignment method, their post-processed
map. Numbers come from `05_score.py` via `outputs/rescore_post_inscene.json`.

## The reproduction anchor is now essentially exact

| system | mean mAcc, 8 scenes |
|---|---|
| ConceptGraphs (unbounded point-cloud map) | **0.402** |
| *their published class-agnostic row (Table II)* | *0.406* |
| our 32 KB fixed-size trace | **0.324** |

We reproduce their published Replica figure to **99%**. That closes the "you
didn't run their system properly" objection with a number rather than an
argument, and it is what makes everything else on this page quotable.

Per scene (mAcc, theirs / ours) — we lead on **room2** and **office1**:

| room0 | room1 | room2 | office0 | office1 | office2 | office3 | office4 |
|---|---|---|---|---|---|---|---|
| .386/.270 | .370/.245 | **.329/.337** | .377/.324 | **.285/.299** | .484/.371 | .359/.278 | .625/.466 |

### The two bugs that had to be fixed first

Both were in OUR scoring of THEIR system, and both flattered us:

1. **Wrong map.** `03_export_cg.py` picked by `max(getsize)`. Their
   post-processing pass removes points, so `_post` is the *smaller* file and
   max() reliably chose the un-post-processed map. Their eval takes the newest
   pkl.gz (`eval_replica_semseg.py:123`) — the `_post` one.
2. **Wrong candidate class set.** Their eval builds `ignore_index` as the
   excluded classes *plus every class absent from the scene GT* (lines 93–105)
   and suppresses all of it before the argmax (line 139). Their objects can only
   take labels that occur in the room; ours could pick any of 51 and be marked
   wrong.

Fixing both lifted theirs ~0.13 and ours ~0.11 — our trace is bundled from their
labels, so their label quality is our input quality. **Every relative conclusion
from 2026-08-14 survived; every absolute number changed.**

### Verified against their code, not their docs

- `n_exclude 6` = those six *named* classes (`other, floor, wall, ceiling, door,
  window`), not the six most frequent.
- stride 5 is their documented Replica setting, so ours matches. **No stride-1
  re-run is needed** — this was the leading suspect and it is dismissed.
- our CLIP label assignment already matched theirs exactly: ViT-H-14
  `laion2b_s32b_b79k`, prompt `"an image of {c}"`, L2 normalise, argmax.

Remaining known differences, both needing their dataset: they score on the full
SLAM reconstruction cloud (we use 30k backprojected points), and their GT is
Replica-semantic h5 (ours is vMAP renders). Their room0 F-mIoU came out 0.534
against a published 0.360, so these are not a simple scaling.

## Capacity is not the constraint

`dimension_sweep.py`, 512–16384 dims (4–128 KB), 3 seeds:

- mAcc gains **+0.006** going 32 KB → 128 KB.
- Across the whole **32× memory range** mAcc moves only ~0.06.
- At **4 KB**, room2 already scores 0.272 against their 0.329.

So the 0.078 clean-conditions gap is frontend or decode, **not memory** — which
is what makes "fixed 32 KB" a claim rather than an arbitrary choice.

**Caveat:** F-mIoU is *not* flat (room0: 0.370 → 0.431 across 32 → 128 KB). mAcc
saturates once each class wins its own region; F-mIoU is frequency-weighted and
keeps sharpening large regions. Quote both.

## The headline: we fail more gracefully

`degradation_sweep.py`, 3 scenes × 3 failure modes, same degradation rate applied
to both sides, each scored by its own prediction rule. Fraction of each system's
**own** undegraded score retained at the worst level:

| failure mode | theirs | ours |
|---|---|---|
| keep 5% of points | 47% | **76%** |
| 50% of labels corrupted | 53% | **92%** |
| position jitter σ = 0.5 m | 46% | **78%** |

**Unanimous across all 9 sweeps**, with absolute crossovers — we start behind and
finish ahead:

- office4 at 5% kept: theirs 0.224, **ours 0.354**
- office4 jitter 0.5 m: theirs 0.307, **ours 0.430**
- room0 at 50% label corruption: theirs 0.200, **ours 0.255**

Mechanism is representational, not tuning: bundling averages a corrupted label
into near-orthogonal noise the cleanup absorbs; an explicit nearest-neighbour
list has no averaging, so a bad point is simply wrong wherever it is closest.

**So the claim is not "we are 0.078 behind."** It is: *matches an unbounded
point-cloud map to within 0.078 under ideal input, is not capacity-limited, and
overtakes it under sparse, mislabelled or noisy input* — the regime a robot
actually operates in.

**Caveats.** On room2 *drop*, their map holds better through the middle of the
range (96% vs our 70% at 25% kept) and crosses only at the extreme — report
curves, not endpoints. And we could not perturb *inside* their mapping (the SAM
detections went with a reclaimed Colab VM, ~6 h to regenerate), so this degrades
the shared input and re-runs each backend's own rule: **representations, not
implementations.**

## Negative results (all measured, all worth keeping)

- **Multi-scale FPE loses**, 5/5 configurations, and worse with more scales.
  Direct kernel measurement (`kernel_shape.py`) refuted all three mechanisms I
  proposed: *not* dilution (peak/clutter is *better* for bundles, 40–46 vs 27),
  *not* broken shift-invariance (spread ±0.005), *not* decode-grid aliasing
  (ranking identical at grid 192). It is **width**: a bundle's half-width sits on
  its *narrowest* member (0.16/0.10 vs member-alone 0.12/0.07), so the coarse
  scale never reaches half-max and you pay 1/k amplitude for a scale you don't
  get. Rule: **one component per observation, shaped well.**
- **Per-class thresholding (SSPictR-style) is a dud.** room0 +0.014 at τ=3.0 but
  with 83.6% of cells falling back to argmax, and classes-above-zero goes *down*
  (10 → 9). It does not rescue suppressed classes, which was the whole
  hypothesis.
- **`normalised` decode is scene-dependent, not a general win.** 8 scenes: mean
  +0.010, **median −0.003, only 3/8 improved**. office4's +0.105 is the entire
  effect. Do not adopt it globally.
- **The λ-tracks-extent rule is untested, not refuted.** The 8-scene sweep on
  *our* frontend gave r = +0.275, but the mAcc span across all six kernel ratios
  at fixed size is ~0.007 — a flat surface, so "best ratio" was noise and the
  correlation was noise against room shape. Needs re-running on the cgfront data.
- **λ = 0.45,0.27 remains selected on room0**, the scene we report. At grid 192
  it reads 0.226, not 0.234 — the headline was mildly flattered by grid
  quantisation. And 0.60 → 0.45,0.27 changes kernel *size* as well as shape;
  the equal-magnitude control (`0.27,0.45` → 0.181 vs 0.234) is what isolates
  shape and is the comparison to quote.

## The methodological lesson

Three separate times, a conclusion recorded as settled turned out to hold only
in the configuration it was measured in:

- `normalised` was recorded as "no gain" (0.184 vs 0.187) on 2026-08-14. That was
  at the inherited λ=0.6 against the buggy labels; at the corrected configuration
  it is worth +0.105 on office4.
- "cap saturates by 200" was measured in the same stale configuration.
- The λ ratio "result" was read off a surface too flat to carry one.

**λ, the decode grid, the insertion cap, the decode rule and the label quality
are coupled.** Testing one in isolation answers a question about that
configuration only. Sweep jointly, or re-test after anything upstream changes.

## Next

1. **Self-audit of our own handicaps** (running): gridless decode — our
   prediction is quantised to 0.08 m grid cells, theirs is per-point — and the
   per-class cap, which discards ~half of room0's 13,124 observations on a
   saturation finding measured in the stale configuration.
2. **Per-class λ** — one component per observation, width matched to object size.
   Well-motivated by the multi-scale result.
3. **λ transfer sweep on cgfront**, where the signal is ~3× our frontend's.
4. **B3 — DINOv2-large instance keys.** Never started; independent of all of the
   above.
5. Rewrite the published artifact — currently wrong on both numbers and
   mechanism.

## Reproduce

```
outputs/rescore_post_inscene.json      corrected 8-scene head-to-head
outputs/dimension_sweep.json           capacity curve
outputs/degradation_sweep.json         graceful-failure curves
outputs/degradation_sweep_tables.txt   the same, formatted
outputs/decode_rule_sweep_tables.txt   argmax / normalised / threshold
collab_tasks/scripts/                  every sweep script; all CPU, no Colab
```
