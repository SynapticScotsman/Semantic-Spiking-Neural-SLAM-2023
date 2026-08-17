# Gap-closing batch: results

**Date:** 2026-08-17 · **Scenes:** all 8 Replica (cgfront) · **Status:** measured, guard-validated
**Spec/plan:** `docs/superpowers/specs/2026-08-16-gap-closing-batch-design.md`, plan + Amendment A
**Artifacts:** `outputs/batch1/*.json` (guard schema), `collab_tasks/batch1/` (all harnesses)

Every number below comes from `report.py`'s guard: 8 scenes x 5 seed tuples,
paired per-tuple deltas, a verdict only when `|mean| >= 2*sd`. Predictions were
frozen in commit `da1d16d` (Stage 1) and `957429b` (Stage 2) BEFORE running.
Prediction record: **7 HIT / 5 MISS of 12** — scored mechanically, not by us.

## RETRACTED: there is no adopted configuration

**First reported (2026-08-17 morning):** z-score decode + tau 0.5 abstain
adopted at Δ mAcc +0.0155 ± 0.0050, 12% of the gap closed.

**Retracted the same day.** A workflow audit flagged that the guard checked
variance across *seed draws* but never concentration across *scenes*. Measured:

| variant | mean, 8 scenes | **excl. office4** | office4 alone |
|---|---|---|---|
| `zscore_control` | +0.0163 | **+0.0024** | +0.1140 |
| `combo_tau0.5_abstain` | +0.0155 | **+0.0018** | +0.1119 |

Six of eight scenes improve, so the *direction* is real, but essentially all the
*magnitude* is one scene. `report.py` now computes a leave-one-scene-out mean and
returns **SCENE-DEPENDENT** rather than ADOPT-CANDIDATE when the effect does not
survive removal of its best scene. Re-scoring every batch-1 result under that
rule leaves **no adopted mechanism**:

| mechanism | verdict (breadth-aware) | leave-one-out min | carried by |
|---|---|---|---|
| h2 z-score | SCENE-DEPENDENT | +0.0024 | office4 |
| stage2 combo | SCENE-DEPENDENT | +0.0018 | office4 |
| h1 threshold | SURVIVES (mF1 only) | −0.0354 | office3 |
| h3 / h4 / h6 | UNDECIDABLE | −0.0115 / −0.0017 / −0.0057 | office4 / office4 / room1 |
| h5 conf | KILLED (exact 0) | 0.0000 | — |

**Our score therefore stands at 0.324 against their 0.402. The gap is not
closed.** What the batch bought is instrumentation, six honest verdicts, and a
diagnosis — not an improvement.

**office4 is an outlier worth understanding in its own right**: baseline mAcc
0.466 there against ~0.30 elsewhere, and it carries the apparent gain of three
separate mechanisms. Any batch-2 candidate must pre-register a breadth clause
(`scenes_ge >= 6`) and report the office4-excluded mean beside the headline.

## Stage-1 verdicts (guard table)

| mechanism | verdict | best | Δ mAcc | Δ mF1 |
|---|---|---|---|---|
| h2 z-score decode | **ADOPT-CANDIDATE** | zscore_control | +0.0163 ± 0.0053 | +0.0120 ± 0.0018 |
| h1 threshold | SURVIVES (via F1) | tau2.0_abstain | −0.0322 ± 0.0022 | +0.0376 ± 0.0036 |
| h3 per-class λ | UNDECIDABLE | k1.5 | −0.0068 ± 0.0101 | +0.0036 |
| h4 cap | UNDECIDABLE | cap100 | +0.0088 ± 0.0051 | −0.0078 |
| h5 conf weight | KILLED (exact 0 — conf is constant 1.0) | — | 0 | 0 |
| h6 geometric filter | loses to its **random control**, all 3 r | r3.0 | −0.026 | −0.005 |

Notes that matter:

- **The winner is the mechanism we killed twice.** The z-score decode was
  recorded "no gain" (2026-08-14, one seed, stale config) and "scene-dependent,
  median −0.003" (2026-08-16, one seed, 8 scenes). Across the seed battery it
  is +0.0163 on **6 of 8 scenes**, resolved on both metrics. Single-draw testing
  had buried a real effect — the battery exists for false negatives as much as
  false positives.
- **h6's matched random-drop control earned its keep**: filtering looks like
  −0.026 vs baseline, but random dropping costs only −0.005 — the *geometry*
  itself is −0.021. Centroid-distance rejection deletes real object extent
  (sofa arms, rug edges). Bleed is not fixable by outlier rejection.
- **h4 is the audit's coin-flip zone, correctly refused**: +0.0088 ± 0.0051.
  The pre-battery plan would have stamped SURVIVES on seed 7 alone.
- h2's novel spread/mass variants (p,q) all failed badly (−0.04 to −0.15);
  the win is the crude global z-score. Prediction `h2-acc` technically HIT but
  its scope resolved to `zscore_control`, not a p/q variant as its text says —
  the p/q claim proper is a MISS.

## Gap anatomy — where ConceptGraphs actually beats us

`gap_anatomy.py`: the 8,386 cells where their transferred labels are right and
ours are wrong, classified with the decomposition tree (reference draw):

| category | share of gap | adopted decode recovers |
|---|---|---|
| **local_loss** (both classes have nearby obs; we lose a local competition) | **64.6%** | 8.4% |
| **bleed** (a distant class wins the cell) | **29.6%** | 3.2% |
| near_tie | 5.8% | **35.4%** |
| unreachable / misplaced | 0% | — |

The batch's decode fixes near-ties — 6% of the gap — which is exactly why it
closes 12% and no more.

**Interpretation (the paper's frame):** their nearest-neighbour readout is a
LOCAL-evidence decoder — the nearest labelled point is the local evidence. Our
unbinding is a GLOBAL-evidence decoder — field height reflects every
observation of a class everywhere, through the kernel. Where local point
density should decide, we answer with global mass; that is the 64.6%. And the
same property is why we degrade gracefully (1.7x retained under frontend
corruption, measured 2026-08-16): global evidence is robust where local
evidence is brittle. **The gap and the robustness are two faces of the same
representational choice.** Closing the rest of the gap means giving the decode
locality without giving up the superposition — that is the next research
question, and it is now measured, not speculative.

### CORRECTED 2026-08-18: the "global mass" mechanism did not survive per-cell measurement

`local_loss_forensics.py` asked, for every local_loss cell, whose observation
point is actually nearer — measured against the CAPPED stream the trace
contains (cap 400, reference draw). 14,134 cells:

| branch | share | median d(GT) | median d(winner) | what it means |
|---|---|---|---|---|
| winner's point is NEARER | **62%** | 0.26 m | **0.03 m** | the stream's own nearest label is wrong; ANY local decoder over our stream answers the same wrong class. Frontend label placement, not decode. |
| GT's point is nearer | **38%** | **0.04 m** | 0.15 m | the decode loses despite proximity — see below |

In the 38% branch, none of the candidate mechanisms discriminates: global
mass is **cap-equalised** (median 400 vs 400), the winner has more obs within
0.45 m in only 47% (coin flip), within 0.9 m in 50%, and the GT class is more
spread out in only 46%. What is true: **80% of these cells have the winner's
point also inside one lambda (0.45 m), and 52% inside two grid cells
(0.16 m)**. The competition happens at cm scales INSIDE the kernel width,
where lambda cannot tell 4 cm from 15 cm — while their continuous point cloud
resolves it exactly. sofa→cushion (physically interleaved objects) alone is
29% of this branch.

So the paragraph above is retracted as a *mechanism*: the 65% is not "global
mass beats local density". It is (a) wrong local labels in the shared stream,
62%, which no decode can fix, plus (b) sub-resolution ties, 38%, which are a
*resolution* problem (lambda and grid), not an evidence-globality problem.
The robustness half of the story survives — kernel smoothing is still why
degradation is graceful — but the research question changes from "give the
decode locality" to **"resolve below lambda without giving up the kernel"**
(per-class lambda for compact classes — h3, UNDECIDABLE at batch-1 power — is
the natural candidate, now with a mechanism-level reason to retest it
targeted at interleaved compact classes). Stated as hypothesis, not result.

## Blocker-3 killers (2026-08-17 evening): both routes FAIL their gates

Scoped by a 14-agent workflow (4 forensic lenses, adversarial verification: 2 of
8 claims survived), then two pre-registered killer experiments, both run same
day, both CPU-local.

**Route A — class-name retrieval by one unbind** (`collab_tasks/scripts/
class_query_killer.py`, gate frozen in the scoping before the script existed):
delta R@1 vs a class-agnostic null must be >= +0.20. **Measured +0.059** over 8
scenes at 0.75 m. FAIL, not retrofitted. Label parity vs `vsa_labels.npz`
1.0000 on all 8 scenes, so the reduction is correct. Diagnostics recorded, not
claimed: the hit sets barely overlap (13 of 48/39); 16 of the null's 39 hits
are on classes the trace NEVER observed (62 of 147 GT classes, 42%, absent from
the cgfront stream); observed-classes-only delta is +0.30 at every radius but
that denominator was chosen post hoc — hypothesis only. When the unbind hits it
is exact: 7 of room0's 8 hits within 25 cm, median 2 cm.

**Route B — CLIP through phasor projection** (`collab_tasks/scripts/
clip_phasor_retention.py`, gate B1: retention >= 0.80 at d=4096): **measured
0.674** (3 W draws, sd 0.08). FAIL. Ladder 0.409 / 0.674 / 0.798 at d = 1024 /
4096 / 16384 — dimension-limited, not a wall, but 0.80 needs a 128 KB trace.
Measured cone (E0, 500 real room0 crops, ViT-B/32, raw-argmax agreement with
stored labels 1.0000): img-img same 0.877 / cross 0.746; img-TEXT correct
0.292 / wrong 0.228 — the text channel works on a 0.064 margin over 101
classes and O(1/sqrt(d)) projection noise eats it. Mean-centring — the fix that
rescued the adversarial reviewer's SYNTHETIC probe — makes REAL features worse
(0.466 vs 0.674): on real CLIP the shared cone carries class signal.

**Table III status change:** their 20 affordance+negation query strings were
published all along (their Appendix A4; redundant copy on the repo's `query`
branch, commit 75e3ad5, with 316 AMT captions — the two sources differ on 1 of
20 strings). Still unreleased: the manual relevance judgements, and their GT
indexes their own object map. The open task is human labelling of which Replica
GT instances answer each query — feasible, ours to do, must be reported as a
re-annotation.

**Export bugs fixed** (commit c3aae37): `03_export_cg.py` now persists their
per-object `clip_ft` (was silently discarded every export — an appearance
experiment on their frontend needed a fresh 417 MB download because of this);
`cg_frontend_to_trace.py` now carries their `obj` id (was replaced by a running
index, destroying per-object identity).

Corrected in passing: the brief's "0.80 R@1 affordance" was wrong — 0.80 is
their LLM on NEGATION; Replica affordance LLM is 0.57 (CLIP 0.43). The 1.00s
are the REAL Lab scan, n=10.

## Method notes for reuse

- Baseline zero proven by **exact label identity** (100% on all 8 scenes)
  against `vsa_labels.npz` at the reference tuple, hard-stop enforced in every
  screen run.
- Seed battery: `(base, codebook, cap)` tuples; tuple 1 moves **41% of
  labels** — the draw noise is enormous, and any single-draw comparison of
  decode variants is untrustworthy at the ±0.01 scale.
- `UNDECIDABLE` is a verdict: an effect inside the noise band is unmeasured,
  not absent. h3 and h4 are open questions at higher power, not negatives.
- Everything ran CPU-local off a warm cache: Stage 1 ~38 min, Stage 2 ~4 min.
