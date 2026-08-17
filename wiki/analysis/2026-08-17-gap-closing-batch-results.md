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
