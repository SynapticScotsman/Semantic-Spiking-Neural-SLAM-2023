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

### RE-MEASURED under the hashed codebook (2026-08-18, after 3161182)

The clip_ft analyses below were first computed under the SEQUENTIAL class
codebook. Name-hashing redraws ~38% of labels, so they were re-run against a
cache verified label-identical to a fresh build on all 8 scenes (parity
1.000000 each, BASELINE_MACC matched). **Every structural finding holds.**

| quantity | sequential | name-hashed | delta |
|---|---|---|---|
| shared-failure cells | 8,719 | 8,383 | −336 |
| our wrong cells (local_loss) | 14,134 | 13,873 | −261 |
| % of wrong cells that are real GAP | 38.3 | 39.6 | +1.3 |
| % SHARED in winner-nearer branch | 87.6 | 87.0 | −0.6 |
| baseline mAcc | 0.3235 | 0.3198 | −0.0037 |
| reachable ceiling mAcc | 0.4366 | 0.4371 | +0.0006 |
| objects blamed | 124 | 128 | +4 |
| **objects carrying 50% of cells** | **11** | **11** | **0** |
| % truth in their own top-3 | 48.5 | 51.0 | +2.6 |
| % truth within their top-10 | 93.9 | 91.1 | −2.8 |

The worst single object is unchanged in identity and share: office0 obj 9,
1,105 points labelled `switch`, actually a **chair**, 7.2% of all
shared-failure cells. Old-codebook results preserved at
`outputs/batch1/pre_hash/` so the before/after is auditable.

Not re-run, deliberately: `field_why.py` and `local_loss_forensics.py`. Their
mechanism claim was already retracted by `proximity_ceiling.py`, so
re-measuring a retracted mechanism would be wasted compute. `rank2_experiment`
already ran post-rebaseline (0.320 baseline vs the re-baselined 0.3198), so
the oracle retraction stands as published.

### 2026-08-18 (late): their clip_ft recovered, and the ceiling reframed TWICE

Their per-object CLIP features are finally in hand — all 8 scenes, 611
objects, every row count exact against the published trace, so the
`det -> obj` positional join is valid. Export vendored at
`student_gpu_package/handoff_clipft/`. Three measurements, in order, and the
third retracts the implication of the second.

**1. The shared branch is CONCENTRATED** (`object_identity_join.py`).
The 8,719 cells neither system labels correctly trace to just **124 of 611
objects**, and **11 objects — 1.8% of theirs — carry 50%** of them. 80% comes
from 26 objects, 90% from 38.

**2. And it is RECOVERABLE in feature space** (`relabel_headroom.py`).
For each blamed object, where does the true class sit in *their own* clip_ft
ranking under *their own* in-scene restriction?

| truth at | objects | cells | share |
|---|---|---|---|
| rank 1 | 0 | 0 | 0.0% |
| rank 2 | 31 | 3,302 | 37.9% |
| rank 3–5 | 39 | 3,658 | 42.0% |
| rank 6–10 | 34 | 1,227 | 14.1% |
| rank 11+ | 20 | 532 | 6.1% |

48.5% inside their own top-3; only 6.1% beyond rank 10. **Nothing at rank 1**,
so this is never our decode losing a correct label. The failure has a shape:
**60.6% of cells are a SMALL class name on a BIG object** (switch, camera,
desk-organizer, indoor-plant on chair/table/sofa/blinds) against **0.1%** the
other way — a ~500:1 asymmetry. Most-missed truths: chair (38 objects),
sofa (22), table (17). Worst single object: office0 #9, 1,105 points
labelled `switch`, actually a **chair**, 624 cells.

**3. RETRACTED — it does NOT convert into mAcc** (`rank2_experiment.py`).

| rule | THEIRS* | OURS |
|---|---|---|
| their labels as shipped | 0.363 | 0.320 |
| **ORACLE: GT label per object** | **0.291** | **0.298** |
| size prior λ=0.05 (LOSO) | 0.314 | 0.292 |
| size prior λ=0.40 (LOSO) | 0.159 | 0.151 |

Giving every object its **ground-truth** label makes the score go **down**,
in both columns. Mechanism, measured: mAcc is *unweighted per-class recall*,
and correct labels collapse label diversity — distinct non-excluded classes
predicted falls **85 → 68** across the 8 scenes. Every class that stops being
predicted scores recall 0, and averaging over classes makes that cost more
than getting the common classes right. The GT-free size prior, motivated by
the 500:1 asymmetry, is monotonically harmful and is **killed**.

This independently confirms the diagnostic anomaly recorded 2026-08-16: 10%
label corruption *improved* office4 (0.466 → 0.527) and room2 (105%). Same
cause, opposite direction — corruption ADDS diversity. **Two unrelated routes
now agree that Replica mAcc partly measures class coverage rather than
per-object correctness.** That is a statement about the benchmark, and it is
the most transferable thing found today.

*Validity caveat: the THEIRS column uses a nearest-object-centroid proxy, not
their point-transfer, and reads 0.363 where the published pipeline gives
0.402 — treat it as within-experiment only. The OURS column rebuilds our real
trace and lands at 0.320 against the published 0.324, and both oracle and
baseline arms share the same harness, so the comparison is sound.

### CORRECTED AGAIN 2026-08-18: 62% of our "local losses" were never gap

A category error that sat in every writeup above, including the forensics
earlier the same day. `error_decomposition.decompose()` classifies **every
cell where OUR prediction is wrong** (14,134). `gap_anatomy.py` measured the
**8,386 cells where THEY are right and we are wrong** — the actual gap. Those
are different populations, and local_loss shares were being quoted as "shares
of the gap". `shared_ceiling.py` settles it, using their transferred labels
and their scorer:

| proximity branch | cells | GAP (they win) | SHARED (both wrong) |
|---|---|---|---|
| winner's obs nearer | 8,768 | 1,089 (12.4%) | **7,679 (87.6%)** |
| GT's obs nearer | 5,366 | **4,326 (80.6%)** | 1,040 (19.4%) |
| **total** | 14,134 | 5,415 (38.3%) | 8,719 (61.7%) |

The two branches map almost exactly onto shared-ceiling versus real gap:

- The **62% branch** — where the stream's own nearest label is wrong — is
  **87.6% cells neither system gets right.** A shared input ceiling, like the
  0.609 frontend ceiling. Never our decode deficit, and correctly unreachable
  by any decode, kernel, or lambda change.
- The **38% branch** — where we are nearer and still lose — is **80.6% real
  gap.** That is precisely the population `field_why.py` analysed, so the
  interference / local-mass / flat-kernel split below stands on the right
  cells (80.6% pure, 19.4% shared contamination — stated, not hidden).

**Reachable headroom**, fixing only the cells they get right while keeping the
cells we win: **0.3235 → 0.4366, i.e. +0.113 mAcc.** It exceeds their 0.402
because we retain the 21 classes and 2 scenes where we lead.

Provenance note: the 87.6% figure was first produced by a workflow subagent
that wrote json to `outputs/batch1/` without saving a script. Those files were
deleted and the number re-derived independently by `shared_ceiling.py`, which
reproduces it exactly. The same agent's headroom figure (+0.081) could not be
reproduced and is not carried forward.

### The field-native answer (2026-08-18) — MECHANISM RETRACTED 2026-08-18 evening, see below

> **Read the retraction first.** The kernel measurements in this section are correct and stand. The causal claim built on them — *"proximity is what our kernel discards, and that is the gap"* — was refuted by direct measurement the same day (`proximity_ceiling.py`). Keep the numbers; discard the mechanism.

### The field-native answer (2026-08-18): the losses are NOT near-ties

Paul's objection, and it was right: every account above reports the gap as
*shares of cells*, which is not the VSA answer and let a wrong story survive
twice. `field_transect.py` and `field_why.py` answer it in the field's own
quantities, on the 5,366 cells where our own class's nearest observation is
closer and we still lose.

**The kernel, measured from the trace's own `Bx`** (not the sinc idealisation):

| d | k(d) |
|---|---|
| 4 cm | 0.9872 |
| 15 cm | 0.8295 |
| 26 cm | 0.5408 |
| half-height | 27.6 cm |

Moving from 4 cm to 15 cm away from an observation costs a class **16% of its
field height**. So field height is set by *how much evidence sits nearby*,
almost independently of *which class is closest*. Proximity — exactly the
signal their nearest-neighbour readout is built on — is information our kernel
is nearly blind to. **That is the gap, stated in VSA terms.**

**One real cell** (room0, GT `sofa`, 4.4 cm from a sofa observation, 12.4 cm
from a cushion one): `F_sofa = -1.06`, `F_cushion = +23.26`. Our own class's
field is driven *below zero* within 4 cm of its own evidence. 17 sofa
observations within one lambda of that cell against 73 cushion ones.

**Three distinct signatures**, previously lumped together:

| signature | share | what it is |
|---|---|---|
| interference (`f_gt <= 0`) | 7.7% | bundling crosstalk cancels our own field |
| local mass (winner >= 2x obs in one lambda) | 22.9% | what per-class lambda attacks |
| flat kernel (similar mass) | 69.5% | kernel cannot separate the two |

**The correction that matters:** median loss is **1.87x** in field height, p90
**5.67x**, and 12.4% are lost by more than 5x. These are *not* near-ties. That
retro-explains the whole batch-1 result — h1, h2 and the abstain family all
operate on the margin, and near_tie was only 5.8% of the gap, i.e. the entire
reachable population. It also explains h3b: per-class lambda attacks the 22.9%
local-mass slice only, which is why its direction was right (monotone in gamma,
inverse hurts) and its magnitude never resolved.

The 7.7% interference slice is the new and least comfortable finding: no
kernel width, threshold, or per-class lambda can reach it. Only fewer competing
items per trace, or a codebook with less overlap.

### h3b: the targeted retest ran same day — UNDECIDABLE, not adopted

`h3b_compact_lambda.py`, predictions frozen in `3fd8e21` before the run.
Selection is geometry-only: per-CLUSTER RMS spread < 0.35 m (single linkage,
0.30 m grid) — chosen because the first-draft single-centre rule missed
`cushion` entirely (nine compact cushions look "spread out" scene-wide);
caught and fixed BEFORE any screen ran. 48/85 scene-classes compact.

| variant | Δ mAcc (5 tuples) | verdict |
|---|---|---|
| c0.4 | −0.0280 ± 0.0101 | KILLED |
| c0.6 | −0.0017 ± 0.0113 | UNDECIDABLE |
| **c0.8 (best)** | **+0.0065 ± 0.0074** | UNDECIDABLE (needs 0.0148 to resolve) |
| rand0.6 (matched control) | −0.0033 ± 0.0051 | UNDECIDABLE |
| inv0.6 (falsifier) | −0.0133 ± 0.0111 | UNDECIDABLE |

Predictions, scored mechanically: **3 HIT / 2 MISS.** The two misses are the
ones that matter: breadth (5/8 scenes, carried by room0 — drop it and the
mean is +0.0045) and **targeting** (c0.6 beats its matched random control by
only +0.0016 ± 0.0079, unresolved). The direction is consistent with the
mechanism everywhere it can be read — gain is monotone in gamma toward
mild shrink, over-shrink is killed, and shrinking the extended classes
instead *hurts* (−0.0133) — but the mechanism's distinctive signature,
*targeting matters*, was not demonstrated. Under the standing rules this is
an open question at higher power, not an adoption and not a negative.
Transparency note: the frozen PRED prose still described the first-draft
rule; the code and constants that ran are the cluster rule and did not
change after the freeze commit.

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


## RETRACTION + RE-BASELINE (2026-08-18, evening)

Two independent things landed together. Both change what can be quoted.

### 1. The proximity mechanism is refuted

`collab_tasks/batch1/proximity_ceiling.py`, 8 scenes, tuple 0, all at EQUAL
quantisation (the first version of this measurement compared a grid-scored
field against a kNN scored at exact positions — not like-for-like, and the
confound was the finding):

| decoder | mAcc |
|---|---|
| our field, on grid | 0.3235 |
| perfect proximity (NN k=1), on grid | 0.3288 |
| NN k=1, exact positions | 0.3676 |
| kNN-5, exact positions | **0.3844** |
| ConceptGraphs | 0.4020 |

- **Proximity alone is worth +0.0053** — a third of the 0.017 seed-noise
  floor. A *perfect* proximity decoder over our own stream barely beats our
  field. The published claim that "the kernel discards proximity and that is
  the gap" is **RETRACTED**. The kernel facts (k(4cm)=0.9872, k(15cm)=0.8295,
  half-height 27.6 cm) are unaffected and still stand.
- **Our own 96x96 grid costs +0.0388** — larger than proximity, larger than
  superposition's total cost (+0.0098), and larger than any of the seven
  screened mechanisms. We imposed that 0.08 m quantisation; their labels never
  had it.
- **The real lever is bandwidth ADAPTIVITY.** kNN-5's only structural
  difference from our field is an adaptive bandwidth, and it reaches 0.3844.

### 2. The stream ceiling supersedes the +0.113 headroom

Measured on the same run: only **21.3%** of eval points have any observation
of their own GT class within one lambda, and **74.1%** have their GT class
absent from the stream entirely. So the best decoder over OUR stream tops out
at **0.3844**, not 0.402. The `shared_ceiling.py` figure of +0.113 conflates
cells CG wins because our decode is worse (addressable) with cells CG wins
because their map carries geometry our stream never had (not addressable from
the memory at all). **Roughly +0.018 of the gap is unreachable by any memory
change.**

### 3. The class codebook changed — every stored baseline was invalidated

`class_phasors` (vsa_cognitive_mapping/object_grounding.py) now derives each
class key from a **hash of the class NAME** rather than a sequential RNG draw.
This is a correctness fix and it stands (confirmed by Paul 2026-08-18): under
the old scheme a class got whatever vector its list position landed on, so
`chair` in a 3-class scene was bit-identical to `book` in a 4-class scene.
That made the class list part of the transmitted payload — the "32 KB trace"
was 32 KB *plus an exactly-ordered vocabulary* — and made traces from scenes
with different vocabularies unmergeable, which the merge line depends on.

Consequences, all measured:

- **All 8 stored `vsa_labels.npz` baselines failed label parity** (0.44–0.78
  agreement). The guard's A3 gate caught it and hard-stopped the h7 run.
- **The guard earned its keep.** `run_screen` reads baselines from the disk
  cache but builds variants fresh, so it was about to compare an old-codebook
  baseline against new-codebook variants — an artifact that would have looked
  like a large, broad, entirely fake effect. The 40 mixed cache files were
  quarantined to `outputs/batch1/cache_stale_mixed_codebook/`.
- **The headline is unchanged**: 0.3237 -> 0.3197 mean mAcc, delta −0.0040,
  inside the ±0.017 noise band. Per-scene shifts reach 0.050, as expected —
  hashing is effectively a different draw of class keys.

**Canonical re-baseline command** (reproduces `class_fields` bit-for-bit —
verified agreement 1.000000 on room0, which preserves the two-implementation
cross-check between `04_vsa_labels.py` and `collab_tasks/batch1/common.py`):

```
python student_gpu_package/04_vsa_labels.py --scene <scene>_cgfront     --labels-from-points --max-per-class 400 --length-scale 0.45,0.27 --grid 96
```

Note the non-default args: `--max-per-class 400` (04's default is 60) and
`--length-scale 0.45,0.27` (04's default is 0.6 isotropic). Running 04 with
its own defaults does NOT reproduce the batch-1 baseline.

**Everything measured against the old baselines must be re-read as
provisional** until re-run: batch-1's seven verdicts, h3b, gap_anatomy,
shared_ceiling, field_why, and the proximity decomposition above. Paired
deltas at matched seeds should survive, but that is a hypothesis until
measured.
