# Relational recall from the map — execution plan (2026-08-06)

## The idea (Paul's, this date)

SOTA spatial memories (FARM, ConceptGraphs) answer "the box near the lamp"
by STORING relations and parsing queries with VLMs. The cognitive map does
not need to store relations: position codes already contain every spatial
relation implicitly, and field algebra applies them at query time —
unbind CHAIR → chair field; unbind LAMP → lamp field; "chair near the lamp"
= the product F_chair · F_lamp-proximity, all from one 32 KB trace, no
vision model at query time. Directional predicates are one bind
(F_lamp ⊗ S(Δ) shifts the field exactly, by the homomorphism).

Strategic role: (a) attacks the measured 72% wrong-instance failure with
ZERO new storage — spatial context distinguishes identical twins that
appearance cannot; (b) is the principled answer to FARM's referring-
expression capability, from algebra instead of a relation database;
(c) target application council: multi-robot/bandwidth/lifelong mapping,
opponents Kimera-Multi/Hydra-Multi (systems), ConceptGraphs (semantics
protocol), FARM cited as the query-language frontier.

## Phases (tasks #33–38)

**R1 — the disambiguation battery (NOW, CPU, data on disk).**
`vsa_cognitive_mapping/relational_recall.py` on the 5 multi-instance
Replica scenes. Deterministic query generation from GT (no hand-picking):
for every GT instance t of a multi-instance class c, anchor class a =
the different-class GT instance nearest t; the query "the {c} nearest the
{a}" is kept iff t is the unique correct answer by ≥0.5 m margin over the
next c-instance (computed from GT, stated on output). Score with the
established instance-correct protocol (radius 0.75 m, same as
instance_recall.py — no new thresholds). Designs compared on the SAME
class-keyed trace: class field alone (baseline ~25%) · relational
(F_c · F_a, both unbound from the one trace; proximity = the FPE kernel
already in the code, ℓ=0.6, no new parameter) · appearance keys (43%
reference) · appearance × relational. Gate: relational ≥ appearance ⇒
headline section.

**R2 — view+statement queries.** The realistic FARM-style ask: a held-out
VIEW of the instance plus the verbal anchor ("this thing, near the lamp"):
appearance-key field × anchor proximity. Same scoring.

**R3 — directional/displacement predicates.** "Two metres east of the
lamp" = F_lamp ⊗ S(2,0): demonstrate exactness (homomorphism), build the
figure; "between A and B" as product of two shifted kernels. Paper figure
+ inspector panel, not a benchmark.

**R4 — the vertical axis.** "on/under" needs z: third bound factor
S(z) (world z already computed by the depth localiser) or per-class
height stats. Scoped experiment on office scenes (monitors ON desks).

**R5 — language front-end revival.** Parse "the X near the Y" with the
existing MiniLM pipeline (or a 10-line grammar — parsing is not the
contribution); ground with R1/R2 machinery. Score under ConceptGraphs'
query-eval protocol when the head-to-head handoff exists (their metric,
their taxonomy — no invented metrics).

**R6 — deliverables per standing rules.** Belief-field-product inspector
panel (show F_chair, F_anchor, product side by side, photos), talking-guide
section, tracker entries, README updates, commit/push to results-sites.

## Success criteria (stated before running)

- R1 primary: relational instance-correct > 43% (the appearance-key
  reference) on the same scenes/protocol; any anchor-conditional analysis
  reported (queries with near anchors vs far).
- The combination (appearance × relational) should dominate both singles;
  if it does not, report the interference honestly.
- R3: shift exactness at machine precision (it is the homomorphism).

## R1 result (2026-08-06, same day): measured, gate NOT passed, mechanism understood

Battery: 24 unambiguous GT-anchored queries (40 discarded at 0.5 m margin —
Replica rooms are cramped; margin dominates), 5 scenes, 3 decodes tried
(the stop threshold — no further tuning without a decision):

| decode | instance-correct | wrong-inst | off-instance |
|---|---|---|---|
| class field alone | 29% | 71% | 0% |
| field PRODUCT F_c·F_a | **46%** | 21% | 33% |
| peak-selector (anchor picks among F_c modes) | 29% | 58% | 12% |
| appearance keys (57 view-queries) | **72%** | 9% | 19% |
| appearance × anchor | 47% | 23% | 30% |

Findings: (1) the product HALVES wrong-instance errors from the class map
with zero added storage — the idea works; (2) its failure mode is now
characterised: the product of two kernels peaks BETWEEN target and anchor
(33% off-instance); (3) the selector fix underperforms because close
same-class instances blur into one field lobe at ℓ=0.6 — candidate
enumeration fails before selection can help; (4) appearance keys dominate
when a view exists (72%) and the naive anchor product interferes with them.
Honest headline available today: "query-time field products halve
wrong-instance errors at zero storage; view-based appearance keys remain
stronger where views exist; the two do not yet combine."

Fork (Paul's call) before more decode work:
  A. accept the product result, document the gap artifact, move to R3
     (displacement predicates — exact by homomorphism, no artifact) and R2;
  B. principled probabilistic decode (fields → calibrated likelihoods,
     masked argmax of F_c within the anchor's support) — one more
     mechanism, risk of parameter creep;
  C. hybrid architecture: relational selection at the INSTANCE layer
     (clusters), VSA supplies the fields — mirrors how FARM uses symbols,
     concedes pure-algebra purity.

## R2 + C-scout results (2026-08-08, multi-agent): the fork RESOLVED

All on the identical R1 battery (identity-checked: class rerun reproduces
29/71/0 bit-for-bit).

| design | instance-correct | wrong-inst | wrong |
|---|---|---|---|
| R1 class / product (recorded) | 29% / 46% | 71% / 21% | 0% / 33% |
| CLUSTER (observation clusters + Euclidean) | **71%** | 17% | 12% |
| PURSUIT (trace candidates + field-value sel.) | 25% | 50% | 25% |
| **PURSUIT-EUCLID (trace candidates + Euclidean)** | **67%** | 21% | 12% |

R2 (view+statement, n=57): verbal anchors do NOT rescue appearance keys via
field values — product rescues 4 / damages 19 (net −15); scale-free
conditional rule contains damage (3) but recovers ~nothing (net −2).

**The unified mechanism finding (three independent confirmations):**
field-VALUE ranking is blind beyond the kernel's ~1 m support — R1's
selector, R2's anchors, and PURSUIT's selection all fail for this one
reason. Selection must be METRIC. And the metric layer need not be stored:
matching pursuit enumerates 85% of instances from the 32 KB trace alone,
and Euclidean selection over those candidates hits 67% vs the 71%
observation-cluster ceiling (gap = enumeration misses in the two sparsest
scenes).

**Resolution: paths A and C merge.** The architecture: ONE additive trace
(mergeable, O(1) updates, capacity-law-governed) + a tiny derivable index
(pursuit candidates, ~16 B/instance, reconstructable after any merge or
loss — an index, not a second source of truth). Relational queries run
metrically over candidates; displacement predicates stay exact algebra
(R3); nothing FARM-like is stored. Concession vs FARM, stated: relational
SELECTION is metric/symbolic, not binding algebra; relations still computed
at query time from positions, store still one vector.

Remaining in this thread: R3 figure (agent in flight), the R2 follow-up
with metric anchoring (view answer snapped to nearest pursuit candidate,
anchor via candidate distances), R4 z-axis, R5 language, R6 deliverables.

## Honest bounds to carry

Field blur sets relational precision (approximate by design); "left of"
needs a query-supplied frame; parsing is delegated, grounding is ours;
identical twins are distinguishable ONLY relationally (the complementarity
claim, to be shown as a stratified row: twin-pairs vs distinct-pairs).
