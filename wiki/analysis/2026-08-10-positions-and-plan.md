---
title: Four points of view, and the plan that follows from them
type: analysis
status: active
created: 2026-08-10
tags: [paper-plan, icra, t-ro, collaborators, supermap, farm, scope]
---

# Four points of view, and the plan

Source for the collaborator positions: `Telluride-VSA-Maps (1).txt`
(Lorin, Shay, Naitri, plus Paul's 2026-07-30 results page). Source for
our own position: `paper/main.tex` and tracker entries (a)-(bk).

---

## Part 1 — The four points of view, in plain language

Everyone here agrees on the substrate (phasor vectors, bind/bundle/unbind,
fractional power encoding for space and time). What differs is **what the
vector is for**.

### View 1 — Ours, in the ICRA draft: the vector REPLACES the structure

> "Throw the structure away. Keep one 32 KB vector. Measure what still works."

The scene graph and the trace are opposite design points, and we
characterise the trace on its own: how well it relocalises, how well it
grounds objects, how it merges, how its interference scales.

- **Strongest card:** it is finished and measured. Benchmarks exist, the
  merge result is exact and checkable, the capacity law predicts.
- **Weakest card:** it concedes the things graphs are good at (exact
  instance identity, arbitrary relational queries), so a reviewer looking
  for ambition can call the scope narrow.
- **The question it answers:** how much can a fixed-size vector still do?

### View 2 — Naitri: the vector is an OVERLAY on the structure

> "The graph knows where things are now. The vector remembers where they have been."

Keep SuperMap or FARM authoritative for current state, identity and
geometry. Add a fixed-budget VSA that stores the approximate history and
returns candidates the graph then verifies exactly.

- **Strongest card:** it does not fight the graph where the graph wins.
  The claim ("at fixed memory, we retain more useful history than a
  truncated or sampled explicit log") is testable and nobody owns it yet.
- **Weakest card:** it depends on somebody else's system running.
  SuperMap's code is unreleased; FARM's is released.
- **The question it answers:** at a fixed memory budget, who remembers
  more of what happened?

### View 3 — Lorin: isolate the backend and race them

> "Same eyes, different memory, same test."

One perception front end, two map backends (4D scene graph vs SSP/VSA),
the same navigation task suite. Plus hybrid variants: the SSP as a fast
index into the graph, as a replacement for edge and existence tracking,
or as the per-node confidence.

- **Strongest card:** this is exactly the fair comparison Paul already
  demanded ("same front end then different backend"), and its metrics
  (footprint, query big-O, wall-clock, update cost) are the axes where
  the trace actually wins.
- **Weakest card:** it needs a working task suite (HM3D/Habitat) and a
  lot of integration engineering before it produces a single number.
- **The question it answers:** holding perception fixed, does the map
  representation change task performance?

### View 4 — Shay: the algebra already expresses what they compute

> "Their mechanisms are our identities."

An equation-by-equation mapping from SuperMap and FARM onto FHRR:
6 exact correspondences, 4 bounded approximations, 10 reformulations,
0 weak analogies.

- **Strongest card:** it is the intellectual case for the whole
  programme, and it hands us four concrete upgrades we do not have
  (below).
- **Weakest card:** it is theory until an experiment runs, and one of its
  recommendations contradicts one of our measurements.
- **The question it answers:** is the correspondence real, and what does
  it buy that a hand-written predicate library does not?

### Where all four agree

- The state should be **fixed-size**, not growing.
- Relations should be **computed at query time, not stored** (FARM
  independently made the same choice, so this is agreement, not
  advantage).
- Comparisons must hold the **perception front end fixed**.
- The systems axes (bytes, update cost, query cost) are where the case is
  strongest, not raw accuracy.

### Where they diverge, and it matters

**Replace (view 1) versus overlay (views 2 and 3b).** Our ICRA draft
argues the trace as an alternative to the graph. The collaborators are
converging on the trace as a component inside or beside a graph. These
are not contradictory: you cannot argue the overlay credibly without
first knowing what the bounded map does alone, which is what our draft
measures. But they are different papers with different claims, and the
framing needs deciding rather than drifting.

### The one real technical tension

Shay's pipeline **whitens** semantic embeddings before fractional power
encoding. Our measured rule is **centre or z-score, never whiten**
(-47.6% signal, classroom, 5 seeds; though harmless on chess).

These may not actually conflict, and the distinction is precise:

| | what is whitened | how it is used |
|---|---|---|
| our measurement | the content vector | as a **binding key** (bound directly into the trace) |
| Shay's pipeline | the semantic embedding | as an **argument to the encoder** (FPE'd, then bound) |

Shay's own argument is that FPE outputs are quasi-orthogonal by
construction, so the pathology whitening fixes is different in his case.
This is a one-day experiment and it should be run before either claim is
written down as general.

### What Shay's note gives us that we do not have

| idea | what it fixes for us | cost |
|---|---|---|
| **Submap-local frames** | our standing shared-frame merge caveat: a loop closure updates one submap-to-world transform instead of invalidating every stored position | cheap; changes a caveat from "open" to "known fix, unmeasured" |
| **Fusion update + the κ statistic** | re-observation currently grows N; remove-and-replace is exact, so N counts objects. κ (bundle norm) is a threshold-free data-association / change score | 1-2 days, strong T-RO material |
| **Residue (CRT) codes** | separates the RANGE budget from the CROSSTALK budget, which our χ law currently conflates; also the best grid-cell anchor in the programme | medium; T-RO scope |
| **Structured orthogonal random features** | kills the phase-matrix storage (~12 MB at d=4096), which is the same embarrassment as our "decoder grids dominate the honest map state" note | small, strictly better on both axes |

And two convergences worth telling him about, because they are
independent derivations of the same thing: his **participation-ratio
effective load** is the PR term in our measured χ law, and his
**sequential extraction** under decreasing noise is our matching-pursuit
result.

---

## Part 2 — The plan

### A. ICRA 2027 paper (Sept 15, 2026) — the only hard deadline

| # | item | who | state |
|---|---|---|---|
| A1 | Choose the abstract/intro angle (or keep the general synthesis) | Paul | 5 open variants delivered |
| A2 | **D2 authorship** — now bigger: Lorin, Shay, Naitri as well as GC-VSA and Waterloo | Paul | blocks acknowledgments and the review round |
| A3 | Merge the Overleaf partial intro/conclusion with the new drafts | Claude + Paul | drafts flagged MERGE NOTE |
| A4 | ConceptGraphs run under **their** scorer (Colab L4) | Paul's compute | fills the one empty table; pre-registration already in the text |
| A5 | Two figures: Replica belief-field montage, chess qualitative | Claude | placeholders in place |
| A6 | Cut to 6 pages using the ladder in `paper/README.md` | Claude | draft runs ~7 |
| A7 | **One paragraph positioning the overlay direction as future work** | Claude | stakes the collaborators' claim without spending pages, and stops the two papers colliding |

### B. Cheap wins from the notes, before the deadline if they land

| # | item | why now | cost |
|---|---|---|---|
| B1 | **Whitening tension experiment** (binding key vs FPE argument) | resolves a contradiction between two of our own documents before either is published | ~1 day CPU |
| B2 | Cite submap-local frames in the merge caveat | turns "open problem" into "known fix, unmeasured", which is a much better limitation to have | 1 paragraph |
| B3 | κ consistency statistic on the Replica change data | a threshold-free data-association score we currently lack | 1-2 days |

B1 is the one that genuinely matters before submission. B2 is free. B3 is
optional for ICRA and strong for T-RO.

### C. T-RO evolutionary version (submit ~Feb-Apr 2027, after the ICRA decision)

The deliberately-cut delta, now with the collaborator additions:

- frame recovery inside the algebra + **submap-local frames**, measured
- K-scaling, 2 to 16 robots, and the bandwidth protocol
- formal χ derivation, including the **range-versus-crosstalk split**
  that residue codes make explicit
- the full ConceptGraphs head-to-head across all 8 scenes
- the **fusion update and κ**
- **structured orthogonal random features**, which fixes the honest
  map-state number
- the relational layer if the stopping rule is legalised

### D. The collaborators' paper (separate, parallel, not ours to lead)

Naitri's overlay plus Lorin's head-to-head is a second paper. Our
contribution to it is the trace, the capacity law, and the
index-not-store architecture; Shay's validation ladder rungs 1 to 3 are
its spine.

Practical note: **FARM's code and FARM-Scenes are released; SuperMap's
are not** (verified 2026-07-30). If a hybrid target has to be picked to
make progress this year, FARM is the one that can actually be run.

### Decisions only Paul can make

1. **D2 authorship**, now across five or more people. Blocks prose.
2. **Framing**: does the ICRA paper stay "the opposite design point", or
   soften to "a component that can stand alone"? (Recommendation: stay,
   and add A7's paragraph. The standalone result is the prerequisite, and
   softening it costs the paper its spine.)
3. **Who leads which paper**, so the ICRA submission and the
   overlay/hybrid paper do not overlap in claims.
4. **FARM or SuperMap** as the hybrid target. (Recommendation: FARM, on
   code availability alone.)
