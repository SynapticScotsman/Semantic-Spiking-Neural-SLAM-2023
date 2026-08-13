---
title: ASTM Paper Plan — Algebraic Spatio-Temporal Memory (v2, fact-checked)
type: analysis
status: active
created: 2026-07-29
updated: 2026-07-30
source_paths:
  - raw/2026-07-29-astm-fact-check.md
  - wiki/sources/2026-07-29-supermap-paper.md
  - wiki/experiments/2026-07-29-vsa-cognitive-map-classroom-results.md
tags: [paper-plan, astm, vsa, cognitive-map, supermap, benchmark, tracker]
---

## Summary

Working plan and **live tracker** for the ASTM (Algebraic Spatio-Temporal
Memory) paper. **v2 supersedes v1 entirely** after a five-round external
fact-check ([raw/2026-07-29-astm-fact-check.md](../../raw/2026-07-29-astm-fact-check.md))
— three independent review passes converged; treat the position below as
**frozen**. The ASTM draft report itself is **DO-NOT-CIRCULATE** until its
unrun-work-as-done claims are fixed (see Round 4 list in the raw doc).

## Frozen position (converged bottom line)

> "Scene graphs are exact, mutable and relational; VSA memories are bounded,
> associative and approximate. We provide the first rigorous robot-level
> comparison on identical perception data — collected on our own Spot
> platform with independent ground truth — and identify feature isotropy as
> the condition that makes learned semantic representations work in the
> algebraic regime."

Anything beyond this = hypothesis, future work, or cut.

## Key corrections adopted from the fact-check (vs v1)

1. **Framing**: NOT "graphs must serialize to LLMs" (Taskography kills that
   — planners run directly on graphs). The honest contest: **exact/mutable/
   growing graph state vs bounded/associative/approximate VSA state**; the
   real baseline to beat is an **exact temporal event table**, not an LLM.
2. **Novelty**: what⊗where⊗when + algebraic queries has in-house prior art —
   **GC-VSA** (Krausse, Neftci, Sommer, Renner, NICE 2025 (IEEE)
   [corrected 2026-07-30 post-verification]; synthetic only)
   and the **Waterloo SSP lineage** (Komer 2019; Voelker 2021; SSP-SLAM
   Dumont 2023). Both lineages are **collaborators** → *unification story*,
   not novelty defense: first joint real-world deployment of both threads.
3. **Cleanest scientific contribution** (no prior work found in 3 sweeps):
   the **isotropy ⇔ VSA-capacity bridge** — both lineages' random atoms were
   isotropic by construction; real learned features break exactly that
   (365× crosstalk); isotropization is the condition for leaving simulation.
4. **Terminology (mandatory)**: "fixed-dimensional associative map state"
   (report ALL memory: dictionaries, bases, decoders, codebooks, tracks);
   "online symbol/prototype registration" (NOT "zero-shot class expansion");
   **milliseconds** not microseconds; crosstalk metric must be defined
   precisely; fixed dimension ≠ fixed information quality.
5. **SuperMap code/benchmark NOT released** (placeholder repo as of
   2026-07-29) — fallback: our own staged collections (we own the data,
   Round 5).
6. **Deadlines corrected** (v1 had NeurIPS-2026 impossible + inverted
   sequencing) — see ladder below.
7. **Strike list** (never write): "viable successor to the 3D scene graph",
   "definitive mathematical substrate", "practically eliminating crosstalk",
   "impossible to hallucinate", "guaranteed by the law of large numbers".
   "Torus-JEPA"/"SIGReg-on-torus" are internal terms — never cite as
   established. LoopNav does NOT test JEPA models — claim was false, cut.

## Contributions (4 + optional 5th)

1. **System**: fixed-dimensional algebraic event memory on a real robot —
   multi-trace architecture (below), ten query families by single-unbind +
   closed-form range kernels + vocabulary scans; no resonator needed
   (multi-unknown factorization deferred to GC-VSA-team follow-up).
2. **Science**: isotropy ⇔ associative-capacity bridge, with the
   isotropization ablation ladder (raw → centring → unit-norm → PCA-removal
   → ZCA → soft-ZCA → random orth → learned proj → phase map → isotropy
   regularizer), measuring isotropy AND retained semantics.
3. **Calibrated abstention**: null-conditioned evidence
   e = (s_max − μ₀(D,K,τ))/σ₀(D,K,τ); AUROC/ECE/Brier/risk–coverage.
   Ask Furlong: KDE/quasi-probability treatment of event-trace similarities.
4. **Systems comparison** on identical perception data, four-baseline
   ladder: (A) exact temporal event table (indexed — will win at 1.5k
   events; our win condition = accuracy–latency–energy–bytes **Pareto
   frontier at scale**); (B) deterministic temporal graph, no LLM; (C) LLM
   as query-COMPILER into constrained grammar, deterministic execution;
   (D) generative VLM answerer (hallucination + cost). Plus DynaMem,
   embedding retrieval, CountSketch. **ReMEmbR = mandatory baseline**;
   frame benchmark axes as extending the NaVQA family.
5. **(Optional) Dataset release**: our own Spot RGB-D + LiDAR-inertial
   dynamic-indoor data WITH temporal-query annotations — no verified
   competitor offers exactly this; makes the benchmark axes adoptable.

## Adopted architecture: multi-trace design (from cog-maps session)

One trace per query shape — VSA "secondary indexes"; never store more
factors in a trace than its target query can supply (measured basis: the
22× under-unbinding attenuation, 0.247 vs 0.011):

| Trace | Contents | Answers (one unbind) |
|---|---|---|
| `M_what_where` | Σ C ⊗ S(x) | where are chairs, ever / what's here, ever |
| `M_what_when` | Σ C ⊗ T(t) | when were chairs seen |
| `M_where_when` | Σ S(x) ⊗ T(t) | occupancy history of a spot |
| `M_event` | Σ C ⊗ S(x) ⊗ T(t) | supply two factors, decode the third |

Plus: **temporal range kernels** T₍a,b₎ = ∫ₐᵇ T^t dt (closed-form per
frequency component; a time *range* costs the same as a time *point*);
**vocabulary scans** for small discrete unknowns ("which object moved?" =
per-class decode at two windows, ~2K sweeps, ms, deterministic); instances
as atoms inside M_event (C ⊗ I_track ⊗ S ⊗ T), scan the instance
vocabulary — stays fixed-footprint. Writes fan out ×4 (trivial); each trace
has its own capacity budget; marginals are the high-SNR pathways.

## Common evaluation protocol

Event stream E = {(fᵢ, cᵢ, jᵢ, xᵢ, tᵢ, pᵢ)} fed identically to every
representation; sweep D ∈ {1024…32768}, N ∈ {100…30,000}; 15 test
conditions incl. pose noise, track switches, loop-closure correction, class
imbalance; report encode/unbind/cleanup/decode/total latency **separately**
(decode may dominate); percentile metrics; independent ground truth
(surveyed positions/fiducials — we own the platform, we control this);
joules per completed CORRECT query on actual platforms (hardware = future
work; no borrowed 400×/45× (also 392×, 5.5×, 6×) numbers
[corrected 2026-07-30 post-verification: was "420×/84×"] — those are other
tasks/hardware).

## Deadline ladder (as of 2026-07-29)

| Date | Venue | Action |
|---|---|---|
| **Jul 31** | IROS 2026 late-breaking | only if trivial to assemble — 2 days |
| **~Aug 29** | NeurIPS-26 workshops | workshop paper (flag-plant) |
| **Sept 15** | **ICRA 2027 full paper** | **PRIMARY TARGET — 7 weeks** |
| Sept–Oct | CoRL 2026 workshops (Austin Nov 9–11) | backup workshop |
| Nov 11 | NICE 2027 | neuromorphic-angle paper |
| Dec 31 | RA-L → ICRA-27 transfer cutoff | fallback |
| ~May 2027 | NeurIPS 2027 | isotropy-science paper if split out |

Sequencing note: ICRA deadline comes BEFORE any leisurely workshop path —
the measured core must aim at **Sept 15**.

## State of evidence (2026-07-30) — what is MEASURED, by contribution

| # | Contribution | Status | Headline numbers (artifacts under `outputs/classroom/`) |
|---|---|---|---|
| 1 | System (multi-trace + router) | **running** | 4 traces @ hd 8192; conditional recall 8–11 cm; range kernels probe-side, 14.5 ms; router + which_moved live in demo (`astm_traces.py`, `workshop_demo.py`) |
| 2 | Isotropy ⇔ capacity bridge | **measured (offline)** | 0.88 coherent cos → 365× crosstalk → whitening → 27×; heading 28%→98%; eff. rank 8→64. Open: online whitening; full ablation ladder |
| 3 | Calibrated abstention | **v1 working** | per-decode null stats in map state; z≥3 ≙ sim≈0.0037; 93% confident-correct, 80% abstention precision; scales 0.84→0.94 with D. Open: Furlong AUROC/ECE/risk–coverage |
| 4 | Systems comparison | **rung A only** | exact-table baseline auto-scores every query; D×N matrix: 60 cells / 7.7 min, saturation D=4096, **Pareto knee D=4096 (118 MB, 0.80 acc, 2.7 ms)**, no capacity cliff at N=4498. Open: baselines B–D, ReMEmbR |
| + | Working memory (new) | **measured** | per-class λ (0.9/0.995); tracks person 0.12–0.51 m fresh, honest "forgotten" when stale; 2 novel findings (mass imbalance; normalization-cancels-decay → evidence floor) |
| + | VSA-JEPA probe | **decided** | transport exact (1.0000); prediction ≡ persistence (0.727 ≫ null 0.015) → JEPA must beat *persistence*, model appearance dynamics only |
| + | Demos | **live + shareable** | 4-panel live demo (vantage/object/now modes, ~15 ms ticks); 15.4 MB static export (`share/classroom_demo/`); explainers ×3 |

## Next actions (priority order, as of 2026-07-30)

1. **Staged own-data collection** (position GT at scale) — REQUIRED, not
   optional: school_run1's LIO-SAM odometry is diverged (km-scale, longest
   clean segment 4–10 s). Plan the moved-object protocol with the student.
2. **Baseline ladder B–D + ReMEmbR harness** (WP3) — the comparison table is
   now the biggest gap between "measured" and the frozen position.
3. **VGPI pose recovery for school_run1** (le-marmotte; pointclouds config
   exists) — rescues E2 position scaling AND is the three-lineage
   integration demo (one lineage localizes so another can remember).
4. **Online whitening** (running mean/cov estimate) — the biggest deployment
   gap in contribution #2.
5. **Collaborator emails** (see asks below) — settle scope/authorship EARLY.
6. Venue: IROS LBR window closes **Jul 31** (decision now or never);
   NeurIPS-26 workshop ~Aug 29 as flag-plant; **ICRA Sept 15 primary**.

## Work packages (tracker — update statuses here)

| WP | Work | Owner | Status |
|----|------|-------|--------|
| P0 | Multi-trace module: trace set + range kernel + query router + vocab-scan fallback | Paul/Claude | **DONE + upgraded 07-30** (fast decode, multi-scale time, calibration) |
| WP1 | Canonical event-stream exporter; school_run1 extraction; staged moved-object collection | Paul + student | classroom done; school_run1 heading/time done, **position blocked (dataset odometry diverged)**; staged collection = next action #1 |
| WP2 | Memory hardening: per-class norm, λ-decay, calibration (Furlong machinery), online whitening | Paul | **3 of 4 done 07-30** (per-class norm ✓, λ-decay ✓ incl. floor finding, calibration v1 ✓); online whitening open |
| WP3 | Baseline ladder A–D + ReMEmbR + DynaMem + CountSketch harness | student | **rung A done** (exact table in bench + sweep); B–D not started |
| WP4 | Capacity curves (measure, don't just cite) | shared | **first matrix DONE 07-30** (60-cell D×N, Pareto knee D=4096); protocol's 15 stress conditions open |
| WP5 | Isotropization ablation ladder | Paul | whitening subset done + sweep evidence; remaining rungs (ZCA/soft-ZCA/random-orth/learned) open |
| WP6 | Writing; scoping + authorship with GC-VSA team + Dumont + Furlong (settle EARLY) | Paul | pending — asks sharpened (below) |

## Collaboration asks (settle before Sept 15)

- **Krausse** (GC-VSA + Telluride repo co-author): (a) resonator scope —
  paper 1 or follow-up? (b) NEW: his le-marmotte demo has a distance-driven
  1/e map-washout knob — convergent with our per-class λ working memory
  (independent, both mid-July); align terminology + cite each other.
  (c) his static-demo pattern already adopted for our share export.
- **Furlong**: KDE/quasi-probability calibration — we now have working
  z-score abstention + a measured miscalibration finding (null probes must
  match trace RMS amplitude) to hand him.
- **Dumont**: dynamics thread — bring the probe result (prediction ≡
  persistence; the learnable part is appearance dynamics only).
- Capacity-curve ownership; author ordering.

## Draft abstract skeleton (v2 — measured vs target separated)

Measured (safe to claim now, on our own Spot data): 8192-d, ~4,500 events,
8–11 cm conditional recall, crosstalk 365×→27× (defined metric),
2.7–15 ms CPU queries, calibrated abstention (93% confident-correct / 80%
abstention precision), D×N capacity matrix with Pareto knee at D=4096
(118 MB honest map state), per-class decaying working memory with honest
forgetting, cross-implementation validation (two codebases, same numbers).
Target (must be earned before claiming): baseline-ladder comparisons B–D,
Furlong calibration metrics, position scaling on clean GT (staged
collection), online whitening. Skeleton = frozen position quote + Round-4
revised contribution statement + three-lineage unification + Furlong
calibration. v1 abstract is RETIRED (contained "microseconds" + unrun
comparison claims).

## Progress log

- **2026-08-12 (au)** — **ConceptGraphs head-to-head SCORED under their own
  protocol; frontend, not memory, is the measured bottleneck; three-track
  direction decided and Track B shipped.** Full derivation, numbers, and
  status: [2026-08-12-frontend-bottleneck-and-comparison-direction.md](2026-08-12-frontend-bottleneck-and-comparison-direction.md).
  Room0 under their `n_exclude 6` protocol: our VSA trace 0.091 mAcc vs
  their published 0.406; deleting the memory (explicit instance list, same
  detections) scores LOWER (0.066); an oracle instance list (same
  representation, perfect frontend) scores 0.634 — the loss is
  frontend-attributable, matching the existing miss taxonomy, now
  reproduced at the dense per-point level under their own scorer. Decision:
  Track A (CPU-only, rewrite ICRA §VII's semantic-comparison paragraph
  around this ladder + promote the same-frontend 3-backend comparison,
  no GPU dependency) — designed, not yet in `paper/main.tex`. Track B
  (collaborator GPU packages, `collab_tasks/`: B1 open-vocab frontend swap,
  B2 ConceptGraphs' own 8-scene run, B3 DINOv2-large keys) — **shipped**,
  B1 has working tested code. Track C (adopt FARM's published
  referring-expression protocol rather than inventing a benchmark) —
  scoped only, third priority, needs a check that FARM's query
  set/eval code is actually public before any code is written.

- **2026-08-04 (at)** — **VPR FRONTEND (EigenPlaces): Paul's frontend
  hypothesis CONFIRMED, the paper's mean-term mechanism CONFIRMED live, and
  bind-yaw goes redundant-and-harmful exactly as predicted. Held-out
  classroom localisation is now USABLE: 0.35 m median with a 64 KB trace.**
  New modules `vpr_frontend.py` (torch.hub gmberton/eigenplaces ResNet50-2048,
  whole-frame, ImageNet norm shortest-side 384, schema-piggybacked as
  `crop_embeddings_eigenplaces.pt`) and `recall_cdf.py` (the (as) standing
  metric as a tool; regression-reproduces (as) exactly). Geometry per scene:
  classroom cos +0.278 / eff-rank 11.3, TUM +0.485 / 18.6, school +0.230 /
  28.6 — far better conditioned than YOLO (0.94) but still mean-bearing.
  **Classroom battery (blocked seed 0, ~93 queries, oracle 0.16 m):**

  | frontend · treatment | system | median | ≤0.5 m | ≤1 m | ≤2 m | worst |
  |---|---|---|---|---|---|---|
  | dinov2-crops · zscored+yaw (as) | VSA argmax | 0.96 | 38.5% | 52.7% | 79.1% | 8.1 |
  | eigenplaces · raw | kNN ceiling | **0.43** | 51.6% | 72.0% | **100%** | 1.7 |
  | eigenplaces · raw | VSA argmax | 3.12 | 23.7% | 25.8% | 36.6% | 7.3 |
  | eigenplaces · **zscored** | **VSA argmax** | **0.35** | **52.7%** | 68.8% | 93.5% | 7.2 |
  | eigenplaces · zscored | VSA top-3 | 0.33 | 55.9% | 74.2% | 98.9% | 2.3 |
  | eigenplaces · zscored + bind-yaw | VSA argmax | 1.00 | 38.7% | 49.5% | 77.4% | 7.8 |

  **(1) Frontend verdict — Paul right:** kNN ceiling 0.78→0.43 m, tail
  5.4→1.7 m. The descriptor was the bottleneck.
  **(2) Mechanism verdict — the paper right, demonstrated live:** raw
  EigenPlaces FAILS at argmax (3.12 m; the residual mean-term crosstalk bump
  out-competes the true peak — top-3 at 0.62 m proves the truth sits at a
  secondary mode) and **z-scoring alone fixes it to 0.35 m** — the free
  treatment recovering the whole ceiling. The centre-don't-whiten claim now
  has a *constructive* demonstration on a brand-new encoder.
  **(3) VSA argmax ≈ its kNN ceiling** (0.35 vs 0.43 median; kNN keeps a
  small ≤1 m edge, VSA a heavier single-outlier tail; top-3 closes both).
  With conditioned content, the superposition cost at this N is ~nil — the
  grid decode's kernel-smoothing over ALL stored frames even beats 1-NN on
  median. State carefully, never as "beats kNN" unqualified.
  **(4) Yaw verdict — as pre-registered:** bind-yaw HURTS EigenPlaces
  (0.35→1.00 m). Viewpoint-invariant content makes the heading factor pure
  crosstalk. Together with (am): key diversity and content conditioning are
  SUBSTITUTES — use exactly one rescue, not both.
  **(5) TUM desk orbit — honest scale-mismatch finding:** EigenPlaces zscored
  no-yaw: kNN ceiling only 0.82 m / 29.3% ≤0.5 m (oracle 0.25) in a 2.4 m
  space; VSA argmax 1.22 / top-3 0.87, ≤2 m 82.6–98.9% (no collapse, tails
  gone, but fine-grained position unresolved). A street-scale VPR descriptor
  treats the whole desk as ONE place — viewpoint invariance works *against*
  sub-metre discrimination on an orbit. The classroom's distinct corners sit
  at the granularity where it works.
  Queued: school time-recall + χ row on EigenPlaces (running), 2×2 cell 4
  (VPR on the class task), 5-seed treatment table, inspector regeneration as
  the dual-frontend demonstration.

- **2026-08-04 (au)** — **School time recall on the VPR frontend: the
  treatment ordering replicates a FOURTH time, at 10k scale, and improves.**
  `time_recall.py` fixed to read timestamps from the encoder file itself (the
  crop file only covers detection-bearing frames; the VPR file covers all
  11,211 → KeyError). EigenPlaces school_run1 (11,211 frames / 455 s, blocked
  3 seeds, oracle 5.0 s, const 98.3 s): signal **raw 23.5%±8.1, centred
  43.5%±12.3, zscored 42.6%±14.3, whitened128 1.1%±13.2**. Versus
  dinov2-crops (ar): centred 30.0→43.5% (better frontend helps time recall
  too), whitened 8.5→1.1% (whitening now ~useless — the whitening-worst claim
  holds on a purpose-trained encoder at 10k scale). Two-robot merge at full
  scale: max |merged − joint| 1.97e-15, 24/24 identical answers, merged
  battery median 80.7 s. `outputs/school_run1/time_recall.json`.

- **2026-08-04 (av)** — **Double dissociation COMPLETE (all four cells
  measured) + inspector rebuilt as the dual-frontend demonstration.** Cell 4:
  EigenPlaces frame vectors on the CLASS task via the crops-vs-frames
  protocol (each crop assigned its frame's VPR vector; 9,056 crops, chance
  0.312): gap 30 **0.505** vs crops 0.912; gap 300 **0.429** vs crops 0.769.
  The 2×2 (class task gap 300 / place task battery median): object-level
  crops **0.769** / 0.96 m; place-level EigenPlaces 0.429 / **0.35 m** — the
  diagonal wins both ways; "granularity matches query" is now a measured
  claim, and the C-JEPA object-level premise and the VPR result are not in
  tension. `export_recall_inspector.py` rebuilt per the approved plan:
  Section B on EigenPlaces·zscored·no-yaw (battery reproduces: VSA 0.353 m
  vs kNN 0.426 m — memory ≈ ceiling), CDF-led header (standing rule (as)),
  new Section C = live-computed 2×2 table + 16 strided query-crop →
  retrieved-crop photo pairs (gap 300; 13/16 class-correct), merge exactness
  1.11e-16 with 117/117 identical answers. Same artifact URL refreshed
  (5e19782a). Still queued: 5-seed eigenplaces treatment table, χ-row
  write-up if not already in (at), T1 derivation, T5.

- **2026-08-04 (aw)** — **5-seed EigenPlaces treatment table + the honest
  seed-spread caveat on the 0.35 m headline.** `heldout_eval.py` fixed the
  same way as time_recall (timestamps now read from the encoder file itself —
  the crop file lacks detection-free frames → KeyError on frame-level
  frontends). Classroom, blocked, 744 queries, seeds 0–4
  (`heldout_eval_eigenplaces_5seed.json`): raw 2.05±0.77 m (40.7%), centred
  0.87±0.24 (78.5%), **zscored 0.84±0.21 (79.8%±6.5)**, whitened128
  4.57±0.36 (**−47.6%** — worse than the constant predictor, the strongest
  whitening-worst datum yet). Same protocol on dinov2-crops: zscored
  1.14±0.20 (71.6%) — so the frontend swap is a solid 5-seed improvement
  (1.14→0.84 m), but smaller than the seed-0 story. Per-seed forensics
  (recall_cdf seeds 0–4): **seed 0 is the friendly end of the split range** —
  full-744 argmax 0.46 m (kNN ceiling 0.41) vs seeds 1–4 argmax 0.81–1.09
  (ceilings 0.61–0.69, oracle 0.25–0.37 vs 0.16). The kNN ceiling moves in
  tandem with the VSA, and top-3 (0.60–0.77) stays ≈ ceiling at every seed —
  the memory-tracks-its-ceiling claim survives seed variation; the absolute
  0.35 m number does NOT generalise across splits and must always be quoted
  as "seed-0 battery" (the inspector already labels it so). Standing rule
  addition: headline medians from a single split carry the seed tag.

- **2026-08-04 (ax)** — **7-SCENES CHESS: first literature-comparable
  benchmark result, on REAL held-out traverses.** Paul unconvinced by
  self-collected single-pass data → decisions: 7-Scenes only,
  storage-matched baselines, one scene <5 GB. New `tools/prepare_7scenes.py`
  (selective color+pose extraction, ~1.4 GB kept, official split parsed from
  the zip; floor plane auto-picked (x,z); yaw = camera view dir; per-seq
  poses.csv keeps tx/ty/tz for 3D error) and
  `vsa_cognitive_mapping/cross_recall.py` — the benchmark protocol: memory =
  4 train traverses (4,000 frames), queries = 2 physically separate test
  traverses (2,000), **treatment stats fit on memory only (no query leak)**,
  3D translation error of the retrieved frame reported next to 2D (only 3D
  goes beside published numbers). Chess, EigenPlaces, no yaw:

  | system (zscored) | bytes | 3D median | 2D ≤0.5 m | 2D ≤1 m |
  |---|---|---|---|---|
  | kNN exact (ceiling) | 32.8 MB | 0.29 m | 88.5% | 98.1% |
  | PQ m=8 | 2.1 MB | 0.29 m | 89.6% | 98.5% |
  | **VSA trace** | **32 KB** | **0.36 m** | 76.4% | 93.0% |
  | VSA top-3 (2D) | 32 KB | — | 91.5% | 98.9% |

  Literature anchor: DenseVLAD retrieval baseline 0.21 m median on chess
  (arXiv:1903.07504 table) — our kNN ceiling 0.29 m is the same class and
  ballpark; the VSA lands within 0.07 m of ITS OWN ceiling at ~1000× fewer
  bytes, and **PQ cannot exist below ~2 MB here** (the 256×2048 codebook
  alone is 2 MB) — the VSA's 32 KB has no PQ counterpart at this N.
  Treatment mechanism replicates on a public benchmark: raw VSA 0.75 m 3D vs
  centred/zscored 0.37/0.36 m. **Two honest nuances, both now measured:**
  (1) whitened128 is NOT worst here (argmax 0.34 m 3D, ≈ zscored) — the
  whitening-worst claim is scene/encoder/N-dependent (classroom 5-seed had
  it at −47.6% signal); scope it, never state it universally. (2) bind-yaw
  on chess mildly HELPS the tail (top-3 ≤0.5 m 91.5→94.7%, ≤1 m 99.9%;
  argmax ≤1 m 93.0→97.9%) despite viewpoint-invariant content — a desk
  orbit makes heading informative where the position kernel is broad; the
  classroom no-yaw verdict does not transfer to orbit-style trajectories.
  Substitution law needs this scope note. **Multi-session merge upgraded to
  real sessions:** 4 separately-built session memories, added: max |Δ|
  1.58e-15, identical answers 2000/2000 on the full test battery. Artifacts:
  `outputs/cross_recall_7scenes_chess{,_yaw}.json`; embeds
  `outputs/7scenes_chess_seqXX/crop_embeddings_eigenplaces.pt` (geometry:
  cos +0.40-0.49, eff-rank 11-24 — desk-scale coherence like TUM).
  Next candidates (Paul's call): more scenes via student GPU; fire as a
  second CPU scene; the table into the workshop draft.

- **2026-08-04 (ay)** — **Scene-graph comparison SCOPED (not yet run):
  Replica room0, ConceptGraphs query protocol, split CPU/student-GPU.**
  Verified facts: ConceptGraphs (arXiv:2309.16650, ICRA'24) evaluates on the
  8 NICE-SLAM/iMAP Replica renders (room0-2, office0-4, 2000 RGB-D frames +
  per-frame poses each; single Replica.zip from
  cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip — full zip ~all 8 scenes,
  size to confirm at download; per-scene mirrors exist, verify then).
  Their tables: (a) Replica closed-set semantic segmentation, aggregate
  mAcc 40.63 / F-mIoU 35.95 — WRONG arena for us (dense point labelling,
  not a memory task; we'd lose by construction and it measures their
  segmentation stack, not the map). (b) Text-query object retrieval,
  R@1/2/3 over descriptive (20) / affordance (5) / negation (5) queries
  (CLIP-scored vs LLM-scored variants; e.g. descriptive CLIP R@1 0.59) —
  RIGHT protocol shape: query → object instance, scored at instance level,
  no mm anywhere. **Recommended fair test:** room0 only; both systems
  consume the SAME 2000 frames; identical closed-vocab class-query battery
  (+optional CLIP open-vocab descriptive queries); success = answer within
  r=1 m of a GT instance of the queried class (Replica semantic mesh gives
  GT instance centroids), report R@1/R@3 + success rate; then the systems
  columns where the real contrast lives: per-scene map bytes, map build
  hardware, update cost, and a two-session merge column (split the
  trajectory; ours merges by addition, theirs has no merge op — reported as
  capability, not gotcha). Their repo's demo scene IS Replica room0, so the
  student can run their official pipeline unmodified. **CPU (this machine):**
  room0 download, pose conversion (NICE-SLAM traj.txt = 16-number rows →
  same CSV path as prepare_7scenes), YOLO+dinov2/CLIP crop embeds, VSA
  object map, battery scoring, inspector page. **Student GPU:** official
  ConceptGraphs run on room0 (SAM+CLIP+LLM stack), their eval tooling for
  GT instance centroids export (small JSON back to us), their map
  size/build-time profile, optional CLIP embeds for our open-vocab rows.
  Caveat to carry: their published R@k numbers may be on their real-world
  scans — the strict same-data comparison is vs the student's room0 run of
  their code, not vs the paper table; the paper table is quoted as protocol
  precedent only.

- **2026-08-04 (az)** — **T1 BREAKTHROUGH: the χ law's empirical structure
  is pinned, with an out-of-sample falsification test.** Working note:
  `wiki/analysis/2026-08-04-t1-chi-law-empirics.md`. The law:
  χ(N) = a_μ·N + √N·√(α/PR_eff + γ/hd)·h(N,τ), h∞ = √(1+τ/τ₀).
  (1) μ-term exactly linear to 3×10⁵ (χ/N constant 0.0083). (2) The single
  hd=4096 collapse g·PR^0.25 ≈ 0.039 was FALSIFIED by a new hd=1024 run
  (C scaled ×1.85 ≈ hd^-1/2, A only ×1.12) and replaced by the quadrature
  model g² = α/PR_eff + γ/hd, which predicts the held-out A@hd1024 cell to
  **1.4%** — and identifies the real-data "whitened floor is
  encoder-independent" finding as the γ/hd projection-noise term (only
  more dimensions lower it; no treatment can). (3) Persistence: coherent
  for N≲τ then √-reverting to a PERMANENT elevation h∞=√(1+τ/τ₀), τ₀≈60
  (elevation²−1 ∝ τ verified across τ=10 vs 2000); the μ-term is the τ→∞
  limit, which is why centring and key diversity are the two rescues.
  Remaining for T1: derive the constants, one h(N,τ) expression, and the
  no-per-cell-fit closure on the real 16 cells; multi-seed pass on α
  (C-vs-A disagree ×1.8 on single seeds). Two falsifiable predictions
  recorded in the note (hd-doubling asymmetry; fps-subsampling τ test —
  the latter is CPU-cheap on school_run1). Also this session: 
  object_grounding synthetic unit test passed (6/6 instances at ≤0.13 m,
  merge 2.6e-13; cluster-splitting noted as provisional-mode caveat).
  New data: `outputs/synthetic_hd1024/rows.jsonl`.

- **2026-08-05 (ba)** — **Replica room0 CPU half COMPLETE: grounding
  battery + bounded-insertion fix + inspector.** Data landed via
  range-extraction (5.15 of 12.4 GB transferred; resume + retries added
  after two flaky-network failures); 2,000 frames validate (3.1×2.2 m,
  render size 1200×680 matches intrinsics); YOLO found 8,028 detections /
  23 classes; all 8,028 depth-placed. PROVISIONAL protocol (GT centroids
  await the student's semantic-mesh export; targets = second-half
  clusters, memory = first half): raw trace R@1 44% — diagnosed via new
  n_mem column as (i) classes never stored (bottle/oven n_mem=0), and
  (ii) **dominant-class crosstalk: couch at 1,150 of 2,521 detections
  drowns book at 5 — the χ law at class level.** Fix: **bounded per-class
  insertion** (`--max-per-class 60`, reservoir-style, online-compatible,
  one trace, merge exact by construction): **R@1 44→62% all classes,
  77% on in-memory classes**; bed 1.25→0.27 m, book 3.45→0.23 m; merge
  1.56e-12, 16/16 identical answers. Remaining in-memory misses (sink,
  tv, cup) look like render phantoms (reflections/depth-through-glass;
  room-margin filter already drops 2,669 outside-the-room points) —
  visible in the inspector. Inspector (standing rule) published:
  artifact 506ec46c — per class: map with target rings + top-3 answer
  crosses, target crop photo vs the memory's grounded crop photo, HIT/MISS
  pill, no curation. Tools: `object_grounding.py` (+n_mem column,
  room-margin, cap), `export_grounding_inspector.py`,
  `prepare_replica.py` (range-fetch + resume). Bounded insertion is a
  paper-worthy mechanism finding: the χ law predicts the failure AND the
  fix's effect direction. Awaiting student: ConceptGraphs room0 run + GT
  export (brief in START_HERE §7) → rebuild vs true GT, then the
  head-to-head table.

- **2026-08-05 (bb)** — **Consistency ≠ truth resolved: metric renamed,
  CLIP second-labeler added, REAL ground truth extracted on CPU, and the
  same-frontend backend table built (Paul's fairness directive).**
  (1) The provisional metric is now printed/stored as **consistency-R@k**
  with an explicit consistent-mislabel warning. (2) `--verify-crops`:
  CLIP ViT-B/32 must agree with YOLO (top-3) per crop; dropped 1,212/5,359
  observations and the phantom CLASSES (toilet, sink, oven, person)
  vanish from the battery; in-memory consistency 88%. (3) TRUE GT without
  the student: `tools/replica_gt_from_renders.py` backprojects the vMAP
  release's per-pixel instance+depth+pose renders (HF kxic/vMAP
  demo_replica_room_0.zip, 6.2 GB; depth scale auto-detected 1000;
  frame-alignment gate passed — GT walls enclose our camera path) → 89
  instances, 20 COCO-comparable, `outputs/replica_room0/gt_instances.json`
  (student's mesh export demoted to cross-check). REPLICA2COCO mapping
  with judgement calls stated (stool→chair, table→dining table).
  (4) **TRUTH-R@1 = 83%** (5/6 GT-comparable classes; book 0.09 m, chair
  0.04, couch 0.01, table 0.18, plant 0.23; vase FAILS at 4.4 m and is
  the perfect teaching case: consistency 0.07 m — CONSISTENT BUT FALSE,
  a stable detector error, same phenomenon as toilet-armchair).
  (5) **Same-frontend backend table** (identical observation stream):
  VSA trace 32 KB / instance list ~1 KB / raw store — all tie at
  truth-R@1 83% (frontend-limited; the trace gives nothing away). Honest
  framing recorded: at one-room scale bytes favour the explicit list; the
  trace's wins are constant-size in N, EXACT merge (vs association
  problem), and one algebra carrying object+place+time. (6) 2×2 completion
  path: student brief updated to also export ConceptGraphs' observation
  stream (their-frontend × our-backend cell). SOTA context for framing:
  GT-scored open-vocab semantic segmentation on Replica — ConceptFusion
  24.2 mAcc, ConceptGraphs 40.6, HOV-SG 38.1, KeySG 45.8; ConceptGraphs
  text-query R@1 0.59. Inspector rebuilt as the TWO-METRIC page
  (consistency + truth pills, four-quadrant explanations, GT squares on
  the map; artifact 506ec46c refreshed). All numbers:
  `outputs/replica_room0/object_grounding.json`.

- **2026-08-05 (bc)** — **Map joining: survey + measured head-to-head; the
  merge claim is now precise, cited, and quantified.** New note
  `wiki/analysis/2026-08-05-map-joining-vs-alternatives.md` (survey of 7
  merge families with comms numbers — Kimera-Multi 24–146 MB/experiment vs
  our 32 KB/robot — plus the novelty statement: bundling-as-map-merge
  between robots appears to be NEW; closest prior art is within-agent HDC
  aggregation (Neubert & Schubert CVPR'21) and federated HDC averaging).
  New `vsa_cognitive_mapping/merge_comparison.py` — room0, 4 robots,
  disjoint quarters, truth-scored: VSA addition 83% R@1 / 0 association
  steps / 1.8e-12 exact; instance-list associative merge 67% (86 greedy
  decisions, 5 classes drift — the scene-graph merge failure mode measured);
  count-grids 67% at 108 KB; raw union 83% unbounded. Standing caveat
  recorded verbatim in the note: the algebra ASSUMES the shared frame, it
  does not solve alignment; correlation-sweep frame recovery is the open
  robustness experiment. Also: belief-field heatmaps added to the grounding
  inspector (the distribution IS the map; argmax is a readout — Paul wants
  this featured as a representation claim), GT squares recoloured for
  visibility. Full 8-scene Replica run resumed (tasks #30-32): room1/office0
  fetching, remaining five queued; merge_comparison reruns per scene.
  Belief-field artifact: 506ec46c.

- **2026-08-06 (bd)** — **FULL 8-SCENE REPLICA RUN COMPLETE. Pooled
  truth-R@1 88.1% at 1 m (37/42 GT classes; Monte-Carlo chance anchor
  18.2%), 78.6% at 0.5 m (chance 5.9%), one 32 KB trace per scene,
  2,357–10,465 observations/scene.** Per scene (1 m): room0 83, room1 100,
  room2 83, office0 83, office1 100, office2 100, office3 83, office4 80.
  Frontend = unverified YOLO+depth on every scene (room0 rerun to match
  after a verified/unverified inconsistency was caught). Vocabulary
  coverage stated per scene (COCO-80 sees only 9–31% of GT instances —
  the battery examines that subset honestly). **Miss taxonomy: all 5
  misses are frontend (2 never-detected, 3 wrong-observation
  label/depth); memory-attributable misses = 0** — on every scene the
  trace matched or beat the instance-list ceiling on stored classes, and
  on office1 it grounded a table from a SINGLE stored observation the
  clustering backend's min-count discarded. Backend means (8 scenes):
  VSA 87% / instance list 84% / raw store 81%. **4-robot joining means:
  VSA addition 89% (worst 80%), zero association decisions, ~1e-12 exact;
  associative merge 84% (worst 67%) with 622 decisions and 3–8
  classes/scene drifting; grids 82% at ~95 KB; raw union 85% unbounded.**
  Chance anchors, coverage denominators, dual operating points (1 m /
  0.5 m), and the miss taxonomy follow the interpretable-results
  discipline. Deliverables: summary artifact e72578bd (quant page),
  8 per-scene belief-field inspectors (grounding_inspector_<scene>.html),
  `tools/aggregate_replica_truth.py`,
  `outputs/replica_truth_aggregate.json`. SOTA framing unchanged:
  segmentation mAcc numbers cited as different-metric context; the
  same-metric head-to-head is the student's ConceptGraphs run scored by
  our script (her brief covers it). Tasks #30–32 closed.

- **2026-08-06 (be)** — **METRIC CORRECTION (Paul, standing rule) + student
  end-to-end package shipped.** Paul: "don't invent metrics" — the
  truth-R@1@1m protocol is bespoke and therefore incomparable; recorded as
  permanent memory (feedback-no-invented-metrics): adopt the target
  benchmark's metric/scorer/GT verbatim; homemade numbers are diagnostics
  only, never headlines. Consequence: `student_gpu_package/` (pushed,
  results-sites branch) makes BOTH systems scoreable under ConceptGraphs'
  OWN eval — our trace produces dense per-point labels (argmax over class
  fields, CLIP-relabelled to the Replica vocabulary with their prompt
  style) so their mAcc/F-mIoU scorer judges both; 5 stages, their pipeline
  unmodified, one scorer, provenance recorded. CPU smoke passes on room0.
  **Honest pre-registered finding: our all-classes mAcc ≈ 0.076 vs their
  ~0.40** — structure classes (wall/floor/ceiling, majority of points) are
  unlabelable by an object memory while SAM segments everything; which
  classes their eval actually scores is determined by their code at her
  run (README records it). The 88.1% grounding number is hereby demoted
  to internal diagnostic; the paper's semantic comparison waits for the
  their-metric result. Existing pages to be re-worded accordingly in the
  next pass.

- **2026-08-08 (bg)** — **RELATIONAL RECALL R1 measured (Paul's field-algebra
  idea): products halve wrong-instance errors at zero storage; gate vs
  appearance keys NOT passed; mechanism characterised.** Plan + full fork
  analysis: `wiki/analysis/2026-08-06-relational-recall-plan.md`. New
  `vsa_cognitive_mapping/relational_recall.py`: deterministic GT-anchored
  battery ("the {c} nearest the {a}", unique-with-0.5m-margin filter, 24
  kept / 40 ambiguous — Replica rooms are margin-starved), zero new
  parameters (proximity = the FPE kernel). Results: class 29/71/0
  (inst-correct/wrong-inst/off), field PRODUCT 46/21/33, peak-selector
  29/58/12, appearance (57 view-queries) 72/9/19, app×anchor 47/23/30.
  Two mechanisms understood: the product of two kernels peaks BETWEEN
  target and anchor (the 33% off-instance artifact), and close same-class
  instances blur into one lobe at ℓ=0.6 so peak-selection lacks candidates.
  Paul's decision: path A (bank product result, advance R3 displacement
  exactness + R2 view+statement) with multi-agent execution, plus a
  path-C scout (instance-layer selection incl. matching-pursuit field
  enumeration). Three agents dispatched; tasks #34/#35/#39 in progress.

- **2026-08-08 (bh)** — **RELATIONAL SPRINT COMPLETE (3 agents + closing
  run): the fork resolved, the mechanism unified, the exactness flagship
  measured.** Full tables in the plan page. (1) **R3 displacement
  predicates are EXACT**: unbinding with sem[X] ⊗ S(−Δ) equals the
  physically shifted field to machine epsilon (max |Δfield| 1.5e-16
  office3 / 8.0e-17 room0 over a 4×4 Δ-sweep); argmax errors sub-grid
  (≤0.09 m at 0.12 m cells); sign convention measured 16/16; the
  "between A and B" product composition has a MEASURED envelope (all 15
  pairs: works when kernels overlap, |AB| ≲ 1.3 m at ℓ=0.6; locks to one
  anchor beyond). Figure outputs/relational_displacement.png. (2) **R2
  honest negative**: verbal anchors via field VALUES damage more
  view-answers than they rescue (4 rescued / 19 damaged; scale-free
  conditional rule → net −2). (3) **C-scout decisive**: observation
  clusters + Euclidean selection 71/17/12; matching-pursuit enumeration
  recovers 85% of instances FROM THE TRACE ALONE; field-value selection
  25% (blind past kernel support) but **PURSUIT-EUCLID (trace candidates
  + metric selection) = 67/21/12** — within one query of the explicit
  ceiling, fully trace-derived. **Unified mechanism, three independent
  confirmations: field values cannot rank "nearest" beyond ~1 m; metric
  distance over trace-derived candidates can.** Architecture resolution:
  one additive trace + a derivable candidate index (~16 B/instance,
  reconstructable by pursuit after merge/loss); vs FARM the sole
  concession is metric selection — no stored relations anywhere. New
  tools: relational_view_battery.py, relational_instance_layer.py,
  tools/relational_displacement.py. Tasks #33-35, #39 complete; queued:
  metric-anchored R2 follow-up, R4 z-axis, R5 language, R6 deliverables.

- **2026-08-08 (bi)** — **SKEPTIC PANEL on the relational sprint (3
  adversarial agents; all major findings upheld; corrections applied).**
  Full corrections block now heads the relational plan page. Summary:
  (1) the 46% product row was UNREPRODUCIBLE (decode edited away after
  recording) — restored as a named design, rerun reproduces 46/21/33
  exactly, artifact now on disk; (2) entry (bh)'s "from the trace alone /
  fully trace-derived" RETRACTED — every pursuit number uses GT instance
  counts (twice for euclid), stopping rule unimplemented, k+1 sensitivity
  unmeasured on the headline design; (3) cross-design orderings on the
  n=24 battery are statistically unestablished (17 distinct targets; ~10
  post-hoc designs; cluster-vs-euclid = one query) — all downgraded to
  hypotheses; (4) circularity: the metric decoders execute the battery's
  own generative min-min rule — 67/71% measure position recovery, not
  relational inference; (5) the margin filter excludes the near-twin
  regime that motivated the sprint (kept-margin median 1.29 m > scoring
  radius); (6) "between" works 2/15 pairs (kernel-overlap envelope),
  figure pair degenerate; shift exactness stands but is an identity
  check, not a capability; (7) all numbers are GT-associated upper bounds
  (memory-side GT filtering, transductive normalisation). The relational
  thread is now banner-labelled INTERNAL DIAGNOSTIC per the
  no-invented-metrics rule. Survives: the R2 negative, exactness as
  implementation check, pursuit-as-mechanism pending a legal stopping
  rule.

- **2026-08-10 (bk)** — **COLLABORATOR NOTES INGESTED (Telluride-VSA-Maps):
  three contributors, and they point at a DIFFERENT paper than our ICRA
  draft. Scope decision now live for Paul.** Source:
  `Telluride-VSA-Maps (1).txt` (Lorin, Shay, Naitri + Paul's 07-30 results
  page). (1) **Lorin**: two threads — same-front-end head-to-head
  (SuperMap-style 4D scene graph vs SSP/VSA backend on HM3D/Habitat VLN
  tasks) and a HYBRID (SSP as a fast index layer over a scene graph, SSP
  replacing edge/existence tracking, SSP as per-node confidence). Metrics:
  fixed footprint, query big-O, wall-clock, update cost. (2) **Naitri**:
  states the objective as a **fixed-budget VSA episodic OVERLAY on an
  open-vocab 4D scene graph** — graph authoritative for current state, VSA
  for approximate history, returning candidates for exact verification;
  central hypothesis is retained-history-at-fixed-memory, not replacement;
  three experiments (fixed-budget temporal memory vs truncated/sampled
  histories; relational retrieval vs FARM incl. compositional anchors;
  repeated-visit robot eval). (3) **Shay**: equation-level FHRR ↔
  SuperMap/FARM correspondence note with an E/A/R/W taxonomy (6 exact, 4
  approximate, 10 reformulation, 0 weak). Convergences with our measured
  record: his participation-ratio effective load N_eff IS our χ law's
  PR_eff term, derived analytically; his "**the bundle should be an index,
  not the primary store**" is exactly the relational-sprint architecture
  resolution (bh); his sequential-extraction/matching-pursuit under
  decreasing noise is our pursuit result, and he ties it to R@5/R@10 as
  the deliverable. New material we do NOT have: submap-local frames as the
  loop-closure/shared-frame fix (directly addresses our standing merge
  caveat); the exact fusion update (remove-and-replace so N counts objects
  not observations) plus the bundle-norm consistency statistic κ as a
  threshold-free data-association score; residue/CRT codes to decouple the
  RANGE budget from the CROSSTALK budget (grid-cell anchor); structured
  orthogonal random features to kill the Θ storage cost (~12 MB at d=4096
  — relevant to our "decoder grids dominate map state" honesty note);
  geometric decay giving N_eff bounded independent of N. **TENSION to
  resolve, not paper over:** Shay's pipeline whitens semantic embeddings
  before FPE, while our measured rule is centre/z-score, never whiten
  (−47.6% classroom 5-seed). These may not actually conflict — ours is
  measured on content used as a BINDING KEY, his is preprocessing for an
  FPE ARGUMENT (encoded, not bound), which is a different pathway; worth
  an explicit experiment and a precise statement either way. **Scope
  question for Paul:** our ICRA draft frames the trace as the OPPOSITE
  design point to scene graphs (standalone characterisation); the
  collaborators are converging on HYBRID/overlay. Not a contradiction (the
  standalone characterisation is the prerequisite for arguing the overlay)
  but the framing, and possibly the paper split, is his call.

- **2026-08-10 (bj)** — **ICRA PAPER BUILD-OUT: complete IEEE-style draft
  in `paper/` (main.tex + refs.bib + figures/), targeting the Overleaf
  project (6a75e41dde34d8bf1ef1b38f; auth-gated, manual sync).** Scope
  decision encoded: ONE track — the fixed-size algebraic map. Four
  pillars, all already measured: system + conditioning discipline
  (Sec. III), χ capacity law with the 1.4% held-out prediction and the
  imbalance→bounded-insertion story (IV), benchmark characterisation
  (chess (ax), Replica ×8 (bd) with miss taxonomy, drift bounding (ah))
  (VI), exact merge with the 4-robot head-to-head (bc)/(bd) and the
  shared-frame caveat verbatim (VI-C). Deliberately out: isotropy science
  (→ workshop draft), multi-robot at scale + frame recovery (→ paper 2),
  relational/instance (Limitations ¶ only, internal-diagnostic). Standing
  rules enforced in-text: no invented metrics (grounding labelled
  characterisation; ConceptGraphs comparison pre-registered under THEIR
  scorer as a \todo slot), multipliers with encoder+N, ≤0.5/1/2 m
  fractions, seed tags, banned phrases swept. Intro+conclusion are
  complete drafts flagged MERGE NOTE (Overleaf partials unseen — Chrome
  extension disconnected). Blockers: D2 authorship (author block
  placeholder), their-scorer run (table slot), ieeeconf.cls swap at
  submission. 7 \todo markers; env/cite/ref/brace checks pass. Cut
  ladder for the ~1-page overrun in paper/README.md.

- **2026-08-06 (bf)** — **INSTANCE vs SEMANTIC recall measured (Paul's
  diagnosis confirmed) + the 2025 primer restored and reviewed.**
  New `vsa_cognitive_mapping/instance_recall.py`: 1,309 held-out
  instance queries over 5 Replica scenes, three memory designs, per-instance
  frame-split. Multi-instance classes (the diagnostic): class⊗pos is
  class-correct-but-WRONG-INSTANCE **72%** of the time (25% instance-correct)
  — the associative gap is structural, exactly as Paul reasoned. The
  VSA-native fix — each observation's own appearance phasor as the key
  (app⊗pos) — nearly doubles instance accuracy (43%); hybrid
  (app⊗class⊗pos) keeps 43% and cuts outright errors to 3%. Two-cause
  decomposition of the remaining gap: weak YOLO 256-d appearance keys
  (DINOv2-key rerun queued) and genuinely IDENTICAL instance models in
  Replica, for which appearance cannot individuate even in principle —
  individuation beyond that is spatial/contextual (instance tokens at
  insertion, or query-side context binding; ties directly to C-JEPA's
  identity anchors). `outputs/instance_recall.json`. Also: talking-guide
  deep-dive links fixed (context-aware artifact vs relative), merge section
  rewritten in plain language, and the Telluride-era demo page RESTORED to
  docs/sites (vsa_primer_2025.html, unchanged, live widgets) with a
  claim-by-claim review page (vsa_primer_review.html, artifact 0c469127):
  homomorphism/merge STANDS; heading construction STANDS (no-DC
  annihilation + multi-lobe petals spelled out) while heading-BINDING is
  SCOPED by the substitution law; √(2D/k) capacity SCOPED as the
  clean-atom slice of the χ law; whitening SPLIT into
  whiten-vocabularies / centre-streams; instance blindness + the
  no-invented-metrics rule recorded as REVISED.

- **2026-08-04 (as)** — **ABSOLUTE-TERMS CORRECTION (user-prompted, correct):
  held-out localisation is POOR, and "signal %" framing obscured it. CDFs are
  now mandatory alongside signal.** Same 91-query classroom battery as the
  photo inspector (bind-yaw, zscored dinov2, blocked seed 0; oracle 0.29 m):

  | method | median | ≤0.5 m | ≤1 m | ≤2 m | worst |
  |---|---|---|---|---|---|
  | VSA memory (argmax) | 0.96 m | **38.5%** | 52.7% | 79.1% | **8.1 m** |
  | VSA (best of top-3 modes) | 0.90 m | 39.6% | 57.1% | **92.3%** | 4.7 m |
  | plain kNN, same descriptors | 0.78 m | 38.5% | 57.1% | 86.8% | 5.4 m |

  Half the queries are worse than ~1 m in a 6.9 m room; that is bad
  localisation and must be said in those words. **Decomposition of the
  badness:** (1) most of it is the **descriptor/data ceiling** — exact
  nearest-neighbour on the same features also only manages 0.78 m median and
  the identical 38.5% ≤0.5 m, so mean-of-crops descriptors across ≥1.5 s
  viewpoint gaps on single-pass data are the bottleneck, not the memory;
  (2) the **superposition cost** is ~0.2 m of median plus a heavier tail;
  (3) the tail is largely **argmax-on-multimodal-fields** — scoring the best
  of the top-3 modes collapses ≤2 m from 79% to 92% and worst from 8.1 to
  4.7 m, so the truth is usually near a secondary mode (consistent with (aq)).
  **What this does NOT threaten:** the paper's claims (treatment orderings,
  the capacity law, merge, drift bounding) never rested on absolute VPR
  quality, and the honest-evaluation instrument exposing this is it working as
  designed. The drift-governor use case remains fine (0.36 m fused) because
  odometry does the fine work and the memory only corrects drift.
  **What it DOES require:** (a) never present "recall" as localisation
  competence — the D4/ICRA framing must not oversell it; (b) **standing rule
  added: every recall table reports ≤0.5/≤1/≤2 m fractions (or the time
  equivalents) alongside signal %**; (c) top-k readout is promoted from
  nice-to-have to the recommended readout; (d) if localisation quality ever
  becomes a goal, the route is better place signatures (layout-aware pooling,
  multi-pass data), since kNN's ceiling shows descriptors are the bottleneck.

- **2026-08-04 (ar)** — **THIRD SCENE, THIRD QUERY TYPE, SAME VERDICT: new
  module `time_recall.py` runs the through-memory test on `school_run1` via
  its VALID factor (time; positions are void) — and whitening is again the
  worst treatment while z-scoring is best.** Content ⊗ ctx_time(t), blocked
  splits, 3 seeds, dinov2 crops, 10,069 frames over 455 s, decode over a
  400-cell time grid, brackets as ever (oracle 5.1 s, constant 110.3 s):
  raw **20.5±8.9%**, centred **30.0±5.2%**, zscored **36.4±10.3%**,
  whitened128 **8.5±4.2%** (barely above the constant predictor). The
  through-memory treatment ordering now replicates on **three scenes × three
  query types**: position (classroom, af), position-given-heading (TUM
  bind-yaw, am), and time (school_run1). "z-score/centre > raw > whitening"
  is no longer a single-task claim. Absolute recall is honest-modest (median
  83 s over a 455 s walk) — the ordering, not the magnitude, is the result.
  **Two-robot merge demo on real school data with a valid factor:** robot A =
  first half of the walk (0–221 s, 3,166 frames), robot B = second half
  (221–440 s); merged vs jointly-built memory agree to **1.81×10⁻¹⁵**;
  24/24 held-out queries return **identical** answers (max field difference
  1.0×10⁻¹²). `outputs/school_run1/time_recall.json`.

- **2026-08-04 (aq)** — **TUM error forensics: the "consistently wrong" guesses
  are two distinct, named mechanisms — and neither is memory malfunction.**
  (Same seed-0 bind-yaw run as the test visual; scratch forensics script; three
  hypotheses tested.)
  **H1 CONFIRMED (raw+yaw): faithful-but-ambiguous.** The decoded peak sits
  within **0.12 m median of the best stored match's position** under the
  memory's own weighting (content-cos × heading-kernel), vs 0.284 m from the
  truth, and is closer to the best match than to the truth on **75% of
  queries**. On the worse half, decoded stays glued to the best match (0.13 m)
  while that match sits **0.60 m from the truth**. The memory returns exactly
  what associative recall should return — the same (appearance, heading)
  recurs on a *different sweep pass* ~0.6 m away. Position aliasing in the
  data, not retrieval failure; same family as the documented multimodal-argmax
  limitation (one more reason to report top-k modes).
  **H2 CONFIRMED (zscored/centred): near-pure density pull.** Error-direction
  vs to-centroid cosine = **+0.86** for zscored (vs +0.22 raw), median error
  1.89 m — into the grid's padded margin. Deleting the mean deletes the
  carrier; the residual field is dominated by the crosstalk mass, whose bump
  sits at the stored-pose density mode → every answer drifts the same way =
  literally "consistently wrong". This is the mechanistic face of the negative
  signal rows in (am).
  **H3 EXONERATED: time alignment.** Real rgb timestamps vs the uniform-30fps
  assumption: median offset 1 ms, max 4 ms; at p95 camera speed 0.70 m/s the
  worst pose-association error is **3 mm**. The FolderSequence uniform-time
  synthesis is fine on this sequence.

- **2026-08-03 (ap)** — **T4 complete: 500-rep bootstrap CIs close the batch.
  Both scene-level ladder claims are now interval-backed.** Classroom gap-300:
  raw 0.720 [0.704, 0.731], zscored 0.722 [0.707, 0.735], whitened128 **0.623
  [0.602, 0.628]** — the classroom whitening penalty is far outside CI overlap
  (raw−wht ≈ 0.10 vs interval widths ~0.03): **real**. TUM gap-300: raw 0.476
  [0.449, 0.494], zscored 0.469 [0.443, 0.486], whitened128 0.468
  [0.439, 0.481] — fully overlapping: the TUM "no penalty" is likewise
  interval-backed, and raw-vs-zscored differences at this gap are **not
  distinguishable** on either scene (consistent with (ai)'s "one family").
  T4 items now all closed: W-seed stable (ah shares reproduce at seeds 1/2),
  hd-8192 spot check ran (table in `outputs/robustness/heldout_hd8192.log` —
  extraction into the WP7 metric section still to do), fraction-matched gap
  framework in `outputs/robustness/fraction_matched.md`, bootstrap CIs both
  scenes. **Phase-2 experiment queue: T2 ✓ T3 ✓ T4 ✓ T6 ✓ — remaining: T1
  (derivation, mine), T5 (writing), seeds/polish for the synthetic figure,
  kNN codebook-byte accounting check, and the writing WPs.**

- **2026-08-03 (ao)** — **Persistence configs land: hypothesis CONFIRMED, with a
  bend that upgrades the claim. Coherent growth is a within-correlation-length
  regime; only the mean is O(N) forever.** (ρ=0.9995 ⇒ correlation time ~2,000
  items; single seed; `rows.jsonl` + updated `synthetic_scaling.png`.)

  | config @3e5 | χ | vs C (5.26) | fitted slope |
  |---|---|---|---|
  | D2 iso + persistent | 33.1 (58.3 @1e6) | 6.3× | +0.31 overall; ~0.6–0.8 pre-bend |
  | F low-rank + persistent | 68.5 | 13× | +0.40 overall; elevated pre-bend |
  | B mean (from an) | 2,490 | 470× | +0.94, no bend |

  **(1) The bend.** Persistent correlation produces ELEVATED, near-coherent
  growth while N ≲ the correlation length (χ already 2.05 at N=100 — 18× the
  isotropic level), then **bends back toward √N once N spans many correlation
  lengths**. Only the mean — persistence of infinite length — stays O(N)
  forever (B: slope 0.94 to 2,490×, no bend).
  **(2) This EXPLAINS the real-data O(N) without contradiction:** the
  classroom's measured slope ~1.0 after centring is a *within-correlation-
  length* observation — N=2,429 items from a walk whose place-cluster
  persistence spans the whole sequence. The synthetic shows what happens
  beyond: reversion to incoherent scaling with a persistence- and
  spectrum-dependent prefactor (F > D2 > C confirms the spectrum multiplies
  the prefactor).
  **(3) NEW, practically valuable claim:** naive O(N) extrapolation from
  small-N measurements OVERSTATES crosstalk at scale — asymptotic growth is
  √N for everything except the mean. Capacity forecasting must separate the
  μ-term (kill it: centre, or bind a diverse key) from the finite-persistence
  term (bends on its own). **T1's law now has its full shape:**
  χ ≈ N·‖μ‖-term + f(τ,N)·persistence-term + √N·g(Σ) — with f coherent for
  N≲τ and √-reverting beyond.
  **Caveats:** single seed; visible non-monotonicity (χ dips 10k→30k on D2) =
  probe noise, needs 2–3 seeds before the figure ships; correlation *shape* is
  AR(1), whereas real walks are closer to piecewise-constant clusters — worth
  one block-structured config as a check.

- **2026-08-03 (an)** — **T2 synthetic sweep complete to N=10⁶ — two clean
  confirmations, two INFORMATIVE FAILURES, and a sharper hypothesis: the growth
  EXPONENT is governed by correlation PERSISTENCE, the spectrum only sets the
  PREFACTOR, and the mean is persistence of infinite length.** (New module
  `synthetic_scaling.py`; controlled dials m/α/ρ through the exact pipeline;
  streaming bundle, log-uniform probes; two bugs found+fixed en route:
  checkpoint-skipping chunks and a rank-reducing whitener vs the shared W.
  `outputs/synthetic/rows.jsonl`, `synthetic_scaling.png`. hd=4096, d_in=256.)

  | config | dial | χ @3e5 | slope | expectation |
  |---|---|---|---|---|
  | C isotropic i.i.d. | — | 5.26 (9.63 @1e6) | **+0.48** | ✓ textbook √N |
  | B + mean (80% energy) | μ | **2490** | **+0.94** | ✓ coherent O(N) |
  | E low-rank, whitened | Σ→flat | 6.84 | **+0.50** | ✓ back to isotropic law |
  | A low-rank (PR 3.6/256) | Σ | 14.78 | **+0.48** | ✗ predicted ~1, got √N |
  | D isotropic + AR ρ=0.9 | ρ | 5.80 | **+0.39** | ✗ predicted 0.5–1, got ≈C |

  **The failures are the finding.** Static low rank does NOT produce coherent
  O(N) growth — independent draws from an anisotropic Σ have larger but
  RANDOM-SIGN pairwise correlations → incoherent √N with a bigger constant
  (A ≈ 2.8× C at matched N). Short-range AR redundancy (corr time ~10) doesn't
  either. Yet real *centred* crops grow at slope ~1.0 (ah). So the O(N) on real
  data after centring must come from **long-lived, sign-biased structure** —
  place clusters that persist for minutes, i.e. a *piecewise-constant local
  mean* global centring cannot remove. Hypothesis: **exponent ← persistence
  length; prefactor ← spectrum; μ = the infinite-persistence limit.** Decisive
  follow-up RUNNING: ρ=0.9995 (corr time ~2000) isotropic + low-rank configs —
  predicted slope →1 over the measured range.
  **This unifies (am):** binding a diverse key (yaw) phase-scatters exactly the
  persistent component — key diversity is a persistence destroyer, which is why
  it substituted for centring. One mechanism now explains centring, the
  whitened floor, the O(N)-after-centring puzzle, and the yaw result.
  T1's law should be written in these terms: χ² ≈ (coherent/persistent
  fraction)·N² + (incoherent prefactor from Σ)·N, per unit signal.

- **2026-08-03 (am)** — **T3 lands and REVISES THE CENTRAL RECOMMENDATION: key
  diversity substitutes for centring. Yaw binding cures the degenerate-raw
  pathology on both scenes, gives the whitening claim its second scene, and
  beats the no-triple alternative.** (`--bind-yaw` stores
  `content ⊗ ctx_pos ⊗ ctx_head`, yaw KNOWN at query; `--place-radius 0.5`
  is the two-factor alternative with view-averaged content. Blocked, 5 seeds,
  json copies per scene×variant.)

  Blocked signal vs the position-only baselines ((ai) classroom / (al) TUM):

  | scene · variant | raw | centred | zscored | whitened128 |
  |---|---|---|---|---|
  | classroom · none (ai) | 31.7 (dinov2) / −38.7 **cells=1** (yolo) | 70.6 / 53.6 | 71.6 / 58.6 | −23.3 / −17.2 |
  | classroom · **bind-yaw** | **65.1 / 63.3** (cells 50/20) | 66.6 / 63.3 | 68.6 / 61.2 | −25.8 / −31.1 |
  | classroom · place-sig | 37.0 / −38.9 (yolo still cells=1) | 65.9 / 53.1 | 67.4 / 58.3 | −29.1 / −20.1 |
  | TUM · none (al) | 16.8/17.0 **cells≤3** | −64.9/−92.1 | −68.6/−83.3 | −171.8/−196.3 |
  | TUM · **bind-yaw** | **66.2 / 66.7** (cells 14/12, median 0.33 m) | −82.7/−64.5 | −90.2/−80.4 | −180.1/−194.8 |
  | TUM · place-sig | 16.3/16.9 (cells ≤2) | −71.3/−88.2 | −76.4/−81.9 | −173.6/−203.7 |

  **(1) THE MECHANISM, and it feeds T1 directly: binding a diverse third key
  scatters the coherent mean term.** Without yaw, near-parallel raw contents
  leave the same residual for every query → constant predictor. With yaw bound,
  the per-item heading phasors randomise the phases of the shared-mean
  component across stored items — the μ-term's coherent pile-up is destroyed
  *by the key*, not by preprocessing. Consequences visible in the table:
  raw yolov8n goes **−38.7% (cells=1) → 63.3% (cells=20)** on the classroom
  and **17% → 66.7%** on TUM; and centring's benefit **evaporates** (classroom
  centred 66.6 vs raw 65.1, within noise) because the mean it removes is no
  longer harmful.
  **(2) On the orbit, centring HURTS with yaw bound** (−65 to −90 vs raw's
  +66): TUM's appearance–heading coupling means the shared mean *carries* the
  usable signal once heading is factored; deleting it deletes the channel.
  So the recommendation is now conditional: **bind diverse keys OR centre;
  never whiten** — with key diversity the preferred, algebra-native fix.
  **(3) THE WHITENING CLAIM GETS ITS SECOND SCENE.** Under bind-yaw, TUM is a
  *working* memory (raw 66%, median 0.33 m in a 2.42 m space) and whitened128
  still sits at **−180 to −195%** — whitening destroys a functioning memory on
  scene 2, exactly as on scene 1 (−26/−31 under the same variant). The
  sharpest claim is no longer single-scene. Consistent with (ah): whitening's
  distinctive act is spectrum-flattening, and that is where retrieval dies —
  the mean term was never its real contribution.
  **(4) The no-triple alternative LOSES.** Place signatures fix nothing the
  triple fixes: yolo stays degenerate on the classroom, TUM stays collapsed
  (a 0.5 m radius on a desk orbit averages everything into one signature).
  Answer to the design question: take the triple — it costs nothing when yaw
  is known (associativity), and it *self-mitigates* the mean problem.
  **Caveats:** yaw is taken from the pose stream at query (IMU-realistic, but
  say it); TUM raw+yaw recall works partly through appearance–heading coupling
  — on an orbit that is the only channel there is; TUM's space is 2.42×1.34 m
  so metres are small even when signal is high; classroom zscored dips
  slightly under the triple (71.6→68.6), the price of the extra factor when
  content already worked.

- **2026-08-03 (al)** — **The critical test (through-memory held-out on TUM)
  COLLAPSED — informatively. The whitening-below-chance claim stays
  classroom-only; the task itself is ill-posed on a desk-orbit trajectory.**
  `heldout_eval` gained `--dataset` (poses via `sequences.py`, sanity-checked;
  aborts on unusable streams). Run: TUM fr1/desk, blocked, 5 seeds, 3 encoders
  × 4 treatments, mocap poses (`heldout_eval_5seed.json`). Mocap space is
  **2.42 × 1.34 m** (oracle 0.08 m, constant 0.50 m). Result: **every real
  treatment scores below the constant predictor** (centred −65 to −92%,
  z-scored −69 to −95%, whitened −172 to −196%), and "raw" leads (17–18%) only
  by being a **constant predictor itself** (cells = 1–3 — the degeneracy flag
  working as designed). **Interpretation:** on a handheld orbit, appearance is
  governed by *viewing direction*, not position, and the memory binds
  appearance to **position only** — so appearance→position recall is ill-posed
  on this trajectory for every treatment. This is a *task-transfer* failure,
  not evidence about treatments. **Survives:** whitening is the worst
  treatment on both scenes; the (af) classroom claim stays single-scene and
  must be hedged in the draft. **Named next test:** bind orientation —
  `appearance ⊗ S(x,y) ⊗ H(ψ)` via the existing `ctx_head` circular encoder
  (small extension to `heldout_eval.evaluate`) — or pick a locomotion-style
  second sequence with real translation. Until one of those runs, the
  through-memory whitening result has **one** environment. Pages:
  `outputs/tum_fr1_desk/results_tum.html` (full TUM results, honest verdict),
  `outputs/cjepa_explainer.html` (C-JEPA visual explainer).

- **2026-08-03 (ak)** — **WP4 and WP5 COMPLETE; C-JEPA ingested. The second
  environment confirms what is universal and demotes what was
  scene-specific.**
  **WP4 final** — paired blocked held-out (5 seeds, identical splits):
  sigreg-raw collapses to **5.5±22.5%** signal vs dinov2-raw 31.7±15.4%;
  sigreg-centred **70.6±6.8%** = dinov2-centred 70.6±7.1%; z-scored 71.2 vs
  71.6 — identical within noise. Both axes (χ and held-out) now agree: the
  trained SIGReg head is *worse raw* off-distribution and *adds nothing* over
  post-hoc centring of its own input. `heldout_eval_sigreg.json`.
  **WP5 final — TUM fr1/desk** (613 frames, 3,218 crops, 22 classes, chance
  0.168; handheld Kinect, mocap GT; `outputs/tum_fr1_desk/`):
  - **Timkey REPRODUCES**: top dim 7.4%, d50 = 43/256 — spectral, not
    rogue-dims, on a second camera/scene.
  - **Encoder grouping REPRODUCES**: dinov2 0.710 > resnet50 0.687 ≫ yolov8n
    0.476 at gap 300 (dinov2 edges resnet50 here; classroom had them swapped —
    consistent with (ai)'s "grouping, not strict ordering").
  - **The raw-cosine whitening penalty is NOT universal — THIRD scene, third
    behaviour**: classroom −0.097 (0.720→0.623), school_run1 −0.004, TUM
    **+0.017 at k=32** (0.476→0.493), ≈0 at k=128, and only full ZCA costs
    (0.440). The universal statements that survive: centring/z-scoring are
    retrieval-safe everywhere; full ZCA always costs; and the DISSOCIATION
    holds (IsoScore-1.0 cells span 0.440–0.493 while raw-0.031 sits at 0.476 —
    isotropy still predicts nothing about retrieval). The *through-memory*
    negative-whitening result (af) remains classroom-only until heldout runs
    on a posed second scene (TUM has mocap poses; heldout/crosstalk still
    assume HF odometry — small extension, noted).
  - **Diversity→rank now has three inconsistent points**: (effK, eff-rank) =
    classroom (6.0, 6.7), school_run1 (7.1, 4.6), TUM (7.4, 8.9). No monotone
    relation in either direction — (ab)'s falsification stands, now stronger:
    class diversity simply does not predict effective rank.
  **C-JEPA ingested** (uploaded PDF; ICML 2026, Nam/Le Lidec/Maes/LeCun/
  Balestriero, arXiv:2602.11389v2): object-level latent masking on frozen
  VideoSAUR/DINOv2 slots; ~+20 pts counterfactual VQA; 1% tokens / 8× MPC;
  influence-neighborhood theorem; "causal" explicitly = predictive-under-
  masking, not identifiability. Full note + six connections + do-not-claim
  list: `wiki/sources/2026-08-03-cjepa-object-level-masking.md`. Headline
  connections: entity-granularity premise = training-side twin of our
  crops-vs-frames result; masked-object completion = partial-cue retrieval in
  predictor form (pivot bridge); their identity anchor is additive binding;
  slot-latent geometry through our χ/heldout machinery is a candidate
  post-freeze experiment (their code stated public, unverified).

- **2026-08-03 (aj)** — **WP4 partial (salvaged): the SIGReg latent does NOT
  transfer off-distribution — through the memory it is WORSE raw than its own
  input, and identical after centring.** Batch-2 agents died (WP5 on the
  session usage limit; WP4 finished its runs but failed to report — numbers
  recovered from its console transcript). Artifacts on disk:
  `sigreg_transfer.py`, `crop_embeddings_sigreg.pt` (9,056 detections through
  the trained head: Linear+BN from `jepa_run1_scratch/final.pt`),
  `crosstalk_scaling_sigreg_decomp.png`. χ at matched N=2,429, hd=8192,
  sigreg latent (128-d) vs its dinov2 input (384-d, entry ah):

  | variant | sigreg χ | dinov2 χ |
  |---|---|---|
  | raw | **94.04** | 79.83 |
  | centred | 52.32 | 51.50 |
  | zscored | 45.14 | 45.71 |
  | whitened | 7.62 | 9.87 |

  **Mechanism, plainly visible:** the sigreg-raw phasors carry signed mean
  off-diag cos **+0.4506** — on its *training* distribution the head drove the
  mean to −0.012 (ae), but applied to per-crop embeddings the BatchNorm's
  running statistics no longer match and a large shared mean reappears. The
  learned mean-removal is **distribution-bound**; post-hoc centring is computed
  on the data at hand and always applies. After centring, sigreg ≈ dinov2 to
  within noise — the trained head adds nothing over post-hoc treatment of its
  own input. Slopes unchanged (raw/centred/zscored ~1.0; whitened 0.859).
  **This strengthens the paper's recommendation** ("centre the data you have")
  and is a fair, honest framing: a *transfer* failure, not a refutation of
  SIGReg on-distribution. Caveats: 128-d vs 384-d not dimension-matched;
  whole-frame-CLS→per-crop is a genuine distribution shift; the paired
  blocked held-out run (dinov2 vs sigreg, 5 seeds) is re-running directly.
  WP5 (TUM fr1/desk): download + config + embed-crops completed before the
  limit — 613 frames → **3,218 crops** (yolov8n cos 0.861, eff-rank 8.9,
  k95 55, class margin +0.058 — same regime as the classroom); encoder passes
  + ladder + report re-running directly.

- **2026-08-03 (ai)** — **WP2 lands: the 5-seed blocked table. Centred/z-scored
  dominate everything; the strict encoder ordering does NOT survive error bars;
  whitening is never competitive.** (`heldout_eval` now emits stds; blocked =
  contiguous held-out segments + ±45-frame eviction; oracle 0.27 m, constant
  predictor 3.30 m; 833 stored / 819 queries; hd=4096; seeds 0–4;
  `heldout_eval_5seed.json`.) Blocked signal, mean±std:

  | encoder | raw | centred | zscored | whitened128 |
  |---|---|---|---|---|
  | dinov2 | 31.7±15.4% | 70.6±7.1% | **71.6±7.5%** | −23.3±12.6% |
  | resnet50 | 8.5±16.9% | **68.4±5.1%** | 64.3±7.8% | −21.4±16.4% |
  | yolov8n | −38.7±30.1% (cells=1) | 53.6±4.9% | **58.6±6.3%** | −17.2±21.6% |
  | untrained | −38.6±29.9% (cells=1) | 52.2±4.3% | **55.4±4.1%** | −29.2±26.2% |

  **(1) The treatment effect dwarfs the encoder effect** — centred/z-scored sit
  50–72% while raw and whitened sit ≤32% and mostly negative; separation ~5σ+.
  **(2) (af)'s crisp ordering claim is SOFTENED:** dinov2 71.6±7.5 vs resnet50
  64.3±7.8 overlap at 1σ, so the honest statement is the *grouping*
  {dinov2, resnet50} > {yolov8n, untrained} (~1.5–2σ), not a strict 4-way
  ordering. **(3) "z-scoring wins everywhere" also softens:** centred beats
  zscored on resnet50 (68.4 vs 64.3), zscored edges centred on dinov2 —
  statistically tied; the paper should say "centre or z-score" as one family.
  **(4) whitened128 has negative mean signal in every blocked cell**; yolov8n /
  untrained overlap zero at 1σ (split-to-split variability), but no whitened
  cell comes within 70 points of its centred counterpart — "never competitive"
  stands with error bars. **(5) raw yolov8n/untrained stay constant predictors**
  (cells=1) at −39%. Contiguous (deterministic, std 0) unchanged from (af).
  This is the publishable table for the workshop draft's Section 6 (slot C14).

- **2026-08-03 (ah)** — **WP1 + WP3 land (agent batch 1). The mean-removal share
  of whitening's gain is ENCODER-DEPENDENT (40–83%), and only whitening bends
  the growth exponent. Kalman drift-bounding survives 10 seeds.** WP2 (5-seed
  blocked table) failed to report in the batch — code change landed
  (`heldout_eval.py` now emits `signal_std`/`median_std`), run relaunched
  directly. NeurIPS workshop skeleton written:
  `wiki/drafts/2026-08-03-neurips-workshop-draft.md` (3 title candidates,
  claims ledger, [PENDING WPn] slots; target venue deliberately "TBD" — D1a).

  **WP1 — decomposition of the whitening χ-gain, matched N=2,429, hd=8192**
  (`crosstalk_scaling.py` generalised to raw/centred/zscored/whitened variants;
  figures `crosstalk_scaling_{yolov8n,resnet50,dinov2,untrained}_decomp.png`):

  | encoder (raw emb cos) | raw χ | centred | zscored | whitened | centred frac | zscored frac |
  |---|---|---|---|---|---|---|
  | yolov8n (0.94) | 180.85 | 42.73 | 37.71 | 9.11 | **0.80** | 0.83 |
  | untrained (0.999) | 201.80 | 69.91 | 55.92 | 11.03 | **0.69** | 0.76 |
  | resnet50 (0.45) | 87.96 | 45.32 | 24.54 | 9.83 | 0.55 | **0.81** |
  | dinov2 (0.39) | 79.83 | 51.50 | 45.71 | 9.87 | **0.40** | 0.49 |

  (fraction = (χ_raw − χ_x)/(χ_raw − χ_whitened) at endpoint N.)
  **(1) (ae)'s mechanism claim is REFINED, not overturned.** Mean-removal
  dominates for the coherent encoders (~70–80% of the gain on yolov8n /
  untrained) but carries only **40%** on dinov2 — there is no single honest
  "X% is mean-removal" number; quote per-encoder or as the 40–83% range, with
  raw coherence as the driver. z-scoring adds a large chunk only on resnet50
  (0.55→0.81; strongly unequal per-dim variances at 2048-d).
  **(2) NEW: centring/z-scoring do NOT change the growth law** — χ slope stays
  0.98–1.01 (coherent O(N)) for raw/centred/zscored on every encoder; **only
  full whitening bends the exponent, and only to ~0.89–0.92**. So the paper's
  clean statement: *z-scoring captures 49–83% of whitening's capacity gain at
  zero retrieval cost; the residual requires the PCA rotation — exactly the
  operation that (af) shows destroys held-out retrieval.* Prior multipliers
  reproduced exactly (19.9×/8.9×/8.1×/18.3×).

  **WP3 — VSA Kalman over 10 odometry-noise draws** (`vsa_kalman.py` gains
  `--noise-seeds`, expensive artifacts built once; split fixed blocked/seed 0 so
  spread is attributable to noise alone; `vsa_kalman_seeds.json`):
  headline (σ=0.02/bias 0.004, fixed a=0.02): **filtered 0.364±0.051 m** vs
  dead reckoning **0.868±0.503 m**, raw recall 0.620 m (seed-independent);
  jumps>0.5 m **242 → 6.5±2.4**; filtered max jump 3.496 m = the true
  trajectory's own discontinuity (3.485 m). Sweep: dead reckoning 0.87→**17.6 m**
  as noise ×20 while best-filtered goes 0.35→**0.62 m** — and its std
  *collapses* (±0.034 → ±0.005). **Saturation claim HOLDS with error bars.**
  Honest caveats carried: filter beats dead reckoning **9/10** at the lowest
  noise (one draw had unusually good odometry, 0.270 m); the sweep's
  best-filtered column is oracle gain selection (fixed a=0.02 is the
  deployment-honest number); the ~0.62 m ceiling *equals the raw-recall
  median* — at high noise the best gain climbs to 0.60 and the filter leans on
  the measurements, which is the mechanism of the bound, stated plainly.

- **2026-08-03 (ag)** — **VSA Kalman filter: the memory BOUNDS odometric drift.
  Both filter steps are native to the algebra.** New module `vsa_kalman.py`.
  Predict is one bind — `ctx_pos` is a group homomorphism, so
  `S_pred = S_prev ⊗ ctx_pos(dx, dy)` moves the belief exactly, O(D), no grid.
  Update is one bundle: `S_post = normalise(a·S_meas + (1−a)·S_pred)`. The grid
  decode is needed only to *display* a position, never to run the filter.
  (Upstream `classroom_associative_memory.py` already had a fixed-weight
  `--kalman-filter`; this generalises it and measures it.)

  **Setup honesty:** propagating with *true* pose deltas would make dead
  reckoning exact and the filter trivially perfect. Odometry is therefore
  corrupted with per-step Gaussian noise plus systematic bias, so the question
  is the real one — does recall pull a drifting track back? This is also
  architecturally honest: pose already comes from LIO-SAM and the VSA is a
  memory, so "relative motion from odometry, absolute correction from memory"
  is the design, not a shortcut.

  **Headline (classroom, dinov2 crops, z-scored, blocked memory of 814 frames,
  queried over all 2,429):**

  | track | median | mean | p90 | max jump | jumps >0.5 m |
  |---|---|---|---|---|---|
  | dead reckoning | 0.743 | 0.718 | 1.047 | 3.524 | 2 |
  | VSA recall (raw) | 0.620 | 1.111 | 3.159 | **8.664** | **242** |
  | **VSA Kalman (a=0.02)** | **0.418** | **0.458** | **0.714** | 3.496 | **4** |
  | true trajectory | — | — | — | 3.485 | 2 |

  It beats **both** inputs, which is what a filter is supposed to do, and the
  jumpiness is gone: 242 physically impossible jumps (worst 8.66 m) become 4,
  with a 3.50 m maximum against the true trajectory's own 3.485 m.

  **(1) THE RESULT WORTH KEEPING: filtered error saturates while dead reckoning
  diverges.** Sweeping odometry quality:

  | noise σ / bias | dead-reckoning median | final drift | best filtered | optimal gain |
  |---|---|---|---|---|
  | 0.02 / 0.004 | 0.743 | 1.09 m | **0.409** | 0.05 |
  | 0.05 / 0.010 | 1.859 | 2.72 m | **0.613** | 0.05 |
  | 0.10 / 0.025 | 3.803 | 5.62 m | **0.627** | 0.35 |
  | 0.20 / 0.050 | 7.606 | 11.23 m | **0.628** | 0.60 |
  | 0.40 / 0.100 | 15.212 | 22.46 m | **0.627** | 0.60 |

  Odometry degrades **20×** (0.74 → 15.2 m) and the fused estimate does not move
  (0.41 → 0.63 m). The memory **bounds** drift at roughly its own recall
  accuracy. That is the honest claim for this component — not "the VSA localises
  better than odometry" (it does not; raw recall is 0.620 against dead
  reckoning's 0.743 at low noise), but "**a fixed-size associative memory caps
  unbounded odometric drift at O(D) per frame**".
  **(2) Optimal gain rises monotonically with odometry noise** (0.05 → 0.05 →
  0.35 → 0.60 → 0.60), exactly as Kalman theory requires — a good internal
  consistency check that the recursion behaves like a filter.
  **(3) NEGATIVE: the adaptive gain does not earn its complexity.** Gating on
  peak-vs-field z gives 0.523 median / 0.747 mean / 2.025 p90 / 7 jumps, worse
  on every metric than fixed a=0.02 (0.418 / 0.458 / 0.714 / 4). The z signal is
  too narrow-range here (median 7.9, p90 11.4, max 11.5) to discriminate. Use a
  constant gain until there is a better confidence signal — the calibrated
  per-(trace, probe-kind) z from `TraceSet.calibrate` is the obvious candidate
  and was not used here.
  **(4) Single-realisation caveat, and a bug it exposed.** Odometry noise
  initially shared the RNG that `make_split` had already advanced, so the
  headline run and the sweep drew different noise for identical parameters —
  dead-reckoning median 0.350 vs 0.777. Fixed with a dedicated stream
  (`seed + 777`), but the 2× spread shows these are **single-draw numbers and
  should be averaged over seeds** before publication.
  **(5) Speed note:** decoding the posterior every step was 99% of the cost and
  is unnecessary — the recursion is pure O(D) vector algebra. Collecting the
  track and decoding once as a batched matmul took the sweep from >10 min
  (timed out) to seconds.

- **2026-08-02 (af)** — **First real held-out protocol. A random split is
  worthless here; whitening goes NEGATIVE on honest splits; z-scoring wins; and
  encoder quality only becomes visible once leakage is removed.** New module
  `heldout_eval.py`: build the memory over one frame set, query with a disjoint
  set, decode position, bracket by an **oracle** (nearest *stored* pose) and a
  **constant predictor** (ignore the query, answer the centroid of stored
  poses). Classroom, 2,429 frames, 30% held out, hd=4096, grid 48, 3 seeds.

  | split | encoder | raw | z-scored | whitened128 |
  |---|---|---|---|---|
  | **random** (leaky) | dinov2 | 40.0% | 91.9% | **95.4%** |
  | | untrained | −25.4% | 82.7% | **95.1%** |
  | **blocked** (honest) | dinov2 | 34.5% | **70.2%** | −26.0% |
  | | resnet50 | 8.2% | **66.0%** | −24.9% |
  | | yolov8n | −29.2% | **59.6%** | −24.2% |
  | | untrained | −28.4% | **56.9%** | −37.1% |
  | **contiguous** | dinov2 | 56.6% | **61.2%** | 7.1% |
  | | untrained | 56.8% | 55.8% | −1.0% |

  **(1) A RANDOM SPLIT IS INVALID at 15 fps.** Held-out frames keep their
  neighbours in the memory, so the **oracle is 0.01 m** — the nearest stored
  pose is a centimetre away — and everything scores ~95%. An **untrained**
  network ties DINOv2 (95.1% vs 95.4%). Use `blocked` (contiguous held-out
  segments plus a ±45-frame eviction radius, oracle 0.30 m) or `contiguous`.
  Note a first attempt at blocking by scattering individual frames produced an
  **empty memory** — at 30% held out the mean spacing between queries is ~3
  frames, so any useful eviction radius deletes everything. Blocks, not points.
  **(2) A CONSTANT-PREDICTOR BASELINE IS MANDATORY.** Scored against "a random
  stored frame", raw features gave a suspiciously uniform **73.1% for all four
  encoders, identical to 3 dp**. The `cells` diagnostic explains it: raw
  yolov8n and raw untrained decode **every query to ONE grid cell**
  (`cells=1, spread=0.00`). Their vectors are so aligned (cos 0.88–0.997) that
  unbinding leaves the same residual regardless of query. It is a constant
  predictor, and a centroid beats a random frame in a 6.9 m room. Against the
  constant baseline those rows are correctly **negative (−25 to −29%)**.
  **(3) WHITENING TO 128 IS WORSE THAN USELESS ON HELD-OUT DATA** — −24% to
  −37% blocked, 7% to −1% contiguous. It decodes to *many* distinct cells
  (500+), so it is not degenerate; it is confidently scattering. This is the
  third and strongest confirmation of (ae): the pipeline's default actively
  destroys generalisation, and closed-set scoring hid it completely.
  **(4) Z-SCORING WINS EVERYWHERE HONEST** — 57–70% blocked, 54–61%
  contiguous, best or joint-best in every encoder × split cell. Consistent with
  (ae)'s frame-level result and (y)'s crop-level ladder.
  **(5) ENCODER QUALITY ONLY APPEARS ONCE LEAKAGE IS GONE.** Random split: all
  four within 0.5 points. Blocked: **dinov2 70.2 > resnet50 66.0 > yolov8n 59.6
  > untrained 56.9** — a 13-point spread in the sensible order. So the encoder
  four-way in (aa) should be re-scored on blocked splits before publication.
  **Clarification recorded** (asked directly): the VSA **never estimates pose**.
  Robot pose is LIO-SAM (LiDAR + IMU); object positions are D455 depth
  deprojected through that pose (`_localize_detections`, `classroom_pipeline.py:417`).
  The memory is *handed* a pose and stores appearance bound to it, so a query
  returns the pose stored with the most similar stored appearance — associative
  recall of a recorded pose, not visual localisation. That is exactly why an
  untrained encoder could match DINOv2 under closed-set scoring: the task needs
  the descriptor to be *consistent*, not *good*.

- **2026-08-02 (ae)** — **MECHANISM FOUND: crosstalk is driven by the COMMON
  DIRECTION, not by the spectrum's shape. "Centre, don't whiten" beats the
  pipeline's current default on both axes.** Learned-vs-imposed isotropy run on
  the JEPA line (`jepa` branch, `run1_to_transitions.py` → 235 classroom frames
  at 3 Hz, `jepa_run1_scratch.yaml` trained 150 epochs, SIGReg λ=0.1 on a 128-d
  head over frozen DINOv2 ViT-S/14). Scored with a **chance and oracle
  baseline** — essential, since the room is 6.87 × 6.71 m and a raw "3.6 m
  error" is otherwise uninterpretable: chance 5.24 m, oracle 2.02 m at a 30-frame
  (10 s) exclusion. "Signal" = fraction of the chance→oracle gap recovered.
  Held-out = the 165 frames the head never trained on.

  | representation | IsoScore | eff-rank | signal (held-out) | χ |
  |---|---|---|---|---|
  | raw DINOv2 384-d | 0.115 | 8.4 | 46.9% | 10.55 |
  | **centred only** | **0.115** | **8.4** | **50.7%** | **5.24** |
  | **z-scored per-dim** | 0.127 | 10.5 | **51.8%** | **4.67** |
  | SIGReg head 128-d | 0.047 | 7.0 | 38.8% | 2.88 |
  | whitened k=64 | 1.000 | 64 | 11.5% | 0.98 |
  | whitened k=32 | 1.000 | 32 | −1.4% | 1.57 |
  | whitened k=128 | 1.000 | 128 | **−3.9%** | 0.58 |
  | SIGReg + whitening | 1.000 | 128 | 1.1% | 0.46 |

  **(1) THE MECHANISM.** Centring leaves IsoScore and effective rank *bit-identical*
  (0.115, 8.4 — they are computed on the centred covariance, so it cannot change
  them) yet **halves χ, 10.55 → 5.24**. Two representations, identical spectra,
  2× different crosstalk. So the coherent interference that dominates bundling
  comes from the **shared mean component adding in phase across stored items**,
  not from the covariance being squashed. Flattening the spectrum (whitening)
  attacks the wrong term.
  **(2) z-scoring PARETO-DOMINATES raw.** Better place signal (51.8% vs 46.9%)
  *and* 2.3× less crosstalk (4.67 vs 10.55), for the cheapest possible
  operation. There is no trade-off to make here — it is free.
  **(3) The pipeline's current default is actively harmful.** PCA-whitening to
  128 components leaves **−3.9% signal on held-out frames — at or below chance**.
  k=32 is also at chance (−1.4%). Only k=64 retains anything (11.5%), a quarter
  of raw's. The extra crosstalk reduction whitening buys (0.58 vs 4.67) is
  worthless because there is no retrievable signal left. **This supersedes
  entry (y)'s "provisional, classroom-specific" hedge on cutting 128 → 32: the
  answer is neither, use centring or z-scoring.**
  **(4) SIGReg reduces crosstalk WITHOUT increasing isotropy** — IsoScore 0.047,
  *lower* (more anisotropic) than raw's 0.115, yet χ 2.88 vs 10.55. Same
  mechanism: its training log shows signed mean cosine driven to −0.012 while
  |cos| stays 0.244. It kills the common direction, not the anisotropy. Its
  held-out and all-frames signal are **identical (38.8%)** — no train/test gap
  on 70 training frames. But plain z-scoring beats it (51.8% vs 38.8%) at a
  fraction of the cost, so the learned-isotropy angle is **deflated, not
  vindicated**.
  **(5) First genuinely held-out numbers in this project.** The JEPA split
  (first 30% train / last 70% test) is the only held-out protocol here; the
  whitening rows going *negative* on held-out data while looking merely poor on
  all-frames data is exactly the kind of thing closed-set scoring hides.
  **Caveats:** 235 frames, one room, one walk; oracle only 2.02 m because a
  single-loop walk barely revisits; χ scales ~N so these values (N=235) are not
  comparable to entry (ac)'s (N=2,429); frame-level, whereas (y)'s ladder was
  crop-level. **Next:** re-run the crop-level ladder with centring and z-scoring
  scored against crosstalk, and re-check whether the 22×/19.9× multipliers in
  (ac) are largely a mean-removal effect.

- **2026-08-01 (ad)** — **RETRACTION: every pose-dependent `school_run1` number
  is void. Its LIO-SAM odometry is catastrophically diverged — quantified.**
  Entry (h) recorded this qualitatively; here is the measurement, against the
  classroom as control:

  | | classroom | school_run1 |
  |---|---|---|
  | poses / duration | 496 / 82.5 s | 1,559 / 454.5 s |
  | extent | 6.9 × 6.7 m | **1,896.9 × 2,398.8 m** |
  | path length | 24.4 m | **23,124 m** |
  | net displacement | 0.4 m | 1,923.7 m |
  | median step | 3.3 cm | 7.8 cm |
  | max single step | 0.2 m | **242.8 m** |
  | jumps > 1 / 5 / 50 m | 0 / 0 / 0 | **250 / 209 / 140** |
  | max implied speed | 1.2 m/s | **1,594.8 m/s** |

  A Spot walks at ~1.6 m/s. The 209 jumps match entry (h) exactly.
  **RETRACTED — do not use:** (a) the `school_run1` recall numbers from the
  video export (median 1,167–1,385 m across three encoders — they measure the
  odometry, not the memory); (b) **the `school_run1` crosstalk run reported
  earlier this session** (raw χ 34.44 → whitened 5.45, "6.3×", slopes +0.826 /
  +0.761). Position phasors are `ctx_pos` at ℓ = 0.75 m over a **2 km** spread,
  so every position is mutually orthogonal — a regime no real room produces, and
  the likely cause of those flatter slopes. The classroom four-encoder table in
  (ac) is unaffected: it uses classroom poses, which are clean.
  **CORRECTION to (ab)'s recommendation.** I wrote that `school_run1` is "the
  natural N-axis for Gate 0". That is only true for **pose-free** measurements —
  crop isotropy, effective rank, class retrieval, the encoder comparison — all of
  which stand, since none of them touch odometry. Anything binding position
  (crosstalk, capacity-with-place, and Gate 0 itself if the task is spatial)
  **cannot** use this sequence. The scale-up N remains blocked on either fixing
  the SLAM or a different dataset; 3RScan remains the recommended route.
  **Guard added** so this cannot recur silently: `tools/export_video_page.py`
  now runs a pose sanity check (jumps > 5 m, implied speed) and writes a `warn`
  block into the payload; the player template renders a red banner, retitles the
  section "Recall on diverged odometry — what failure looks like", and states
  that the flythrough and encoder-comparison panels use no poses and are
  unaffected. The failure is kept and labelled rather than hidden, because it is
  a genuine demonstration that an associative map is exactly as good as the poses
  bound into it.

- **2026-08-01 (ac)** — **CROSSTALK RE-MEASURED ON FOUR BACKBONES. The
  multiplier is encoder-dependent; the FLOOR is not. Retire "22×" as a headline.**
  `crosstalk_scaling.py` gained `--content` (any crop-embedding file → per-frame
  mean descriptor) and `--tag`, so the same metric runs on any backbone; the
  figure filename is now tagged (a fixed name was silently overwriting).
  Matched N=2429, identical frames, identical construction, hd=8192:

  | content | raw emb cos | raw χ | **whitened χ** | χ gain | raw ratio | **wht ratio** | ratio gain |
  |---|---|---|---|---|---|---|---|
  | yolov8n crops | 0.930 | 180.85 | **9.11** | **19.9×** | 0.95 | **0.27** | 3.5× |
  | resnet50 crops | 0.447 | 87.96 | **9.83** | **8.9×** | 0.60 | **0.24** | 2.5× |
  | dinov2 crops | 0.383 | 79.83 | **9.87** | **8.1×** | 0.52 | **0.25** | 2.1× |
  | untrained crops | 0.999 | 201.80 | **11.03** | **18.3×** | 1.00 | **0.26** | 3.9× |

  **(1) The headline multiplier more than halves on a competent encoder** —
  19.9× (YOLO) → 8.1× (DINOv2) on the growth-law metric, 3.5× → 2.1× on the
  recall-relative one. The public 365×/22× figures were measured on YOLOv8n's
  penultimate detection feature, the most anisotropic representation available.
  **Do not quote a bare multiplier again; quote it with its encoder.**
  **(2) THE RESULT WORTH BUILDING ON: whitening lands every backbone on the
  SAME floor.** Whitened χ spans just **9.11–11.03** (1.2×) and whitened ratio
  **0.24–0.27** (1.1×), while raw χ spans **79.83–201.80** (2.5×). Even the
  *randomly initialised* network — raw cosine 0.9992, the worst raw χ at 201.80 —
  reaches χ = 11.03 after whitening. The destination is encoder-independent; only
  the distance travelled differs. This is "whitening moves them onto the
  isotropic curve", now measured across four backbones including an untrained
  control, and it is a far stronger claim than any single multiplier.
  **(3) It answers the reviewer question the encoder table raised.** "Why not
  just use a better encoder?" — because a better encoder buys ~2.3× of the way
  (180.85 → 79.83) and then *stops*; whitening is what reaches the floor, and the
  floor is where capacity is actually set. Both are needed and they are not
  substitutes.
  **(4) The constant-factor-not-scaling-law claim SURVIVES, now robustly.**
  log-log slopes across all four: raw **+0.987 to +1.004**, whitened **+0.886 to
  +0.915**. Coherent O(N) raw, still ~O(N^0.9) whitened, in every condition.
  Whitening shifts the curve down; it does not bend it — and that is now shown on
  four encoders rather than one.
  Figures: `crosstalk_scaling_{yolov8n,resnet50,dinov2}_crops.png`.
  **Caveat:** these use per-frame *mean-of-crops* descriptors (2,429 frames), not
  the frame-level `embeddings.pt` (1,239) the original 85×/3.8× came from, so the
  YOLO row is the like-for-like anchor, not the old number. χ scales ~N, so
  endpoint values are not comparable across different N.

- **2026-07-31 (ab)** — **Second scene FALSIFIES the low-rank-from-few-classes
  hypothesis, and RETRACTS my own whitening claim from entry (y).** New module
  `room_diversity.py`. `school_run1` crops extracted (41,697 detections over
  11,211 frames — **9.3× the classroom's canonical 4,498**). Diversity is
  *measured*, not assumed: class entropy H and effective class count exp(H).

  | scene | n | cos | eff-rank | IsoScore | effK | top1 | +z | +whiten128 |
  |---|---|---|---|---|---|---|---|---|
  | classroom | 9,056 | +0.883 | **6.7** | **0.0222** | 6.0 | 0.720 | 0.722 | **0.623** |
  | school_run1 (matched N) | 9,056 | +0.868 | **4.6** | **0.0139** | 7.0 | 0.854 | 0.863 | 0.852 |
  | school_run1 (full) | 41,697 | +0.868 | 4.6 | 0.0141 | 7.1 | 0.866 | 0.875 | 0.863 |
  | school_run1 (full, gap 1357) | 41,697 | +0.868 | 4.6 | 0.0141 | 7.1 | **0.842** | 0.850 | 0.838 |

  **(1) HYPOTHESIS FALSIFIED.** The prediction (handoff §7, tracker (y)) was
  that a semantically *richer* scene spreads features over *more* directions.
  `school_run1` is measurably richer — effK **7.1 vs 6.0**, 45 classes vs 40 —
  and its effective rank is **lower** (4.6 vs 6.7) with IsoScore **lower**
  (0.0141 vs 0.0222). Direction is opposite to the prediction. Class diversity
  does not drive effective rank, at least not positively, so "low-rank *because*
  few classes" cannot be asserted. The anisotropy is still spectral (Timkey d50
  = 52, consistent with everything else today) — what fails is the *causal
  story*, not the characterisation.
  **(2) RETRACTION — the entry (y) claim does not generalise.** I reported that
  PCA whitening hurts appearance retrieval (classroom: 0.720 → 0.623, −0.097).
  On `school_run1` the same operation costs **−0.004** (0.842 → 0.838). The
  whitening penalty is **scene-dependent, not a property of appearance keys**.
  Entry (y)'s recommendation to cut `LanguageMap` from 128 to 32 components is
  therefore **provisional and classroom-specific** — it must be re-tested per
  scene before being adopted as a default.
  **(3) Isotropy ≠ retrieval, fourth independent demonstration.**
  `school_run1` is *more* anisotropic on both metrics and retrieves *far*
  better (0.842 vs 0.720). Together with crops-vs-frames (z), the ladder (y) and
  the encoder four-way (aa), four separate designs now show the same
  dissociation. This is the most robust result of the day and should anchor the
  paper's framing: the covariance spectrum predicts *bundling capacity*, not
  *key quality*, and conflating the two is the error to call out.
  **(4) Confound checked and rejected.** Gap 300 is 12% of the classroom's
  2,478 frames but only 2.7% of school_run1's 11,211, making the raw comparison
  unfair. Re-run at a proportionally matched gap of **1,357 frames**:
  school_run1 still scores **0.842** vs classroom 0.720. Also reported at
  matched N (9,056) since eff-rank and IsoScore both move with sample size.
  **Consequence for Gate 0:** `school_run1` at 41,697 detections is now the
  largest real event stream available and is the natural N-axis for the
  crossing-point figure — roughly 3× the ~13k PQ-kNN crossover, i.e. the first
  regime where the byte-budget claim can actually be tested rather than assumed.
  **Minor artifact bug:** `room_diversity.py` writes a fixed
  `outputs/classroom/room_diversity.json`, so the second (gap-1357) run
  overwrote the two-scene table; the numbers above are the durable record until
  the output path is parameterised by scene+gap.

- **2026-07-31 (aa)** — **Encoder four-way on identical boxes: Liang CONFIRMED
  exactly, Godey INVERTED, Timkey negative everywhere.** New modules
  `encoder_comparison.py` (re-crops from `detections_crops.csv` so the encoder
  is the only variable; DINOv2 preprocessed exactly as upstream `init-am`
  does — square bilinear 224 + ImageNet norm, **not** the HF processor's
  shortest-edge-256 + centre-crop, which slices content off tall boxes) and
  `encoder_report.py`. All 9,056 crops, 40 classes, chance 0.312:

  | encoder | dim | cos | eff-rank | /dim | IsoScore | top dim % | d50% | top1@300 |
  |---|---|---|---|---|---|---|---|---|
  | yolov8n (CNN, detection) | 256 | +0.883 | 6.7 | 0.026 | 0.0222 | 5.3% | 49 | 0.720 |
  | resnet50 (CNN, ImageNet) | 2048 | +0.332 | 23.2 | 0.011 | 0.0108 | 3.4% | 57 | **0.787** |
  | dinov2-S/14 (ViT, SSL) | 384 | +0.299 | 18.4 | 0.048 | **0.0454** | 2.5% | 46 | 0.769 |
  | resnet50 **untrained** | 2048 | **+0.999** | **1.0** | 0.000 | **0.0000** | 1.3% | 185 | 0.578 |

  **(1) Liang et al. NeurIPS 2022 replicated to three decimals.** Predicted
  untrained ResNet ≈ 0.99 mean pairwise cosine; measured **+0.999**, with
  effective rank **1.0 of 2048** — the entire representation collapses to one
  direction. Training *reduces* anisotropy massively (0.999 → 0.332). Extreme
  anisotropy is the architectural default, not a learned pathology.
  **(2) Godey et al. EACL 2024 does not hold here — it inverts.** Godey:
  anisotropy tracks self-attention, CNNs largely fine. Measured: the **ViT is
  the best-behaved trained encoder** (lowest cos 0.299, highest IsoScore
  0.0454), and the **most anisotropic trained encoder is a CNN** (YOLOv8n,
  0.883). Anisotropy here tracks the **training objective**, not the
  architecture: YOLO's "embedding" is a detection head's penultimate feature
  never trained to be a descriptor, while ResNet-50 (classification) and DINOv2
  (instance discrimination) both optimise global discriminative representations.
  Publishable either way, per the handoff.
  **(3) Timkey negative across all four**, including the untrained net: top
  dimension carries 1.3–5.3% of the mean cosine and 46–185 dims are needed for
  half, against 76–99% from a *single* dimension in GPT-2/BERT/XLNet. The
  untrained row is the cleanest possible separation of the two phenomena —
  eff-rank **1.0** (maximally low-rank) with top-dim share **1.3%** (no rogue
  coordinate) — i.e. a dense shared *direction*, not a coordinate-aligned spike.
  **(4) THREAT TO THE FRAMING, must be met head-on:** a better backbone buys
  both better isotropy *and* better retrieval, so "why not just use a good
  encoder instead of whitening?" is now an obvious reviewer question. The 365×
  (and 22×) crosstalk figures were measured on **YOLO features — the worst
  encoder tested**, which overstates the benefit of whitening relative to a
  competent descriptor. Re-measure crosstalk on DINOv2/ResNet features before
  quoting any multiplier. The surviving claim: even the best encoder here sits
  at IsoScore 0.045 against 1.0, so the correction still matters — but its
  magnitude is encoder-dependent and must be reported as such.
  **(5) METRIC DEFINITION IS NOW LOAD-BEARING** (handoff §8's discipline item,
  concretely): mean cosine and IsoScore **rank the encoders differently** —
  cosine says YOLO worst, IsoScore says ResNet-50 worse than YOLO (0.0108 vs
  0.0222) because it uses proportionally fewer of its 2048 dims. Retrieval
  disagrees with both, ranking ResNet-50 *best* despite its second-worst
  IsoScore. **Third independent demonstration that isotropy is not sufficient**
  (after crops-vs-frames and the ladder). The paper must name which metric its
  headline multiplier refers to.
  **(6) A real descriptor helps, modestly:** +0.067 (ResNet) / +0.049 (DINOv2)
  over YOLO at gap 300. DINOv2 gets within 0.018 of ResNet-50 using **5.3× fewer
  dimensions**. z-scoring only helps the most anisotropic encoder (YOLO
  0.720→0.722) and hurts the better-behaved ones (ResNet 0.787→0.772).
  **Upstream note:** `init-am`'s `detect_and_embed_classroom.py` embeds DINOv2 at
  **frame** level (`run_embed_dino` → CLS of the whole resized frame), so
  `run_dino_comparison.sh` compares YOLO vs DINO on the weaker representation
  for both — the per-detection gap is upstream's, not a porting artifact. Their
  `--pca-whiten` rationale is stated as spreading a tight 0.70–1.00 cosine band,
  a *presentation* objective that entry (y) shows diverges from retrieval.
  Their `--trim-stationary` likely explains part of the 9,056 vs 4,498 gap.
  **BUG FIXED:** `embed-crops` inherited `cmd_embed`'s hardcoded `rgb_d455`, so
  `--out-dir outputs/school_run1` was silently re-cropping the *classroom*;
  now takes `--repo`/`--rgb-config` (no data was corrupted — killed before its
  save step).

- **2026-07-31 (z)** — **Crops vs frames head-to-head: the crop stage is
  justified, and it yields a counterexample worth more than the justification.**
  Identical protocol, identical detections, identical class labels, identical
  gap control — the only change is which vector represents a detection (its own
  crop, or the embedding of the frame it came from). Class top-1, chance 0.313:

  | gap (frames) | CROP | FRAME | above chance |
  |---|---|---|---|
  | 30 (~2 s) | **0.879** | 0.520 | +0.566 vs +0.207 |
  | 300 (~20 s) | **0.720** | 0.435 | +0.406 vs +0.121 |
  | 900 (~60 s) | **0.691** | 0.455 | +0.378 vs +0.142 |

  Crops carry **3.4× the signal above chance** at 20 s. **Structural reason,
  quantified:** 995 of 1,210 frames hold more than one detection and 877 hold
  more than one *distinct class*, so **88.6% of detections (3,985) sit in a
  mixed-class frame where the frame vector is identical across different
  objects**. For those, per-object cueing is impossible in principle, not merely
  inaccurate. **THE COUNTEREXAMPLE (paper-grade):** frame features are *more*
  isotropic than crops on both metrics — IsoScore 0.0315 vs 0.0222, eff-rank 9.0
  vs 6.7 — and retrieve *far worse*. So **isotropy is not sufficient for a good
  key; the vector must describe the right referent.** This is a clean empirical
  rebuttal to any naive "more isotropy is better" reading of the capacity story,
  it supports the predicted optimum-not-monotone shape, and it should be a
  figure. z-scoring also splits the two: neutral-to-positive on crops (0.720 →
  0.724 at gap 300) but harmful on frames (0.435 → 0.379). **Side effect worth
  noting:** stride-1 crop extraction yields **9,056 detections over 2,478
  frames**, against the canonical **4,498 over 1,210** — roughly double the
  event count, which shifts N in every crosstalk and crossover figure and should
  be reconciled before Gate 0 (only 4,498 of the 9,056 have a matching frame
  embedding, which is why the head-to-head runs on that subset).
  **Prerequisite noted:** `school_run1` has `embeddings.pt` and
  `detections.csv` but **no crops**, so the impoverished-vs-diverse room test
  needs `embed-crops` run over its ~11k frames first.

- **2026-07-31 (y)** — **TIMKEY DIAGNOSTIC: our anisotropy is NOT the NLP
  rogue-dimension pathology. It is genuinely spectral.** New module
  `vsa_cognitive_mapping/isotropy_ladder.py` (Q2's ladder + the Timkey test),
  writes `outputs/classroom/isotropy_ladder.json`. Decomposition is exact, not
  sampled: with unit-normalised vectors, `mean_{i≠j} cos = Σ_d [(Σ_i u_i[d])² −
  Σ_i u_i[d]²]/(n(n−1))`, so per-dimension contributions sum to the measured
  mean. **Result (9,056 crops, 256-d): top dimension contributes 5.3% of the
  mean cosine; top 5 = 18.4%; 49 of 256 dims needed for 50%; 151 for 90%.**
  Frames behave the same (top dim 5.0%, top 5 14.9%, 53 dims for 50%). Compare
  Timkey & van Schijndel EMNLP 2021 on transformers: **top dim alone gives 0.76
  (GPT-2 L12), 0.88 (BERT L11), 0.99 (XLNet)**. We are two orders of magnitude
  away from that concentration. **This confirms the hypothesis in the handoff
  brief §7** — low-rank-from-few-classes, not narrow-cone-from-rogue-dimensions
  — and it is *good* for novelty: the cheap known fix does not apply, and no
  paper relates codebook effective rank to bundle capacity.
  **Two separable phenomena, established:** (1) a large **common mean offset** —
  centring alone collapses mean cosine 0.883 → 0.007 at zero retrieval cost
  (0.720 → 0.716); (2) a **genuinely low-rank covariance** that centring does
  not touch at all (eff-rank stays 6.7, k95 stays 44 of 256). Whitening attacks
  (2); Timkey's fix attacks neither, because there is no rogue dimension to
  remove. NOTE a presentation confound: the `drop top-k` rungs centre first, so
  their +0.007 is centring's effect, not the drop's — the clean Timkey number is
  the contribution decomposition above, computed on raw L2-normalised vectors.
  **LADDER RESULT — isotropy and retrieval trade off MONOTONICALLY, and the
  cost scales with how many directions are equalised** (class top-1, gap 300
  frames ≈ 20 s, chance 0.312): raw IsoScore 0.0222 → **0.720**; z-scored
  per-dim IsoScore 0.0388 → **0.722**; PCA-whiten 32 IsoScore 1.0 → 0.693;
  whiten 64 → 0.654; whiten 128 → **0.623**; full ZCA → **0.592**. Control rung
  (random orthogonal) reproduces raw exactly on every rotation-invariant metric,
  validating the implementation. **Two actionable consequences.** (a)
  **Per-dimension z-scoring is the best rung on this axis** — it nearly doubles
  IsoScore (0.0222→0.0388), lifts eff-rank 6.7→10.9, and *slightly improves*
  retrieval, at the lowest possible cost. (b) **The language pipeline whitens to
  ≤128 components and should not** — PCA-32 reaches the same IsoScore 1.0 for a
  2.7-point retrieval cost where 128 costs 9.7 points, because the discarded
  directions were nuisance that whitening was amplifying. Retest `LanguageMap`
  at 32 before Gate 0. **UNRESOLVED and load-bearing:** all retrieval here is
  raw cosine, *not* retrieval through the memory, where isotropy buys the ~22×
  crosstalk reduction. The net trade is still unmeasured, and the honest paper
  claim must pair the 22× with its discriminability cost rather than quoting it
  alone. Also still class-level only — no instance ground truth exists in this
  dataset. **Free next test** (handoff §7): the low-rank hypothesis predicts
  worse anisotropy in semantically impoverished rooms — `school_run1` vs
  classroom, machinery now exists.

- **2026-07-31 (x)** — **BLOCKING BUILD GAP CLOSED: per-detection crop features
  exist.** New `embed-crops` subcommand in `classroom_pipeline.py` (additive —
  `cmd_embed` and every existing artifact untouched). Crops each YOLO box with
  an 8% context pad, embeds the crop, and writes `crop_embeddings.pt` +
  `detections_crops.csv` (row order = embedding row order, so `det_id` is a
  positional join key) + `crop_isotropy.json`. **9,056 crop vectors, 256-d,
  40 classes, over all 2,478 frames** (37 flagged tiny at min_side<16 px).
  Isotropy, like-for-like on identical frames: crops **cos mean +0.882 /
  eff-rank 6.7 / k95 44** against frames **+0.941 / 8.1 / 43** — crops are
  less anisotropic and spread over more directions, but still live in a narrow
  cone. Within-class cos +0.949 vs between-class +0.874, **margin only +0.076**.
  **Retrieval probe** (`crop_retrieval_probe.json`, leave-one-out top-1 cosine,
  class-level, 40 classes, chance 0.312) with a **temporal gap control** —
  without it the nearest neighbour is the same object one frame later and the
  number is meaningless: gap 0 → **0.960**; gap 30 frames (~2 s) → **0.886**;
  gap 300 frames (~20 s) → **0.720**. So the partial-cue premise is **viable at
  category level** (0.72 vs 0.31 chance across a 20 s gap), and the crop stage
  is worth its compute. **Counter-intuitive finding: PCA whitening HURTS
  appearance retrieval at long gaps** — 0.623 whitened vs 0.720 raw at gap 300,
  while being neutral-to-positive at gap 0 (0.965 vs 0.960) and gap 30 (0.890
  vs 0.886). Reading: the top PCA directions carry the class-discriminative
  signal and the low-variance directions carry viewpoint/lighting nuisance;
  whitening promotes the nuisance to equal footing, which only bites once the
  temporal gap makes nuisance variation large. **This contradicts the pipeline's
  whiten-everything default** (whitening is what fixed heading 28%→98%) — the
  two key types want opposite treatment: spatial/heading keys need isotropy for
  low superposition crosstalk, appearance keys need preserved structure for
  retrieval. **NOT yet resolved:** this probe measures raw cosine retrieval, not
  retrieval *through* the memory, where whitening's ~22× crosstalk reduction may
  repay the ~10-point discriminability loss. That trade has not been measured
  and must be before the factor set is fixed. **Also still open:** all of this
  is *class*-level; there is no instance ground truth in this dataset, so "can
  it find *this* chair" remains unanswered, and the +0.076 class margin on
  YOLOv8n's embedding head is thin enough that a real descriptor (DINOv2 / CLIP
  image encoder / re-ID model) should be trialled before Gate 0 — it is a cheap
  swap now and an expensive discovery later. Sequence is also single-session,
  one room, one lighting condition, so 0.72 is an optimistic ceiling for
  cross-visit re-finding.

- **2026-07-31 (w)** — **Demo page restructured (demos first) + full fact-check
  applied.** `astm/docs/demo_page.html` on branch `astm` (commit `ce56cef`).
  Order is now replay → English → time → merge → **under the hood** → limits;
  the mechanism explainer moved *after* the demos. A claim-by-claim audit
  against code and exported data found and fixed **13 substantive errors**. The
  two that mattered most: (1) the page asserted a *single* memory answered every
  panel — the English panel in fact queries a **second** `LanguageMap` memory
  built over `events_object.csv` with text-derived class atoms, different
  extent, different weighting, and "where" meaning object location rather than
  robot vantage; (2) the "something to sit on" miss was described as landing on
  the **couch** and therefore *semantic* — it actually lands **0.14 m from a
  different chair**, i.e. an **instance-selection** failure of argmax over a
  23-instance class. Also corrected: 4 replay steps exceed 1 m (not 1);
  footprint at the page's own config is **414 MB grids / 420 MB total** (the
  118 MB figure was a coarser D=4096 run *and* pairs with 256 KB traces, not
  512 KB); vocabulary is **203** non-class nouns / 241 rows (not "~180" — the
  same error sits in `language_query.py:53-54`); time windows are **28 s of an
  83 s walk** (not five minutes) and cost *about* the same as a point query
  (9.8–10.2 vs 9.0–10.2 ms, not "exactly"); the merge split is **interleaved
  from one walk**, not two robots on different ground, and its 0.000 m offset is
  **grid-forced** (0.1 m sweep, true offset on a node); kernel half-height width
  is **0.30ℓ ≈ 0.23 m** with the marker at ℓ sitting on the *second null*;
  **7 of 9** `person` abstentions are honest, 2 are genuine misses (35 and 16
  detections within one time length-scale at t=275/963); crosstalk slopes are
  **+0.97 raw vs +0.86 whitened** (the caption claimed identical); GC-VSA
  (arXiv:2503.08608) is **"inspired by"** CANs and models grid-cell *position*,
  not heading; the widgets are an **independent JS reimplementation**, not the
  production code path; "Live replay" → "The walk, replayed" (it is a static
  export). Limits section gained the omissions already recorded here: nothing
  held out, ground truth **estimator-matched** (GT bandwidths = the encoder's
  own length scales), **decode is 91% of query time**, the exact table already
  wins on bytes at the demonstrated N, `school_run1` attempted and failed,
  argmax cannot separate instances, online whitening ~10× worse, and `vsa.py`
  ported from the workshop `jepa` branch. Additions: "worst frame" button
  (jumps to the 6.85 m failure), orientation strip, demo→explainer cross-links,
  and the observation that **all 4 failures carry only 1–4 detections** (mean
  2.25 vs 3.65) — a pattern, not a law, since some 1-detection frames resolve.
  Replay caption now **computes its own statistics** from the payload rather
  than hardcoding them. Verified in-browser: no console errors, all 13 canvases
  paint, all anchors resolve, no horizontal overflow. **Unfixed / known gaps:**
  the export script that generates this page is still not committed (noted on
  the page itself), and the "3–15 ms" hero claim was narrowed to the only
  latencies actually exported (time panel, 3.0–10.2 ms) — replay and language
  export **no** latency at all.

- **2026-07-30 (v)** — **Episodic-memory pivot proposed and stress-tested; NOT
  yet adopted.** New page
  [`wiki/analysis/2026-07-30-episodic-memory-pivot.md`](2026-07-30-episodic-memory-pivot.md).
  Reframes ASTM from rival current-state index to *fixed-dimensional associative
  sketch of observation history*; flagship becomes lifelong object re-finding
  under change (partial cues in, **search action** out). Three proposal claims
  failed against numbers already in this repo: (1) **memory-budget win
  unearned** — the 34 B/event baseline (`crossover_analysis.py:79-84`) cannot
  answer a crop-cued query; an honest PQ-kNN baseline is ~40 B/obs vs 512 KB
  traces → crossover **~13k observations**, i.e. the proposed patrol regime sits
  AT OR BELOW it, and kNN accuracy is flat while ours decays O(N) (slopes +0.97
  raw / +0.86 whitened — whitening is a 22x constant, not a scaling-law change).
  (2) **"objects recovered per metre" measures the planner** and is gamed by
  abstaining (it lowers the denominator) — split into planner-free memory
  metrics (rank of true location; oracle-planner expected search cost bracketed
  by oracle-memory ceiling and uniform-prior floor; abstention priced as
  exhaustive search) and downstream system metrics. (3) **Dropping wall-clock is
  REJECTED** — `_range_kernel` (:328) is the strongest asset here, a visit IS an
  interval so `time_range` already is the visit probe, and the AUROC 0.905
  calibration was fitted with time as a factor; visit becomes a **second**
  factor and only the continuous-time *decode* path is retired. **Revised
  baseline threat**: a kNN vector DB over per-observation embeddings, not a
  scene graph — no such code exists. **BLOCKING BUILD GAP**: `cmd_embed`
  (`classroom_pipeline.py:163`) embeds the WHOLE FRAME; there are zero
  per-object appearance vectors in 9,519 lines, so every partial-cue claim
  currently reduces to a 38-class COCO label. **Honesty item**: `G_flat` is
  5,040 x 8,192 complex64 ≈ 330 MB of the 420 MB `memory_bytes()` total —
  deployed footprint exceeds FARM's entire scene memory (23–125 MiB), so the
  O(D)-query claim is false as built; adopting le-marmotte's 121-vector
  re-anchored readout projects 420 MB → ~11 MB and 7 → ~1 ms (**arithmetic, not
  measured**). **Data**: 3RScan/RIO is the strongest real fit and was absent
  from the plan (1,482 scans / 478 changing environments, per-instance 6DoF
  change transforms = detector-independent GT, no simulator); FindingDory
  (Apache-2.0) second; ReplicaCAD as action rig; GOAT-Bench does not fit.
  **Gate 0 (2 weeks, before anything else)**: fix a byte budget B, sweep N to
  1e6, squeeze four systems into B — if the curves do not cross, the pivot dies.
  Re-fit estimated ~24 person-days + dataset.

- **2026-07-30 (t)** — Crosstalk scaling + crossover economics + probe fixes
  (agent). (1) `crosstalk_scaling.py` finally DEFINES the public metric
  (hd=8192, N=50..1239 — embeddings.pt is the stride-2 embed, so 1239 IS
  "all frames" — x3 subsample seeds). Recall-relative ratio is ~flat in N
  (slopes +0.01 raw / +0.09 whitened; endpoint 0.93x vs 0.20x, 4.6x apart).
  Growth-law metric chi(N)=N*mean|offtarget| (per-item-signal units): raw
  slope **+0.97 = O(N) coherent CONFIRMED** (phasor mean cos 0.88), but
  whitened slope **+0.86, NOT sqrt(N)** — whitening buys a ~22x constant
  factor (endpoint 85x vs 3.8x), not a scaling-law change (residual scene
  correlations, mean |cos| 0.07). **365x/27x reproduced under NO definition
  tried — retire or re-derive.** Figure crosstalk_scaling.png.
  (2) `crossover_analysis.py`: events.csv measured 78.7 B/event (34 packed);
  scan 18 ns/event; ASTM query 5.7 ms D=4096 / 10.9 ms D=8192 (grid 72;
  entry j's 2.7 ms was the grid-48 sweep). Crossovers: table > traces-only
  at N*=3.3k (D=4096, ALREADY passed at N=4498) / 6.7k (8192); table > FULL
  state (210/420 MB incl. decoders) only at N*=2.7M/5.3M events
  (594x/1187x demonstrated N); latency: scan > ASTM at N*=0.31M/0.61M; an
  INDEXED table (1.8 us) is never beaten. Figure crossover_analysis.png.
  (3) Re-run fixes: probe_jepa content-only null (same derangement)
  = +0.004 vs persistence 0.727, conclusion now persistence-baseline only;
  two-loop test 4 whitening refit on EVEN frames only: median 0.08→0.08 m,
  100→99.8% ≤0.5 m (near-invariance is the finding); language paraphrase
  split n=12: grounded median 0.14 m (9/12 ≤1 m, top-1 7/12), direct
  0.20 m (9/12 ≤1 m).

- **2026-07-30 (u)** — Post-verification wording/citation corrections applied
  (agent). Docs-only pass over entry (r)'s literature findings: isotropy claim
  narrowed + prior-art citations added (Mu&Viswanath/Ethayarajh/Su/Ganesan),
  365× hedged as static ratio, "never grows"→trace-state wording, AUROC=sim
  caveat, language-query differentiators vs VLMaps/CLIP-Fields/NLMap,
  √(2D/K), FPE attribution (Plate→Komer 19→Frady 21), Frady/Kleyko/Sommer 18
  decay attribution, GC-VSA venue NICE 2025, References list on the robotics
  page. Files: results_robotics.html, demo explainers (md+html),
  code_explainer.html, working doc, this tracker (lines 44/120).

- **2026-07-30 (s)** — **Evaluation hardening (agent).** The 4 eval-design
  defects from (r) fixed (astm_traces.py / calibration_eval.py /
  astm_sweep.py); defaults verified to reproduce all logged numbers BEFORE
  switching (old-.pt bench bit-identical modulo timings; calibration_eval
  baseline re-run matched 0.872/0.878 exactly).
  (1) Estimator-matched GT: `bench --gt-bandwidths "0.4,0.75,1.5"` re-scores
  the same answers per KDE bandwidth + a bandwidth-free "supported" column
  (within 1 m / 40 fr of ANY matching event): 15/20, 15/20, 14/20 correct;
  18/20 supported. ONE flip — where(person) all-time fails only at bw=1.5
  (GT mode shift on a diffuse class; answer stays event-supported). All
  other bench claims survive all bandwidths; the 2 unsupported answers
  (person range queries) are both abstained.
  (2) Null calibration v2 per (decode, trace, probe-kind), 13 cells — NEW
  DEFAULT (measured misapplication of the old single null: range-cell nulls
  3.3-4.5x lower, marginal what_where nulls 2.6x higher than the shared
  point null). Bench: confident 15/16 correct (94%), abstained 4/4 wrong
  (100% abstention precision) vs old 14/15 (93%), 5 abst / 4 wrong (80%) —
  the correct-but-abstained range query is now confident. Old .pt files
  fall back to legacy nulls (router reports which null it used).
  (3) Guard band on unanswerable calibration labels (window +/-2*time_l;
  place radius+2*pos_l; --no-guard-band reproduces old): AUROC z 0.905 vs
  0.872, ECE 0.063, Brier 0.104, z now BEATS raw sim (0.875). Honest
  attribution: the gain comes from (2) — new nulls + OLD labels already
  0.906; the guard itself ~neutral here (109->107 unanswerable). z>=3
  operating point moved along the curve (cov 56.3% / risk 0.403 vs
  41.9%/0.289); fixed-coverage risk slightly better (0.476 vs 0.491 @70%).
  (4) Sweep reseeds class atoms per replicate (class_seed_mix = cell seed,
  new sweep default; build default mix=None unchanged). Re-run seeds {0,1}
  (reduced from 3), 40/40 cells, 5.4 min: accuracy-by-D 0.70/0.74/0.78/
  0.83/0.86 vs (j) 0.70/0.73/0.80/0.83/0.87 — D-scaling claim SURVIVES;
  seed variance did NOT widen (mean |s0-s1| 0.050 vs 0.077 on the same
  seeds pre-fix; 2 seeds = coarse). Pre-hardening sweep artifacts archived
  in outputs/classroom/sweep/archive_prehardening/; astm_traces.pt rebuilt
  with v2 nulls (range cells calibrated at width t_max/3; residual
  width-dependence documented in calibrate()).

- **2026-07-30 (r)** — **Three-agent verification sweep complete.**
  (V1) REPRODUCTION: 11/11 headline claims reproduced from bare logged
  commands, 0 drift — effectively bit-level at fixed seeds; only timings
  vary (cite as approximate); no missing artifacts or flag drift.
  (V2) ADVERSARIAL CODE REVIEW: no claim-breaking bugs; algebra clean
  everywhere attacked. Seven evaluation-design weaknesses, headline three:
  ground truth is estimator-matched (exact-table KDE bandwidth = FPE length
  scale — re-score under multiple bandwidths); one null per decode-type
  misapplied across marginal/point/range probes (calibrate per trace x
  probe-kind); calibration battery's "unanswerable" queries lack a guard
  band (deflates our own AUROC — conservative direction). Plus: sweep seeds
  don't reseed class atoms; JEPA above-null margin is vacuous (use
  persistence framing only); two-loop "held-out" whitening leaks (label
  fix); language battery has 3 identity freebies (report n=12 split).
  (V3) LITERATURE: isotropy claim = OVERCLAIM as framed — narrow to the
  robot-measured anisotropy->recall-collapse bridge; MUST cite
  Mu&Viswanath 18, Ethayarajh 19, Su 21, and Ganesan et al. NeurIPS 21
  (learned vectors break HRR). SNR must read sqrt(2D/K) (our own factor-2).
  The 365x/27x crosstalk metric is undefined in code and does not evidence
  O(N)-vs-sqrtN — define + run the N-sweep or drop "coherent". Language
  differentiation = fixed-size state + algebraic time-composition (NOT
  "participates in the algebra" — VLMaps queries are also algebraic).
  Decay traces = Frady/Kleyko/Sommer 2018 prior art; keep our two traps as
  engineering findings. Citations verified: GC-VSA **NICE 2025** (fact-check
  IJCNN wrong — fix line 44); strike-list guard numbers wrong: real
  VSA-OGM pair is 400x/45x not 420x/84x (fix line 120). "Map state never
  grows" and "ground-truth furniture cluster" = overclaims to reword.
  Top-3 reviewer attacks: (1) ground-truth circularity + N=1 environment;
  (2) isotropy-as-rediscovery unless narrowed + N-sweep run; (3) bounded
  state loses to an exact table by ~3 orders at demonstrated scale
  (crossover ~1e6-1e7 events) — compute crossover or say "projected".
  28-item required-citation list in the V3 report (task transcript).

- **2026-07-30 (q)** — **Citations verified at source + branch + robotics page.**
  Three collaborator papers fetched and verified: GC-VSA = Krausse, Neftci,
  Sommer, Renner, **NICE 2025, arXiv:2503.08608** (the fact-check's "IJCNN
  2025" appears to be wrong — arXiv page says NICE; correct before
  submission); **HyperSpace** = Snyder, Capodieci, Gorsich, Parsa,
  arXiv:2604.15113 (modular VSA framework; independently reports
  cleanup/similarity dominating runtime — matches our decode-dominates
  finding); **VSA-OGM** = Snyder et al., npj Unconventional Computing 3:13
  (2026) — the source of the 400x/45x numbers the strike-list forbids
  borrowing (occupancy task). Deliverables: `astm` branch committed on the
  VSACognitiveMapping clone (26be178, from init-am: module + docs + 15
  figures + README with provenance/results/citations; NOT pushed pending
  approval); robotics-audience results page
  `outputs/classroom/results_robotics.html` (+ inlined share version,
  published as artifact) covering entries a-p with 8 figures.

- **2026-07-30 (p)** — Two-loop pseudo-timeline experiment (agent). New
  `two_loop_experiment.py`: FAKE second loop by frame interleaving (even
  original t → loop 1 at t/2, odd → loop 2 at 620+(t-1)/2; positions
  unchanged) — a MACHINERY test for revisit structure (interleaved frames are
  ~0.1 s apart, appearance nearly identical; NOT robustness). 4/4 tests pass
  (hd=16384, log weighting): (1) when(place) from M_where_when is bimodal at
  5/5 positions, peak pairs 616-634 frames apart above a fake-place null
  (single-loop control unimodal); (2) cross-loop where(class) negative
  control: 4/6 top classes <0.15 m, outliers person/sports-ball are mode-flips
  BELOW the range-matched fake-class gate, gated which_moved max 0.18 m; (3)
  injected tv loop-2 cluster relocation 1.68 m → which_moved rank #1, measured
  1.51 m; (4) content⊗ctx_pos episodic memory from EVEN frames recalls ODD
  (revisit) frame positions at median 0.08 m / p90 0.17 m, 100% <0.5 m
  (shuffled-content null median 4.35 m, 9% <0.5 m). Figure:
  `outputs/classroom/two_loop_experiment.png`. Runtime 160 s.

- **2026-07-30 (m)** — Hybrid decay clocks (agent). `build_now --decay-per
  {event,frame,metre,both}`: lambda_eff = lambda_time**dt_frames *
  lambda_dist**dd_metres, clocks count from each class's LAST WRITE (frame row
  + cumulative travel), so a paused robot forgets under frame but not metre.
  New `--lambda-{static,dynamic}-dist` (0.999, 0.7 /m; 1/e washout =
  -1/ln(lambda) m), mode+rates in memory_now.pt meta; default `event` verified
  bit-identical to the previous build. New `compare_clocks` cmd: the walk DOES
  contain a stationary stale window (person stale 75 f at t=1163, 0.12 m
  moved; 50-frame travel min/median 0.028/1.035 m) — there metre keeps
  w=14.8 while event/frame/both collapse to <1e-3 (forgotten); moving window
  (t=551, 2.06 m) all time-clocks w~0.03, metre w=28. Figure:
  `outputs/classroom/decay_clocks_comparison.png`. Distance knob credit:
  Krausse le-marmotte 1/e-washout, independently convergent with our
  per-class frame decay — the product form composes both.

- **2026-07-30 (o)** — **Language→VSA queries WORK (the wedge, measured).**
  New `language_query.py`: MiniLM text latents → whitened (over ~180-noun
  vocab — isotropy medicine applied to the language modality) → random
  phasor projection; memory rebuilt with TEXT-DERIVED class atoms over
  object-position events (log class-weighting). Two mechanisms, 15-query
  paraphrase battery: (A) soft grounding (text-sim route to atoms): 10/15
  top-1 grounding, median err 0.11 m; (B) **direct latent unbind** (free
  text as the unbinding key, no codebook routing): median err **0.17 m**,
  12/15 within 1 m — "luggage"→suitcase 0.03 m, "cold storage for
  food"→fridge 0.24 m, "cutting tool for paper"→scissors 0.11 m, all
  phrases never stored. Failures are LM-semantic, not VSA-mechanical:
  ambiguous phrases ("something to sit on" spreads over couch/table/chair
  → mixture peak off; "a computer display"→laptop — defensible). The two
  mechanisms fail on DIFFERENT queries (complementary; 'seating furniture'
  grounded fails 5.9 m, direct 0.19 m). Prior-art context: CLIP-Fields/
  VLMaps do language-queryable maps without VSA; this is the LM-latent-as-
  unbinding-key demo the research programme's wedge names. Figure:
  outputs/classroom/language_query.png.

- **2026-07-30 (l)** — **Calibration metrics + synthetic moved-object
  (agent).** (1) `calibration_eval.py`: 544-query labeled battery (435
  answerable over all 27 classes with >=3 events, 109 unanswerable; base
  error 61.9%). AUROC z 0.872 vs raw sim 0.878; per decode type z==sim
  exactly (z is a per-type monotone affine of sim, so **null calibration
  adds nothing over raw sim within a decode type** — its value is the
  interpretable threshold, not discrimination). ECE (in-sample Platt, 10
  bins) 0.067 z / 0.066 sim; Brier 0.140/0.136. Risk-coverage: default
  z>=3 gives coverage 41.9% at selective risk 0.289 (vs 0.619 at full
  coverage). Figure: calibration_eval.png. (2) `moved_object_synthetic.py`:
  relocated all 265 second-half 'tv' events by (+1.50,+0.75) m (2.5/1.5
  infeasible in-bounds); hd=16384 (half-window range kernels keep ~2% of
  components — at 8192 tv sits ~2.5σ over the fake-class null). Gated
  which_moved (min_sim = fake-class range-matched null mean+3σ = 0.00094;
  the stored point-probe null does NOT transfer to range kernels): modified
  ranks tv **#1, 1.71 m measured vs 1.68 m injected**; control tv 0.38 m
  (natural 0.24 m). Conditional where(tv, half): 0.29/0.07 m err; timeless
  marginal is bimodal (field 0.0396/0.0404 at the two truths) and argmax
  loses the old place (1.54 m) — the event trace is necessary. Figure:
  moved_object_synthetic.png.

- **2026-07-30 (n)** — **Frequency bias FIXED + object-position events (agent,
  astm_traces.py).** (1) `build --class-weighting {conf,balanced,log}` with
  tercile bias-bench: the sqrt-N bias manifests as POSITION ERROR (crosstalk
  hijacks rare-class peaks), not abstention — under conf, LOW-tercile mean
  err 5.15 m / MID 1.93 m; **log damping fixes everything** (LOW 0.12 m,
  MID 0.04 m, HIGH unharmed at 0.05-0.22 m; rare-class z the highest of all
  three modes). Balanced fixes MID + 2/3 of LOW. (2) `export --place-mode
  object`: depth-localized events (2.1% fallback); ASTM chair where-query
  peak lands **0.086 m** from the largest furniture cluster vs 2.11 m for
  the vantage build — conditional queries now answer "where IS X at t".
  Bonus: null floor drops (0.00183 vs 0.00272) — object positions decorrelate
  place vectors. (3) Hygiene filters (--min-class-events, --class-blocklist)
  at the canonical-stream boundary. No regressions (default paths
  byte-consistent; full bench matches).

- **2026-07-30 (k)** — **Online whitening eval (agent).** Closes the
  "whitening is offline" gap with numbers: streaming Welford whitening
  (per-frame top-64 eigh, sign-aligned; centering-only before frame 100)
  reaches oracle-level isotropy fast — windowed mean |off-diag| cosine
  matches oracle by t≈300 (0.133 vs 0.132; raw 0.94, oracle whole-walk
  0.084) and stays <1.5× oracle for the whole walk. Recall cost is
  transient: grid-decode L1 = 0.09 m oracle vs causal 1.58/0.85/0.15 m
  (early/mid/late thirds) — near-oracle by the final third; whole-walk
  causal 0.86 m. Calibration lap does NOT suffice: frozen-K L1 =
  1.73/1.38/1.15 m (K=100/300/600), and even on the late third frozen-600
  (0.29 m) loses to continual causal (0.15 m). EMA (half-life 200) worse
  than exact Welford everywhere (1.30 m). Surprise: causal codes stay
  ~orthogonal to final-stats re-encodings even late (sim 0.29) — degenerate
  eigen-subspaces keep rotating, so store codes as encoded, never re-encode.
  Script vsa_cognitive_mapping/online_whitening_eval.py; figure
  outputs/classroom/online_whitening.png.

- **2026-07-30 (j)** — **D×N sweep (agent).** First rung of the evaluation
  matrix: full 60-cell grid (D 1024..16384 × N 500..4498 × seeds 0,1,2) ran
  in 7.7 min, 0 skipped/failed (grid 48, n_null 100, fixed 15-query battery;
  ground truth = each cell's own subsampled exact table). Accuracy
  (seed-mean, avg over N) 0.70/0.73/0.80/0.83/0.87 at D=1024..16384 —
  saturation starts at D=4096 (+0.03/doubling after). NO capacity cliff at
  N=4498: D=1024 degrades mildly (0.78@N=1000 → 0.67@N=4498) while
  D=16384 is best at full stream (0.91, max 0.93). Pareto knee = D=4096
  (118 MB map state incl. decoders; 256 KB traces; ~2.7 ms median query).
  Calibration: confident-correct rate climbs 0.84→0.94 with D; abstention
  precision peaks 0.89@4096, falls to 0.58@16384 (few, borderline
  abstentions left). Residual errors = where@window range-kernel queries on
  diffuse classes (tv 55/60, person 50/60 wrong; both improve with D).
  Artifacts: outputs/classroom/sweep/ (2 CSVs + 4 figures);
  script vsa_cognitive_mapping/astm_sweep.py.

- **2026-07-30 (i)** — **Semantic map upgrades (agent): object binding +
  working memory.** (1) `build --place-mode object`: binds SEM_class to
  depth-derived OBJECT positions (2.3% fallback to robot pose). Chair field
  peaks now land 0.13/0.19/0.34 m from the three biggest object_map chair
  clusters, off-trajectory where furniture is — vs robot-mode peak 2.05 m
  from clusters on the trajectory. "Where IS the chair" answered, not "where
  was it seen from". (2) `build_now`/`query_now`: per-class decayed M_now
  (lambda 0.995 static / 0.9 dynamic). Person blob tracks latest sighting
  (0.12-0.51 m when fresh) and is honestly flagged "forgotten" when decayed
  (w~0, sim 0.13 vs 0.77-0.93 fresh). TWO measured VSA findings for the
  paper: (a) raw decayed bundles are undecodable for rare classes (static
  mass ~100x -> 3.09 m crosstalk drag); fix = per-class weight-normalized
  bundle; (b) weight normalization WITHOUT an evidence floor silently cancels
  decay (trace and weight shrink together) — a 1e-3 floor restores real
  forgetting. Figures: working_memory_person.png,
  semantic_query_chair_memory_object.png. memory.pt untouched
  (backward-compatible; new files memory_object.pt / memory_now.pt).

- **2026-07-30 (h)** — **E2 school_run1: pipeline scales, dataset's odometry
  does not.** Config-driven pipeline ran unattended (11,211 frames embedded,
  38,141 detections, memory built: 3,737 stored @ hd 8192, whitened).
  Evaluate: heading L1 0.156 rad and time hold at 4.5x stored frames;
  position L1 118 m — diagnosed as **diverged LIO-SAM odometry in the
  dataset itself** (x span -1788..109, y span -89..2310 in a ~30 m building;
  209 pose jumps >5 m, max 243 m; longest clean segment 4-10 s). school_run1
  position mapping is untestable with provided poses. Consequences: (1)
  vindicates Round-5 fallback — our own staged collection is REQUIRED for
  the scaling claim, not optional; (2) heading/time scaling numbers are
  usable; (3) le-marmotte VGPI (one-vector lidar localization, validated on
  this dataset family; school_run1 pointclouds config exists) is a candidate
  pose-recovery path = natural three-lineage integration point. Artifacts:
  outputs/school_run1/{embeddings.pt, detections.csv,
  associative_memory_stride.pt, evaluate.png}.

- **2026-07-30 (g)** — **ASTM engine upgrades (agent).** (1) Fast decode:
  probe-side factors fold into the 1-D residual before one matvec — range
  grid decode **168 -> 14.5 ms**, algebraically identical (|dSim| <= 5e-10),
  no per-query 330 MB allocation. (2) Multi-scale time (`--time-scales
  5,20,80`, bundle|bind): NEITHER dominates — bundle rescues sparse-window
  range queries (4.01 -> 0.22 m) but stays below confidence threshold and
  adds one confident-wrong mode-hop; bind sharpens correct answers (13 -> 7
  fr) but drops sims 3-4x into abstention. Dominant error everywhere =
  multimodal-time mode-picking, not kernel width. (3) Calibrated abstention
  (contribution #3, working): per-decode-type null distributions in the .pt;
  z>=3 lands at sim ~0.0037, reproducing the empirical >=0.006/<=0.002
  split; bench: 93% confident-correct, 80% abstention precision.
  Methodological catch: unit-amplitude random null probes miscalibrate —
  nulls must match event-trace RMS component amplitude. Figure:
  outputs/classroom/multiscale_time_comparison.png.

- **2026-07-29 (f)** — **JEPA predictability probe run**
  (`vsa_cognitive_mapping/probe_jepa_predictability.py`, figure
  `outputs/classroom/jepa_predictability.png`). Question: is s_{t+1}
  predictable from z_t = s_t ⊗ ctx_pos(p_t) via the bind z_t ⊗ T(Δpose)?
  Classroom data, 1,238 consecutive pairs, hd=8192, whitened content (PCA-64,
  seed 0), pipeline position bases (seed 100, ℓ=0.75). Numbers (mean/median/
  p10): (a) content persistence 0.727/0.739/0.590; (b) place-code transport
  sanity 1.0000 (exact — FPE group property); (c) bound-state prediction
  0.727/0.739/0.590 (== a, identically: lossless transport makes prediction
  inherit persistence); (d) shuffled null 0.016/-0.001/-0.012. Horizon decay
  (mean, h=1/2/4/8/16): 0.727/0.657/0.541/0.382/0.195 vs flat null ~0.015.
  **Conclusion: mixed — transport is lossless (b=1.000) so prediction adds
  nothing over content persistence (a≈c), but binding transport preserves it
  0.711 above the shuffled null; go-signal for VSA-JEPA hinges on beating the
  persistence baseline, not the null.**
- **2026-07-29 (e)** — **Unified showcase demo + explainer shipped (2 agents).**
  workshop_demo.py now has a fourth panel: the ASTM multi-trace router live in
  the browser (`/api/astm/*`) — conditional where/when/what queries incl.
  range kernels and which_moved, verified in-UI: where(chair,t=150) decodes
  the robot's true t=150 position, per-stage latency shown (total ~7 ms;
  encode 0.7 / unbind 0.1 / decode 6.1). All four memory systems now
  demonstrable on one page at localhost:8020. Explainer written twice over:
  `wiki/analysis/2026-07-29-demo-explainer.md` (full: analogy, pipeline,
  panel guide, numbers table, limitations, three-lineage unification) +
  `outputs/classroom/demo_explainer.html` (screen-share version). Writer
  agent correctly refused an unsourced number from the brief (flagged in
  wiki/log.md) — the documented 0.067 |mean| off-diag stands.
- **2026-07-29 (d)** — **Demo 80x faster + class recall; le-marmotte surveyed.**
  Fixed workshop_demo.py hot path (their phasor_cross_correlation re-normalized
  the full decoder matrices every call; hoisted to precomputed unit-norm
  conjugate complex64 mats): reverse query 800 ms -> **10 ms** server-side.
  Added class->location recall (`/api/where_class` + dropdown): phase-normalized
  bundle of stored frames containing the class as a unit-modulus unbinding key
  on M_pos — 7 ms, 21 classes. Surveyed `neuromorphs/le-marmotte-vsa-grid-cells-
  in-silicon`: VGPI lidar positioning in ONE 2048-d phasor vector (~16 KB map),
  0.13 m SE(2) relocalization on OUR Spot dataset, online one-vector SLAM,
  fpga/fpga-1bit precision-reduction branches (iCE40), and Krausse's
  `interactive-demo` branch — fully in-browser static demo (onnxruntime-web
  YOLO + Quick,Draw sketch -> query M via class prototype labels). THIRD
  in-house lineage for the unification story; their demo architecture
  (precompute assets, browser-side matvec worker) is the model for a public
  ASTM demo. Clone at scratchpad; ingest as wiki source pending.
- **2026-07-29 (c)** — **Workshop code integrated + interactive demo live.**
  Both Telluride zips extracted to `external/` (py3.9 compat patch: one
  `from __future__ import annotations` line × 17 files). Ran THEIR
  detect_and_embed (2,478 frames, 8,878 detections) and THEIR build
  (stride-3, whitened, hd=8192, 826 stored). Their own `evaluate`
  independently reproduces our July numbers: position L1 0.183 m, heading
  0.122 rad. New `vsa_cognitive_mapping/workshop_demo.py` serves their
  demo.mp4 as a live page (scrub/play 1,652 held-out frames, reverse query
  ~0.8 s/tick, click-to-query forward direction) — verified: time recall
  0.2–0.8 s from truth; forward query at a trajectory point recalled a
  stored frame 2 cm away. Cross-lineage validation point for the paper:
  two implementations, same data, same numbers.
- **2026-07-29 (b)** — **P0 v0 built and benched** (`astm_traces.py`):
  canonical event stream (4,498 events / 38 classes), 4 traces at hd=8192,
  closed-form range kernels (probe-side), decode-target query router,
  Baseline-A exact table (KDE-modal ground truth) + 21-query bench.
  Results: marginals 0.05–0.22 m / 11–13 fr; conjunctive point queries
  0.09–0.15 m; class queries correct; point-query latency 3–16 ms
  (range-kernel grid probes ~250 ms — decode dominates, as Round 4
  predicted; optimization TODO). **Emergent calibration evidence: answers
  with peak sim ≥ ~0.006 accurate; ≤ ~0.002 unreliable (sparse-window
  cases)** — direct empirical seed for contribution #3. Honest memory
  accounting: traces 512 KB, but 420 MB total (decoder grids dominate).
- **2026-07-29 (a)** — v1 written; five-round fact-check ingested same day;
  v2 rewrite (this page). Corrections: deadlines, SuperMap unreleased,
  terminology, four-baseline ladder, unification framing, multi-trace
  architecture adopted. Next: IROS-LBR go/no-go decision.

## Related Pages

- [Fact-check raw doc](../../raw/2026-07-29-astm-fact-check.md)
- [SuperMap source page](../sources/2026-07-29-supermap-paper.md)
- [Classroom results](../experiments/2026-07-29-vsa-cognitive-map-classroom-results.md)
- [JEPA→FHRR canonical plan](2026-05-16-jepa-fhrr-vsa-canonical-plan.md)
- [SIGReg VSA reframe](2026-05-11-sigreg-vsa-reframe.md)

## Open Questions

- IROS LBR (Jul 31): go/no-go?
- Resonator scope: paper 1 or GC-VSA follow-up? (team call)
- Does capacity hold at school_run1 scale? (WP4 first)
- Furlong KDE calibration: applicable to event traces?
- Dataset release: include as 5th contribution or separate paper?
