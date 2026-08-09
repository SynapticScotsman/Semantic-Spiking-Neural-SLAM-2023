# Abstract + Intro variants for evaluation (Opus panel, 2026-08-10)

Five agents, one fact sheet (identical measured numbers + honesty rules +
style rules: no em dashes, British spelling, VSA-as-algebra naming), five
rhetorical angles. Each ends with its own honest self-assessment.

**Status:** `main.tex` now carries a deliberate GENERAL SYNTHESIS of all
five (2026-08-10) rather than any one angle: it opens on growing
structures and the machinery they oblige, states the design point with
the "hour or a week, same 32 KB object" hook (variant 1), presents
query / relocalisation / merge / filter / law as identities of the
algebra (variant 5) with the merge bit-equivalence hook (variant 3), and
closes on an explicit not-claimed paragraph (variant 4). The law's
predictive framing (variant 2) is compressed into one sentence there and
carried in full by Sec. IV. Pick a sharper angle later and I graft it in;
the five below stay as the source material.

Quick comparison:

| variant | opening move | strongest asset | biggest risk |
|---|---|---|---|
| 1 Systems-first | "same 32 KB after an hour or a week" | concrete, checkable hook; merge as culmination | capacity law underplayed; "just compression" reading |
| 2 Law-first | "maps fail in the field; this one fails by equation" | 1.4% prediction + falsified alternative; kills "bag of tricks" | notation-heavy start for ICRA; chi-to-metres gap |
| 3 Merge-first | multi-robot merge pain (MB + estimates) vs addition | the only categorical (not incremental) claim leads; bit-level equivalence hook | reads as a partial system to end-to-end multi-robot reviewers; internal 89-vs-84 number sits closest to the headline |
| 4 Honest-contrarian | state the three losses first, then price them | disarms every reviewer objection on page one | skim reader retains only the losses |
| 5 Algebra-first | subsystems become identities | homomorphism + merge exactness up front; intellectual hook | instance blindness is the direct cost of the central claim |

---

<!-- VARIANT 1 -->

# Variant: Systems-first (the 32 KB map)

## Suggested title

**A 32 KB Map: Vector Symbolic Memory for Semantic Relocalisation, Query, and Exact Multi-Robot Merge**

## Abstract

The map in this paper is one vector: 8,192 phase angles, 32 KB, allocated before deployment and unchanged in size no matter how long the robot drives. Landmarks enter by binding an appearance atom to a fractional-power position code and adding the result into a running sum; queries (where was a chair, what is here, when was this place last occupied) are one unbind plus a kernel readout, milliseconds on a laptop CPU, with no nodes or edges to traverse. We characterise this trace as a robotic memory. On the 7-Scenes chess cross-traverse split with EigenPlaces descriptors, the trace relocalises to 0.36 m median 3D error at 32 KB, against 0.29 m for an exact kNN store at 32.8 MB and 0.29 m for product quantisation at 2.1 MB; product quantisation cannot reach this budget, since its codebook alone is roughly 2 MB. A capacity law predicts a held-out condition to 1.4% and predicted a class-imbalance failure and its fix. Maps from four robots merge by elementwise addition: exact to float reordering, zero data-association decisions, one 32 KB message each. The memory does not out-localise nearest neighbours on its own descriptors and does not perform dense semantic segmentation; exact merge assumes a shared frame and vocabulary.

## Introduction

A robot that has driven for an hour and a robot that has driven for a week hand us the same object: 8,192 complex phase angles, four bytes each, 32 KB in total. Nothing was appended, no node was inserted, no edge was linked, and the query cost did not move. That fixed artifact, and what a robot can still do once its map has been compressed into it, is the subject of this paper.

The prevailing alternative is to grow structure. Scene graphs [ConceptGraphs], [Hydra] and instance databases accumulate one record per observed object, then maintain them: association decisions to decide whether a new detection is an old instance, merges and splits when that decision was wrong, and traversal at query time over a graph whose size tracks the environment. The growth is not incidental; it is the representation. It also sets the communication bill. A recent multi-robot mapping experiment [Kimera-Multi] moves 24-146 MB between agents, because what must be shared is structure.

We take the opposite starting point: fix the artifact first, then ask what capability survives. Our memory is a vector symbolic algebra [Plate] in the Fourier holographic reduced representation, with dimension D = 8,192. The algebra supplies three operators: binding (elementwise complex multiply), superposition (vector addition), and unbinding (multiply by the conjugate). Continuous space enters through fractional power encoding, S(x) = X^x, which satisfies S(a) * S(b) = S(a + b) exactly (measured to 1.5e-16), so translation is multiplication and a spatial kernel falls out of the inner product. A whole map is one sum, Sum c * S(x, y), over every landmark ever seen. Older literature reads VSA as "architecture"; here the algebra is the substrate and the map is the architecture we build from it.

The systems consequences are immediate. Memory is allocated before deployment, so there is no growth to bound and no eviction policy to tune. Query is one unbind and a readout, so latency is independent of how much has been stored; there is no structure traversal because there is no structure to walk. Merging two maps is adding two vectors, which is exact: the sum equals the map that would have been built jointly, to roughly 1e-15 of float reordering. Using bundling as the between-robot merge operator appears to be new. The standing caveat is that this exactness assumes a shared coordinate frame and a shared atom vocabulary; the algebra assumes the frame, it does not solve it.

We then measure what the budget costs. On the official 7-Scenes [7-Scenes] chess cross-traverse split with EigenPlaces descriptors, an exact kNN store reaches 0.29 m median 3D error at 32.8 MB, product quantisation with m = 8 matches it at 2.1 MB, and the 32 KB trace reaches 0.36 m (76.4% of 2D errors within 0.5 m, 93.0% within 1 m; 91.5% and 98.9% for a top-3 readout). Product quantisation cannot exist at our budget: its codebook alone is around 2 MB. Published DenseVLAD retrieval on this scene is 0.21 m. We state plainly that the memory does not out-localise nearest neighbours on its own descriptors, and that a detector-fed object memory cannot label walls and floors, so it is not a dense semantic segmenter. Our claim throughout is capability at the perception ceiling under a fixed byte budget, not accuracy records. Instance identity remains a limitation.

Capacity is not left to empirical luck. We derive a law, chi(N) = a_mu * N + sqrt(N) * sqrt(alpha/PR + gamma/D) * h(N, tau), which predicted a held-out condition to 1.4%, predicted a class-imbalance failure mode and its fix (bounded per-class insertion lifted grounding from 44% to 62%), and isolates an encoder-independent floor: whitening four backbones collapses raw chi from 79.8-201.8 to 9.1-11.0 at N = 2,429. The same algebra also bounds drift: predict is one bind, update is one bundle, and with odometry degraded twentyfold (dead reckoning 0.87 to 17.6 m) fused error moves only 0.35 to 0.62 m (0.364 +/- 0.051 m against 0.868 +/- 0.503 m over ten noise draws).

Our contributions are:

1. **System.** A fixed 32 KB vector symbolic map (D = 8,192 FHRR) supporting spatial, semantic, and temporal queries in one unbind plus kernel readout on a laptop CPU, with a filter native to the algebra that holds error near 0.6 m under 20x odometry degradation.
2. **Capacity law.** A closed-form capacity expression that predicts held-out performance to 1.4%, predicts a class-imbalance failure and its fix, and exposes an encoder-independent whitened floor.
3. **Benchmark characterisation.** An honest byte-versus-error placement on the official 7-Scenes chess split (32 KB / 0.36 m against 32.8 MB / 0.29 m exact and 2.1 MB / 0.29 m quantised), plus an internal Replica characterisation in which every miss is perception-attributable and memory-attributable misses are zero.
4. **Exact merge.** Elementwise addition as a multi-robot merge operator: exact to float reordering, 89% mean grounding with zero association decisions against 84% (worst scene 67%) for instance association with 622 greedy decisions, at one 32 KB message per robot, under a stated shared-frame and shared-vocabulary assumption.

## Why this angle works (agent's self-assessment)

- **The hook is concrete and falsifiable.** "Same object after an hour or a week: 32 KB" is a systems fact a reviewer can check in one table, and it frames every later number as a consequence of a design choice rather than as a competitive claim. It also lets the honesty constraints land as engineering trade-offs (0.36 m is what 32 KB buys) instead of as apologies, which is the safest way to survive a skeptical reviewer who will notice the 0.29 m baselines anyway.
- **It sequences the contributions in the order a systems reader cares about.** Fixed allocation, constant query cost, no traversal, then merge as addition. The merge result (zero association decisions, one 32 KB message each) is the strongest and most defensible claim in the paper, and this framing makes it the natural culmination rather than an appended experiment.
- **Weaknesses.** The angle underplays the capacity law, which is arguably the deepest scientific contribution and here reads as one dense paragraph a theory-minded reviewer may find rushed. It also invites the "so it is just lossy compression" reading, and the introduction only rebuts that implicitly (via query capability and merge algebra) rather than head-on. Finally, leading with bytes risks a reviewer concluding the paper is about a compression trick and reading the 0.36 m as the headline result, when the intended headline is what the compressed object can still be queried and merged for; the 7-Scenes number is also single-scene, which the abstract does not foreground.

---

<!-- VARIANT 2 -->

# Variant: Law-first (failures predicted by an equation)

## Suggested title

**A Map That Fails by Equation: A Capacity Law for Vector Symbolic Robot Memory, and the Engineering It Dictates**

*(Alternates: "Predicting Where the Map Breaks: An Interference Law for 32 KB Vector Symbolic Maps"; "From Capacity Law to Field Fix: Algebraic Robot Mapping with Predictable Failure")*

## Abstract

Robot maps usually fail in ways discovered in the field. We present a map whose failures are predicted on paper, and we show the prediction working. Our map is a single FHRR vector symbolic trace, D=8192 phasors (32 KB, fixed before deployment), holding continuous geometry through fractional power encoding that is exact to 1.5e-16, and read by one unbind plus a kernel readout in milliseconds on a laptop CPU. Retrieval degrades under a three-term interference law: an exactly linear shared-mean term (chi/N constant at 0.0083 up to 3x10^5 items), a fluctuation term combining content overlap and a content-independent projection floor in quadrature, and a temporal persistence factor saturating at a permanent elevation. A single-power-law alternative is falsified, and the quadrature model predicts a held-out cell to 1.4% (0.0321 predicted, 0.0316 observed). The law's terms dictate engineering: it explains why whitening collapses four heterogeneous backbones onto one floor, and it predicted a deployed class-imbalance failure whose bounded-insertion fix raised grounding from 44% to 62%. We characterise the map on 7-Scenes chess (0.36 m median at 32 KB) and on exact four-robot merging (89% versus 84% with zero association decisions).

*(196 words)*

## Introduction

A robot map is a bet about what will still be retrievable later. Most mapping systems settle that bet empirically: the map works until it does not, and the boundary is discovered in deployment, in a corridor with too many chairs or after a fifth traverse of the same room. Retrieval degradation in learned and quantised representations is real but rarely has a closed form, so the practical response is to over-provision memory and hope. We take the opposite route. We build the map out of an algebra whose interference we can write down, measure the terms separately, and then let the equation dictate the engineering.

The substrate is a vector symbolic algebra (VSA). The older literature reads the acronym as "architecture" [Plate]; we use "algebra" for the operator set (bind, bundle, unbind) and reserve "architecture" for the map built from it. We use FHRR phasor vectors at D=8192. Binding is elementwise complex multiplication, superposition is vector addition, and continuous space enters through fractional power encoding S(x)=X^x with S(a)*S(b)=S(a+b) holding to 1.5e-16 in our measurements [Frady]. The whole map is one trace, Sum_i c_i * S(x_i,y_i): 8,192 phase angles at 4 bytes each, 32 KB, a size fixed before deployment rather than grown during it. A query is one unbind and one kernel readout, milliseconds on a laptop CPU.

Because superposition is linear, we can characterise what an added item does to every other item. We measure an interference law chi(N) = a_mu*N + sqrt(N)*sqrt(alpha/PR + gamma/D)*h(N,tau), and we evidence its three terms separately rather than fitting one curve. The shared-mean term is exactly linear: chi/N stays constant at 0.0083 out to 3x10^5 items and never bends. The fluctuation term has two sources that add in quadrature: content overlap alpha/PR, which conditioning treatments act on, and a content-independent projection floor gamma/D, which they cannot touch. A single-power-law alternative is falsified by a D=1024 run, and the quadrature model predicts a held-out cell to 1.4% (predicted 0.0321, observed 0.0316). The third term is temporal: interference grows coherently while N < tau, then reverts as sqrt to a permanent elevation h_inf = sqrt(1+tau/tau0) with tau0 around 60. Nothing here is decoration. The gamma/D floor explains a measurement we would otherwise call a coincidence: whitening lands four very different backbones (YOLOv8n, ResNet-50, DINOv2-S, and an untrained ResNet-50), whose raw chi spans 79.8-201.8 at N=2,429, onto the same floor at 9.1-11.0.

The law then makes predictions about deployments, and we report both the prediction and the outcome. It says rare classes drown in a bundle dominated by common ones; in Replica, couch is stored 1,150 times and book 5 times, and a bounded per-class insertion rule derived from that prediction raised grounding from 44% to 62%. It says the mean term is what conditioning should attack, and that centring or z-scoring the content stream and binding a diverse key (which phase-scatters the mean) are substitutes; they measure as substitutes. It also says whitening flattens the spectrum where retrieval lives, so whitening should be avoided even though it improves chi: on classroom this costs 47.6% signal over 5 seeds, while on chess it is harmless. The effect is scene-dependent and we state it as such.

We are explicit about what this is not. We do not beat k-nearest-neighbour retrieval on its own descriptors, and we do not do dense semantic segmentation; our pre-registered all-classes mAcc of roughly 0.076 against roughly 0.40 for [ConceptGraphs] under their own scorer is a gap we report rather than reframe (that run is pending). Instance identity is a stated limitation, and exact merging assumes a shared frame and vocabulary: the algebra does not solve alignment. The claim is capability at a fixed and very small memory ceiling, with the failure boundary known in advance.

Our contributions are:

1. **System.** A single-trace FHRR map at D=8192 (32 KB, fixed pre-deployment) with exact fractional power encoding, millisecond CPU queries, and drift bounding by one bind and one bundle: with odometry degraded 20x (0.87 to 17.6 m dead reckoning), fused error moves only 0.35 to 0.62 m (10 draws: 0.364+/-0.051 m versus 0.868+/-0.503 m).
2. **Capacity law.** A three-term interference law with each term separately evidenced, a falsified single-power-law alternative, a held-out prediction accurate to 1.4%, and conditioning rules (centre or bind-diverse; never whiten) derived from the terms rather than tuned.
3. **Benchmark characterisation.** 7-Scenes chess relocalisation on the official cross-traverse split with z-scored EigenPlaces: 0.36 m median 3D at 32 KB (76.4% <=0.5 m 2D, 93.0% <=1 m; top-3 91.5%/98.9%) against a 32.8 MB kNN ceiling at 0.29 m and PQ m=8 at 2.1 MB / 0.29 m, where the PQ codebook alone is about 2 MB so no PQ operating point exists at 32 KB ([DenseVLAD] 0.21 m for context); plus an internal Replica characterisation over 8 scenes with zero memory-attributable misses.
4. **Exact merge.** Elementwise addition merges maps to about 1e-15; across 4 robots this gives 89% (worst scene 80%) with zero association decisions, against 84% (worst 67%) and 622 decisions for association-based merging, at one 32 KB vector per robot versus 24-146 MB per experiment for [Kimera-Multi], under a stated shared-frame and shared-vocabulary assumption.

## Why this angle works (agent's self-assessment)

- **It converts the paper's weakest competitive position into its strongest scientific one.** We cannot win a localisation leaderboard, and a benchmark-first framing invites reviewers to score us on 0.36 m versus 0.29 m and stop reading. Law-first reframes the same numbers as a characterisation of a predicted operating point, and the 1.4% held-out cell plus the falsified single-power-law alternative are genuinely rare in a robotics mapping paper: they are the kind of evidence a reviewer cannot get from an ablation table. The class-imbalance story (predicted on paper, fixed in deployment, 44% to 62%) is the closing argument, and it is the single most persuasive sentence we have.
- **It gives every engineering choice a parent, which kills the "bag of tricks" objection.** Conditioning rules, the whitening prohibition, the bounded per-class insertion cap, and the 32 KB fixed footprint all descend from named terms in one equation rather than arriving as tuned hyperparameters. This also makes the honesty constraints load-bearing instead of defensive: the gamma/D floor is a term that says some interference cannot be conditioned away, so admitting limits reads as the theory working rather than as hedging.
- **The weaknesses are real and the reviewer will find them.** First, the law is validated on interference chi, not directly on task metrics, so a sceptic can ask whether predicting chi to 1.4% predicts a metre of localisation error; we assert the link but the paper as described does not close it, and that is the most likely rebuttal question. Second, the strongest deployed prediction (class imbalance) is qualitative in its predictive step: the law says rare classes drown, but 44% to 62% was not forecast numerically in advance, so a hostile reading calls it post hoc. Third, this angle front-loads notation and delays any picture of a robot, which is a real risk with an ICRA audience; it also spends abstract words on the law that a benchmark-first variant would spend on 32 KB versus 24-146 MB, the comparison most likely to make a tired reviewer sit up. Finally, the whitening result is scene-dependent (-47.6% on classroom, harmless on chess), which slightly undercuts a rule presented as derived rather than empirical.

---

<!-- VARIANT 3 -->

# Variant: Merge-first (adding maps)

## Suggested title

**Maps That Add: Exact Multi-Robot Map Merging in a 32 KB Vector Symbolic Memory**

Alternates:
- *Merging by Addition: An Algebraic Map Representation for Multi-Robot Semantic SLAM*
- *One Map from Many, Without Association: Bundling as a Merge Operator*

## Abstract

Merging two robot maps is an estimation problem. Occupancy grids need an SE(2) transform search then cell fusion; pose-graph systems need inter-robot loop closures, pairwise-consistency outlier rejection and distributed optimisation, with one reported system moving 24-146 MB per experiment; multi-robot scene graphs add per-layer node association. Every output is an estimate: order- and iteration-dependent, never identical to a jointly built map. We present a semantic spatial memory whose merge operator is elementwise addition. A map is one 32 KB trace in a fractional-power-encoded FHRR vector symbolic algebra (D = 8192): binding composes what with where, bundling accumulates, and a query is one unbind, milliseconds on CPU. Given a shared vocabulary and coordinate frame, M_A + M_B equals the jointly built trace to float reordering (about 1e-15), commutative and associative, with no association decisions. Four session memories on 7-Scenes chess sum to answer all 2,000 test queries identically to the joint build; four robots over 8 Replica scenes reach 89% mean grounding against 84% for instance-list association (622 greedy decisions). The same trace localises to 0.36 m median where a 32.8 MB kNN ceiling gives 0.29 m. The algebra assumes the frame; it does not recover it.

## Introduction

Two robots explore a building and return with a map each. Combining them is not an addition; it is a pipeline. Occupancy grid methods search over SE(2) transforms for the alignment that maximises cell agreement, then fuse cells [Birk-Carpin]. Pose-graph systems detect inter-robot loop closures, reject outliers by pairwise consistency, and run distributed optimisation [DOOR-SLAM, Kimera-Multi]; Kimera-Multi reports 24-146 MB moved per experiment. Multi-robot scene graphs add per-layer node association, deciding which of my rooms and objects are which of yours [Hydra-Multi]. Neural implicit maps merge by consensus optimisation over many communication rounds. Each is well engineered, and each produces an estimate: order- and iteration-dependent, never identical to the map the same robots would have built together, with association errors that persist.

We present a memory in which this step is not a pipeline. A map is a single vector, and merging two maps is elementwise addition: M_A + M_B. It is commutative and associative, so merge order cannot matter; the payload is one 32 KB vector per robot regardless of how much that robot saw; and there are no association decisions to get right or wrong. The merge is exact rather than approximate, reproducing the jointly built trace up to floating-point reordering (about 1e-15). On 7-Scenes chess, four independently built session memories summed answer all 2,000 test queries identically to the jointly built memory. With four robots on identical observation streams across 8 Replica scenes, addition reaches 89% mean grounding (80% on the worst scene) with zero association decisions, against 84% (67% worst) for instance-list association with 622 greedy decisions, 82% for count grids at roughly 95 KB, and 85% for an unbounded raw union. Bundling as a between-robot merge operator appears novel; the closest prior art aggregates HDC descriptors within one agent [Neubert-Schubert] or averages models in federated HDC [FedHD].

We state the limit of this result up front. Exactness holds under a shared atom vocabulary and a shared coordinate convention. Much of the machinery we contrast against exists precisely to recover the unknown inter-robot transform; our algebra assumes that frame, it does not solve it. Misaligned traces sum to an inconsistent superposition, and do so silently: there is no residual to inspect and no error signal. The fair comparison is therefore against the back-end merge step given the frame, not against complete multi-robot front-ends, and we run it that way.

An addition operator matters only if what it adds is a usable map, and the rest of the system is what earns that. We build on a vector symbolic algebra (VSA; older literature reads the acronym as architecture: the algebra supplies bind, bundle and unbind, and the map is the architecture we build from them). We use FHRR phasors at D = 8192 with fractional power encoding, so position composes exactly: S(a)*S(b) = S(a+b) to 1.5e-16. Landmarks are bound to positions and bundled into one trace of 8,192 phases at 4 bytes, 32 KB in total; a query is one unbind and a kernel readout, milliseconds on CPU. Because everything shares one superposition, we can say in advance how much it holds: a capacity law predicts held-out recall to 1.4%, and predicted a class-imbalance failure that bounded per-class insertion fixed (44% to 62%). The same two operators bound drift online: predict is one bind, update is one bundle, and under 20x odometry noise dead reckoning degrades from 0.87 m to 17.6 m while the fused estimate stays at 0.62 m.

We are explicit about what this is not. At 32 KB the trace reaches 0.36 m 3D median on the official 7-Scenes chess cross-traverse split, where exhaustive kNN over the same EigenPlaces descriptors reaches 0.29 m at 32.8 MB: this is not state-of-the-art localisation, it is not dense segmentation, and instance identity remains a limitation. Our claim is capability at the perception ceiling.

Our contributions are:

1. **System**: an FHRR trace at D = 8192 that holds a semantic map in 32 KB and answers localisation and semantic queries in milliseconds on CPU.
2. **Capacity law**: a closed-form predictor of recall against load, accurate to 1.4% on held-out runs, which predicted and then fixed a class-imbalance failure.
3. **Benchmark characterisation**: 0.36 m 3D median at 32 KB on the official 7-Scenes chess cross-traverse split, against a 32.8 MB kNN ceiling and 2.1 MB product quantisation, both at 0.29 m.
4. **Exact merge**: elementwise addition as the between-robot merge operator, equal to the jointly built map given a shared frame and vocabulary, with that assumption stated as the limit it is.

## Why this angle works (agent's self-assessment)

- **It leads with the one claim that is categorically, not incrementally, better.** Every other number in the paper is "competitive at much smaller size"; the merge result is "identical to the joint build, checkable in one line". Opening on merging gives a reviewer a crisp thing to be surprised by within two sentences, and the 7-Scenes result (four summed session memories answering all 2,000 queries identically) is the strongest single artifact we have, because it is a bit-level equivalence rather than a metric comparison.
- **The frame caveat is load-bearing and this framing forces it into the open, which is both the defence and the ceiling.** A merge-first opening invites the obvious objection ("you assumed away the hard part") immediately, so we answer it in the abstract's last sentence and again before the contributions. The honest cost is that the claim is capped at the back-end merge step given the frame; a reviewer who wants end-to-end multi-robot SLAM will read this as a partial system, and no rewording changes that.
- **The weakest supporting evidence sits closest to the headline.** The 4-robot 89% vs 84% grounding comparison comes from an internal characterisation protocol on Replica, not a benchmark-comparable scorer, so the number carries less weight than its prominence suggests; we should present it as a controlled ablation (same observations, three merge strategies), not as a benchmark win. Secondary risks: the merge-first opening pushes the capacity law and the 7-Scenes size/accuracy trade-off into supporting roles, and an HDC-literate reviewer may consider bundling-as-merge obvious, so the novelty framing must stay carefully scoped against [Neubert-Schubert] and [FedHD] rather than claimed broadly.

---

<!-- VARIANT 4 -->

# Variant: Honest-contrarian (price the trade)

## Suggested title

**What a Bounded Associative Map Costs, and What It Buys: Pricing the Trade Against Scene Graphs**

## Abstract

Scene graphs are exact, mutable and relational. The memory we study is bounded, associative and approximate. We do not claim the second replaces the first; we measure what each side of that trade costs. Our map is a single 32 KB vector symbolic algebra trace (FHRR phasors, D=8192) fixed before deployment, queried by one unbind and a kernel readout in milliseconds on CPU, with no structure to traverse. We state the costs first. On 7-Scenes chess the trace localises at 0.36 m median 3D against 0.29 m for exact k-nearest-neighbour over its own descriptors; it cannot label walls, floors or ceilings, so under ConceptGraphs' all-classes dense metric we pre-register an expected 0.076 mAcc against their 0.40; it assumes a shared frame; instance identity is unreliable on multi-instance classes. What that buys: 32 KB against 32.8 MB at 0.07 m of accuracy, a merge that is elementwise addition (exact to 1e-15, zero association decisions, 2,000/2,000 chess queries identical to a jointly built map), and a capacity law predicting a held-out condition to 1.4%. Within this envelope, every observed miss is perception's.

## Introduction

A scene graph is the right representation for a great many robot tasks. It is exact, it is mutable, it names its relations, and you can walk it. Nothing in this paper argues otherwise. What we argue is that the representation arrives with obligations that are rarely priced: data association on every update, correspondence and optimisation on every merge, and structure-specific traversal code on every query. Growth in the number of objects is not by itself the problem; the machinery that growth obligates is. When two robots meet, [Kimera-Multi] exchanges 24-146 MB per experiment and solves a correspondence problem; [Hydra] maintains layers whose consistency must be actively repaired; [ConceptGraphs] builds an explicit instance list whose identity decisions are load-bearing.

We study the opposite corner. A vector symbolic algebra ([Plate]) supplies bind, bundle and unbind; the map is the architecture built from them. Ours is one FHRR trace of 8,192 phase angles at four bytes each: 32 KB total, fixed in size before deployment, holding every landmark superposed. Position enters by fractional power encoding, S(x)=X^x with S(a)*S(b)=S(a+b) exact to 1.5e-16, so spatial composition is algebra rather than lookup. A query is one unbind followed by a kernel readout, milliseconds on CPU. There is no traversal because there is no structure.

This buys something and it costs something, and we think the honest move is to lead with the costs. Three of them are real and we do not soften them.

First, the memory does not out-localise exact k-nearest-neighbour over its own input descriptors: 0.36 m against 0.29 m median 3D on 7-Scenes chess. Superposition is lossy, and lossy compression loses.

Second, it cannot label structure classes. Walls, floors and ceilings are not object-like and our encoder does not produce them, so under ConceptGraphs' own all-classes dense metric we expect roughly 0.076 mAcc against their roughly 0.40. We state that expectation here, with their scorer, before the run, because a number stated afterwards is not a prediction.

Third, merging assumes a shared coordinate frame. We do not solve alignment; misaligned traces sum silently into inconsistency, and that is a genuine hazard, not a footnote. Separately, wrong-instance rate reaches 72% on multi-instance classes in internal diagnostics: instance identity is a limitation of this memory, not a feature of it.

Now the price on the other side. The 0.07 m accuracy gap on chess is bought at 32 KB against 32.8 MB, and the comparison is not merely favourable but structural: product quantisation with m=8 matches the kNN ceiling at 2.1 MB, yet cannot exist at 32 KB because its codebook alone is about 2 MB. Two-dimensional recall is 76.4% within 0.5 m and 93.0% within 1 m; top-3 reaches 91.5% and 98.9%. Under our internal Replica characterisation protocol across eight scenes (not benchmark-comparable, and we label it as such), memory-attributable misses are zero: all five pooled misses trace to detection or depth, and on identical observations the trace recalls 87% against 84% for an explicit instance list and 81% for a raw store. Merging is elementwise addition: exact to about 1e-15, zero association decisions against 622 for an instance-association merge that degrades to 84% mean and 67% worst, and four chess sessions answer 2,000 of 2,000 queries identically to a jointly built map. One 32 KB vector crosses the link per robot. Fused with odometry, drift is bounded where dead reckoning is not: 0.87 to 17.6 m under 20x noise against 0.35 to 0.62 m fused, and 0.364 +/- 0.051 m against 0.868 +/- 0.503 m over ten draws.

We also explain the envelope rather than merely reporting it. A capacity law, chi(N) = a_mu*N + sqrt(N)*sqrt(alpha/PR + gamma/D)*h(N,tau), predicts a held-out condition to 1.4%, predicted the class-imbalance failure together with its bounded-insertion fix (grounding 44% to 62%), and identifies a gamma/D projection floor that is encoder-independent: raw chi spans 79.8-201.8 across four named backbones at N=2,429 and collapses to 9.1-11.0 whitened. The law also dictates conditioning discipline (centre, z-score, or bind a diverse key), and warns that whitening is harmful in most held-out cells (-47.6% on classroom, five seeds) while harmless on chess.

Our contributions are:

1. **System.** A 32 KB fixed-size FHRR map with exact fractional-power spatial composition and constant-time unbind queries, with its failure envelope stated in advance.
2. **Capacity law.** A closed-form predictor of recall capacity, validated to 1.4% on a held-out condition, that anticipates failures and prescribes conditioning.
3. **Benchmark characterisation.** 7-Scenes localisation priced against exact kNN and PQ at matched budgets, plus a pre-registered ConceptGraphs comparison under their scorer.
4. **Exact merge.** Multi-session and multi-robot map union by elementwise addition: 1e-15 exact, zero association decisions, given a shared frame.

## Why this angle works (agent's self-assessment)

- **It defuses the three reviewer objections by making them the thesis.** A reviewer who was going to write "this loses to kNN", "it cannot do all-classes semantics", and "the merge assumes alignment" finds all three already on page one, with numbers attached. That converts adversarial reading into verification, and it earns credibility for the harder claims (zero memory-attributable misses, exact merge) that a hyped intro would have spent.
- **The pre-registration is the strongest move available.** Stating 0.076 mAcc against 0.40, in the target's own metric, before the run, is a costly signal no competing variant can cheaply match; it also inoculates the paper if the run comes back worse than hoped, since the paper predicted the shape of the failure rather than discovering it.
- **The weaknesses of this angle are real.** (a) It front-loads defeat, and a skimming reviewer or an area chair reading only the abstract may retain "0.36 loses to 0.29" and "0.076 against 0.40" and stop; the buys have to survive that skim, which is why the 32 KB against 32.8 MB and the PQ codebook-floor argument sit as close to the losses as possible. (b) The intro is dense with numbers and reads as a results section in miniature, which risks leaving the reader without a single memorable image of the system. (c) The framing concedes the general case to scene graphs, so the paper's scope is narrow by construction: it wins its envelope but cannot claim the field, and a reviewer looking for ambition may score it low on impact even while agreeing with every sentence.

---

<!-- VARIANT 5 -->

# Variant: Algebra-first (identities replace subsystems)

## Suggested title

**A Map You Can Compute With: Vector Symbolic Algebra as the Query Interface for Spatial Memory**

*(alternatives: "One Algebra, Four Subsystems: Query, Merge, Filter and Range as Identities over a Spatial Trace"; "Multiply to Move, Add to Merge: An Algebraic Spatial Memory for Multi-Robot Semantic Mapping")*

## Abstract

We present a spatial memory in which the map is an algebraic object and the usual map subsystems are identities rather than code. A robot's map is a single vector: one FHRR phasor trace in C^D, D=8192, holding 8,192 phase angles in a fixed 32 KB. Inserting an observation is a multiply then an add. Asking where the chair was is a multiply by a conjugate. Translating an entire spatial belief field is one multiply, because fractional power encoding gives a group homomorphism S(a)S(b) = S(a+b) that we measure exact to 1.5e-16. Merging two robots' maps is addition. Four consequences follow and we measure each: query needs no index because there is no structure to traverse; merge is exact to ~1e-15, with four independently built 7-Scenes chess sessions answering all 2,000 test queries identically to the jointly built map, and four robots reaching 89% mean grounding with zero association decisions against 84% for instance association; a Kalman-style filter runs as bind then bundle, holding 0.35-0.62 m where dead reckoning diverges 0.87-17.6 m; and a time interval costs the same single unbind as an instant. Addition presumes a shared frame and vocabulary; alignment remains outside the algebra.

*(191 words)*

## Introduction

A robot map is usually a data structure with services bolted around it: an index to answer queries, a protocol to merge maps between robots, a filter to fuse motion with observation, a special path for range queries over time. Each service is code someone writes, tunes and debugs, and each has its own failure modes. We take a different route. We choose a representation whose *algebra* already contains those services, so that the query engine, the merge protocol and the filter are not subsystems but identities.

The representation is a vector symbolic algebra (VSA) over FHRR phasor vectors [Plate], [Kanerva]: unitary vectors z in C^D with D = 8192, every component a unit complex number. Three operators act on them. Bind is elementwise complex multiplication, unbind is multiplication by the conjugate, and bundle is vector addition. Continuous position enters through fractional power encoding [Komer], S(x) = X^x taken elementwise, and this is where the algebra earns its keep: S(a)S(b) = S(a+b) is a group homomorphism, exact in our measurements to 1.5e-16, so translating an entire spatial belief field costs one bind and loses nothing. Similarity between S(x) and S(x') is a fixed radial kernel, so readout is a dot product rather than a search. Heading rides on a circular base with integer frequencies and no DC component, so marginalising over heading is a single bundled probe. The map itself is one trace, a sum of weighted place vectors, Sum c S(x,y): 8,192 phase angles at 4 bytes each, 32 KB, fixed before the robot is switched on and unchanged by how much it then sees.

Four subsystems collapse into that algebra, and we measure each collapse. *Query* becomes one unbind and one kernel readout, milliseconds on a laptop CPU; there is no structure traversal because there is no structure. *Merge* becomes addition, agreeing with a jointly built map to about 1e-15: on the 7-Scenes chess cross-traverse split, four independently built session memories, summed, answer all 2,000 test queries identically to the map built jointly. In a four-robot experiment, addition reaches 89% mean grounding (worst robot 80%) with zero association decisions, against 84% (worst 67%) and 622 association decisions for an instance-association merge, and the communicated object is one 32 KB vector per robot where Kimera-Multi reports 24-146 MB per experiment [Kimera-Multi]. *Filtering* becomes predict-by-bind and update-by-bundle: under 20x odometry noise, dead reckoning drifts from 0.87 m to 17.6 m while the fused estimate stays between 0.35 m and 0.62 m (over ten draws, 0.364 +/- 0.051 m against 0.868 +/- 0.503 m), and we decode only for display. *Range queries* become closed-form range kernels, so asking about a time interval costs the same single unbind as asking about an instant.

One caveat is mandatory and we state it before any result. Addition merges maps only under a shared frame and a shared vocabulary. The algebra does not solve alignment; misaligned traces sum silently into inconsistency, and detecting that is outside what we claim.

We are equally explicit about what the system is not. It is not state-of-the-art localisation: on 7-Scenes chess with z-scored EigenPlaces descriptors, exhaustive kNN reaches 0.29 m median 3D error at 32.8 MB and product quantisation matches it at 2.1 MB, while our 32 KB trace gives 0.36 m (76.4% within 0.5 m in 2D, 93.0% within 1 m, and 91.5%/98.9% at top-3), with published DenseVLAD at 0.21 m for context. It is not dense semantic segmentation, and we report a pre-registered comparison under ConceptGraphs' own scorer [ConceptGraphs]. Instance identity is a stated limitation, with 72% wrong-instance responses on multi-instance classes in our internal diagnostic. The claim is capability at the perception ceiling: the algebra adds query, merge, filter and range behaviour at fixed cost, without claiming to improve the descriptors underneath.

Our contributions are:

1. **System.** A spatial memory in which one FHRR trace of fixed 32 KB supports insertion as multiply-add, query as unbind, exact translation as one bind, and a Kalman-style filter entirely inside the algebra.
2. **Capacity law.** A closed-form law chi(N) = a_mu N + sqrt(N) sqrt(alpha/PR + gamma/D) h(N,tau), predicting held-out capacity to 1.4%, predicting a class-imbalance failure that bounded per-class insertion then fixed (44% to 62%), and exposing an encoder-independent whitened floor gamma/D (raw chi 79.8-201.8 across four named backbones at N = 2,429 collapsing to 9.1-11.0).
3. **Benchmark characterisation.** A cost-accuracy account against kNN and PQ baselines on the official 7-Scenes chess cross-traverse split, plus an internal Replica grounding protocol, with conditioning rules that are measured rather than assumed (centre or z-score, or bind a diverse key; never whiten, which costs 47.6% on classroom over five seeds while being harmless on chess).
4. **Exact merge.** Addition as a merge operator, verified identical to joint construction to ~1e-15 across sessions and evaluated across four robots, with the shared-frame and shared-vocabulary precondition stated as a limitation rather than assumed away.

## Why this angle works (agent's self-assessment)

- **The hook is a genuine intellectual claim, not a framing device.** "Subsystems become identities" is falsifiable and each of the four instances carries a measured number in the same paragraph that states it, so the reviewer never spends more than two sentences in pure mathematics before hitting metres, megabytes or decision counts. The homomorphism exact to 1.5e-16 and the ~1e-15 merge agreement are the strongest facts in the whole fact set, and this angle puts both in the first half of the abstract where a skim reader will actually see them.
- **It reframes the honesty constraints as scope rather than damage.** Because the contribution is stated as algebraic capability at fixed cost, the admission that kNN wins on localisation accuracy reads as the expected consequence of a 1000x smaller footprint rather than as a failed comparison. The risk is the reverse: a hostile reviewer may read "the algebra gives you all this for free" and then find that the 0.36 m versus 0.29 m gap, the pending segmentation number and the 72% wrong-instance rate all arrive in one dense paragraph, which can land as a pile-up. A revision might distribute those admissions rather than clustering them.
- **The weak points are the capacity law and multi-instance identity.** The capacity law is the paper's most original theory but it is genuinely awkward under an algebra-first frame: it is a statement about interference and noise, not about identities, so contribution 2 sits slightly outside the story the introduction tells and currently arrives as a formula the reader has no motivation for. Instance identity is worse: the whole appeal of "no structure to traverse" is precisely why the system cannot distinguish two chairs, so the 72% figure is not an incidental limitation but the direct cost of the central claim, and an alert reviewer will notice that this angle does not say so. If this variant is chosen, both should be addressed in Section 2 rather than papered over.
