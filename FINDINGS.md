---
title: Object-centric VSA view memory — findings, recipe, and pitfalls
type: analysis
status: active
created: 2026-08-15
updated: 2026-08-17
source_paths:
  - sspslam/objectmap/
  - experiments/run_view_localisation.py
  - experiments/turntable_dataset.py
  - experiments/run_object_map.py
provenance:
  exact: [§2, §4, §9]        # algebraic identities, verified to machine precision
  synthetic: [§5, §7, §10]   # rendered turntable + HOG — illustration, not measurement
  pending: [§6-rotation]     # awaits wiki/analysis/2026-08-17-3d-cogmaps-and-rotation-encoding.md
tags: [vsa, cognitive-map, object-centric, viewpoint, anisotropy, fpe, errata]
---

# Object-centric VSA view memory — findings, recipe, and pitfalls

## What this is

Give every object its own little turntable in memory. Walk round a chair and
you store a handful of snapshots, each tagged with the angle you took it from.
Later, show the memory a fresh photo and it tells you which side you are
standing on — no pose input, appearance alone.

The whole object file is **one fixed-size vector**. Angles are stored on a real
circle, so walking a full lap brings you back to exactly where you started, and
"rotate the object 40°" is a single multiply.

```
object file    V = (1/K) Σ_k  c(z_k) ⊗ S_view(φ_k)
```

Plainly: `c(z_k)` is a fingerprint of what the object looked like, `S_view(φ_k)`
is the angle you saw it from, `⊗` staples them together, and the sum is all of
it in one vector.

## What works

| | plainly | §|
|---|---|---|
| The circle is a real circle | 360° gets you back to the exact same code, to 1e-16. Rotating is one multiply. | §2 |
| Reading it is one FFT | You get the *whole* answer — how likely every angle is — not just a best guess. Instant. | §4 |
| Tracking over time | Frame by frame the answer jumps around; feed it through a filter and it settles. 17° → 6°. | §12 |
| One known starting angle | Fixes the symmetric objects completely. 83° → 2°. | §12 |
| The whole scene in one vector | Six objects, twelve views each, one vector, unbind an ID to query it. 72× smaller than a list, same accuracy. | §16 E0, E2 |
| Two vectors per object, not one | Name the object with an unbound appearance prototype, read the angle with the view book. Identification 0.44 → 0.89 for one extra vector. | §16 E1 |
| Twelve sides is the answer | Store a view every 30°. The same number for every object that isn't symmetric, and the cliff past it is sharp. | §16 E4 |

## What doesn't

**It can't guess sides you never looked at.** Leave a 30° hole in the orbit and
the error inside that hole is ~27°, tracking the hole size all the way up to
chance. It fills in *between* stored views and does not reach past them (§0 E4).

**Symmetric objects can't be told apart, and no amount of data fixes it.** A
cube looks identical from four sides. That is not a bug in the estimator —
there are genuinely four right answers. Tracking plus one known starting angle
is the only fix (§12).

**Sharpening the code makes it worse, not better.** I assumed a finer angular
code would help. It doesn't: 25.6° error with a broad code, 33.0° with a sharp
one. When your stored views are far apart, *reach* beats precision (§14).

**Building the brain's middle stage doesn't help.** Face patches go
view-specific → mirror-symmetric → view-invariant, so we built the mirror stage
deliberately. It identifies objects *worse*, by −0.15, and equally so for
symmetric and chiral objects — the regime argument that should have rescued it
doesn't (§16 E1). The code works exactly as designed; the design doesn't pay.

**It is not more accurate than just keeping a list — it ties one.** Store the
same views in a list, take the nearest: 10.0°, against the object file's 9.0°,
difference −1.0° with a confidence interval straddling zero (§16 E2). What the
VSA buys is size, not accuracy — the whole map, six objects and every view, in
**one vector** instead of 172,872 floats. **Twelve to seventy times smaller,
same answer.** That is the honest pitch, and it is not the one this document
was making before §16.

Getting even that far needed one setting fixed. At the repo's old
`max_harmonic=8` the object file really did lose, 25.5° against 10.0°, and got
*worse* the more views it was given. The number of distinct frequencies in the
code is what sets how many views a bundle can hold, and 8 was too few (§16 E2).

## The six rules that cost us the most

1. **Frames are not samples.** 72 frames round an orbit are worth about **6**
   independent observations. I quoted `n=216, p=7.6e-9` once; the real n was
   ~19 and the p-value was fiction. Retracted in §0 E8.
2. **Never split by alternate frames.** A held-out view sitting 5° from a
   stored one measures memory, not generalisation. It made us look 3× better
   than we were (§0 E4).
3. **Don't whiten the features.** It gives perfect-looking statistics and
   destroys the answer — 11° → 86°, against 90° for guessing (§5, and `astm`
   measured it first on real robot data).
4. **Run the dumb baseline first.** Everything in §§4–14 was unfalsified rather
   than validated until §16 compared it against a list and a nearest-neighbour.
   It lost. The claim had to be rewritten, not the experiment (§16 E0).
5. **Sweep the hyperparameter before blaming the method.** §16 E0 concluded the
   object file loses to a list. It was measured at the repo's default
   `max_harmonic`, which E2 then showed is the single knob that controls how
   many views a bundle holds. Setting it correctly turned a loss into a tie —
   and E0 had swept the vector dimension over sixteen-fold without touching it.
6. **One object is not evidence.** In §16 E1 the cube was the single object
   whose identification improved under mirroring, and it looked like a clean
   confirmation of the published account. Adding four more symmetric objects
   flipped it from +0.13 to −0.06. It had been an artefact of which distractors
   happened to be in the set.

## The one idea underneath the front end

Every crop is part *"which object this is"* and part *"which way it's facing."*
The first part doesn't change as you walk round — a chair is still that chair
from behind — so it sits under every similarity measurement like a floor.
Call the floor's share `λ`:

```
ρ(Δ) = λ + (1 − λ)·r(Δ)
```

Measured similarity = floor + (what's left) × the real angle signal. Removing
the floor is a **contrast knob, not an information gain** — it reorders nothing,
but it rescales three numbers we use to make decisions. Holds to RMS 0.002–0.016
with nothing fitted (§14).

## Before you quote anything

**Read §0 first.** Every degree in this document comes from a *rendered*
turntable with a HOG descriptor — it demonstrates a mechanism, it is not a
measurement of any real encoder or robot. The algebra (§2, §4, §9, and the
filter's predict step) is exact and separately verified. DINOv2 has never run
here; egress policy blocks the weights.

## See it

`docs/view_circle.html` — pick an object, scrub the orbit, watch the likelihood
succeed or fail. The cube's four bright ridges are the clearest single view of
why symmetry is unfixable without an anchor.

```bash
python experiments/build_view_circle_page.py     # rebuild it from the code
```

---

## 0. Errata and provenance — read before quoting anything below

Added 2026-08-17, after reading `astm/docs/RESULTS_SO_FAR.md` and
`astm/docs/provenance-audit.md` on `neuromorphs/VSACognitiveMapping@gpu-tasks`.
Several claims in the original draft were stated more strongly than the
evidence supports. The corrections are here rather than silently patched in.

### E1. What kind of thing each number in this document is

Three categories, and they are not interchangeable:

| tag | what it means | where |
|---|---|---|
| **exact** | an algebraic identity, verified numerically to machine precision. Quotable as stated. | §2 periodicity and binding-as-rotation; §4 the one-FFT ≡ scan identity; §9 |
| **synthetic illustration** | measured on a **rendered** turntable with a HOG descriptor. Illustrates a mechanism; is **not** a measurement of any real system, encoder, or robot. **Do not report as a result.** | §5 all figures; §7; the degree figures in §4 and §10 (and see E4 — §4's figures are additionally superseded by the blocked re-run) |
| **pending** | awaits the 3-D/rotation handoff, which may supersede it | §6 rotation/quaternion recommendation |
| **retracted** | quoted with an invalid sample size; see E8 | the p-value in §5's appearance-rate refutation |

No number in this document was measured on real imagery, on DINOv2, or on a
robot. The dataset is `experiments/turntable_dataset.py` — 6 procedurally
rendered objects × 72 azimuths.

### E2. §5 is an independent replication, not a new finding — priority is `astm`'s

`RESULTS_SO_FAR.md` finding 1 ("isotropy predicts capacity, not key quality")
and finding 2 ("centre or z-score, do not whiten") already establish this **on
real robot data**, before anything here was run:

> Whitening harder degrades retrieval monotonically: 0.720 → 0.693 → 0.654 →
> 0.623 → 0.592 as 32, 64, 128 then all 256 directions are equalised.
> … Whitening to 128 components scores **below chance** on held-out data.
> — `astm/docs/RESULTS_SO_FAR.md`, findings 1–2 (measured)

§5's whitening result reproduces that conclusion on a different read-out
(view-direction rather than place retrieval) and on synthetic data. It is
corroboration, not discovery. Cite theirs.

### E3. §5's "drop the leading PCs" recipe does **not** transfer as written

The spectra are not comparable:

| | top dimension's share | dims for half the variance |
|---|---|---|
| `astm` crops, DINOv2, 256-D — *measured* | 5.3% | **49 of 256** |
| §5 here, HOG on rendered turntable — *synthetic* | 17.8% | ~3 |

The turntable spectrum is roughly an order of magnitude more concentrated than
real crop embeddings. Deleting 2–5 directions is a strong intervention on the
first and a weak one on the second, so the *effect size* in §5's recipe table
will not carry over, and may vanish entirely.

What may survive is the **criterion**, not the cut depth: selecting directions
by *between-object variance share* is a different axis from the
rogue-dimension analysis in `RESULTS_SO_FAR.md` finding 3, and has not been
tested against real data at all. Treat §5's recipe as a **hypothesis to test**,
not a recommendation to apply.

### E4. The split was not blocked — re-run, and the original figures were optimistic

§4 and §5 originally held out *alternate* azimuths, so a held-out view sat 5°
from a view still in the object file. `RESULTS_SO_FAR.md` corrects exactly that
pattern — a split leaving neighbours of the query in memory measures
memorisation, not generalisation.

**Re-run** with contiguous held-out arcs (`experiments/run_blocked_split.py`):
alternating kept/held arcs so no held-out query is adjacent to a stored side,
latent statistics and object files fitted on kept views only. Chance is 90°;
a constant predictor also scores 90°.

| held-out arc | nearest kept view | **in-gap median** | <15° | control (kept views) |
|---|---|---|---|---|
| interleaved *(original)* | 5.0° | **8.0°** | 0.66 | 7.0° |
| 10° | 5.0° | 10.8° | 0.58 | 8.0° |
| 20° | 7.5° | 21.2° | 0.36 | 5.0° |
| 30° | 10.0° | **27.5°** | 0.18 | 6.0° |
| 45° | 15.0° | 36.8° | 0.08 | 7.5° |
| 60° | 17.5° | 59.0° | 0.06 | 7.0° |
| 90° | 25.0° | 79.2° | 0.04 | 6.0° |

Three things follow, and the first is a correction:

1. **The original 8° was optimistic by roughly 3×.** At a 30° gap — the first
   split with genuinely no adjacent stored side — it is 27.5°, and the fraction
   within 15° collapses from 0.66 to 0.18. Any headline "~8–10° median" in
   earlier drafts of this document should be read as 27.5° at a 30° gap.

2. **The object file does not extrapolate, and that is the real result.** The
   control column is flat at 5–8° for every gap width: the file is undamaged,
   and error in the gap tracks gap width almost linearly to chance at 90°.
   Viewpoints you never observed are not recoverable — the code interpolates
   within roughly the view kernel and no further. This is the honest version of
   §4 and arguably more useful than the original number.

3. **A design rule falls out.** To localise to better than 15°, stored sides
   must be spaced no wider than about the view kernel's half-width (16° at
   `max_harmonic=8`). That is the number `merge_tol` should be set from, and it
   converts directly into a coverage requirement on the orbit: ~22 sides for a
   full circle, not "a few diverse sides".

Per object at the 30° gap: `console` 18.8°, `L_block` 19.7°, `chair` 21.0°,
`mug` 27.3°, `pot` 29.0°, `cube` 90.0° — the cube still aliasing correctly.

### E5. Never quote a bare whitening or conditioning multiplier

`RESULTS_SO_FAR.md`: the public 365× and 22× figures were measured on the most
anisotropic representation available and are 8.1× on DINOv2. Any ratio produced
by the experiments here is single-encoder (HOG) and single-dataset; quote the
encoder and the dataset with it or not at all.

### E6. Prior art for §11 step 5

`RESULTS_SO_FAR.md` finding 6 already implements a Kalman-style filter over
this algebra for **position** — predict is one bind via the FPE homomorphism,
update is one bundle, and it caps odometric drift (dead reckoning 3.803 m →
0.627 m at σ=0.10). §11's "close the loop" is therefore not new for position.
The object-centric analogue — feeding a *view-direction* likelihood back as a
heading correction — is the part that does not yet exist.

### E8. Views on an orbit are not independent samples, and one p-value here was fiction

§5's refutation of the appearance-rate hypothesis was reported as
"ρ = +0.380, n = 216, p = 7.6e-9". The **n is wrong**, and so the p-value is
meaningless. Consecutive views on an orbit are strongly correlated — that is
the whole content of §13 — so 216 held-out frames are nowhere near 216
independent observations.

Integrating the appearance autocorrelation
(`experiments/run_frontend_diagnostics.py`) gives an effective sample size of
about **6 independent views per 72-frame orbit** (median 6.2 on conditioned
keys, 2.8 raw). Across six objects with half the ring held out, the effective
n is nearer **19 than 216** — an inflation of roughly 11×.

What survives and what does not:

- The **sign and the ranking survive**. ρ is positive under every
  leave-one-object-out exclusion (+0.220 to +0.417), and the aliasing split
  (single-peak 24.0° against multi-peak 80.5°) is a difference between two
  large groups, not a correlation coefficient.
- The **p-value does not survive**, and should never have been quoted. Nor
  should any confidence interval computed by resampling frames.
- Anything resampling frames as if they were independent needs a
  **hierarchical bootstrap** — objects, then contiguous arcs, then seeds —
  which is the same discipline `RESULTS_SO_FAR.md` applies and which this
  document had not.

This is the identical error to the one the blocked-split correction fixed in
E4, in a different guise: E4 removed leakage between train and test, E8 is
about leakage *within* the test set inflating confidence. The fix for both is
to treat the arc, not the frame, as the unit.

### E7. What remains unaffected

The algebra is untouched by all of the above. The view circle is exactly
2π-periodic, binding is exactly rotation on it, and the one-FFT likelihood is
identical to the explicit scan to 6.7e-16. Those are proofs with a numerical
check, not empirical claims, and E1–E6 do not weaken them. The *architecture*
argument in §1 — two memories rather than one bound blob — is likewise
structural: binding a bundle of random-looking terms into the spatial argument
destroys the correlation peak for reasons that do not depend on the dataset,
though the specific metre figures quoted are synthetic.

## 1. The two memories

```
scene map     M_scene = (1/N) Σᵢ  IDᵢ ⊗ S_allo(pᵢ)          which things, where
object file   Vᵢ      = (1/K) Σₖ  c(zₖ) ⊗ S_view(φₖ)        what one looks like, from which side
```

| symbol | what | why not otherwise |
|---|---|---|
| `ID` | random unitary atom, minted on first sighting | a name must survive being seen from a new side; a crop doesn't |
| `S_allo(p)` | ordinary **aperiodic** SSP of world position | unchanged from SSP-SLAM; this is the navigation index |
| `c(z)` | `unit((z−μ)W)` — centred crop embedding, random projection, L2 | a *value* on the circle, not the circle |
| `S_view(φ)` | **periodic** FPE of object-centred azimuth | the orbit; wraps exactly after a full turn |
| `CLASS` | second atom, its own bundle | `unbind(CLASS)` → "where are all the chairs" |

Robot pose appears nowhere in a file. `φ` is a *relation* between object and
camera, so it is computed at query time from the two poses — or, better,
recovered from appearance (§4).

**Do not fold them into one blob.** `ID ⊗ S_allo ⊗ V` leaves `S_allo ⊗ V`
after unbinding `ID`, and `V` is a bundle of random-looking terms, so the
spatial peak scatters:

```
mean position error   bundle ID ⊗ unit(S_allo + V):  0.03 m
                      bind   ID ⊗ S_allo ⊗ V     :  7.13 m     (14 × 14 m room)
```

It breaks the other direction too — unbinding `ID ⊗ S_view(φ)` from that blob
leaves `S_allo ⊗ c`, not `c`. If you want one vector per object, *superpose*
the roles (`object_vector(mode='bundle')`); `mode='bind'` reproduces the
failure so it stays measurable.

---

## 2. The view circle: integer harmonics

An ordinary SSP has real-valued phases, so `S(x)` never repeats — right for a
room, wrong for an orbit. Restrict the phase matrix to **integer harmonics**:

```
S(φ) = ifft( exp(i·k·φ) ),  k ∈ ℤ

S(φ + 2π)   = S(φ)                    exact, to 1e-16
S(a) ⊗ S(b) = S(a + b mod 2π)         binding is rotation on the circle
```

So orbiting by Δ with no new image is one bind, a full turn is the identity,
and the ±π seam is not a seam. Verified to machine precision.

`max_harmonic` sets angular resolution the way a length scale sets spatial
resolution:

| `max_harmonic` | lobe half-width | worst sidelobe |
|---|---|---|
| 6 | 21° | −0.16 |
| 8 | 16° | −0.13 |
| 12 | 11° | −0.10 |

Two details worth keeping:

- **Taper the harmonic multiplicities.** There are more dimensions than
  distinct harmonics, so each repeats. Equal repeats give a Dirichlet kernel
  ringing to −0.31; Fejér weights (multiplicity ∝ `M+1−k`) halve that to
  −0.13 for a slightly wider lobe.
- **The kernel cannot be made non-negative.** Exactly one dimension carries
  DC, so the kernel averages to `1/d` over the circle while peaking at 1, and
  must dip below zero somewhere. The taper controls how much, not whether.

**Sphere.** `view_dims=2` gives a periodic FPE over `(azimuth, elevation)`.
Formally a torus; keep elevation in `[−π/2, π/2]` and it behaves as a view
sphere. Everything below generalises with a 2-D FFT.

---

## 3. Angles: two of them, never conflate

- **`view_azimuth(obj, robot, obj_yaw)`** — measured *at the object*, from
  its own front, towards the camera. *Which side am I seeing?* This is what
  `S_view` encodes.
- **`camera_bearing(obj, robot, robot_yaw)`** — measured *at the robot*, from
  its heading, towards the object. *Where in my field of view is it?* Used to
  point a camera; **never stored**.

They differ by a half turn plus the object's yaw. Spinning the robot on the
spot sweeps the bearing through a full turn and does not move the view
azimuth at all — the object presents the same face regardless of which way
you happen to be facing. Storing the wrong one silently corrupts the book.

---

## 4. View-direction localisation — the map-localisation twin

This is the operation worth having. **Localising in a room:** unbind `ID` from
the scene map, correlate the residue against a grid of `S_allo(x)`, take the
peak — that's where you are. **Localising on an object:** correlate an
observed crop key against the object file read at every viewpoint, take the
peak — that's the direction you're looking from. Same operation, different
manifold, and *no pose input*: appearance alone fixes the angle.

Naively that's one unbind per hypothesis. It doesn't have to be. Because the
harmonics are integers, the score is a Fourier series in the angle with
integer frequencies:

```
score(φ) = ⟨ c ⊗ S(φ), V ⟩
         = (1/d) Σ_f  Ĉ_f · conj(V̂_f) · exp(i·k_f·φ)
```

Bin `Ĉ_f·conj(V̂_f)` by harmonic and take **one inverse FFT** — the entire
likelihood field over the circle, exactly. Verified identical to a 720-point
scan to 6.7e-16, and ~640× faster. On the view sphere it's a 2-D inverse FFT.

```python
loc = m.localise_view("chair_a", crop_embedding)
loc.phi       # peak: estimated viewing direction
loc.field     # the whole likelihood over the circle
loc.margin    # peak minus best competing peak outside the main lobe
```

**Return the field, not just the argmax.** Its shape is the informative part.
A symmetric object produces several equal peaks and no confidence weighting
separates them — treat it as a likelihood over viewpoint and fuse it with
odometry, the same way you'd fuse a place-field readout rather than trusting
it blind. `margin` is the honest symmetry detector.

On the turntable (6 objects, 72 azimuths, K=12 sides on file, 151-D) —
**synthetic illustration on a non-blocked split. Superseded by the blocked
re-run in §0 E4: at a 30° held-out arc these become 27.5° median overall, and
the file is shown not to extrapolate.** Kept here only for comparison:

```
chair     median  8.3°      mug      median  7.8°
L_block   median  7.8°      console  median  6.5°
pot       median 20.2°      cube     median 78.3°
```

**The cube is the metric working, not failing.** It is 90°-symmetric, so its
likelihood genuinely has four equal peaks. Any estimator that "does well" on
the cube is measuring something other than viewpoint. Keep an object like it
in your test set permanently.

---

## 5. Anisotropy: the actual result

> **Provenance: synthetic illustration.** Every figure in this section is HOG
> on a rendered turntable. It corroborates `astm/docs/RESULTS_SO_FAR.md`
> findings 1–2, which measured the same conclusion on real robot data and have
> priority. The "drop the leading PCs" recipe below does **not** transfer as
> written — see §0 E2, E3, E4.

This is where the interesting finding is. Sweep partial whitening, `α = 0`
centre-only through `α = 1` full PCA whitening, stats fitted on train
azimuths and applied to held-out ones:

| α | eff. rank | cross-object cos | stability (5° apart) | view-direction error |
|---|---|---|---|---|
| 0 centre only | 30 | −0.066 | **+0.762** | **11.0°** |
| 0.25 | 57 | −0.045 | +0.616 | 12.0° |
| 0.5 | 107 | −0.017 | +0.298 | 23.3° |
| 0.75 | 175 | −0.005 | +0.043 | 82.5° |
| 1.0 whitening | **215/215** | **−0.006** | +0.009 | **85.8°** (chance = 90°) |

Whitening delivers *everything* the naive anisotropy fix asks for — perfect
cross-object orthogonality, maximal effective rank — and destroys viewpoint
completely. **The mechanism: the viewpoint manifold is itself high-variance
structure.** Whitening cannot distinguish "anisotropy shared across objects"
(which you want gone) from "the smooth ridge that is one object turning"
(which *is* the signal), so it flattens both.

Whitening is simply too blunt. Split each PC's variance into between-object
and within-object:

```
PC0  17.8% var   83.8% between-object     <- says WHICH OBJECT
PC1  11.4%       70.5%                    <- says WHICH OBJECT
PC3   8.3%       56.0%
PC5   4.1%       56.3%
PC4   5.3%        2.2%                    <- says WHICH WAY
PC7   3.0%        0.5%                    <- says WHICH WAY
```

The identity-carrying and viewpoint-carrying directions are **largely
disjoint**. So delete the first kind and leave the rest at unit scale:

| treatment | cross-object | stability | view-direction error |
|---|---|---|---|
| centre only | −0.066 | +0.762 | 11.0° |
| **drop top 2–5 PCs** | −0.041 | +0.681 | **10.0°** |
| full whitening | −0.006 | +0.009 | 85.8° |

Better on *both* axes than centre-only. The curve is flat between k=2 and
k=12 — it is not a knife-edge.

### The recipe

```
z → subtract mean (or z-score)
  → project onto PCs fitted on a held-out sample
  → DELETE components whose between-object variance share > ~0.4
  → leave every survivor at unit scale        ← not λ^(−α/2)
  → random projection to d
  → L2 normalise
```

Two caveats I have not resolved:

- I selected by **rank**, which is only a proxy for the share. Selecting by
  share directly would keep PC4 (2.2% between-object, pure viewpoint) that
  `k=5` currently throws away. Untested; probably free improvement.
- Between-object share needs instance labels. During online mapping you have
  provisional instance IDs from data association, which is enough to
  estimate it, but it's a chicken-and-egg loop I haven't closed.

---

## 6. Consequences for `npc-av-learning2025`

> **Provenance: mixed.** The commutativity argument is exact algebra. The
> practical rotation-encoding recommendation is **pending** the 3-D handoff
> (`wiki/analysis/2026-08-17-3d-cogmaps-and-rotation-encoding.md`), which was
> not reachable from this session and may supersede it.

Three of these bear directly on that architecture.

**The contrastive objective and the view circle pull in opposite
directions.** Training embeddings to maximise same-class / minimise
cross-class similarity is an explicit gradient *towards* between-object
variance — the exact directions §5 says to delete before estimating
viewpoint. Optimise hard enough and you flatten the viewpoint manifold the
same way whitening does, and the agent loses the ability to know which side
it's looking at. The finding that the two subspaces are largely disjoint is
the good news: you can have both, but keep them separate — let the
contrastive head own the identity subspace, and build `S_view` on the
complement. Worth measuring the between-object share of your learned space
before and after contrastive training; if it rises sharply, that's the
trade being made.

**Quaternions: SO(2) is exact, SO(3) is not.** Binding in FHRR is circular
convolution, which is **commutative**, so it can only represent a commutative
group exactly. Azimuth alone is SO(2) — commutative — so `S(a)⊗S(b) =
S(a+b)` holds exactly, which is what makes the orbit and the closed-form
localiser work. Full 3-D rotation is SO(3), non-commutative, and no amount of
FPE over quaternion components fixes that: `R₁R₂ ≠ R₂R₁`, but the binding of
their codes always commutes. Three honest options:
1. **Torus approximation** — encode `(azimuth, elevation[, roll])` as
   independent circles (`view_dims=2` does the first two). Composition is
   only correct for rotations about a fixed axis order, but for a camera
   orbiting an upright object that is the actual situation.
2. **Restrict to the orbit you actually traverse** — a turntable *is* SO(2).
3. **Non-commutative VSA** if you truly need SO(3) composition, at which
   point the closed-form FFT localiser no longer applies.

Encoding quaternion components with ordinary real-phase FPE has the same
problem as encoding an angle with a real-phase FPE: no periodicity, so a full
turn doesn't return, and the code drifts instead of closing.

**DINOv2 specifics I could not test here** (egress policy blocks
`huggingface.co`, `dl.fbaipublicfiles.com` and `download.pytorch.org`; see
§8): ViT patch tokens include **high-norm outlier tokens** parked in
uninformative patches, which dominate an unweighted mean-pool. That's
anisotropy with nothing to do with viewpoint. `run_view_localisation.py`'s
`--encoder dinov2` path drops tokens above 2× the median norm before pooling;
using a registers-variant checkpoint is the cleaner fix. **Re-run the §5
sweep on real DINOv2 before trusting the numbers** — the *ordering* of the
treatments should transfer, the degrees will not.

---

## 7. What "around the shape" could mean

You were unsure between two structures; they are genuinely different and I
have only measured the first.

**Coarse — one descriptor per side** (implemented, all numbers above):

```
V = (1/K) Σₖ  c(zₖ) ⊗ S_view(φₖ)
```

**Fine — patch tokens on a surface coordinate** (not implemented):

```
V = (1/K) Σₖ Σ_p  c(z_{k,p}) ⊗ S_surf(u_p) ⊗ S_view(φₖ)
```

where `u_p` is where the patch sits on the object — patch grid position, or a
genuine surface coordinate if you have geometry. This makes the object file a
*spatial layout of parts* rather than one vector per side, which is what
"encode them around the shape" most naturally means. It should help exactly
where the coarse version fails: partial occlusion (some patches still match)
and symmetry (a handle at a known surface location breaks a tie the pooled
descriptor can't). It costs capacity linearly in patches, so `K × P` terms
bundled instead of `K` — with the `1/√terms` SNR law that is the thing to
measure first.

They layer: coarse as the index, fine underneath for verification.

---

## 8. Why the numbers here are from a rendered turntable

> **Provenance: synthetic illustration.** See §0 E1.

Every dataset and weights host is blocked by this environment's egress
policy — 403 on CONNECT, refused before TLS: `cave.cs.columbia.edu`,
`huggingface.co`, `openml.org`, `zenodo.org`, `archive.org`, `figshare`,
`kaggle`, `drive.google.com`, `dl.fbaipublicfiles.com`,
`download.pytorch.org`. Open: all of GitHub, `storage.googleapis.com`, PyPI.

So `experiments/turntable_dataset.py` renders 6 objects × 72 azimuths with a
z-buffered rasteriser — real self-occlusion, true azimuth labels, and a
deliberately symmetric object. Locally you have none of these limits: point
the loader at COIL-20/100, CRIB, or your MuJoCo renders and nothing
downstream changes. It only needs `(images, object_index, azimuth, names)`.

**Everything in §5 is HOG-on-grayscale, not DINOv2.** Grayscale on purpose —
colour would give object identity away for free and hide what the geometry is
doing. Treat the orderings as hypotheses to re-test on your own front end.

---

## 9. Minimal local implementation

If you want it without the repo:

```python
import numpy as np

def circular_ssp_space(d=151, max_harmonic=8):
    """Integer harmonics, Fejér-tapered, conjugate-symmetric."""
    n = (d - 1) // 2
    base = np.arange(1, max_harmonic + 1)
    w = (max_harmonic + 1 - base).astype(float)
    counts = np.floor(w / w.sum() * (n - len(base))).astype(int) + 1
    counts[np.argsort(w)[::-1][:n - counts.sum()]] += 1
    k = np.repeat(base, counts)                       # (n,)
    phases = np.concatenate([[0], k, -k[::-1]])       # (d,) conj-symmetric
    return phases

def S_view(phases, phi):
    return np.fft.ifft(np.exp(1j * phases * phi)).real

bind   = lambda a, b: np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real
unbind = lambda a, b: np.fft.ifft(np.fft.fft(a) / np.fft.fft(b)).real

def object_file(keys, angles, phases):
    return np.mean([bind(c, S_view(phases, a)) for c, a in zip(keys, angles)], 0)

def view_likelihood(book, key, phases, n=720):
    """Whole likelihood over the view circle in one inverse FFT."""
    k = np.rint(phases).astype(int)
    A = np.fft.fft(key) * np.conj(np.fft.fft(book)) * (-1.0) ** k
    w = np.zeros(n, complex)
    np.add.at(w, k % n, A)                            # bin by harmonic
    field = np.fft.ifft(w).real * n / len(book)
    return np.linspace(-np.pi, np.pi, n, endpoint=False), field
```

`unbind` by division is fine here because `S_view` is unitary; in the library
it's the involution, which is the same thing for unitary vectors and stabler
for everything else.

---

## 10. Running what's in the repo

```bash
pip install numpy scipy matplotlib scikit-image        # nengo not needed
python experiments/test_object_map.py                  # 38 checks
python experiments/run_view_localisation.py --plot     # §4 and §5, ~8 s
python experiments/run_object_map.py --plot            # §1, the full map
python experiments/run_nn_baseline.py                  # §16 E0, ~6 min
python experiments/run_mirror_stage.py --symmetric-set # §16 E1, ~9 min
python experiments/run_residue_code.py                 # §16 E2, ~20 min
python experiments/run_k_curve.py --symmetric-set      # §16 E4, ~15 min
```

Swap the front end with `--encoder dinov2` (needs torch + transformers +
weights), or replace `load_turntable()` with a COIL/CRIB/MuJoCo loader.

Key entry points:

| what | where |
|---|---|
| periodic view FPE, closed-form likelihood | `sspslam/objectmap/viewspace.py` |
| crop embedding → key, standardisation | `sspslam/objectmap/appearance.py` |
| per-object view book, angular merging | `sspslam/objectmap/objectfile.py` |
| scene map, write path, all read-outs | `sspslam/objectmap/scenemap.py` |
| view azimuth vs camera bearing | `sspslam/objectmap/geometry.py` |

---

## 11. Open, in rough order of value

**Superseded by §16**, which reorders this list against the literature and
against E0's result. Kept for the front-end items 1–4, which §16 does not cover.

Items 1–2 are now partly forced by §0: the recipe in §5 is untested against a
real spectrum, and the split needs redoing before any degree figure is quoted.

1. Select PCs by **between-object share** rather than rank (§5) — cheap, and
   probably free accuracy.
2. Re-run §5 on **real DINOv2**; check whether high-norm patch tokens change
   the picture, and whether the between-object/viewpoint subspaces are as
   cleanly separated in a learned space as in HOG.
3. Measure the between-object share of the **contrastively trained** space,
   before and after training (§6). This is the experiment that says whether
   the two objectives are actually in conflict in your setup.
4. **Patch tokens on a surface coordinate** (§7), scored on occlusion and
   symmetry, where the pooled version should fail.
5. ~~Close the **loop**~~ — **done**, see §12. Feeding the likelihood into a
   Bayes filter on the circle takes per-frame 17.0° to 6.0°, and to 4.0° with
   one known starting direction. Note the position version already exists — §0 E6,
   `RESULTS_SO_FAR.md` finding 6, where predict is one bind and update is one
   bundle. What is new here is the *view-direction* likelihood as the update
   term, which is the point at which "I know which side I'm seeing" becomes
   part of SLAM rather than an offline query.
6. **`K` selection.** Currently merging by a fixed angular tolerance. Ought to
   be driven by the view kernel's lobe width and by how fast appearance
   actually changes — objects with a discontinuity (a handle appearing) need
   sides packed more tightly there than on a smooth flank.


---

## 12. Continuity: a Bayes filter on the view circle

Decoded frame by frame, the estimate jumps 50–180° between consecutive crops.
A camera orbiting an object cannot do that. Both steps of the fix are native to
the representation:

```
predict   B_k <- B_k · exp(-i k Δ) · exp(-σ²k²/2)      shift, then blur
update    b   <- b · softmax_β(likelihood)
```

The first factor of the predict step **is** binding by `S_view(Δ)` — the same
operation as `orbit()`. The second is per-harmonic damping. This is the
view-circle twin of `RESULTS_SO_FAR.md` finding 6.

`experiments/run_view_tracking.py`, one full orbit, 72 frames 5° apart, object
files holding kept views only (30° held-out arcs), σ=2°/frame, β=3, odometry
noise 1.5°. A jump counts as impossible above 15°, three times the true step.

| read-out | median error | largest jump | impossible jumps |
|---|---|---|---|
| per frame, no memory | 17.0° | 180° | 89 / 426 |
| + Bayes filter, free start | **6.0°** | 175° | 25 / 426 |
| + one known starting direction | **4.0°** | 42° | 11 / 426 |

**Continuity fixes the coverage failure and not the aliasing one.** The filter
coasts across held-out arcs on prediction instead of guessing, which is where
most of the error went. But four equal branches give it nothing to choose
between:

| cube | median error | largest jump | impossible |
|---|---|---|---|
| per frame | 83.0° | 180° | 27 |
| + filter, free start | 87.0° | 175° | 10 |
| + known start | **2.0°** | 11° | **0** |

That is the useful result: **continuity converts a per-frame ambiguity into a
single global one.** A symmetric object costs one piece of information for the
whole trajectory rather than one per frame — fix the branch once at the start
and it stays fixed.

Two implementation notes. A belief with two near-equal modes has a MAP that
flips on noise alone, showing up as a 180° jump the belief never made; the
read-out prefers the mode nearest the previous estimate unless another beats it
by a margin. And β matters more than σ — at β=50 the appearance evidence
overwhelms the prior and the filter buys almost nothing (45 impossible jumps);
at β=3 it binds.

*Provenance: synthetic illustration, same rendered turntable and HOG front end
as §5. The algebraic identity in the predict step is exact.*


---

## 13. What the front end alone determines

Two things get called anisotropy. They have different causes, and only one of
them is the encoder's fault.

**Between-object** — the leading principal directions say *which object this
is*: PC0 83.8% between-object variance, PC1 70.5%. Removable, and §5 removes
them.

**Across-view** — seven views spanning 30° of orbit occupy about **3.8
effective dimensions** out of 1764. That number is **identical before and
after conditioning** (3.8 both ways, per object 3.2–4.0). It is not an encoder
pathology: it is what a continuous trajectory produces, for any encoder. It is
the same phenomenon as scene redundancy in a video, where frames close in time
are close in pose and therefore close in appearance.

### The two are linked, but not the way you would guess

An object's identity does not change as you walk around it. So the
identity-carrying directions are **constant along the orbit**, and act as a DC
pedestal on the appearance autocorrelation, lifting every lag equally.
Removing them does not create angular information — it unmasks what was there:

| | appearance half-width | N_eff of 72 | local rank |
|---|---|---|---|
| raw HOG, centred | 28° | 2.8 | 3.8 |
| top 2 PCs dropped | **18°** | **6.2** | 3.8 |
| *view kernel, for comparison* | *16°* | | |

**Conditioning buys contrast, not information.** Half-width nearly halves and
effective sample size more than doubles, while the intrinsic local
dimensionality does not move at all.

### Three numbers, no VSA required

All computable from the crops alone, before any object file exists:

| | what it is | what it predicts |
|---|---|---|
| **half-width** | lag at which similarity falls to half | the finest angular distinction the descriptor supports. If it exceeds the view kernel, extra `max_harmonic` is wasted |
| **alias peak** | highest similarity at a **circular** lag beyond the kernel | aliasing. cube **0.910 at 90°**, next highest 0.581 — the four-fold symmetry, visible in the descriptor before any map is built |
| **N_eff** | independent views per orbit | the sample size to use in any statistic |

**The alias peak must be measured in circular distance.** Scanning linear lag
on a 72-view ring treats 355° as far away when it is the adjacent frame; done
that way, five of six objects report their own neighbour as an alias and the
diagnostic is worthless. That bug was in the first version of this analysis.

### Consequences

1. **The encoder sets the achievable angular resolution.** The VSA kernel can
   only be as sharp as the descriptor's autocorrelation. At 18° against a 16°
   kernel these happen to be matched here; for the `pot` (55°) the kernel's
   resolution is unusable and no VSA change recovers it.
2. **Aliasing is screenable without building anything.** One autocorrelation
   per object tells you which instances will be unlocalisable, and at what
   offset — which is the information the continuity filter needs an anchor for
   (§12).
3. **Sample by decorrelation length, not by frame.** ~6 independent views per
   orbit is the number that should size a split, a bootstrap, or a coverage
   claim. This is where the blocked split of E4 comes from, derived rather
   than imposed.
4. **This is how to compare encoders cheaply.** Half-width, alias peak and
   N_eff rank a front end for this task in seconds, with no map, no split and
   no decode — which is the practical way to evaluate DINOv2 against HOG when
   the weights become reachable.

*Provenance: synthetic illustration — rendered turntable, HOG front end. The
circular-distance requirement and the DC-pedestal argument are structural and
do not depend on the dataset; the degrees do.*


---

## 14. Formalising it: the identity pedestal

§13 observed that conditioning nearly halves the appearance correlation length
without changing the local dimensionality. That has an exact form, and it is
worth writing down because three separate quantities turn out to be the same
effect seen from different angles.

### The decomposition

For object *o* at viewing angle φ, split the crop embedding into a global mean,
a part constant along the orbit, and a part that actually varies:

```
z_o(φ) = μ + a_o + v_o(φ),        ⟨v_o(φ)⟩_φ = 0
```

`a_o` is the object's identity — it does not change as you walk around it.
Define the **identity fraction**

```
λ_o = ‖a_o‖² / ( ‖a_o‖² + ⟨‖v_o(φ)‖²⟩_φ )      ∈ [0, 1]
```

Since ⟨v_o⟩ = 0 the cross terms vanish on average, and the autocorrelation of
the L2-normalised key separates:

> **Law 1.**  ρ_o(Δ) = λ_o + (1 − λ_o)·r_o(Δ)

where `r_o` is the autocorrelation of the view-varying part alone. This is an
affine contrast map, and it has **no free parameters** — λ is measured, not
fitted.

**Tested** (`experiments/run_pedestal_model.py`), pointwise over all lags:

| object | λ | half-width of ρ | half-width of r | max &#124;ρ − model&#124; | RMS |
|---|---|---|---|---|---|
| L_block | 0.115 | 25° | 20° | 0.022 | 0.012 |
| console | 0.133 | 20° | 15° | 0.015 | 0.008 |
| cube | 0.366 | 20° | 10° | 0.004 | 0.002 |
| chair | 0.378 | 30° | 15° | 0.005 | 0.002 |
| mug | 0.472 | 55° | 45° | 0.032 | 0.016 |
| pot | **0.746** | 180° | 55° | 0.011 | 0.006 |

### Three corollaries, one mechanism

Everything §13 reported follows from Law 1 rather than being separate findings:

**(a) Contrast.** The alias margin is `ρ(0) − ρ(Δ) = (1 − λ)(1 − r(Δ))`. An
object whose appearance is largely view-invariant has *every* margin
compressed toward zero even when `r` is perfectly informative. The `pot` at
λ = 0.746 keeps only a quarter of its available contrast.

**(b) Half-width.** ρ reaches ½ where r reaches `(½ − λ)/(1 − λ) < ½`, which
occurs at a larger lag. The pedestal **inflates** the measured half-width —
the 28° → 18° of §13.

**(c) Effective sample size.** With `τ = 1 + 2Σ_{k≤m} ρ_k`, Law 1 gives

```
τ = 1 + 2mλ + 2(1 − λ)Σ r_k
```

The `2mλ` term grows with the summation window and has nothing to do with view
correlation. So **N_eff computed on unconditioned keys is not well defined** —
it depends on how many lags you chose to sum. This is why N_eff moved 2.8 → 6.2
in §13, and it means the E8 correction should always be computed after removing
the pedestal.

**So removing the identity component is a contrast stretch of gain 1/(1 − λ).**
It reorders nothing and creates no information — but it rescales the
half-width, the alias margin and the effective sample size, and all three are
used to size design decisions. That is the precise sense in which conditioning
"buys contrast, not information".

### A conjecture that failed

The natural next step was that angular resolution should be the convolution of
the appearance width and the view kernel, `√(w_app² + w_kern²)`, so sharpening
the kernel past `w_app` ≈ 18° would buy nothing. Swept over `max_harmonic`,
five random projections each, at a 30° held-out gap:

| `max_harmonic` | kernel half-width | median error |
|---|---|---|
| 3 | 34° | **25.6° ± 1.7** |
| 4 | 28° | 27.7° ± 5.6 |
| 6 | 20° | 30.1° ± 1.9 |
| 8 | 16° | 30.4° ± 2.7 |
| 12 | 11° | 29.3° ± 2.6 |
| 16 | 8° | 33.3° ± 1.5 |
| 24 | 6° | **33.0° ± 4.7** |

Spread 7.7° against a seed sd of 2.9° — real, at 2.6×, but with **the opposite
sign to the prediction**. Error *rises* as the kernel sharpens.

The conjecture is refuted, and the reason corrects §0 E4's phrasing. At sparse
coverage the binding constraint is not resolution but **reach**: a broad kernel
spans the held-out arc, a sharp one falls into it. So the rule is

> **match the view kernel to the stored-view spacing, not to the appearance
> width** — and never sharpen past coverage, because the resolution gained
> cannot be used and the reach lost is immediate.

E4 stated the same relation from the other side ("store sides no wider apart
than the kernel half-width"); the two together say the two quantities should be
*matched*, and that whichever you cannot change should set the other.

*Provenance: Law 1 and its three corollaries are algebra, tested here to
RMS ≤ 0.016 on the rendered turntable; the decomposition and the corollaries do
not depend on the dataset. The kernel sweep is synthetic illustration, five
seeds, one split.*


---

## 15. Where this sits in the literature

Searched 2026-08-17. Every reference below is marked **[v]** if the citation was
checked against a source in that session, or **[m]** if it is from memory and
still needs verifying before it goes in a paper. Do not cite an **[m]** without
checking it.

### The VSA lineage

| work | what it gives us |
|---|---|
| Plate, *Holographic Reduced Representations*, IEEE TNN 1995 **[m]** | bind and bundle |
| Kanerva, *Hyperdimensional computing*, Cogn. Comput. 2009 **[m]** | the framing |
| [Frady, Kleyko, Kymn, Olshausen & Sommer, *Computing on Functions Using Randomized Vector Representations*, arXiv:2109.03429, 2021](https://arxiv.org/abs/2109.03429) **[v]** | **the one that matters most here.** FPE as a unitary representation of an abelian group, and *kernel design as the explicit knob*. Our integer-harmonic circle is a special case of their framework — the compact one they set up but do not pursue for viewpoint |
| [Dumont & Eliasmith, *Exploiting semantic information in a spiking neural SLAM system*, Front. Neurosci. 2023](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1190515/full) **[v]** | the direct parent — this repo. Constrains itself to quantities known in hippocampus: head direction, **object vector cells**, place, grid |
| [Kymn, Mazelet, Thomas, Kleyko, Frady, Sommer & Olshausen, *Binding in hippocampal-entorhinal circuits enables compositionality in cognitive maps*, NeurIPS 2024, arXiv:2406.18808](https://arxiv.org/abs/2406.18808) **[v]** | current state of the art on the VSA–brain link. Position in a **residue number system**, residues as complex vectors, modular attractor whose modules map to grid modules. Does *space*, not objects |
| [Renner et al., *Visual odometry with neuromorphic resonator networks*, Nature MI 2024](https://www.nature.com/articles/s42256-024-00846-2) and [*Neuromorphic visual scene understanding with resonator networks*](https://www.nature.com/articles/s42256-024-00848-0) **[v]** | the closest existing "rotation as binding". Log-polar coordinates *so that binding becomes equivariant to rotation and scale* — our trick, one manifold over: image-plane rotation, not object viewpoint |
| [Neubert & Schubert, *An Introduction to Hyperdimensional Computing for Robotics*, KI 2019](https://link.springer.com/article/10.1007/s13218-019-00623-z); [CVPR 2021 descriptor aggregation](https://openaccess.thecvf.com/content/CVPR2021/papers/Neubert_Hyperdimensional_Computing_as_a_Framework_for_Systematic_Aggregation_of_Image_Descriptors_CVPR_2021_paper.pdf) **[v]** | VSA in robotics. Read carefully: they do viewpoint-**invariant** recognition. They discard viewpoint; we estimate it |
| [HyperSpace, arXiv:2604.15113, 2026](https://arxiv.org/abs/2604.15113) **[v]** | benchmarks HRR vs FHRR on continuous spatial domains and finds **similarity and cleanup dominate runtime**, not binding. §4's closed form removes exactly that cost — this is the citation for why it is worth having |

**The gap.** Nothing found stores, per object instance, a *periodic view manifold
with appearance as values on it*, read out by unbinding to a viewpoint
likelihood. Every ingredient is published; the combination is not.

**That is a novelty claim, not a superiority claim.** §16 measures the object
file against the obvious non-VSA alternative — a list of stored views and a
nearest-neighbour — and it loses on accuracy at the dimension used here. What it
wins is memory, by a factor of `n_obj × K`. Do not cite the gap as evidence the
method is better.

### The neuroscience — four legs

**1. Object vector cells.** [Høydal, Skytøen, Andersson, Moser & Moser, *Nature*
568:400, 2019](https://www.nature.com/articles/s41586-019-1077-7) **[v]**. MEC
neurons firing at a specific **distance and direction from an object**,
generalising across objects and environments. Nearly `ID ⊗ S_allo` with a vector
to the object. Best citation for the scene-map half — and already cited by the
parent SSP-SLAM paper.

**2. Spatial view cells.** [Rolls et al., *Cereb. Cortex* 9(3):197,
1999](https://academic.oup.com/cercor/article/9/3/197/428888), reviewed in
[*Hippocampus* 2024](https://onlinelibrary.wiley.com/doi/full/10.1002/hipo.23666)
**[v]**. Primate hippocampal neurons coding the allocentric location **being
looked at**, not where the animal is. The information split is stark: **0.47
bits about spatial view, against 0.017 for eye position, 0.005 for head
direction, 0.033 for place.** Direct support for §3 — the viewing relation is
its own quantity, kept separate from self-pose.

**3. View-tuned neurons in IT.** Predicted by Poggio & Edelman, *Nature* 343:263,
1990 **[m]**; found by Logothetis, Pauls & Poggio in monkeys trained on paperclip
objects **[m]** (both referenced by [*3D Object Recognition: A Model of
View-Tuned Neurons*, NIPS
1997](https://papers.nips.cc/paper/1296-3d-object-recognition-a-model-of-view-tuned-neurons)
**[v]**). Objects stored as **a set of views**, novel views handled by
**interpolating across them**. This is the object file, found in 1995 — and our
measured signature (§0 E4: interpolates between stored views, does not
extrapolate past them) is the same behaviour.

**4. Mirror-symmetric view tuning — the one that vindicates the cube.**
Freiwald & Tsao, *Science* 2010 **[m]**, whose ML/MF → AL → AM hierarchy is
quoted and replicated in [Farzmahdi et al., *eLife* 13:e90256,
2024](https://elifesciences.org/articles/90256) **[v]**: **view-specific →
mirror-symmetric → view-invariant.** AL neurons genuinely cannot tell a left
profile from a right one. That is aliasing, in a brain, as a *designed
intermediate stage*. Farzmahdi et al. further show the same tuning emerges
spontaneously in CNNs trained on symmetric object categories — so it is a
property of the data, not the architecture.

**§13's cube result is therefore not a defect we failed to engineer away. It is
what the primate visual system also does, for the same reason.**

That much survives testing. What does **not** survive is the next step —
building the stage deliberately. §16 E1 does exactly that and finds it makes
identification worse, equally for symmetric and chiral objects. Emergent in a
trained network is not the same as useful in a designed one.

### Where the biology does *not* back us

| our choice | status |
|---|---|
| integer-harmonic circle | **engineering convenience.** Grid modules are multi-scale, not integer-harmonic. Cite Frady for the maths, not biology |
| one-FFT likelihood (§4) | **no biological claim.** An algorithmic shortcut |
| Bayes filter on the circle (§12) | **arguably supported** — the head-direction ring attractor with angular-velocity input *is* a circular filter with a velocity-driven predict step |
| SO(3) composition (§6) | **nothing.** No evidence any brain composes 3-D rotations this way. Leave it alone |

One honest wrinkle: object vector cells encode *where the object is relative to
me*, not *which face of it I am seeing*. The view circle is closer to IT view
tuning than to entorhinal vector coding. **The contribution is the stitch** —
MEC-style vector coding for *where*, IT-style view tuning for *what it looks
like from here*, in one algebra. No single cell type does both.


---

## 16. What to run next, and what E0 found

The §15 literature review reorders the queue. E0 and E1 have been run; E0
because it could invalidate the premise, E1 because it was the only one making
a prediction that could fail. Both came back against us, which is the point of
having run them.

| | experiment | what it decides | status |
|---|---|---|---|
| **E0** | list-of-views baseline | is the object file a better estimator, or only a smaller one? | **done — it is only smaller** |
| **E1** | mirror-symmetric intermediate (Freiwald & Tsao) | does deliberate aliasing *improve* identification, as it does in AL? | **done — refuted. It makes identification worse, in both symmetry regimes** |
| **E2** | residue view code (Kymn 2024) | can a modular code hold K views at a fraction of the dimension? | **done — refuted. But it found the capacity knob E0 missed, and that ties E0's score** |
| **E3** | object-vector-cell scene map (Høydal) | should the scene half be allocentric or egocentric-vector? | spec below |
| **E4** | the K curve (Logothetis / Poggio–Edelman) | where is the knee, and does it match IT view tuning? | **done — knee at K=12, store a side every 30°. The per-object prediction is refuted** |

**Naming.** These are §16 E0–E4, the *experiments*. §0 E1–E8 are the *errata*.
The document always writes the prefix; a bare "E4" in this section means the K
curve, and in §0 means the blocked-split correction.

DINOv2 still gates every absolute degree figure in all five (`huggingface.co`
blocked — §8). All five compare decoders on one front end, so their *relative*
conclusions survive the swap; their degrees do not.

---

### E0 — does unbinding actually beat a list of stored views?

> **Corrected by E2.** Everything below was measured at `max_harmonic=8` and
> `16`, the repo defaults. E2 found that `max_harmonic` is the capacity knob,
> and at `k_max=4` the object file **ties** the list at every K rather than
> losing to it — `vsa − nearest` = −1.0° [−6.5, +6.0] at K=12. The conclusion
> "it is a smaller store, not a better estimator" survives; the conclusion
> "it is *worse*" does not. Read this section for the method and the capacity
> mechanism, and E2 for the corrected scoreline.

**Short answer: at the dimension this repo has been measured at, no — it loses,
badly.** At `d=151` with `K=12` views on file, nearest-neighbour over the same
conditioned keys gets **10.0°** median in-gap error and the object file gets
**25.5°**. The whole-scene vector gets 76.8°. §15 claimed a gap in the
literature; this section is the reason that claim has to be stated as *nobody
has done it* rather than *it is better*.

**Longer answer: the loss is a capacity limit, not a dead end.** The gap closes
with dimension, and what the object file actually buys is memory, not accuracy.
At `d=4801` the entire map — six objects, twelve views each — sits in **one
4801-float vector** and decodes to 13.5°, against 345,672 floats for the list at
10.0°. That is **72× less memory for 3.5° more error**, and there is no
list-based method with an analogue at that budget.

Run it: `python experiments/run_nn_baseline.py`.

### Why this experiment exists

Neubert & Schubert (§15) do viewpoint-*invariant* recognition by bundling
descriptors, with no unbinding anywhere. Nothing in §§4–14 had ever been
compared against the obvious non-VSA thing: keep the K conditioned descriptors
in a list, take the nearest. Every claim in this document was unfalsified rather
than validated.

Four decoders, identical inputs — same crops, same conditioning (`drop=2`), same
30° blocked arcs, same K views on file:

| decoder | what it does | can it interpolate? |
|---|---|---|
| `nearest` | cosine to each stored key, report that key's angle | no — it quantises to the store |
| `kernel` | circular mean of stored angles, softmax-weighted by cosine, temperature fitted leave-one-out **on the store** | yes |
| `vsa` | §4's one-FFT likelihood over the object file | yes |
| `vsa-scene` | all six object files superposed into ONE vector via `bind(ID_o, book_o)`, unbound at query time | yes |

### [A] Equal store, `d=151` — the repo's own configuration

| K | decoder | median err | <15° | ID hit | map floats |
|---|---|---|---|---|---|
| 6 | nearest | 20.0° | 0.37 | 0.89 | 5,436 |
| 6 | kernel | 17.2° | 0.44 | 0.89 | 5,436 |
| 6 | vsa | 23.0° | 0.36 | 0.63 | 906 |
| 12 | **nearest** | **10.0°** | 0.63 | 0.97 | 10,872 |
| 12 | kernel | 10.0° | 0.65 | 0.97 | 10,872 |
| 12 | **vsa** | **25.5°** | 0.29 | 0.55 | 906 |
| 24 | nearest | 15.0° | 0.60 | 0.98 | 21,744 |
| 24 | vsa | 29.0° | 0.16 | 0.64 | 906 |
| 36 | nearest | 15.0° | 0.58 | 0.98 | 32,616 |
| 36 | vsa | 30.5° | 0.12 | 0.63 | 906 |

Read the two columns against each other and the mechanism is obvious: **the
list gets better with more stored views and the object file gets worse.**
23.0 → 25.5 → 29.0 → 30.5 as K goes 6 → 36. That is not a property of
unbinding, it is bundle capacity — K appearance keys, mutually correlated by
the §14 identity pedestal, superposed into one 151-D vector. Adding a view adds
more crosstalk than signal.

Identification degrades the same way: 0.97 for the list against 0.55–0.64 for
the object file.

Hierarchical bootstrap over objects → arcs → seeds (2000 draws, 3 seeds,
6 arcs), difference of medians `vsa − nearest`:

| K | diff of medians | 95% CI |
|---|---|---|
| 6 | +3.0° | [−3.5, +17.5] |
| 12 | **+15.5°** | **[+6.0, +64.5]** |
| 24 | +14.0° | [+0.0, +63.0] |
| 36 | +15.5° | [−6.0, +64.5] |

At K=6 it is a tie. At K=12 the interval excludes zero and the object file is
worse. No frame was treated as a sample anywhere in that.

**A note on the statistic.** These are `median(vsa) − median(nearest)` on each
resample, not `median(vsa − nearest)`. The first version of this table used the
latter and reported +8.5° [+1.2, +12.0] at K=12 — a different quantity, and one
that does not match the +15.5° the [A] table implies. The median of a per-crop
difference is pinned near zero whenever two decoders agree on most crops, which
is exactly what happens when they share a stage; §16 E1's end-to-end contrast
degenerated to a hard `+0.000 [+0.000, +0.000]` before the fix. The intervals
here are correspondingly wider, because a difference of medians carries more
variance than a median of differences. The conclusion does not change; the
number quoted for it does.

### [D] Dimension sweep — where the real answer is

`d=151` is where §§4–13 were measured, and it is also where the bundle runs out
of room. Sweeping it, at K=12:

| d | k_max | per-object | scene vector | list floats | scene floats |
|---|---|---|---|---|---|
| 151 | 16 | 25.5° | 67.0° | 10,872 | 151 |
| 301 | 16 | 15.0° | 47.0° | 21,672 | 301 |
| 601 | 16 | 13.5° | 38.0° | 43,272 | 601 |
| 1201 | 16 | 13.0° | 19.5° | 86,472 | 1,201 |
| 2401 | 16 | 11.0° | 15.5° | 172,872 | 2,401 |
| 4801 | 16 | **10.5°** | **13.5°** | 345,672 | **4,801** |

Three things fall out.

1. **The object file needs ~16× the dimension to match the list on accuracy.**
   10.5° at `d=4801` against the list's 10.0°. It never clearly wins on error.
2. **It wins on memory by a factor of K, and the scene vector by a factor of
   `n_obj × K`.** The list must keep `n_obj × K × d` floats; the whole-scene
   vector is `d`. At the matched-accuracy point that is 345,672 against 4,801.
3. **The scene vector needs `d ≳ 1200` to work at all.** Below that, six
   objects × twelve views do not fit and it returns near-chance. This is the
   capacity curve for superposing a whole map, measured rather than assumed.

`k_max=16` beats `k_max=8` at every dimension above 151 — the 16° lobe of the
default is coarser than the 18° descriptor half-width of §13 suggested was
needed, and sharpening it is free once there is dimension to spend.

### [C] Per object, K=12, d=151

| object | nearest | kernel | vsa | vsa-scene |
|---|---|---|---|---|
| chair | 10.0° | 10.0° | 22.2° | 76.5° |
| mug | 10.0° | 10.0° | 22.7° | 52.0° |
| **cube** | **90.0°** | **90.0°** | **90.0°** | **89.8°** |
| L_block | 10.0° | 10.0° | 23.8° | 71.0° |
| pot | 10.0° | 10.0° | 26.3° | 86.7° |
| console | 10.0° | 10.0° | 15.0° | 62.8° |

The cube is at chance for **every** decoder, including the two that have no VSA
in them. That is the control working: aliasing is a property of the object and
the descriptor, not of the representation (§13), and any decoder that looked
good on the cube would be reading something other than pose.

### What this changes

- **§15's gap claim stands as novelty, not as superiority.** The combination is
  unpublished; it is not better, though E2 shows it is not worse either once
  `max_harmonic` is set properly. Anywhere this document implies the object
  file is a better *estimator*, it is wrong — it is a better *store*.
- **The honest pitch is compression and superposition.** One vector for a whole
  scene, queried by unbinding an ID, at 72× less memory than the list. That is
  worth having and has no list-based analogue. Accuracy parity is the price.
- **`d=151` is too small and every earlier number is affected.** §§4–13 were all
  measured there. The absolute degrees in those sections are pessimistic for the
  object file by roughly a factor of two; the *relative* conclusions (which PCs
  to drop, aliasing, the pedestal law) are unaffected because they are properties
  of the descriptor, not the code.
- **Bundle capacity is now the first-order constraint**, which is exactly the
  problem residue number systems are for (§15, Kymn 2024) — see E2 below. That
  experiment was ranked last on the assumption that the descriptor was the
  binding limit. It is not, at K ≥ 12.

*Provenance: measured — rendered turntable, HOG front end, 3 seeds, 30° blocked
arcs, hierarchical bootstrap over objects → arcs → seeds. The degrees are HOG
numbers and would move with a different encoder; the K-dependence, the capacity
curve and the direction of every comparison are structural.*

---

### E1 — mirror-symmetric intermediate: **refuted**

**The prediction failed, and so did the rescue.** Bundling each view with its
mirror image does not improve identification — it makes it **worse**, by
−0.148 [−0.261, −0.058] at `d=2401`, and the interval excludes zero at two of
three dimensions. The falsifier stated in the spec fired. This subsection keeps
the failed conjecture on the page.

Run it: `python experiments/run_mirror_stage.py --symmetric-set`.

#### What was built

Freiwald & Tsao's hierarchy (§15 leg 4) is ML/MF → **AL** → AM: view-specific,
then mirror-symmetric, then view-invariant. Each stage is one line here:

```
book_ML = (1/K) Σ_k  c(z_k) ⊗ S_view(φ_k)                        view-specific
book_AL = (1/K) Σ_k  c(z_k) ⊗ [S_view(φ_k) + S_view(2μ − φ_k)]   mirror-symmetric
book_AM = (1/K) Σ_k  c(z_k)                                      view-invariant
```

`μ` is the mirror axis; reflecting about it sends φ → 2μ − φ. AM is the honest
end of the sequence — bundle the appearance, bind no angle at all. The measured
prediction was: identification improves along ML → AL → AM, pose degrades along
it, and AL degrades *specifically*, keeping distance from the mirror axis and
losing only the sign.

Ten objects (the six-object set plus four built symmetric on purpose — see
below), K=12, 30° blocked arcs, three seeds.

#### The mechanism works. It just doesn't help.

| d | stage | ID hit | pose err | unsigned err | sign acc |
|---|---|---|---|---|---|
| 151 | ML | 0.44 | 35.0° | 24.5° | 0.61 |
| 151 | AL | 0.36 | 73.8° | 24.0° | **0.50** |
| 151 | **AM** | **0.89** | — | — | — |
| 601 | ML | 0.84 | 16.5° | 15.0° | 0.67 |
| 601 | AL | 0.67 | 62.0° | 17.5° | **0.51** |
| 601 | **AM** | **0.90** | — | — | — |
| 2401 | ML | **0.94** | 14.5° | 14.0° | 0.69 |
| 2401 | AL | 0.79 | 64.2° | 15.0° | **0.50** |
| 2401 | AM | 0.91 | — | — | — |

*Unsigned err* is the error in |φ − μ|, the distance from the mirror axis.
*Sign acc* is how often the estimate lands on the correct side of it.

**Read the last two columns first: the AL code is correct.** Signed pose
collapses (14.5° → 64.2°) while unsigned pose barely moves (14.0° → 15.0°), and
sign accuracy sits at exactly 0.50. That is a mirror-symmetric representation
doing precisely what one is supposed to do — it keeps how far round you are and
throws away which way. The axis sweep confirms it is mirroring and not an
artefact of where the turntable starts: AL identification is 0.79–0.81 for
mirror axes at 0°, 30°, 60° and 90°, while ML sits flat at 0.94 as the control.

So this is not an implementation failure. The stage is real and it costs
identification anyway.

#### The rescue hypothesis, and why it also fails

The first run used the six-object set, where the cube was the *only* object
whose identification improved under mirroring (+0.13). That looked like
Farzmahdi et al.'s actual claim — mirror tuning emerges from training on
**symmetric** categories, and faces are near-symmetric, so left/right there is
nuisance variation rather than signal. Most objects on this turntable are
chiral, where left/right *is* the signal. On that reading AL was being tested
outside the regime it was proposed for.

Testing it needs more than one symmetric object, so `turntable_dataset.py`
gained an opt-in set of four built symmetric by construction — `bar` (2-fold),
`cross` (4-fold), `drum` (16-fold), `tripod` (3-fold), flat-painted so the
rasteriser's `abs(n·light)` shading preserves the geometry's symmetry. It is
off by default and no previously published figure moves.

Objects are then sorted by their **measured** §13 alias peak, not by intent:

| object | alias peak | at lag | group | ML ID | AL ID | AL − ML |
|---|---|---|---|---|---|---|
| chair | 0.527 | 20° | chiral | 0.99 | 1.00 | +0.01 |
| mug | 0.680 | 20° | chiral | 1.00 | 0.89 | −0.11 |
| cube | 0.924 | 90° | **symmetric** | 0.83 | 0.78 | −0.06 |
| L_block | 0.521 | 20° | chiral | 0.93 | 0.66 | −0.27 |
| pot | 0.673 | 20° | chiral | 0.97 | 0.98 | +0.01 |
| console | 0.363 | 20° | chiral | 0.98 | 0.79 | −0.19 |
| bar | **1.000** | 180° | **symmetric** | 1.00 | 0.67 | −0.33 |
| cross | 0.997 | 180° | **symmetric** | 0.93 | 0.77 | −0.16 |
| drum | **1.000** | 180° | **symmetric** | 0.83 | 0.78 | −0.06 |
| tripod | 0.565 | 120° | chiral | 0.95 | 0.63 | −0.32 |

| group | AL − ML identification | 95% CI |
|---|---|---|
| symmetric (4 objects) | **−0.150** | [−0.306, −0.021] |
| chiral (6 objects) | **−0.147** | [−0.315, −0.028] |

**Identical, and both negative.** Mirroring does not hurt chiral objects and
help symmetric ones; it hurts both, by the same amount. The regime argument is
dead.

Note also that the cube's +0.13 in the six-object set became **−0.06** once
four more objects were added. It was never evidence — it was a small-set
artefact of which distractors happened to be present. The single datum that
looked like a confirmation did not survive more data, which is the whole reason
for the alias-peak stratification rather than eyeballing one object.

#### What did earn its place: AM, and not where expected

| d | pipeline | ID hit | end-to-end pose |
|---|---|---|---|
| 151 | ML→ML | 0.44 | 90.0° |
| 151 | AL→ML | 0.36 | 90.0° |
| 151 | **AM→ML** | **0.89** | **64.8°** |
| 601 | ML→ML | 0.84 | 19.5° |
| 601 | **AM→ML** | **0.90** | 20.0° |
| 2401 | ML→ML | **0.94** | **15.0°** |
| 2401 | AM→ML | 0.91 | 16.8° |

*(a wrong object scores 90°, so this is the number a robot lives with, not a
pose figure conditioned on already knowing what it is looking at)*

**AM identifies at 0.89–0.91 and does not care about dimension.** It has no
binding in it, so it has no capacity pressure — the §16 E0 constraint simply
does not apply. ML needs `d=2401` to catch it up. At `d=151`, where §§4–13 were
measured, using an unbound prototype to name the object and the view book only
to read the side takes identification from **0.44 to 0.89** and end-to-end pose
from 90° (chance) to 64.8°, for one extra `d`-vector per object.

That is the actionable finding, and it is not the one the biology predicted:
**two vectors per object, not three stages.** An unbound appearance prototype
for *what*, a view-bound book for *which way*. The middle stage is the one to
drop.

#### Where this leaves the biology

§15 said §13's cube "is what the primate visual system also does". That claim
survives — AL neurons are genuinely mirror-symmetric and the cube is genuinely
aliased. What does **not** survive is the inference that building the stage
deliberately should help. Two readings remain open and this experiment cannot
separate them:

1. Mirror symmetry in AL is a *consequence* of the training distribution, as
   Farzmahdi et al. argue, not a computation worth reproducing. Emergent is not
   the same as useful.
2. It pays off for a task this one does not test — faces, or many instances
   within a category, where left/right genuinely is nuisance. Our objects are
   ten distinct instances, so identification is a between-category problem.

Reading 2 is the honest caveat, but note the symmetric group above is the
closest this dataset comes to it and AL lost there too. Anyone wanting to
revive it should test within-category discrimination, not add more symmetric
instances.

*Provenance: measured — rendered turntable (ten objects with `--symmetric-set`,
six without), HOG front end, 3 seeds, 30° blocked arcs, hierarchical bootstrap
over objects → arcs → seeds with the object level restricted to each group. The
degrees and hit rates are HOG numbers. The sign-accuracy result, the axis
independence, and the direction of the identification effect are structural.*

### E2 — residue view code: **refuted, and it corrects E0**

**Residue coding does not transfer, and the reason corrects §16 E0's
headline.** Residue harmonic sets land at or past chance — 87–95° median
against 90° for guessing, with 57–78% gross failures — and no amount of fixing
the comparison rescues them. But diagnosing *why* found the thing E0 missed:
**bundle capacity is set by the number of distinct harmonics, and E0 never
swept it low enough.** At `max_harmonic=4` the object file **ties a list of
stored views at every K**, at twelve times less memory.

Run it: `python experiments/run_residue_code.py`.

#### A residue system is a choice of harmonics, nothing more

With moduli `mᵢ` and `M = ∏mᵢ`, index the circle by `x ∈ [0, M)` with
`φ = 2πx/M`. The residue `x mod m` is carried by
`exp(2πi(x mod m)/m) = exp(i(M/m)φ)`, so **module `m` is exactly the single
integer harmonic `M/m`**:

```
dense band  {1, 2, …, 16}     16 frequencies, top harmonic 16
residue (7,8,9)  {56, 63, 72}   3 frequencies, top harmonic 72
```

Both fill the same `ssp_dim`. The residue set spends the whole budget on three
frequencies — five times the redundancy each — and resolves four times finer,
because the Chinese remainder theorem makes them jointly unambiguous. That was
the capacity argument. `sspslam.objectmap.residue_harmonics` builds the set; no
other machinery was needed.

#### It fails, and it is not the comparison's fault

At `d=601`, `K=12`:

| code | distinct harmonics | top | lobe | median | <15° | >45° |
|---|---|---|---|---|---|---|
| band 8 | 8 | 8 | 16.5° | 16.3° | 0.46 | 0.23 |
| band 16 | 16 | 16 | 9.0° | **13.5°** | 0.54 | 0.22 |
| band 24 | 24 | 24 | 6.2° | 13.5° | 0.56 | 0.19 |
| res 3-4-5 | 3 | 20 | 4.0° | 87.0° | 0.29 | 0.61 |
| res 4-5-7 | 3 | 35 | 2.2° | 93.0° | 0.13 | 0.73 |
| res 7-8-9 | 3 | 72 | 1.0° | 87.8° | 0.12 | 0.71 |
| *list (1-NN)* | — | — | — | *10.0°* | *0.63* | *0.17* |

Three ways it could have been an unfair test, all closed:

1. **Not the sharpness.** §14 showed sharp kernels lose when stored views are
   far apart, and every residue set above has a 1–4° lobe. So resolution-matched
   ones were built too: `res 4-5` has M=20, an 18° step matching the
   descriptor's half-width exactly, and a 13.5° lobe *wider* than band 16's.
   It still fails, at 91.5° with 71% gross failures.
2. **Not the readout.** Kymn reads residues with a resonator network; argmax
   over a superposed likelihood throws the modular structure away. So the
   resonator's fixed point was computed directly — one phase per module, then
   CRT. It changes nothing: 87.0 → 90.0 for `res 3-4-5` at K=12, 93.0 → 88.4
   for `res 4-5-7`. The loss is in the code, not in how it is read.
3. **Not the descriptor.** See below.

#### The mechanism: distinct harmonics are the capacity currency

Query each book with a key that is **in** it. No generalisation left to do, no
descriptor limit to hit — what remains is purely how well the code survives
superposing K items. `d=601`, median error in degrees:

| code | distinct harmonics | K=1 | K=4 | K=12 | K=24 |
|---|---|---|---|---|---|
| res 3-4-5 | 3 | **0.0** | 1.0 | 87.0 | 56.5 |
| res 4-5-7 | 3 | **0.0** | 0.5 | 52.0 | 89.0 |
| res 5-7-9 | 3 | **0.0** | 0.0 | 41.0 | 102.5 |
| band 4 | 4 | 0.0 | 1.0 | 4.0 | 5.0 |
| band 8 | 8 | 0.0 | 1.0 | 3.5 | 5.0 |
| band 16 | 16 | 0.0 | 0.5 | 1.0 | 3.7 |
| band 24 | 24 | 0.0 | 0.5 | **0.5** | **2.5** |

**Every code is exact at K=1.** What separates them is how many items the
bundle holds, and it tracks the number of distinct harmonics monotonically. A
residue system minimises precisely that quantity — three frequencies is the
whole design.

Plainly: with only three frequencies the similarity kernel is a mean of three
cosines, whose sidelobes sit close to the main peak. Superposing K views piles
K sets of those sidelobes on top of each other, and once the pile clears the
gap the peak is no longer the right one. More frequencies means lower
sidelobes means more room to superpose.

**Why it does not transfer from the published result.** Residue coding buys
*unambiguous range per frequency*. The object file's constraint is
*superposition SNR*. Different currency. Kymn et al. represent **one** position
and factorise it with a resonator; we hold **K** items and read a peak off
their sum. The residue advantage is real in their setting and inapplicable in
ours, and no implementation detail bridges that.

#### What it pays for: §16 E0's headline was a hyperparameter artefact

If distinct-harmonic count sets capacity, then E0's signature failure — error
*rising* with K — should move with `max_harmonic`. E0 swept it over {8, 16}.
It never tried 4. Median in-gap error on the real task:

| k_max | lobe | K=6 | K=12 | K=24 | K=36 |
|---|---|---|---|---|---|
| **4** | **28°** | **18.0°** | **9.0°** | **14.3°** | **16.5°** |
| 8 | 16° | 20.0° | 16.5° | 22.5° | 23.5° |
| 12 | 12° | 20.5° | 13.0° | 20.5° | 21.5° |
| 16 | 9° | 20.0° | 11.0° | 20.0° | 21.2° |
| 24 | 6° | 20.0° | 11.3° | 17.7° | 19.0° |
| 48 | 3° | 20.0° | 10.5° | 16.0° | 17.0° |
| *list (1-NN)* | — | *20.0°* | *10.0°* | *15.0°* | *15.0°* |

*(d=2401; the same ordering holds at d=601 and d=151)*

`k_max=4` wins at every K, and at K=12 it beats the list. Re-running E0's full
protocol at `k_max=4`, `d=2401`:

| K | decoder | median err | ID hit | map floats |
|---|---|---|---|---|
| 12 | nearest | 10.0° | 0.98 | 172,872 |
| 12 | kernel | 10.0° | 0.98 | 172,872 |
| 12 | **vsa** | **9.0°** | 0.87 | **14,406** |
| 12 | vsa-scene | 13.5° | 0.75 | **2,401** |
| 24 | nearest | 15.0° | 0.98 | 345,744 |
| 24 | **vsa** | **14.3°** | 0.88 | **14,406** |

Difference of medians, hierarchical bootstrap, `vsa − nearest`:

| K | diff | 95% CI |
|---|---|---|
| 6 | −2.0° | [−6.0, +7.8] |
| 12 | −1.0° | [−6.5, +6.0] |
| 24 | −0.7° | [−3.5, +10.0] |

**Every interval spans zero.** Not a win — a tie. But a tie at 12× less memory
for the per-object books and 72× less for the scene vector, which is a
different claim from the one E0 made.

Note the winning lobe is **28°, wider than the descriptor's own 18°
half-width**. §14 said reach beats precision when stored views are far apart.
This says it more strongly: the best kernel is *broader than the correlation
length of the thing it is matching*, because a broad kernel is also a
low-frequency one, and low-frequency codes superpose better. Two arguments that
looked separate — §14's reach-versus-precision and E0's capacity — are the same
constraint seen from either end.

*Provenance: measured — rendered turntable, HOG front end, 3 seeds, 30° blocked
arcs, hierarchical bootstrap over objects → arcs → seeds. The self-retrieval
table is descriptor-independent by construction and the harmonic-count ordering
in it is structural. The degrees on the real task are HOG numbers.*

### E3 — object-vector-cell scene map

The scene half is the untested half. `ID ⊗ S_allo(p)` is an allocentric position
code; object vector cells (Høydal, §15) are *egocentric vector to object*, and
their headline property is generalising **across environments**.

Test exactly that property. Build the map in one room layout, move every object,
query in the new layout. Allocentric binding should fail; a vector-to-object
code should transfer. Nothing else in this document separates the two, and the
parent SSP-SLAM paper already has the machinery.

This is also the experiment that decides §15's honest wrinkle — whether the
scene half should be OVC-like at all, or whether the view circle is carrying the
whole idea.

---

### E4 — the K curve: **the knee is real, the per-object prediction is not**

**Every non-aliased object needs its stored views no more than 30° apart, and
nothing about the object changes that.** The knee is sharp and identical across
objects whose §13 descriptor half-widths span 15° to 40°. The prediction that
§13's cheap diagnostic would say how densely to orbit a given thing is
**refuted** — and there is a reason it had to be.

Run it: `python experiments/run_k_curve.py --symmetric-set`.

Measured at `max_harmonic=4`, `d=2401`, so §16 E2's capacity confound is gone
and this curve is about coverage, which is what E4 is for.

#### The pooled curve, and why K stops mattering

| K | spacing | median gap to nearest stored view | object file | list |
|---|---|---|---|---|
| 2 | 180° | 42° | 68.5° | 70.0° |
| 4 | 90° | 25° | 40.5° | 40.0° |
| 6 | 60° | 18° | 24.0° | 25.0° |
| **12** | **30°** | **10°** | **13.0°** | **15.0°** |
| 18 | 20° | 10° | 16.5° | 15.0° |
| 36 | 10° | 10° | 19.7° | 15.0° |

Knee at **K=12** for both decoders. Past it, look at the third column: the gap
floors at 10° and stops falling. With held-out arcs 30° wide, a thirteenth
stored view has nowhere closer to sit, so it adds bundle load without adding
coverage. **K is the wrong axis** — it stops being informative exactly where
the geometry saturates, not where the object does.

#### The object file and the list are indistinguishable at every K

Difference of medians, hierarchical bootstrap:

| K | 2 | 3 | 4 | 6 | 8 | 12 | 18 | 24 | 36 |
|---|---|---|---|---|---|---|---|---|---|
| vsa − list | −1.5° | +2.5° | +0.5° | −1.0° | +3.7° | −2.0° | +1.5° | +3.0° | +4.7° |
| interval spans 0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**Nine out of nine.** E2 established the tie at three values of K; this
establishes it across the whole curve from 2 to 36. Wherever the two agree, the
limit is coverage and not the representation — which is the cleanest statement
of what the object file is: a compression of the list that costs nothing in
accuracy.

#### The right axis, and the refuted prediction

What a robot actually controls is not K but **how much of an orbit it walks**.
Sweeping the held-out arc width instead, median error per object:

| object | §13 half-width | 10° | 20° | 30° | 45° | 60° | widest arc filled to <15° |
|---|---|---|---|---|---|---|---|
| chair | 25° | 5.2° | 10.5° | 6.5° | 28.2° | 47.0° | **30°** |
| mug | 35° | 7.0° | 12.3° | 7.3° | 25.8° | 44.5° | **30°** |
| L_block | 25° | 6.0° | 7.5° | 7.0° | 26.3° | 38.5° | **30°** |
| pot | 40° | 8.5° | 11.2° | 12.8° | 29.7° | 53.0° | **30°** |
| console | 15° | 6.5° | 8.7° | 6.0° | 17.5° | 27.5° | **30°** |
| tripod | 15° | 8.0° | 11.7° | 12.0° | 122.3° | 105.2° | **30°** |
| cube | 15° | 87.0° | 95.0° | 90.0° | 85.0° | 92.5° | — |
| cross | 15° | 90.2° | 140.0° | 90.0° | 90.2° | 171.7° | — |
| drum | 35° | 90.0° | 100.7° | 90.0° | 90.0° | 153.5° | — |

Six non-aliased objects. **Six identical answers.** Half-widths spanning 15° to
40° — a 2.7-fold range — produce zero spread in the tolerable arc, so there is
nothing for the half-width to correlate with. The cliff between 30° and 45° is
in the same place for all of them.

**Why the prediction had to fail: half-width is two-sided.** A wide appearance
autocorrelation means the object looks similar over a wide arc. That should let
you reach further across a hole — but it is the same property that makes the
angle hard to pin down once you get there, which §13's own consequence 1 already
said about the `pot`. Reach and contrast move together and in opposite
directions, so their product is flat. One number cannot predict a quantity that
depends on both.

What *does* predict the cliff is the **view kernel lobe, 28°**, sitting right at
the 30°/45° boundary, with the median descriptor half-width (22°) below it.
Consistent with §14 and §16 E2: reach is the binding quantity, and the kernel is
what supplies it.

#### The practical number

**Store a side every 30°, so twelve per object.** Fewer and the file cannot fill
the hole; more and you are paying bundle capacity for coverage you already have.
That is the answer to "how many views do I need", and it is the same answer for
every object that is not symmetric — which makes it a much more useful rule than
the per-object one that was predicted, even though it is the boring outcome.

Comparing 30° against published IT view-tuning widths (Logothetis, §15 leg 3)
is the obvious next move and is **not done here**: that citation is tagged
**[m]**, from memory, and this document does not compare a measured number
against an unverified one.

*Provenance: measured — rendered turntable, ten objects with `--symmetric-set`,
HOG front end, 3 seeds, `d=2401`, `max_harmonic=4`, blocked arcs, hierarchical
bootstrap over objects → arcs → seeds. The 30° figure is a HOG number and would
move with the encoder. That the knee saturates where the gap floors, and that
half-width fails to predict the tolerable arc for the reason given, are
structural.*

### The order

**E3 only**, with E0, E1, E2 and E4 done.

Two items from E1 and E2 are already implemented rather than queued, because
each was one line:

- **An unbound appearance prototype per object** (`ObjectFile.prototype`,
  `ObjectCentricMap.identify`, `recognise`), so naming does not go through the
  capacity-bound view book. E1: 0.44 → 0.89 identification at `d=151`.
- **`max_harmonic` is a capacity setting, not a resolution setting** (E2).
  The repo default of 8 was the wrong side of the optimum; 4 is better at
  every K tested and turns E0's loss into a tie.

*Provenance: E0 and E1 measured (see their own notes). E2–E4 are
specifications — nothing in them is a result, and none of their predicted
outcomes may be quoted as findings.*
