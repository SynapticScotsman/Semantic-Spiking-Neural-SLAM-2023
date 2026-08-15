# Object-centric VSA view memory — findings, recipe, and pitfalls

Everything measured so far, compressed for a local rebuild. Written against
`sspslam/objectmap/`, but §9 is a from-scratch implementation in ~40 lines if
you'd rather not take the dependency.

**One-line summary.** Give every object its own circle of viewpoints encoded
with *integer-harmonic* FPE; store appearance as values on that circle, never
as the circle itself; condition the latents by deleting the few directions
that say *which object* rather than by whitening; then viewpoint becomes
localisable from appearance alone, in closed form, at ~8–10° median.

---

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

Measured on the turntable (6 objects, 72 azimuths, K=12 sides on file,
151-D, held-out azimuths):

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
5. Close the **loop**: feed `localise_view`'s likelihood back as a heading
   correction, the way landmark recall corrects the path integrator in
   SSP-SLAM. That's the point at which "I know which side I'm seeing" becomes
   part of SLAM rather than an offline query.
6. **`K` selection.** Currently merging by a fixed angular tolerance. Ought to
   be driven by the view kernel's lobe width and by how fast appearance
   actually changes — objects with a discontinuity (a handle appearing) need
   sides packed more tightly there than on a smooth flank.
