# Object-Centric VSA Cognitive Map

A mental map of an *object* rather than of a *room*: a sphere of viewpoints
around a thing floating in space, which you can orbit, query, and carry with
the object when somebody moves it.

Same idea as fractional-power-encoded SSPs for space — but the manifold is a
circle around an object instead of a grid over a floor.

See **`FINDINGS.md`** for the results, the latent-conditioning recipe, and the
pitfalls — and **read its §0 errata first**: every degree and metre figure in
both documents is synthetic illustration from a rendered turntable, not a
measurement of any real system.

```
sspslam/objectmap/          numpy + scipy only, no nengo
experiments/run_object_map.py    end-to-end demo, prints measurements
experiments/test_object_map.py   38 assertions
```

```bash
python experiments/test_object_map.py
python experiments/run_object_map.py --plot --save-dir data/objectmap
```

---

## Two memories, not one blob

**Scene map** — which instances exist and where:

```
M_scene = (1/N) Σ_i  ID_i ⊗ S_allo(p_i)
```

**Object file** — what one instance looks like from which side:

```
V_i = (1/K) Σ_k  c(z_k) ⊗ S_view(φ_k)
```

| symbol | what it is | why that and not something else |
|---|---|---|
| `ID` | random unitary atom, minted on first sighting | a name has to survive the object being seen from a new side; a crop does not |
| `S_allo(p)` | ordinary aperiodic SSP of world position | the navigation index, unchanged from SSP-SLAM |
| `c(z)` | `unit((z − μ) W)` — centred crop embedding, random projection, L2 | a *value* stored on the circle, not the circle itself |
| `S_view(φ)` | **periodic** FPE of object-centred azimuth | the orbit; wraps exactly after a full turn |
| `CLASS` | second atom, own bundle | `unbind(CLASS)` gives "where are all the chairs" |

Robot pose is nowhere in the file. `φ` is a relation between the object and
wherever the camera happens to be, so it is computed at query time from the
two poses.

### Why the two memories are not merged

The compact single-phasor form `M_obj = ID ⊗ S_allo ⊗ V` does not work, and
the failure is not marginal. Unbinding `ID` then leaves `S_allo ⊗ V`, and `V`
is a bundle of random-looking terms, so binding it in scatters the spatial
peak:

```
[8] Two memories vs one fully-bound blob
    object      bundle: pos err   bind: pos err
    mean                  0.030 m         7.130 m       (in a 14 x 14 m room)
```

It breaks the other direction too: unbinding `ID ⊗ S_view(φ)` from that blob
leaves `S_allo ⊗ c`, not `c`. (In the demo `table_a` sits exactly at the
origin, where `S_allo` *is* the binding identity — so binding it in is a
no-op and that one object's view read-out survives. That is the mechanism
showing through, not an exception to it.)

If you want one vector per object, superpose the roles instead of binding
them — `ObjectCentricMap.object_vector(oid, mode='bundle')` gives
`ID ⊗ unit(S_allo + V)`, which keeps both read-outs at the cost of extra
cross-talk. `mode='bind'` reproduces the fully-bound form so the difference
stays measurable rather than asserted.

---

## The view circle

An ordinary SSP uses real-valued phases, so `S(x)` never repeats — right for
a room, wrong for an orbit. Restricting the phase matrix to **integer
harmonics** makes the code exactly periodic:

```
S(φ) = ifft( exp(i·k·φ) ),  k integer

S(φ + 2π) = S(φ)                 exact, to 1e-16
S(a) ⊗ S(b) = S(a + b mod 2π)    binding is rotation
```

So orbiting by `Δ` with no new image is a single bind, and a full turn is the
identity to machine precision. From the test output:

```
[PASS] S(phi + 2pi) == S(phi): max diff 2.33e-15
[PASS] a full turn is the identity: max diff 1.11e-16
[PASS] orbit by delta lands on the right side with no image:
       cos to target +0.424 vs another side -0.019
```

`max_harmonic` sets the angular resolution — how wide an arc one stored view
generalises over — the way a length scale sets spatial resolution:

| `max_harmonic` | lobe half-width | worst sidelobe |
|---|---|---|
| 6 | 21° | −0.16 |
| 8 | 16° | −0.13 |
| 12 | 11° | −0.10 |

The kernel cannot be made non-negative: one dimension out of `ssp_dim`
carries DC, so the kernel averages to `1/ssp_dim` over the circle while
peaking at 1. `taper='fejer'` (default) roughly halves the ringing versus the
flat Dirichlet alternative; `CircularSSPSpace.lobe_width()` returns the
number to size `merge_tol` against.

**Sphere.** `view_dims=2` gives a periodic FPE over `(azimuth, elevation)`.
It is formally a torus; keep elevation in `[−π/2, π/2]` and it behaves as a
view sphere. `view_azimuth_elevation()` produces the pair from two 3-D poses.

---

## Filling it from a walk

```python
from sspslam.objectmap import ObjectCentricMap

m = ObjectCentricMap(feat_dim=384, ssp_dim=151, domain_dim=2,
                     bounds=[[-7, 7], [-7, 7]], length_scale=0.6)
m.appearance.fit(sample_of_crop_embeddings)      # centring statistics

for det in detections:                            # per YOLO/DINOv2 crop
    obj_id, is_new = m.observe(
        robot_pos=det.robot_xy,        # from SLAM
        obj_pos=det.world_xy,          # from the detector + depth
        embedding=det.crop_embedding,  # DINOv2 / CLIP crop, NOT a place descriptor
        robot_yaw=det.robot_heading,
        class_name=det.label,          # optional
    )
```

`observe` does the whole write path: crop → `c`; mint `ID` if this is a new
instance; derive `φ` from the two poses; merge the view into the book.

**`K` stays small by construction.** A new view within `merge_tol` of a
stored one merges into it (circular mean of the angle, running mean of the
key) instead of appending. Every frame of a walk past a chair is nearly the
same view, and bundling near-duplicates costs capacity without adding
information:

```
[PASS] merging keeps K far below the detection count:
       K = [10, 10, 9] from [59, 44, 57] detections
```

Revisits are the same code path: same `ID`, new `φ`, another term in the
mean.

---

## Reading it

| question | call | operation |
|---|---|---|
| Where is this instance? | `where_is(id)` | unbind `ID`, correlate vs an `S_allo` grid |
| What is at this place? | `what_is_at(p)` | unbind `S_allo`, clean up the ID |
| Where are all the chairs? | `where_are('chair')` | unbind `CLASS` from the class bundle |
| What does this side look like? | `view_of(id, φ)` | unbind `S_view(φ)`, clean up the key |
| What should I see from here? | `expected_view_from(id, robot_pos, robot_yaw)` | derive `φ` from the two poses, then the above |
| **Which side am I looking at?** | `localise_view(id, embedding)` | correlate the key against the file at every φ — one inverse FFT |
| Orbit by Δ, no new image | `orbit(view_code, Δ)` | bind `S_view(Δ)` |
| Does this match what I expect? | `verify(id, robot_pos, embedding)` | cosine of prediction against the live crop key |
| The chair moved | `move_object(id, p_new)` | rewrite `S_allo`, keep the file |

From the demo (8 objects, 2600 detections, 151-D, 15% of detections held out):

```
[2] where is it        mean error 0.075 m, max 0.200 m
[3] what is at p       8/8 correct
[5] which side         residue identifies the right object's view: 47/49
                       cos to the crop seen there +0.355, to the far side +0.004
[6] held-out poses     cos(prediction, crop actually seen) +0.301
                       cos vs a different object's prediction  -0.013
                       right-object hit rate 0.96 over 384 held-out detections
[7] orbit              correct side ranked first for 9/9 orbits
```

### Two angles, not one

These are different quantities and mixing them silently corrupts the book:

- **`view_azimuth(obj, robot, obj_yaw)`** — measured *at the object*, from
  its own front, towards the robot. *Which side am I looking at?* This is
  what `S_view` encodes.
- **`camera_bearing(obj, robot, robot_yaw)`** — measured *at the robot*, from
  its heading, towards the object. *Where in my field of view is it?* This is
  what you point a camera with, and it is never stored.

They differ by a half turn plus the object's yaw. Spinning the robot on the
spot changes the bearing through a full 5.28 rad while the view azimuth does
not move at all — the object presents the same face regardless of which way
you happen to be facing.

---

## What not to do

**Do not FPE the embedding** (`exp(i·W·z)`). FPE builds a similarity
manifold; that is right for space and viewpoint and wrong for appearance,
because "similar embedding" means "looks alike", which merges two different
chairs rather than two sides of one chair. Measured on look-alike crops:

```
[5] keys of look-alike crops stay near-orthogonal:  mean off-diagonal cos -0.025
    FPE of the same embeddings merges them instead: mean off-diagonal cos +0.920
```

And in the demo's room, where objects share a class bias:

```
                   cos(front, back) same object   cos(front, front) across objects
plain projection              +0.198                        -0.055
FPE of embedding              +0.414                        +0.335
```

A good key keeps both low — different sides are different entries, different
objects are different keys. FPE raises both.

**Do not whiten the appearance keys.** Centring (or z-scoring) removes the
shared mean direction that makes every crop look alike, and is worth doing.
Whitening keeps going and amplifies the low-variance directions where the
embedding is mostly noise. `AppearanceCodec` accepts `'center'`, `'zscore'`
or `'none'` and raises on anything else.

**Do not bind heading to a viewpoint-invariant place descriptor.** A
descriptor built to be the same from every angle carries no angle to bind to;
the product is heading times a constant.

**Do not use a snapshot map `c_i ⊗ S_ego` as a substitute for object files.**
That is a map of places you stood, and it dies the moment the object moves.
The object file does not:

```
[10] moved chair_a to [-0.5 0.5]: decoded [-0.5 0.5], err 0.000 m
     object file survived: 7 views before, 7 after
```

Only `S_allo` was rewritten.

**Do not use the crop as the name.** `ID` is a random atom precisely so that
identity survives a new side, new lighting, or a partial occlusion.

---

## Capacity

Both memories are bundles of unit vectors, so read-out SNR falls off roughly
as `1/√(terms)` — capacity is *objects × distinct views*, not pixels. In the
demo, 8 objects at 151-D give an unbind peak around 0.34 against a noise
floor near 0.1; the cross-talk panel of `object_map.png` shows the margin
directly. Raise `ssp_dim` for more objects; keep `K` small per object by
leaving `merge_tol` near the view kernel's lobe width.

`ObjectFile.coverage(view_space)` reports the fraction of the view circle
within the main lobe of some stored view — a "have I walked far enough around
this thing" number. Prediction quality tracks it: a side never walked past is
a hole in the file, not something the code invents.

---

## Relation to the rest of the repo

This is the same split Dumont's SSP-SLAM already makes — FPE for space,
atoms for what. The added piece is the view circle `S_view`, with appearance
as values on that circle rather than as the circle itself. `S_allo` is the
existing `HexagonalSSPSpace`; the object-centred `φ` is the same code as a
head-direction context, with its origin moved onto the object.

The module is pure numpy/scipy and does not import nengo, so it runs
standalone; the spiking side of the repo is untouched.
