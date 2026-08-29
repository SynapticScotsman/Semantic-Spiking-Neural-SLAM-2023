# Object-Centric VSA Cognitive Map

A memory of an *object* rather than of a *room*. Each thing gets its own
turntable: walk round it, store a few snapshots tagged with the angle you took
them from, and later a fresh photo tells you which side you are standing on.
Push the chair across the room and the memory still holds — only its position
needs rewriting.

Same trick as SSPs use for space, with the manifold swapped: a circle around an
object instead of a grid over a floor.

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

Keep two things separate: a **list of what is where**, and, per object, **what
it looks like from each side**. Merging them into one vector breaks both — the
measurement is below.

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

The tempting move is one vector per object, `ID ⊗ S_allo ⊗ V`. It fails badly.

Plainly: asking "where is it?" is supposed to leave you holding the position
code. With everything stapled together you are left holding *position stapled
to the view book*, and the view book looks like noise, so the position smears
out. Same in reverse — asking "what does this side look like?" leaves you
holding *appearance stapled to position*, which is not appearance.

```
[8] Two memories vs one fully-bound blob
    object      bundle: pos err   bind: pos err
    mean                  0.030 m         7.130 m       (in a 14 x 14 m room)
```

One object in the demo bucks this: `table_a` sits exactly at the origin, where
the position code happens to be the do-nothing element, so stapling it in
changes nothing and that object reads out fine. That is the mechanism showing
through, not an exception to it.

If you do want one vector per object, **stack the two roles instead of
stapling them** — `ObjectCentricMap.object_vector(oid, mode='bundle')` gives
`ID ⊗ unit(S_allo + V)`, which keeps both read-outs at the cost of extra
cross-talk. `mode='bind'` reproduces the fully-bound form so the difference
stays measurable rather than asserted.

---

## The view circle

A normal SSP encodes a line: walk far enough and you are somewhere new. An
orbit is not a line — walk far enough and you are back where you started. If
the code does not know that, 359° and 1° are strangers.

The fix is one line of arithmetic: **use whole-number frequencies**. Then the
code repeats exactly every 360°, and "rotate by Δ" becomes a single multiply.

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

`max_harmonic` is the sharpness knob: how wide an arc one stored snapshot
covers. Same role a length scale plays for space.

| `max_harmonic` | lobe half-width | worst sidelobe |
|---|---|---|
| 6 | 21° | −0.16 |
| 8 | 16° | −0.13 |
| 12 | 11° | −0.10 |

**Sharper is not better.** Measured at a 30° gap between stored views:
25.6° error with a broad kernel, 33.0° with a sharp one. A sharp code has short
reach and falls into the gaps between what you stored. Match the kernel to your
stored-view spacing (`FINDINGS.md` §14).

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

One call does everything: fingerprint the crop, invent a name if this is a
thing you have not seen before, work out which side of it you are on, and file
the snapshot under that angle.

**Near-duplicate snapshots get merged, not appended.** Fifty frames walking
past a chair are fifty near-identical pictures; storing all of them costs
capacity and adds nothing. Anything within `merge_tol` of a stored angle folds
into it instead.

```
[PASS] merging keeps K far below the detection count:
       K = [10, 10, 9] from [59, 44, 57] detections
```

Revisits are the same code path: same `ID`, new `φ`, another term in the
mean.

---

## Reading it

Every question is answered by division. You know part of what went in, so you
divide it out, and whatever is left is the answer — then you match that against
the things you know about.

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

### Frame by frame it jumps; over time it settles

Asked afresh each frame, the answer hops around by 50° or more — which a camera
walking round a chair obviously cannot do. Feeding the answer through a filter
that remembers where you just were fixes it, and both halves of that filter are
operations this code already has: *predict* is one bind by `S_view(Δ)`, exactly
what `orbit()` does, and *update* is a multiply by the new likelihood.

| read-out | typical error | impossible jumps |
|---|---|---|
| per frame, no memory | 17.0° | 89 / 426 |
| + filter | 6.0° | 25 / 426 |
| + one known starting angle | **4.0°** | 11 / 426 |

The last row is what fixes symmetric objects. Tracking alone cannot choose
between a cube's four identical sides, but it turns *four answers every frame*
into *one choice for the whole trip* — so one known starting angle settles it
permanently. `experiments/run_view_tracking.py`, written up in `FINDINGS.md` §12.

---

### Two angles, not one

There are two angles in play and they are easy to confuse. Get them the wrong
way round and the object file quietly fills with nonsense.

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

Each of these was tried and measured, not guessed at.

**Do not FPE the embedding** (`exp(i·W·z)`). FPE makes similar inputs get
similar codes. That is what you want for space and for angle. It is exactly
what you do *not* want for appearance: "similar-looking" then means "same
thing", so two different chairs merge into one instead of two sides of one
chair staying apart. Measured on look-alike crops:

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

**Do not whiten the appearance keys.** Subtracting the average crop is worth
doing — it removes the sameness that every crop shares. Whitening does not stop
there: it also blows up the directions where the embedding is mostly noise, and
the answer goes with it (11° → 86°, against 90° for guessing). `AppearanceCodec` accepts `'center'`, `'zscore'`
or `'none'` and raises on anything else.

**Do not bind heading to a viewpoint-invariant place descriptor.** If a
descriptor is deliberately built to look the same from every angle, there is no
angle left in it to bind to. You get heading times a constant.

**Do not use a snapshot map `c_i ⊗ S_ego` as a substitute for object files.**
That records *where you were standing*, not what the object looks like. Move the
chair and it is worthless. An object file survives, because only the chair's
position changed and not its sides:

```
[10] moved chair_a to [-0.5 0.5]: decoded [-0.5 0.5], err 0.000 m
     object file survived: 7 views before, 7 after
```

Only `S_allo` was rewritten.

**Do not use the crop as the name.** A name has to survive seeing the thing
from behind, in different light, half hidden. A picture of it does not. That is
why `ID` is a random atom with no connection to appearance.

---

## How much fits

Everything is piled into a fixed-size vector, so the more you add, the noisier
each read-out gets — roughly `1/√(number of things piled in)`. What costs you is
*objects × distinct views*, not image resolution. In the
demo, 8 objects at 151-D give an unbind peak around 0.34 against a noise
floor near 0.1; the cross-talk panel of `object_map.png` shows the margin
directly. Raise `ssp_dim` for more objects; keep `K` small per object by
leaving `merge_tol` near the view kernel's lobe width.

`ObjectFile.coverage(view_space)` answers "have I walked far enough round this
thing yet?" — the fraction of the circle within reach of some stored snapshot.
Accuracy follows it closely, because **a side you never looked at is a hole, not
something the code can invent**. To localise better than 15° you need snapshots
no more than ~16° apart, which is about 22 round a full circle.

---

## Relation to the rest of the repo

Nothing here is a new kind of memory. It is the split SSP-SLAM already makes —
smooth codes for *where*, random atoms for *what* — pointed at an object instead
of a room.

This is the same split Dumont's SSP-SLAM already makes — FPE for space,
atoms for what. The added piece is the view circle `S_view`, with appearance
as values on that circle rather than as the circle itself. `S_allo` is the
existing `HexagonalSSPSpace`; the object-centred `φ` is the same code as a
head-direction context, with its origin moved onto the object.

The module is pure numpy/scipy and does not import nengo, so it runs
standalone; the spiking side of the repo is untouched.
