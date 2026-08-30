# docs/

## `view_circle.html`

The interactive diagnostic for the object-centric view circle. Open it straight
from the repo — it is committed built, with every number and every image inlined,
so it needs no server and no network.

Eight sections, in two halves.

**The original four** — what the view circle is and how it fails:

1. *Every viewpoint at once* — the likelihood field for every crop against every
   hypothesised direction. One bright diagonal means it works; several parallel
   lines mean the object genuinely looks the same from several sides.
2. *It fails in two different ways* — aliasing, which no amount of data fixes,
   against coverage, which more stored sides fixes directly.
3. *It cannot teleport* — the circular Bayes filter, and what an anchor buys.
4. *Two things called anisotropy* and *the pedestal law* — §13 and §14.

**The four added after §16's experiment programme:**

5. *Watch it turn* — the orbit as a film: the object rotating, dense optical
   flow beside it, the warp residual showing what the flow cannot explain, and
   the memory's belief updating frame by frame.
6. *Occlusion costs 3°, the gap costs 8°* — where the errors actually live.
7. *One setting was wrong the whole time* — the `max_harmonic` sweep.
8. *Eight experiments, five of them wrong* — the scoreboard.

### Rebuilding

Two generators, one per data block. Neither is optional if you change the
encoder, the split or the filter — the page will otherwise show stale numbers
with a confident provenance banner, which is worse than showing nothing.

```bash
python experiments/build_view_circle_page.py   # <script id="viewdata">
python experiments/build_flow_panel.py         # <script id="flowdata">
```

The two halves are measured at different settings on purpose. The first four
sections are at `ssp_dim=151`, `max_harmonic=8` — the values §§4–13 were
measured with, kept so those figures stay reproducible. The rest are at
`ssp_dim=2401`, `max_harmonic=4`, which §16 E2 shows is the better choice. The
page says which is which wherever they disagree.

### The rule that keeps biting

**Display resolution must never reach the descriptor.** Tiles are cropped from a
224px render; every descriptor and every optical-flow field comes from the 96px
render the experiments use. Letting the display resolution leak into the
analysis once moved one object's error by 47° without changing a line of the
method. Both generators keep the two apart, and both say so in their docstrings.

Nothing here should be edited by hand.
