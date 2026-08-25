# Rendered pages

GitHub shows `.html` as source, not as a page. Open these through a raw proxy:

| page | open | what it is |
|---|---|---|
| **View Circle Diagnostics** | [open rendered ▸](https://raw.githack.com/SynapticScotsman/Semantic-Spiking-Neural-SLAM-2023/claude/object-centric-vsa-map-kf8met/docs/view_circle.html) | Where object-centric view-direction localisation holds and where it breaks. Pick an object, scrub the orbit, watch the likelihood over the whole view circle. Self-contained, ~900 KB, works offline once loaded. |

**Read `FINDINGS.md` §0 before quoting any figure from these pages.** Every
degree in them is synthetic illustration from a rendered turntable with a HOG
front end — not a measurement of any real encoder, dataset or robot. The
algebraic identities (exact periodicity, binding as rotation, the one-FFT
likelihood, the filter's predict step) are exact and separately verified.

Rebuild after changing the encoder, the split or the filter:

```bash
python experiments/build_view_circle_page.py
```

That regenerates the `<script id="viewdata">` block in place from the same
experiment code behind the tables in `FINDINGS.md`, so the page cannot drift
from the results it illustrates.
