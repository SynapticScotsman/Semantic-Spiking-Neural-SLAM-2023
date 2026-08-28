# Working notes for Claude

## How to write here

**Give it straight.** Lead with the answer, then the mechanism. If something
failed, say it failed in the first sentence. No throat-clearing, no "it's worth
noting that", no restating the question before answering it.

**Plain first, technical second.** The maths is welcome — keep it — but put a
plain-English sentence next to it saying what it actually means. "λ is the
fraction of the key that doesn't change as you walk around the object" earns
its place beside the formula. Both, not one or the other.

**Cut the padding.** No summarising what you just said. No listing options you
won't pursue. If a table says it, don't also say it in prose.

**Visuals earn their keep.** Default to a picture when the point is a shape, a
trend, a comparison, or a failure mode. The interactive page in `docs/` is the
house style: measured data, provenance banner, controls that let the reader
poke at the claim rather than take it on trust. Rebuild it with
`python experiments/build_view_circle_page.py`.

## How to report results

This project's standard, inherited from `astm/docs/RESULTS_SO_FAR.md` on
`neuromorphs/VSACognitiveMapping`:

- **Tag every number**: exact (algebraic identity, verified) / synthetic
  illustration (rendered turntable + HOG — never report as measurement) /
  pending / retracted. See `FINDINGS.md` §0.
- **Frames are not samples.** Views on an orbit are correlated; ~6 independent
  views per 72-frame orbit. Never quote n or a p-value computed on frames.
  Hierarchical bootstrap: objects → arcs → seeds. (`FINDINGS.md` §0 E8.)
- **Blocked splits only.** Contiguous held-out arcs, never alternate frames.
- **Seeds, not a seed.** One draw of a random projection moves results by ~3°.
- **Publish the number that hurts.** Refutations of our own hypotheses go in
  the document, not the bin. §14 keeps a failed conjecture on the page.

## Repo orientation

- `FINDINGS.md` — results, errata, provenance. §0 first, always.
- `OBJECT_CENTRIC_MAP.md` — what each piece of the map is and why.
- `sspslam/objectmap/` — the map itself. numpy + scipy, no nengo.
- `docs/view_circle.html` — the interactive diagnostic; generated, not hand-edited.
- `experiments/run_*.py` — every number in the docs traces to one of these.

## Environment

Egress policy blocks `huggingface.co`, `dl.fbaipublicfiles.com` and
`download.pytorch.org`, so DINOv2 cannot run in a remote session. GitHub, GCS
and PyPI are open. Don't substitute another encoder and call it a comparison —
leave the column empty and say why.
