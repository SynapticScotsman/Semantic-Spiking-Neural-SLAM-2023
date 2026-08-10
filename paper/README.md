# ICRA 2027 paper — working directory

Target: **ICRA 2027, deadline Sept 15, 2026.** Overleaf project:
https://www.overleaf.com/project/6a75e41dde34d8bf1ef1b38f

## Publication pipeline (ICRA + T-RO, decided 2026-08-10)

Two-stage, using T-RO's "evolutionary paper" category (extended versions of
your own conference papers with substantial new results are an explicitly
recognised T-RO submission type):

1. **ICRA 2027** (Sept 15, 2026): this draft. Merge as a *property*, χ law
   empirical, at-scale material named in future work.
2. **T-RO evolutionary version** (submit ~Feb–Apr 2027, after the ICRA
   decision): subsumes the ICRA paper + the deliberately-cut delta = frame
   recovery in the algebra, K-scaling 2–16 robots, bandwidth protocol,
   formal χ derivation + no-per-cell-fit real-data closure, full
   ConceptGraphs head-to-head, relational layer if legalised. Cites the
   ICRA paper with a delta statement; text rewritten, not pasted; not
   under review simultaneously.
3. **Insurance**: if ICRA rejects (~late Jan 2027) the work never appeared
   in proceedings, so the T-RO submission qualifies as a *new* paper and
   becomes eligible for ICRA 2028 presentation via PaperPlaza journal
   transfer (T-RO/RA-L→ICRA transfer deadline was Dec 31 for ICRA 2027).
   RA-L remains the fast fallback.

Rule refs: T-RO-at-ICRA presentation is only for NEW (non-evolved) T-RO
papers (ieee-ras.org "Presenting Your T-RO Paper at ICRA"); ICRA 2027 CFP
confirms journal transfer + Dec 31, 2026 deadline.

## Files

- `main.tex` — complete IEEE-conference-style draft (IEEEtran conference
  mode; one-line swap to ieeeconf.cls for PaperPlaza, noted at the top of
  the file). All sections written: abstract, intro, related work (with
  merge-survey table), method, capacity law, setup, results (5
  subsections, 4 tables), limitations, conclusion.
- `refs.bib` — full bibliography. Every entry web-verified 2026-08-10
  (fetch + adversarial recheck) and carries a real clickable URL: arXiv
  abstract page where the paper is on arXiv, else the DOI link, else the
  official project page. No repo-internal links appear in the rendered
  PDF; number provenance lives in `% provenance:` TeX comments only.
- `variants/abstract_intro_variants.md` — five complete alternative
  abstract+intro drafts (systems-first, law-first, merge-first,
  honest-contrarian, algebra-first), each with a suggested title and an
  honest self-assessment, plus a comparison table. **The abstract and
  intro currently in `main.tex` are a deliberate general synthesis** that
  covers all four pillars evenly; swap in a variant (or a graft) once the
  angle is chosen.
- `figures/` — `synthetic_scaling.png` (capacity-law figure, ready);
  remaining figures are `\todo{}` placeholders (list below).

## Getting it into the Overleaf project

Fastest path (existing project): open the project → Upload (top-left
menu) → drag `main.tex`, `refs.bib`, and the `figures/` folder in.
**Merge warning:** the Intro and Conclusion here are complete drafts
written without sight of the partials already in Overleaf (marked with
`MERGE NOTE` comments) — merge or replace, don't keep both copies.
Alternative: Overleaf Menu → GitHub sync against this repo
(`results-sites` branch, `paper/` directory) if the account has the
GitHub integration.

## Scope decision (recommendation, encoded in the draft)

**This paper = one track: the fixed-size algebraic map.** Four pillars,
all measured: (1) the system + conditioning discipline; (2) the χ
capacity law with the 1.4% held-out prediction; (3) benchmark
characterisation (7-Scenes chess, Replica ×8, drift bounding); (4) exact
merge with the 4-robot head-to-head.

Deliberately **left out** (separate papers, per "scope may get too
large"):
- **Isotropy/conditioning science** (dissociation, whitening mechanism,
  rogue-dims) → the NeurIPS workshop paper (Aug 29; skeleton at
  `wiki/drafts/2026-08-03-neurips-workshop-draft.md`). Only the deployed
  *rules* appear here (Sec. III-D).
- **Multi-robot at scale** (frame recovery, K-scaling, bandwidth
  protocols) → paper 2. The merge *property* stays here because it's the
  headline systems differentiator; the merge *system* is future work.
- **Relational/instance query layer** → internal-diagnostic only
  (skeptic-panel bannered); appears one paragraph in Limitations.
- Calibrated abstention, working-memory decay, neuromorphic/spiking
  implementation (NICE 2027 angle).

## What blocks what

| blocker | blocks | status |
|---|---|---|
| **D2 authorship** | author block, acknowledgments, who reviews | open — Paul |
| ConceptGraphs their-scorer run (Colab L4 / student) | the head-to-head table in Sec. VI-B | notebook ready, pre-registration already in the text |
| ieeeconf.cls files | final template compliance | one-line swap, do near submission |
| Figure generation (below) | camera-ready figures | scriptable from existing artifacts |

## Figure checklist

| # | figure | source | status |
|---|---|---|---|
| 0 | **Page-1 hero** — (a) asked/recalled photo pairs at the 50th and 75th error percentile, (b) the chess map with a segment per query, (c) a Replica belief field with GT instances | `python tools/make_hero_figure.py` (reads both inspector payloads; nothing hand-drawn) | **in draft** |
| 1 | χ(N) synthetic scaling | `outputs/synthetic/synthetic_scaling.png` | **in draft** (restyle pass wanted) |
| 2 | Replica belief-field montage (2–3 class queries + one consistent-but-false case) | harvest from `outputs/grounding_inspector_*.html` payloads / `tools/export_grounding_inspector.py` | placeholder |
| 3 | Chess qualitative (query frame → retrieved frame pairs + error CDF) | `outputs/cross_recall_7scenes_chess.json` + inspector | placeholder (optional if page-tight) |
| 4 | Merge schematic (4 traces → sum ≡ joint; vs association decisions) | draw; numbers from `merge_comparison_k4.json` | placeholder (optional) |

## Number provenance

Every number in `main.tex` traces to a tracker entry
(`wiki/analysis/2026-07-29-vsa-query-layer-paper-plan.md`), cited in TeX
comments: chess (ax), Replica pooled (bd), room0 bounded insertion (ba),
truth metrics + CLIP verify (bb), merge survey + head-to-head (bc)+(bd),
χ law (az) + `wiki/analysis/2026-08-04-t1-chi-law-empirics.md`, Kalman
(ag)/(ah), treatments 5-seed (aw), substitution law (am), time recall
(ar)/(au), instance blindness (bf). Audit rule before freeze: re-check
each against its entry; anything untraceable is cut, not paraphrased.

Standing rules enforced in the text: no invented metrics (bespoke
protocols labelled "characterisation", ConceptGraphs comparison under
their scorer); every multiplier with encoder+N; ≤0.5/1/2 m fractions
beside medians; single-split numbers seed-tagged; chance/oracle/constant
brackets; banned-phrase list respected (no "viable successor",
"definitive", "microseconds", bare multipliers).

## Length

Current draft will run ~7 pages + refs against ICRA's 6 (+2 paid). Cut
ladder, in order: Fig. 3 and 4 → merge-survey table (Table I) to one
prose sentence → Sec. VI-E ablations compressed to one paragraph
(pointing to the workshop paper) → range-kernel/multi-trace details
compressed. Do not cut: Limitations, the merge caveat, protocol
discipline.
