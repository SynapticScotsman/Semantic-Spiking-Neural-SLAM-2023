# ICRA consolidation and work board — design

**Date:** 2026-08-18 · **Status:** approved in brainstorming, pending Paul's spec review
**Deadline served:** ICRA 2027 submission, 2026-09-15 (four weeks out)
**Approach:** claim-audit first (approach A), with the thread inventory (approach B)
folded in as a section of the same document.

## Purpose

Consolidate every ongoing thread in the project against one question — *does it
make the Sept 15 paper stronger?* — and produce (a) a single system-of-record
wiki page, (b) a live ICNS command board, and (c) a ranked punch list with each
item assigned to an execution lane that matches the compute actually available.
Threads that do not feed the deadline are parked in an explicit "beyond" lane,
not dropped.

Paul's stated priorities for the paper to sing, in order: results completeness
**on the mapping side**, the narrative spine re-anchored after the week's
retractions, and figures. Compliance/logistics items are tracked but are not
what "sing" means.

## Deliverables

1. **Claim audit** of `paper/icra6.tex` (method, §1 below) — lands inside the
   wiki doc.
2. **Wiki consolidation doc**: `wiki/analysis/2026-08-18-icra-consolidation-and-work-board.md`.
3. **ICNS board**: a branded HTML command board published as a private artifact,
   kept current as items close.
4. **Spine rewrite pass** on `paper/icra6.tex`, scoped by the audit.

## 1. Claim audit (method)

Go through `paper/icra6.tex` and extract every quantitative claim, every
mechanism statement, and every forward promise ("we show that…"). Each gets a
row: *claim → source of truth (wiki page / output file, date) → stamp*.

Stamps:

- **SAFE** — evidence current under the hashed codebook
  (`class_phasors` name-hash, commit `3161182` re-baseline) and survives the
  2026-08-17/18 retractions.
- **STALE** — rests on something retracted (the proximity-is-the-gap mechanism,
  the global-mass story), superseded (the +0.113 reachable headroom → the
  0.3844 stream ceiling), or provisional (any batch-1 verdict quoted before
  re-validation under the new baselines).
- **MISSING** — promised but not measured: the storage-matched kNN fp16 row
  (live `\todo` at icra6.tex:609), the B1 whitening-tension experiment, and
  whatever the mapping-side completeness check turns up.

Known danger zones, checked first:

- numbers generated under the sequential (pre-hash) codebook;
- any of the twelve batch-1 verdicts (all officially provisional);
- the two retracted mechanism narratives;
- whether the paper has caught up with the *good* news: degradation
  holds-and-strengthens on all 8 scenes under the clean codebook
  (commit `01980b9`), the benchmark-measures-class-coverage finding (two
  independent routes agree), and the exact gap decomposition with its
  ~+0.018 provably-unreachable share.

Scoping rule: the audit determines which batch-1 verdicts the paper actually
quotes; only those are re-validated. No blanket re-run of all twelve.

## 2. Wiki consolidation doc

`wiki/analysis/2026-08-18-icra-consolidation-and-work-board.md`, with frontmatter
matching the wiki convention (`type: analysis`, `status: active`). Sections:

- **Canonical evidence base** — one short paragraph per settled result with the
  quotable number and its provenance, all post-rebaseline. The only place
  either of us looks up "what may the paper say". Seed content includes:
  0.3198 mean mAcc (ours, hashed codebook) vs 0.402 (ConceptGraphs published)
  vs 0.406 (their Replica row, our reproduction 99%); stream ceiling 0.3844;
  grid quantisation cost +0.0388; superposition total cost +0.0098; graceful
  degradation 1.7x retained; capacity/χ law; merge exactness; bounded readout
  29.94 MB / 0.41 ms / exact on 145 answerable queries.
- **Claim audit table** (from §1).
- **Thread inventory** — each active thread: state, what it offers the paper,
  verdict **feed** or **park**. Threads in scope: ConceptGraphs head-to-head,
  hashed-codebook re-baseline, degradation/robustness, per-object-ids fetch
  (in flight on Colab), episodic memory / bounded readout / OC-NaVQA,
  Table III re-annotation, whitening tension (B1), κ statistic, submap-local
  frames (B2), JEPA→FHRR, isotropy WP1–WP9, object orbit line, map merge
  thread, 3D cogmaps/rotation.
- **Punch list** — ranked *blocks-submission → strengthens-spine → polish*,
  each item tagged with its execution lane (§4).
- **Beyond lane** — T-RO delta, collaborators' overlay paper, JEPA, isotropy,
  orbit, merge: one line each, owner, earliest sensible date.

## 3. ICNS board (artifact)

An ICNS-branded HTML command board, published as a **private artifact**, kept
current as punch-list items close (standing rule: visual reports are
deliverables and stay CURRENT). Content mirrors the wiki doc, glanceably:

- countdown to Sept 15;
- punch list as status chips (open / running / done / blocked-on-Paul);
- the three-lane work board (§4);
- thread map with feed/park verdicts.

It is a **new** artifact — a command board, not a results line — so it gets its
own URL and a new row in the live-URL list in memory. It does not replace any
existing results artifact. Build steps: invoke `/icns-design` first; use the
vendored system at `docs/brand/` (`layout.html` as base); semantic tokens only
(`--text-body`, `--border`, `--text-muted`); verify every `var(--…)` resolves;
check both colourways (`[data-colourway="plate"]`).

## 4. Work board — three execution lanes

Matched to actually-available compute (Paul, 2026-08-18):

- **CPU-local (Claude, days):**
  - re-validate only the paper-quoted batch-1 verdicts under the hashed
    codebook (scoped by the audit);
  - the storage-matched kNN fp16 row (icra6.tex:609 `\todo`);
  - B1 whitening tension — binding-key vs FPE-argument, ~1 day, resolves the
    contradiction between our classroom measurement and Shay's pipeline before
    either is published;
  - mapping-side completeness runs the audit flags;
  - all four figure jobs: hero panel (c) regeneration from the real-GT
    grounding branch (icra6.tex:120), capacity-law figure legend
    (icra6.tex:500), Replica belief-field montage, chess qualitative (A5).
  - Canonical re-baseline command (non-default args, from the 2026-08-17
    results page): `python student_gpu_package/04_vsa_labels.py --scene
    <scene>_cgfront --labels-from-points --max-per-class 400 --length-scale
    0.45,0.27 --grid 96`.
- **Paul's GPU (sparing, rough code tolerated):** only what genuinely needs
  CUDA and feeds the paper — currently the in-flight per-object-ids fetch
  (commit `1486cf3`) and any ConceptGraphs re-export the audit demands. Short,
  resumable notebook sessions via colab-mcp.
- **Collaborator GPU (handoff-grade only):** nothing paper-critical by
  default — latency (days) and the code-quality bar make it wrong for a
  four-week deadline. It is the lane for *beyond* items, JEPA training
  foremost, plus one staged item: **OC-NaVQA de-risking** (below). Each
  handoff ships as a hardened package modelled on `student_gpu_package`:
  pinned deps, checkpointing, loud errors.

### OC-NaVQA (DAAAM, CVPR 2026) — de-risk now, run later

OC-NaVQA is the only candidate benchmark that passes the no-invented-metrics
rule (their annotations, their scorer, ReMEmbR and ConceptGraphs baselines
already reported on it), and its winnable axes — graceful degradation,
conjunctive partial-cue queries — are our strongest. But a comparable number
requires CODa (`dense_global` poses), NaVQA preprocessing from the ReMEmbR
fork, an RTX-5090-class GPU, and GPT-5-mini in the reasoning loop, with the
CSV schema still unverified. That is a multi-week integration, off the ICRA
critical path. Staging:

1. **De-risk in parallel (collaborator-GPU handoff or a rough Paul run):**
   verify `data/oc-navqa_data.csv` schema, download CODa, run the
   ReMEmbR-fork preprocessing, confirm sequences load end-to-end. Mechanical,
   documented (`INSTALL.md`/`RUNNING.md`), checkable — legitimate handoff
   work.
2. **Decision point ~Sept 1:** if preprocessing is done and a minimal
   end-to-end query works, consider one supplementary sentence/row in the
   ICRA paper ("evaluated-on", not a head-to-head). Otherwise nothing is
   lost — the T-RO opener is staged and ready the week after submission.
3. **Either way** the ICRA paper cites DAAAM/OC-NaVQA in the A7 future-work
   positioning.
- **Paul, non-compute:** D2 authorship, angle choice (A1), Table III
  re-annotation go/no-go. Flagged on the board as the only items that block
  prose.

## 5. Spine rewrite pass

A targeted rewrite of `paper/icra6.tex`, not a fresh draft. Every changed
sentence traces to a row in the canonical evidence base. Scope:

- intro and limitations re-anchored to what survived: kernel smoothing →
  graceful degradation stands; the proximity and global-mass mechanism stories
  come out;
- the stream-ceiling decomposition presented as a *strength*: we can say
  exactly where the remaining gap lives and that ~+0.018 of it is unreachable
  by any memory change;
- the benchmark-coverage finding placed carefully — transferable, but must not
  read as excuse-making;
- the A7 paragraph positioning the overlay direction as future work (stakes
  the collaborators' claim without spending pages), citing DAAAM/OC-NaVQA as
  the staged next benchmark;
- the free B2 edit: cite submap-local frames in the merge caveat, converting
  "open problem" to "known fix, unmeasured".

Out of scope for this pass: the "6-page cut" (stale header comment — the file
is an 8-page submission at 8 pages), any new mechanism claims, anything
quoting a batch-1 verdict that has not been re-validated.

## Order of execution

1. Claim audit (§1) — everything else is scoped by it.
2. Wiki doc (§2) written from the audit; punch list and lanes assigned.
3. ICNS board (§3) published from the wiki doc.
4. CPU-local runs and figure jobs (§4) in punch-list order; board updated as
   items close.
5. Spine rewrite (§5) once the audit says which sentences must change and the
   re-validation says which numbers may be quoted.

Steps 4 and 5 interleave; the board is the coordination point. The OC-NaVQA
de-risk runs in parallel throughout, with its go/no-go decision ~Sept 1.

## Error handling / guard rails

- All re-validation runs go through the existing batch-1 guard
  (`report.py`): label-parity hard stop, seed battery, leave-one-scene-out
  breadth rule. No single-draw comparisons at the ±0.01 scale.
- Stale-cache quarantine rule stands: mixed-codebook caches go to
  `outputs/batch1/cache_stale_mixed_codebook/`, never deleted silently.
- No number enters the paper that is not in the canonical evidence base with
  provenance.
- Standing rules apply: no claims without evidence; qualitative artifact with
  every quantitative delivery; adopt the target benchmark's own scorer
  verbatim; individual files for Paul's review before any commit of results.

## Success criteria

- Every claim in icra6.tex is stamped SAFE, or has a punch-list item that
  makes it SAFE, or is deleted in the spine pass.
- The four figure `\todo`s and the kNN row `\todo` are closed.
- B1 is measured and its answer recorded before submission.
- The board shows, at any moment, what is running, what is blocked on Paul,
  and what is parked — with nothing ambiguous between lanes.
