# ICRA Consolidation & Work Board Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit every claim in `paper/icra6.tex` against the post-rebaseline evidence, produce a system-of-record wiki page + live ICNS command board, close the missing runs/figures, and land the spine rewrite — all serving the 2026-09-15 ICRA deadline.

**Architecture:** Claim-audit first: the audit (Task 1) scopes every downstream task. A wiki page is the system of record; an ICNS HTML artifact mirrors it glanceably. Experimental work runs through the existing batch-1 guard (`collab_tasks/batch1/report.py`); prose changes trace to evidence-base rows.

**Tech Stack:** Python 3 (numpy, matplotlib; CPU-local), existing `collab_tasks/batch1` harness, LaTeX (icra6.tex), ICNS design system vendored at `docs/brand/`.

**Spec:** `docs/superpowers/specs/2026-08-18-icra-consolidation-design.md`

## Global Constraints

- **Deadline:** ICRA 2027 submission, 2026-09-15. OC-NaVQA go/no-go: ~2026-09-01.
- **Canonical trace build** (reproduces `class_fields` bit-for-bit): `python student_gpu_package/04_vsa_labels.py --scene <scene>_cgfront --labels-from-points --max-per-class 400 --length-scale 0.45,0.27 --grid 96`. Note the non-default args; 04's own defaults do NOT reproduce the batch-1 baseline.
- **Guard rules (always):** all results via `collab_tasks/batch1/report.py` — 8 scenes × 5 seed tuples, paired deltas, `|mean| >= 2*sd` to resolve, leave-one-scene-out breadth rule, label-parity hard stop. No single-draw comparisons at the ±0.01 scale. Predictions frozen in a commit BEFORE running.
- **Stale caches:** mixed-codebook caches go to `outputs/batch1/cache_stale_mixed_codebook/`, never silently deleted.
- **No number enters the paper** unless it has a row in the canonical evidence base (Task 2) with provenance.
- **Review-before-commit:** paper edits and new results files go to Paul (SendUserFile) before `git commit`. Wiki/board documentation commits directly (established repo convention), but is also sent.
- **ICNS gotchas:** colourway switches on `[data-colourway="plate"]`, not `data-theme`; semantic tokens only (`--text-body`, `--border`, `--text-muted`); verify every `var(--…)` resolves; check both colourways. Invoke `/icns-design` before building.
- **Benchmark metrics:** ConceptGraphs' scorer verbatim (`student_gpu_package/05_score.py` path via `common.score`). No invented metrics.
- **Environment:** Windows, CPU-only local. Repo root: `C:\Users\30068379\OneDrive - Western Sydney University\Code\Semantic-Spiking-Neural-SLAM-2023`. Run python from repo root so `collab_tasks.batch1` imports resolve.

---

### Task 1: Claim audit of icra6.tex

**Files:**
- Create: `wiki/analysis/2026-08-18-icra-consolidation-and-work-board.md` (frontmatter + §Claim audit only; Task 2 adds the rest)
- Read: `paper/icra6.tex`, `wiki/analysis/2026-08-17-gap-closing-batch-results.md`, `wiki/analysis/2026-08-18-external-read-bounded-readout-and-benchmark-viability.md`, `wiki/analysis/2026-08-16-conceptgraphs-corrected-and-graceful-failure.md`, `wiki/analysis/2026-08-10-positions-and-plan.md`, `outputs/batch1/*.json`

**Interfaces:**
- Produces: the audit table. Columns, exactly: `# | tex line | claim (verbatim, ≤25 words) | type (number/mechanism/promise) | source of truth (path, date) | stamp (SAFE/STALE/MISSING) | action`. Also three summary lists: STALE claims, MISSING claims, and **quoted batch-1 verdicts** (this list is Task 4's entire scope; empty list = Task 4 collapses to a recording step).

- [ ] **Step 1: Read `paper/icra6.tex` end to end.** Extract every sentence containing a number, a mechanism claim ("because", "the reason", "driven by"), or a promise ("we show", "we release"). Record tex line numbers.
- [ ] **Step 2: Stamp each claim** against the evidence sources above. Danger zones first: pre-hash codebook numbers; any batch-1 verdict (all provisional); the retracted proximity and global-mass mechanisms; the superseded +0.113 headroom (→ 0.3844 stream ceiling). Also check the paper carries the good news: degradation holds-and-strengthens (commit `01980b9`), benchmark-measures-coverage (two independent routes), gap decomposition with ~+0.018 unreachable.
- [ ] **Step 3: Write the wiki page skeleton** with frontmatter (`title`, `type: analysis`, `status: active`, `created: 2026-08-18`, `tags: [icra, paper, consolidation, work-board]`) and the §Claim audit section containing the table + three summary lists.
- [ ] **Step 4: Mechanical coverage check.** `grep -nE "[0-9]+\.[0-9]+|[0-9]+\s*(KB|MB|ms|%|×|x)" paper/icra6.tex` — every hit is either in the audit table or listed under an explicit "not a claim" note (section numbers, equation refs, dimensions of the method itself).
- [ ] **Step 5: Send the file to Paul** (SendUserFile) and commit: `git add wiki/analysis/2026-08-18-icra-consolidation-and-work-board.md && git commit -m "wiki: icra6 claim audit (SAFE/STALE/MISSING, batch-1 quote scope)"`

### Task 2: Wiki consolidation doc (evidence base, threads, punch list, beyond)

**Files:**
- Modify: `wiki/analysis/2026-08-18-icra-consolidation-and-work-board.md`
- Modify: `wiki/index.md` (add 1 entry), `wiki/log.md` (add dated entry, matching existing entry format)

**Interfaces:**
- Consumes: Task 1's audit table and summary lists.
- Produces: §Canonical evidence base (the only lookup for "what may the paper say"), §Thread inventory, §Punch list, §Beyond lane. Punch-list item schema: `id | item | rank (blocks-submission/strengthens-spine/polish) | lane (cpu-local/paul-gpu/collab-gpu/paul-noncompute) | status (open/running/done/blocked-on-paul)`.

- [ ] **Step 1: Write §Canonical evidence base.** One paragraph per settled result, quotable number + provenance, all post-rebaseline. Seed rows (verify each against its source before writing; correct here if the source disagrees): ours 0.3198 mean mAcc hashed codebook vs ConceptGraphs 0.402 published (their Replica row 0.406, our reproduction 99%); stream ceiling 0.3844 (kNN-5 exact positions); grid quantisation cost +0.0388; superposition total cost +0.0098; ~+0.018 of gap unreachable by any memory change; graceful degradation 1.7× retained (76/92/78% vs 47/53/46%); merge exactness; capacity/χ law; per-scene trace 256 KB byte-identical across 1.8× observation range, total system 6.25–7.50 MB, materialised-decode contrast 195–342 MB; episodic readout 20× better than chance, median 0.13 m; bounded readout 29.94 MB / 0.41 ms / exact on 145 answerable queries.
- [ ] **Step 2: Write §Thread inventory** — for each: state, what it offers the paper, verdict **feed** or **park**. Threads: ConceptGraphs head-to-head; hashed-codebook re-baseline; degradation/robustness; per-object-ids fetch (in flight, Colab); episodic memory / bounded readout / OC-NaVQA; Table III re-annotation; whitening tension (B1); κ statistic; submap-local frames (B2); JEPA→FHRR; isotropy WP1–WP9; object orbit line; map merge thread; 3D cogmaps/rotation.
- [ ] **Step 3: Write §Punch list** from the audit's STALE+MISSING lists plus the paper's six `\todo`s (lines 69, 120, 237, 500, 609, 843) and the figure jobs, ranked and laned per the schema. Paul-noncompute items flagged: D2 authorship, A1 angle, Table III go/no-go.
- [ ] **Step 4: Write §Beyond lane** — T-RO delta, collaborators' overlay paper, JEPA, isotropy, orbit, merge: one line each, owner, earliest sensible date. Include OC-NaVQA staging summary with the ~Sept 1 decision point.
- [ ] **Step 5: Update `wiki/index.md` and `wiki/log.md`**, send to Paul, commit: `git commit -m "wiki: ICRA consolidation — evidence base, thread inventory, punch list, work board"`

### Task 3: ICNS command board artifact

**Files:**
- Create: `docs/sites/icra_board.html` (from `docs/brand/layout.html`)
- Modify: memory file `feedback-visual-reports-are-deliverables.md` (add the board's URL row)

**Interfaces:**
- Consumes: Task 2's punch list and thread inventory (content is transcribed, not invented).
- Produces: a private artifact URL; the board is updated whenever a punch-list item changes status (every later task's final step).

- [ ] **Step 1: Invoke the `icns-design` skill** (source of truth for the brand) and copy `docs/brand/layout.html` to `docs/sites/icra_board.html`.
- [ ] **Step 2: Build the four board sections:** countdown to 2026-09-15; punch list as status chips (open/running/done/blocked-on-Paul); three-lane work board + Paul-noncompute column; thread map with feed/park badges. Semantic tokens only.
- [ ] **Step 3: Token check.** For every `var(--X)` used: `grep -o "var(--[a-z-]*)" docs/sites/icra_board.html | sort -u`, confirm each is defined in `docs/brand/styles.css` or `docs/brand/tokens/`. Unknown custom properties fail silently — zero unmatched allowed.
- [ ] **Step 4: View both colourways** in the browser (default and `[data-colourway="plate"]`), fix contrast issues.
- [ ] **Step 5: Publish as a private artifact** (Artifact tool, favicon 🗺️, title "ICRA Command Board"), add the URL to the live-URL list in the `feedback-visual-reports-are-deliverables` memory file, send to Paul, commit the HTML.

### Task 4: Scoped batch-1 re-validation under the hashed codebook

**Files:**
- Run: `collab_tasks/batch1/h*.py` — only those on Task 1's **quoted batch-1 verdicts** list
- Modify: `wiki/analysis/2026-08-18-icra-consolidation-and-work-board.md` (stamp updates), `docs/sites/icra_board.html` (status)

**Interfaces:**
- Consumes: Task 1's quoted-verdict list. **If the list is empty:** record "paper quotes no batch-1 verdicts; STALE stamps from this source removed; no re-run needed" in the wiki page and stop — that is a valid completion.
- Produces: per-mechanism verdicts (guard JSON in `outputs/batch1/`), evidence-base rows for any number the paper may now quote.

- [ ] **Step 1: Pre-flight.** Confirm the cache state is clean: baselines in `outputs/batch1/` post-`rebaseline.py` (commit `3161182`), stale caches quarantined. Run `python -c "from collab_tasks.batch1 import common; print(common.BASELINE_MACC)"` and eyeball against the wiki's re-baselined values.
- [ ] **Step 2: Freeze predictions.** For each scoped mechanism write the Amendment-A6 prediction block (expected sign, magnitude band, breadth clause `scenes_ge >= 6`, office4-excluded mean reported beside headline) into the wiki page; commit BEFORE running: `git commit -m "freeze: re-validation predictions"`.
- [ ] **Step 3: Run each scoped screen** from repo root, e.g. `python collab_tasks/batch1/h2_spread_norm.py` (each screen calls `run_screen`, which invokes the guard and writes JSON). Runtime reference: full Stage 1 was ~38 min CPU; a scoped subset is less.
- [ ] **Step 4: Record verdicts** in the wiki page (stamp STALE→SAFE only where the guard verdict supports the paper's sentence; otherwise the sentence goes on Task 8's deletion list). Update board statuses.
- [ ] **Step 5: Send guard JSONs + wiki diff to Paul, then commit.**

### Task 5: Storage-matched kNN fp16 baseline row (icra6.tex:609)

**Files:**
- Create: `collab_tasks/batch1/knn_storage_matched.py`
- Test: inline `_self_test()` in the same file (matches batch1 convention — see `common.py:256`)
- Modify: `paper/icra6.tex:609` (fill the `\todo`), wiki + board

**Interfaces:**
- Consumes: `from collab_tasks.batch1.common import SCENES, load_scene, score, GRID` and the seed battery convention (`default_fields(data, ti)` pairing).
- Produces: one table row: mAcc (mean ± sd across 5 tuples, 8 scenes) for a 32 KB fp16 point-list baseline, plus the guard JSON `outputs/batch1/knn_storage_matched.json`.

- [ ] **Step 1: Read the tex context** around line 609 to fix the exact protocol the row promises (budget, what "storage-matched" covers). Default protocol if the tex leaves freedom (record the choice in the file docstring): budget 32,768 bytes; each stored point = 2 coords fp16 (4 B) + class id uint8 (1 B) = 5 B → 6,553 points; per seed tuple, uniform subsample of the capped observation stream; decode = kNN (k=1 and k=5) evaluated at grid-cell centres — **equal quantisation** with our field (the `proximity_ceiling.py` lesson: never score a point-exact decoder against a grid-scored field).
- [ ] **Step 2: Write `_self_test()` first:** (a) budget accounting asserts `n_points * 5 <= 32768`; (b) on `make_blob_data()`-style toy input, k=1 prediction at a cell equals the label of the nearest stored point. Run `python collab_tasks/batch1/knn_storage_matched.py --self-test` → must fail (module empty), then implement, then pass.
- [ ] **Step 3: Freeze prediction** (band: between NN-grid 0.3288 and kNN-5-exact 0.3844 from `proximity_ceiling.py`; breadth clause) — commit before the full run.
- [ ] **Step 4: Full run** (8 scenes × 5 tuples), score via `common.score`, guard via `report.report`. Write JSON.
- [ ] **Step 5: Fill icra6.tex:609** with the measured number, add the evidence-base row, update board, send script+JSON+tex diff to Paul, commit after his OK.

### Task 6: B1 whitening tension experiment

**Files:**
- Create: `collab_tasks/scripts/whitening_tension.py`
- Read/reuse: `collab_tasks/scripts/clip_phasor_retention.py` (loaders, W-draw machinery), `vsa_cognitive_mapping/classroom_pipeline.py` (binding-key setup)
- Output: `outputs/whitening_tension.json`; wiki §addendum + board

**Interfaces:**
- Produces: two verdicts — Arm A (whitening the **binding key**; must reproduce the known ~−47.6% classroom cost, else the harness is wrong and no Arm B verdict is issued) and Arm B (whitening the **FPE argument**, Shay's usage: preprocess real CLIP crop embeddings with raw/centre/z-score/whiten before fractional-power encoding; metric = retrieval R@1/retention as in `clip_phasor_retention.py`, 5 seeds, verdict by the `|mean| >= 2*sd` rule).

- [ ] **Step 1: Write Arm A as the harness self-check** — replicate the classroom binding-key comparison (centre vs z-score vs whiten, 5 seeds). Gate: whiten arm must lose by the recorded magnitude's sign and order (≈40–55% signal cost). Run; if the gate fails, stop and debug the harness — do not proceed to Arm B.
- [ ] **Step 2: Freeze the Arm B prediction** (Shay's claim: whitening the FPE argument is harmless-or-helpful because FPE outputs are quasi-orthogonal by construction; our nulls: no resolved delta). Commit the frozen block.
- [ ] **Step 3: Run Arm B** on real CLIP features (the E0 crop set used by `clip_phasor_retention.py`; note there that mean-centring HELPED synthetic but HURT real — 0.466 vs 0.674 — so real features are mandatory).
- [ ] **Step 4: Record both verdicts** in the wiki page as the resolution of the positions-doc §"one real technical tension"; state scope precisely (binding keys vs FPE arguments). Update the paper's whitening sentence if it has one (audit will have flagged it). Update board; send to Paul; commit.

### Task 7: Figures batch

**Files:**
- Modify/run: `tools/make_hero_figure.py`
- Create: `tools/make_belief_montage.py`, `tools/make_chess_qualitative.py`
- Locate: the capacity-law figure generator (`grep -rn "law" tools/ paper/figures/` and the `\includegraphics` name at icra6.tex:~500)
- Output: `paper/figures/*.pdf|png`; wiki + board

**Interfaces:**
- Consumes: `collab_tasks.batch1.common.load_scene` + `class_fields` for belief fields; the real-GT grounding branch outputs for hero panel (c) (icra6.tex:120 names the source).
- Produces: four figure files referenced by existing `\includegraphics` names (do not rename targets; overwrite in place so the tex compiles unchanged).

- [ ] **Step 1: Hero panel (c).** Read the `\todo` at icra6.tex:120 and `tools/make_hero_figure.py`'s CLI; regenerate panel (c) from the real-GT grounding branch; rebuild `paper/figures/hero.pdf`/`.png`.
- [ ] **Step 2: Capacity-law figure legend.** Locate the generator, set legend labels to match the notation of Eq. (law) as the `\todo` at icra6.tex:500 requires, regenerate.
- [ ] **Step 3: Belief-field montage.** New `tools/make_belief_montage.py`: for 4 chosen scenes, render the signed class-field (diverging colormap, zero-centred) for 2 illustrative classes each via `class_fields`, plus GT object outlines; export one figure. Keep the field-not-object-list framing (render the field, per the stratify-by-confident rule).
- [ ] **Step 4: Chess qualitative.** New `tools/make_chess_qualitative.py` from the 7-Scenes chess experiment outputs (locate under `outputs/`; the A5 item in the positions doc). Panel: query → field response → retrieved location vs GT.
- [ ] **Step 5: Compile the paper** (`latexmk -pdf icra6.tex` from `paper/`), visually check all four figures at print size, confirm page count is 8, send PNGs to Paul, commit after his OK.

### Task 8: Spine rewrite of icra6.tex

**Files:**
- Modify: `paper/icra6.tex`
- Consumes: Task 1 audit (deletion list), Task 4 verdicts, Task 5 number, Task 6 verdicts.

**Interfaces:**
- Produces: a compiling, 8-page icra6.tex in which every claim is SAFE. Each edit carries a `% evidence:` comment naming its evidence-base row (stripped before camera-ready, kept for review).

- [ ] **Step 1: Mechanism removals.** Delete/rewrite every sentence on the audit's STALE list: the proximity-is-the-gap story, the global-mass story, the +0.113 headroom. Replacement framing per spec: kernel smoothing → graceful degradation stands.
- [ ] **Step 2: Stream-ceiling strength paragraph.** Present the decomposition as a strength: exactly where the remaining gap lives; ~+0.018 unreachable by any memory change; our own grid costs +0.0388 (larger than superposition's total +0.0098).
- [ ] **Step 3: Benchmark-coverage finding** — one careful paragraph (two independent routes: oracle-labels lower mAcc; corruption raised office4/room2), placed in Experiments discussion, phrased as a property of the metric, not an excuse.
- [ ] **Step 4: A7 future-work paragraph** — position the overlay direction (Naitri/Lorin) as future work; cite DAAAM/OC-NaVQA as the staged next benchmark. **B2:** cite submap-local frames in the merge caveat ("known fix, unmeasured").
- [ ] **Step 5: Numbers pass.** Insert the Task 5 kNN row; update any number the re-validation moved; per-scene trace framing per evidence base ("constant in objects and observations" claim, incremental-state vs fixed-infrastructure split — never unqualified "32 KB").
- [ ] **Step 6: Compile, page-count (8), and audit re-check** — re-run Task 1 Step 4's grep; every number must trace. Send the full tex diff to Paul; commit only after his OK.

### Task 9: OC-NaVQA de-risk handoff package

**Files:**
- Create: `collab_tasks/handoffs/ocnavqa_derisk/README.md`, `check_csv.py`, `smoke_load.py`, `requirements.txt`
- Modify: wiki (beyond lane status) + board

**Interfaces:**
- Produces: a collaborator-runnable package whose end state is "CODa + NaVQA sequences preprocessed, one query loads end-to-end", reported back as a filled-in checklist in the README. Go/no-go read on ~2026-09-01.

- [ ] **Step 1: Read the released annotation schema** via WebFetch of the raw CSV (`MIT-SPARK/DAAAM` → `data/oc-navqa_data.csv`, ~29 KB) — header + 5 rows; record columns and QA-pair count in the wiki (closes the "schema unverified" flag).
- [ ] **Step 2: Write `check_csv.py`:** downloads nothing; takes a local path; asserts the recorded columns exist, prints row/QA counts, exits non-zero loudly on mismatch.
- [ ] **Step 3: Write `smoke_load.py`:** given local CODa root + preprocessed NaVQA sequence dir, loads one sequence, checks `dense_global` poses are used (assert on the pose file path — the known gotcha), prints frame count and one QA pair.
- [ ] **Step 4: Write `README.md`** as a numbered runbook: clone DAAAM (BSD-3, follow their `INSTALL.md`/`RUNNING.md`); download CODa (link, `dense_global` warning); preprocess NaVQA via the ReMEmbR fork (`github.com/nicogorlo/remembr` — scripts live THERE, not in DAAAM); run `check_csv.py` then `smoke_load.py`; fill in the result checklist. Pin deps in `requirements.txt`. State the GPU note (RTX-5090-class for their runtimes; GPT-5-mini only needed for the full eval, NOT for this de-risk).
- [ ] **Step 5: Send the package to Paul** to forward to the collaborator; commit; put the ~Sept 1 decision point on the board with status `running (external)`.

### Task 10: Close-out

**Files:** wiki page, board, memory.

- [ ] **Step 1:** Reconcile the board with reality: every punch-list item done/parked/blocked-on-Paul with nothing ambiguous; success criteria from the spec checked one by one (all claims SAFE-or-deleted; five `\todo`s closed — the sixth, funding/authorship, is Paul's; B1 recorded; board unambiguous).
- [ ] **Step 2:** Update memory: `project_icra_consolidation` pointer if not already present; artifact URL row confirmed; wiki `log.md` close-out entry. Final commit.

## Task order & dependencies

1 → 2 → 3 (docs chain). 1 → 4 (scope). 5, 6, 7 independent after 1; 9 independent of everything (start immediately, external latency). 8 gates on 1, 4, 5, 6 (and uses 7's figures). 10 last. Suggested start order: **1, 9, 2, 3, 5, 6, 7, 4, 8, 10** — Task 9 first-ish because its latency is external and the Sept 1 decision needs runway.

## Self-review notes

- Spec coverage: §1→Task 1, §2→Task 2, §3→Task 3, §4 lanes→Tasks 4–7 (CPU), 9 (collab-GPU staged), Paul-GPU items tracked on the board (per-object-ids fetch is in flight, not a task here), §5→Task 8, OC-NaVQA staging→Task 9, order-of-execution→above.
- The paper's Paul-only items (authorship, angle, email, funding) are board rows, deliberately not tasks.
- Type consistency: all harness references use real names from `collab_tasks/batch1/common.py` (`SCENES`, `BASELINE_MACC`, `load_scene`, `class_fields`, `default_fields`, `score`, `run_screen`, `make_blob_data`, `_self_test` convention) and `report.py`'s guard.
