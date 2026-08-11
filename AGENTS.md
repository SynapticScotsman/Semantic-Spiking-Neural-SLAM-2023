# AGENTS.md

This repository uses an LLM-maintained wiki as a persistent knowledge layer on top of experiments and source documents.

The assistant should treat this file as the operating contract for wiki maintenance.

Agent and human quick entrypoint: `wiki/START_HERE.md`.

Cursor-specific workflow overlays:

- `.cursor/rules/agent-workflow.mdc`
- `.cursor/rules/karpathy-guidelines.mdc`
- `.cursor/skills/karpathy-guidelines/SKILL.md`
- `.cursor/commands/wiki-ingest.md`
- `.cursor/commands/wiki-lint.md`
- `docs/CURSOR_AGENT_BEST_PRACTICES_AUDIT.md`

## Mission

Maintain a compounding, cross-linked markdown wiki under `wiki/` that captures:

- What was ingested (papers, notes, experiment outputs, discussions)
- What was learned (claims, evidence, contradictions, open questions)
- What changed over time (chronological log)

Raw inputs remain immutable under `raw/`. The wiki is the mutable synthesis layer.

## Execution Kernel (Adapted)

All wiki maintenance should follow:

1. Read before write.
2. Make the smallest change that works.
3. Verify with evidence before claiming completion.

Additional guardrails:

- Name pages and sections clearly enough for a future reader to infer intent quickly.
- Prefer deleting stale or duplicate wiki content over preserving dead fragments.
- Treat failure paths as part of the spec: record what can break and recovery/rollback notes.

## Canonical Layout

- `raw/` immutable sources and assets
- `wiki/` synthesized markdown pages
- `wiki/index.md` content-oriented catalog
- `wiki/log.md` chronological append-only operations log
- `tools/wiki_ops.py` helper CLI for init/ingest/query/lint

Within `wiki/`, use these categories:

- `sources/`
- `experiments/`
- `concepts/`
- `methods/`
- `datasets/`
- `analysis/`
- `lint/`

## Core Workflows

### 1) Ingest

When asked to ingest a new source:

1. Read the source from `raw/`.
2. Define acceptance criteria for the ingest before editing (for example: pages updated, contradictions surfaced, links added).
3. Create or update a page in `wiki/sources/` (or `wiki/experiments/` when source is an experiment result).
4. Update affected concept/method/dataset pages.
5. Update `wiki/index.md`.
6. Append an entry to `wiki/log.md`.
7. Explicitly flag contradictions and unresolved uncertainty.

Preferred command:

`python tools/wiki_ops.py ingest --source <path-to-source> --kind source`

### 2) Query

When asked a question:

1. Read `wiki/index.md` first to identify relevant pages.
2. Synthesize from wiki pages first; only fall back to raw files if needed.
3. Cite the pages used.
4. If the result is high value, save it to `wiki/analysis/`.
5. Update `wiki/index.md` and append to `wiki/log.md`.
6. Include verification notes (what evidence supported the answer and what remains uncertain).

Preferred command:

`python tools/wiki_ops.py query --question "<question>"`

### 3) Lint (Wiki Health Check)

Perform periodic maintenance:

- Detect broken links
- Detect orphan pages (no inbound links)
- Detect unresolved placeholders (`TODO`, `TBD`, etc.)
- Suggest missing cross-references and missing concept pages

Save lint reports under `wiki/lint/`, then log the run.

Preferred command:

`python tools/wiki_ops.py lint`

## Page Conventions

### Frontmatter

New wiki pages should include YAML frontmatter with:

- `title`
- `type` (source|experiment|concept|method|dataset|analysis|lint)
- `status` (draft|active|archived)
- `created`
- `updated`
- `source_paths` (list; empty when not applicable)
- `tags` (list)

### Content Sections

Use these sections where applicable:

- `## Summary`
- `## Key Points`
- `## Evidence`
- `## Contradictions / Tensions`
- `## Related Pages`
- `## Open Questions`

### Linking

- Prefer relative markdown links between pages.
- Add links both from new pages to related pages and back from key related pages when possible.
- Keep links specific and actionable.

## Safety / Integrity Rules

- Never modify files under `raw/` unless explicitly requested by the user.
- Never delete wiki content unless explicitly requested; prefer marking `status: archived`.
- Preserve historical entries in `wiki/log.md` (append-only).
- If uncertain, record uncertainty explicitly instead of inventing facts.
- One human (repo owner) remains the decision owner for major direction changes.
- For parallel agent work, use bounded write scopes and a verifier pass before closeout.

## Operating Style

- Keep summaries concise and evidence-oriented.
- Distinguish observation vs interpretation.
- Surface disagreements between sources quickly.
- Favor incremental edits over large rewrites.

## Design System (WSU · ICNS)

All HTML the repo publishes — results pages, inspectors, explainers, talk pages —
uses the WSU/ICNS brand. Do not invent a palette or type scale per page.

- Source of truth: the `icns-design` skill (`~/.claude/skills/icns-design/`).
  `README.md` there is the authoritative brand document; invoke it with
  `/icns-design` before designing any page or asset.
- Vendored into this repo at `docs/brand/`:
  - `styles.css` → `tokens/{colors,typography,spacing,video}.css`
  - `assets/WSU_Logo_ICNS_{RGB,White}.png` — supplied lockups, never redraw or recolour
  - `layout.html` — the base page template (lockup, eyebrow/title, rule, stats,
    media plate, callout, table, prose, footer). Start new pages from this.

Token contract — use the semantic names, not the raw ramp:

- Colourway switches on `[data-colourway="plate"]` on `<html>`; the default
  `:root` set is Paper (light). There is no `data-theme` attribute.
- Flip with the colourway: `--bg`, `--text-body`, `--text-muted`, `--brand`,
  `--accent`, `--border`, `--surface-card`, `--surface-panel`, `--good`, `--warn`.
- Do NOT use `--ink`, `--line`, `--panel`, `--paper` for page chrome — those are
  raw Paper values and stay light on the dark colourway.
- Media (figures, clips) sits on the fixed deep plate `--plate-deep` (#05070A) in
  BOTH colourways. `--plate` (#16181B) is the dark page surface, not the media plate.
- Type roles are `--size-*` / `--track-*` / `--measure-body` / `--font-{display,body,prose,mono}`.
- Figure and video colours come from `tokens/video.css` (`--viz-*`), which carries a
  CVD audit — respect its "do not pair" notes.

Content rules: sentence case, Australian spelling, mono uppercase eyebrows with
`·` separators, numbers in mono tabular figures, no emoji, no marketing voice.

Verification: after building a page, confirm every `var(--…)` it uses actually
resolves (an unresolved custom property fails silently and the rule is dropped),
and check both colourways.
