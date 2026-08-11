# CLAUDE.md

Project instructions live in [AGENTS.md](AGENTS.md) — read it first. It is the
operating contract for the wiki, the execution kernel, and the design system.

## Design system (short version)

Any HTML this repo publishes uses the WSU · ICNS brand. Invoke the `icns-design`
skill (`/icns-design`) before designing a page or asset; it is the source of truth.

The system is vendored at `docs/brand/` — `styles.css`, `tokens/`, `assets/` (the
supplied lockups), and `layout.html`, the base template new pages start from.

Two gotchas that fail silently, both covered in full in AGENTS.md:

- The colourway switches on `[data-colourway="plate"]`, not `data-theme`.
- Use the semantic tokens (`--text-body`, `--border`, `--text-muted`). The raw
  `--ink`, `--line`, `--panel` do not flip with the colourway.

After building a page, check that every `var(--…)` resolves — an unknown custom
property drops the whole declaration with no error — and view both colourways.
