from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw"
WIKI_DIR = ROOT / "wiki"
INDEX_FILE = WIKI_DIR / "index.md"
LOG_FILE = WIKI_DIR / "log.md"

EXCLUDED_ORPHAN_FILES = {
    "index.md",
    "log.md",
    "overview.md",
    "open-questions.md",
    "README.md",
}


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_date(ts: dt.datetime | None = None) -> str:
    ts = ts or utc_now()
    return ts.date().isoformat()


def iso_stamp(ts: dt.datetime | None = None) -> str:
    ts = ts or utc_now()
    return ts.strftime("%Y-%m-%d %H:%M UTC")


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "untitled"


def ensure_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def ensure_scaffold() -> None:
    dirs = [
        RAW_DIR / "inbox",
        RAW_DIR / "assets",
        WIKI_DIR / "sources",
        WIKI_DIR / "experiments",
        WIKI_DIR / "concepts",
        WIKI_DIR / "methods",
        WIKI_DIR / "datasets",
        WIKI_DIR / "analysis",
        WIKI_DIR / "lint",
    ]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)

    ensure_file(
        RAW_DIR / "README.md",
        "# Raw Sources\n\n"
        "This folder is immutable source-of-truth input.\n\n"
        "- Put new sources into `raw/inbox/`.\n"
        "- Put downloaded images/assets into `raw/assets/`.\n"
        "- Do not edit source files during wiki maintenance.\n",
    )
    ensure_file(
        WIKI_DIR / "overview.md",
        "---\n"
        "title: Wiki Overview\n"
        "type: concept\n"
        "status: active\n"
        f"created: {iso_date()}\n"
        f"updated: {iso_date()}\n"
        "source_paths: []\n"
        "tags: [overview]\n"
        "---\n\n"
        "## Summary\n\n"
        "High-level map of this research wiki.\n\n"
        "## Key Points\n\n"
        "- Track experimental evidence, not only outcomes.\n"
        "- Keep contradictions explicit.\n"
        "- Save high-value Q&A as durable analysis pages.\n",
    )
    ensure_file(
        WIKI_DIR / "open-questions.md",
        "---\n"
        "title: Open Questions\n"
        "type: analysis\n"
        "status: active\n"
        f"created: {iso_date()}\n"
        f"updated: {iso_date()}\n"
        "source_paths: []\n"
        "tags: [questions]\n"
        "---\n\n"
        "## Active Questions\n\n"
        "- Which representation choices most affect loop-closure robustness?\n"
        "- What ablation should be run next to reduce uncertainty?\n",
    )
    ensure_file(
        INDEX_FILE,
        "# Wiki Index\n\n"
        "Use this catalog to locate relevant pages quickly.\n\n"
        "## Overview\n\n"
        "- [overview](overview.md) - High-level map of this wiki.\n"
        "- [open-questions](open-questions.md) - Current unresolved questions.\n\n"
        "## Sources\n\n"
        "_No source pages yet._\n\n"
        "## Experiments\n\n"
        "_No experiment pages yet._\n\n"
        "## Concepts\n\n"
        "_No concept pages yet._\n\n"
        "## Methods\n\n"
        "_No method pages yet._\n\n"
        "## Datasets\n\n"
        "_No dataset pages yet._\n\n"
        "## Analysis\n\n"
        "_No analysis pages yet._\n\n"
        "## Lint Reports\n\n"
        "_No lint reports yet._\n",
    )
    ensure_file(
        LOG_FILE,
        "# Wiki Log\n\n"
        "Append-only timeline of ingest/query/lint operations.\n",
    )


def append_log(operation: str, title: str, details: list[str]) -> None:
    ensure_scaffold()
    lines = [f"\n## [{iso_stamp()}] {operation} | {title}\n"]
    lines.extend([f"- {detail}\n" for detail in details])
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.writelines(lines)


def upsert_index_entry(section: str, rel_path: Path, summary: str) -> None:
    ensure_scaffold()
    text = INDEX_FILE.read_text(encoding="utf-8")
    lines = text.splitlines()
    section_header = f"## {section}"
    entry = f"- [{rel_path.stem}]({rel_path.as_posix()}) - {summary}"

    try:
        start = lines.index(section_header)
    except ValueError:
        lines.extend(["", section_header, "", entry])
        INDEX_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("## "):
            end = i
            break

    block = lines[start + 1 : end]
    block = [ln for ln in block if ln.strip() not in {"_No source pages yet._", "_No experiment pages yet._", "_No concept pages yet._", "_No method pages yet._", "_No dataset pages yet._", "_No analysis pages yet._", "_No lint reports yet._"}]

    if entry not in block:
        if block and block[-1].strip() != "":
            block.append("")
        block.append(entry)
    lines = lines[: start + 1] + block + lines[end:]
    INDEX_FILE.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def page_frontmatter(title: str, page_type: str, source_paths: list[str] | None = None, tags: list[str] | None = None) -> str:
    source_paths = source_paths or []
    tags = tags or []
    src_str = ", ".join(f'"{p}"' for p in source_paths)
    tag_str = ", ".join(tags)
    return (
        "---\n"
        f"title: {title}\n"
        f"type: {page_type}\n"
        "status: draft\n"
        f"created: {iso_date()}\n"
        f"updated: {iso_date()}\n"
        f"source_paths: [{src_str}]\n"
        f"tags: [{tag_str}]\n"
        "---\n"
    )


def create_ingest_page(source: str, title: str, kind: str, summary: str) -> Path:
    ensure_scaffold()
    source_path = Path(source)
    page_dir = WIKI_DIR / ("experiments" if kind == "experiment" else "sources")
    filename = f"{iso_date()}-{slugify(title)}.md"
    page_path = page_dir / filename

    if page_path.exists():
        raise FileExistsError(f"Page already exists: {page_path}")

    body = (
        page_frontmatter(
            title=title,
            page_type=kind,
            source_paths=[source_path.as_posix()],
            tags=[kind],
        )
        + "\n## Summary\n\n"
        + f"{summary}\n\n"
        + "## Key Points\n\n"
        + "- TODO: Add key findings.\n\n"
        + "## Evidence\n\n"
        + "- TODO: Add concrete evidence with references.\n\n"
        + "## Contradictions / Tensions\n\n"
        + "- TODO: Record contradictions and uncertainty.\n\n"
        + "## Related Pages\n\n"
        + "- TODO: Link related concept/method/dataset pages.\n\n"
        + "## Open Questions\n\n"
        + "- TODO: Add follow-up questions.\n"
    )
    page_path.write_text(body, encoding="utf-8")
    return page_path


def create_query_page(question: str, title: str) -> Path:
    ensure_scaffold()
    filename = f"{iso_date()}-{slugify(title)}.md"
    page_path = WIKI_DIR / "analysis" / filename
    if page_path.exists():
        raise FileExistsError(f"Page already exists: {page_path}")

    body = (
        page_frontmatter(title=title, page_type="analysis", tags=["query"])
        + "\n## Question\n\n"
        + f"{question}\n\n"
        + "## Answer\n\n"
        + "TODO: Add synthesized answer.\n\n"
        + "## Citations\n\n"
        + "- TODO: List wiki pages and raw sources used.\n\n"
        + "## Follow-ups\n\n"
        + "- TODO: Add next questions and experiments.\n"
    )
    page_path.write_text(body, encoding="utf-8")
    return page_path


def markdown_links(text: str) -> list[str]:
    links = []
    for match in re.findall(r"\[\[[^\]|#]+", text):
        links.append(match[2:].strip())
    for match in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        if ".md" in match:
            links.append(match.split("#")[0].strip())
    return links


def normalize_link(from_file: Path, link: str) -> Path:
    if link.endswith(".md"):
        path = (from_file.parent / link).resolve()
    else:
        path = (from_file.parent / f"{link}.md").resolve()
    return path


def lint_wiki() -> tuple[Path, dict[str, int]]:
    ensure_scaffold()
    wiki_files = [
        p for p in WIKI_DIR.rglob("*.md") if p.name not in {"README.md"}
    ]
    existing = {p.resolve() for p in wiki_files}
    inbound: dict[Path, int] = {p.resolve(): 0 for p in wiki_files}
    broken: list[tuple[Path, str]] = []
    placeholders: list[Path] = []

    placeholder_re = re.compile(r"\b(TODO|TBD|PLACEHOLDER)\b", re.IGNORECASE)
    for page in wiki_files:
        text = page.read_text(encoding="utf-8")
        if placeholder_re.search(text):
            placeholders.append(page)
        for link in markdown_links(text):
            resolved = normalize_link(page, link)
            if resolved in existing:
                inbound[resolved] = inbound.get(resolved, 0) + 1
            else:
                broken.append((page, link))

    orphans = []
    for page in wiki_files:
        if page.name in EXCLUDED_ORPHAN_FILES:
            continue
        if inbound.get(page.resolve(), 0) == 0:
            orphans.append(page)

    report_path = WIKI_DIR / "lint" / f"{iso_date()}-lint.md"
    rel = lambda p: p.relative_to(ROOT).as_posix()

    lines = [
        page_frontmatter(
            title=f"Wiki Lint {iso_date()}",
            page_type="lint",
            tags=["lint", "health-check"],
        ),
        "",
        "## Summary",
        "",
        f"- Pages scanned: {len(wiki_files)}",
        f"- Broken links: {len(broken)}",
        f"- Orphan pages: {len(orphans)}",
        f"- Placeholder pages: {len(placeholders)}",
        "",
        "## Broken Links",
        "",
    ]
    if broken:
        lines.extend([f"- `{rel(src)}` -> `{target}`" for src, target in broken])
    else:
        lines.append("- None")

    lines.extend(["", "## Orphan Pages", ""])
    if orphans:
        lines.extend([f"- `{rel(p)}`" for p in orphans])
    else:
        lines.append("- None")

    lines.extend(["", "## Placeholder Pages", ""])
    if placeholders:
        lines.extend([f"- `{rel(p)}`" for p in placeholders])
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Suggested Next Actions",
            "",
            "- Add links from high-level concept pages to orphan pages.",
            "- Resolve or explicitly defer placeholder sections.",
            "- Fix broken links to keep navigation trustworthy.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path, {
        "pages": len(wiki_files),
        "broken": len(broken),
        "orphans": len(orphans),
        "placeholders": len(placeholders),
    }


def cmd_init(_: argparse.Namespace) -> None:
    ensure_scaffold()
    append_log(
        operation="init",
        title="Initialize wiki scaffold",
        details=[
            "Created wiki and raw directory scaffold (if missing).",
            "Initialized index.md and log.md (if missing).",
        ],
    )
    print("Initialized wiki scaffold.")


def cmd_ingest(args: argparse.Namespace) -> None:
    source = args.source
    source_path = (ROOT / source).resolve() if not Path(source).is_absolute() else Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    title = args.title or source_path.stem.replace("_", " ").strip()
    summary = args.summary or "TODO: Replace with concise summary."
    kind = args.kind

    page_path = create_ingest_page(source=str(source_path), title=title, kind=kind, summary=summary)
    rel_page = page_path.relative_to(WIKI_DIR)
    section = "Experiments" if kind == "experiment" else "Sources"
    upsert_index_entry(section, rel_page, f"{kind} page created on {iso_date()}.")
    append_log(
        operation="ingest",
        title=title,
        details=[
            f"Source: `{source_path}`",
            f"Created page: `wiki/{rel_page.as_posix()}`",
            f"Kind: `{kind}`",
        ],
    )
    print(f"Ingest page created: {page_path}")


def cmd_query(args: argparse.Namespace) -> None:
    question = args.question.strip()
    title = args.title or question[:80]
    page_path = create_query_page(question=question, title=title)
    rel_page = page_path.relative_to(WIKI_DIR)
    upsert_index_entry("Analysis", rel_page, f"query logged on {iso_date()}.")
    append_log(
        operation="query",
        title=title,
        details=[
            f"Question: {question}",
            f"Created analysis page: `wiki/{rel_page.as_posix()}`",
        ],
    )
    print(f"Query page created: {page_path}")


def cmd_lint(_: argparse.Namespace) -> None:
    report_path, stats = lint_wiki()
    rel_report = report_path.relative_to(WIKI_DIR)
    upsert_index_entry("Lint Reports", rel_report, f"{stats['broken']} broken, {stats['orphans']} orphan.")
    append_log(
        operation="lint",
        title=f"Wiki health check {iso_date()}",
        details=[
            f"Pages scanned: {stats['pages']}",
            f"Broken links: {stats['broken']}",
            f"Orphan pages: {stats['orphans']}",
            f"Placeholder pages: {stats['placeholders']}",
            f"Report: `wiki/{rel_report.as_posix()}`",
        ],
    )
    print(
        "Lint complete: "
        f"pages={stats['pages']} broken={stats['broken']} "
        f"orphans={stats['orphans']} placeholders={stats['placeholders']}"
    )


def cmd_search(args: argparse.Namespace) -> None:
    ensure_scaffold()
    query = args.query.strip().lower()
    limit = max(1, args.limit)

    hits: list[tuple[int, Path, str]] = []
    for page in WIKI_DIR.rglob("*.md"):
        text = page.read_text(encoding="utf-8")
        lines = text.splitlines()
        score = text.lower().count(query)
        if score <= 0:
            continue
        snippet = ""
        for line in lines:
            if query in line.lower():
                snippet = line.strip()
                break
        hits.append((score, page, snippet))

    hits.sort(key=lambda x: x[0], reverse=True)
    for score, page, snippet in hits[:limit]:
        rel = page.relative_to(ROOT).as_posix()
        print(f"{score:>3} | {rel} | {snippet}")

    if not hits:
        print("No matches found.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wiki maintenance helper CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Initialize wiki scaffold.")
    p_init.set_defaults(func=cmd_init)

    p_ingest = sub.add_parser("ingest", help="Create a source/experiment ingest page.")
    p_ingest.add_argument("--source", required=True, help="Source path (relative to repo root or absolute).")
    p_ingest.add_argument("--kind", choices=["source", "experiment"], default="source")
    p_ingest.add_argument("--title", default=None, help="Optional human-readable title.")
    p_ingest.add_argument("--summary", default=None, help="Optional one-line summary.")
    p_ingest.set_defaults(func=cmd_ingest)

    p_query = sub.add_parser("query", help="Create an analysis page from a question.")
    p_query.add_argument("--question", required=True, help="Question text.")
    p_query.add_argument("--title", default=None, help="Optional page title.")
    p_query.set_defaults(func=cmd_query)

    p_lint = sub.add_parser("lint", help="Run wiki health checks and create report.")
    p_lint.set_defaults(func=cmd_lint)

    p_search = sub.add_parser("search", help="Simple local full-text search over wiki markdown.")
    p_search.add_argument("--query", required=True, help="Case-insensitive text query.")
    p_search.add_argument("--limit", type=int, default=10, help="Max number of matches to print.")
    p_search.set_defaults(func=cmd_search)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
