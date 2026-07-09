#!/usr/bin/env python3
"""Search scholarly literature for mentions of the Telluride Neuromorphic Workshop.

Google Scholar has no public API and aggressively blocks scripted access, so
this tool deliberately does NOT scrape Scholar. Instead it queries services
that index the same literature and are built for programmatic use:

  * OpenAlex        -- free, no key; searches title + abstract + full text
  * Europe PMC      -- free, no key; true full-text search of open-access papers
  * Semantic Scholar -- free; title/abstract search (API key optional, raises limits)
  * Crossref        -- free, no key; publisher metadata search
  * SerpAPI (optional) -- paid service that returns real Google Scholar results
                          through its own licensed infrastructure; enabled only
                          when SERPAPI_KEY is set

Results from all sources are de-duplicated (by DOI, then normalised title)
and written to CSV and JSON.

Usage:
    python telluride_search.py                          # default query, all free sources
    python telluride_search.py --query "\"Telluride Neuromorphic Cognition Engineering Workshop\""
    python telluride_search.py --sources openalex,europepmc --max-results 500
    python telluride_search.py --mailto you@example.com # join OpenAlex/Crossref polite pools

Environment variables:
    SEMANTIC_SCHOLAR_API_KEY  optional, raises Semantic Scholar rate limits
    SERPAPI_KEY               optional, enables the Google Scholar (SerpAPI) source
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Iterator, Optional

import requests

DEFAULT_QUERY = '"Telluride Neuromorphic Workshop"'
USER_AGENT = "telluride-scholar-search/1.0 (research bibliometrics tool)"

# Minimum seconds between requests, per source. Staying well under published
# rate limits is what keeps these APIs happy without any key.
POLITE_DELAY = {
    "openalex": 0.2,        # limit: 10 req/s
    "europepmc": 0.5,
    "semanticscholar": 1.5,  # unauthenticated pool is shared; be gentle
    "crossref": 1.0,
    "serpapi": 1.0,
}


@dataclass
class Paper:
    source: str
    title: str
    year: Optional[int] = None
    authors: str = ""
    venue: str = ""
    doi: str = ""
    url: str = ""
    snippet: str = ""
    cited_by: Optional[int] = None
    also_found_in: list = field(default_factory=list)


class HttpClient:
    """requests wrapper with per-source pacing and 429/5xx backoff."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self._last_call: dict[str, float] = {}

    def get_json(self, source: str, url: str, params: dict, max_tries: int = 5) -> Optional[dict]:
        delay = POLITE_DELAY.get(source, 1.0)
        for attempt in range(max_tries):
            wait = self._last_call.get(source, 0) + delay - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_call[source] = time.monotonic()
            try:
                resp = self.session.get(url, params=params, timeout=30)
            except requests.RequestException as exc:
                print(f"  [{source}] network error: {exc} (attempt {attempt + 1})", file=sys.stderr)
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError:
                    print(f"  [{source}] non-JSON response", file=sys.stderr)
                    return None
            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = resp.headers.get("Retry-After")
                pause = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** (attempt + 1)
                print(f"  [{source}] HTTP {resp.status_code}, backing off {pause:.0f}s", file=sys.stderr)
                time.sleep(pause)
                continue
            print(f"  [{source}] HTTP {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
            return None
        print(f"  [{source}] giving up after {max_tries} attempts", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Source adapters. Each yields Paper objects, paginating until exhausted or
# max_results is reached.
# ---------------------------------------------------------------------------

def search_openalex(client: HttpClient, query: str, max_results: int, mailto: str) -> Iterator[Paper]:
    """OpenAlex `search` covers title, abstract and (where indexed) full text."""
    cursor = "*"
    fetched = 0
    while cursor and fetched < max_results:
        params = {
            "search": query,
            "per-page": min(200, max_results - fetched),
            "cursor": cursor,
            "select": "id,doi,title,publication_year,authorships,primary_location,cited_by_count",
        }
        if mailto:
            params["mailto"] = mailto
        data = client.get_json("openalex", "https://api.openalex.org/works", params)
        if not data:
            return
        for work in data.get("results", []):
            fetched += 1
            authors = "; ".join(
                a.get("author", {}).get("display_name", "") for a in (work.get("authorships") or [])[:10]
            )
            loc = work.get("primary_location") or {}
            src = loc.get("source") or {}
            yield Paper(
                source="openalex",
                title=work.get("title") or "",
                year=work.get("publication_year"),
                authors=authors,
                venue=src.get("display_name") or "",
                doi=(work.get("doi") or "").replace("https://doi.org/", ""),
                url=work.get("doi") or work.get("id") or "",
                cited_by=work.get("cited_by_count"),
            )
        cursor = (data.get("meta") or {}).get("next_cursor")


def search_europepmc(client: HttpClient, query: str, max_results: int, mailto: str) -> Iterator[Paper]:
    """Europe PMC does genuine full-text search over open-access articles,
    so it catches mentions in methods/acknowledgements that metadata-only
    services miss."""
    cursor = "*"
    fetched = 0
    while cursor and fetched < max_results:
        params = {
            "query": query,
            "format": "json",
            "pageSize": min(100, max_results - fetched),
            "cursorMark": cursor,
            "resultType": "lite",
        }
        data = client.get_json(
            "europepmc", "https://www.ebi.ac.uk/europepmc/webservices/rest/search", params
        )
        if not data:
            return
        results = (data.get("resultList") or {}).get("result", [])
        if not results:
            return
        for rec in results:
            fetched += 1
            yield Paper(
                source="europepmc",
                title=rec.get("title") or "",
                year=int(rec["pubYear"]) if rec.get("pubYear", "").isdigit() else None,
                authors=rec.get("authorString") or "",
                venue=rec.get("journalTitle") or rec.get("bookOrReportDetails", {}).get("publisher", ""),
                doi=rec.get("doi") or "",
                url=f"https://europepmc.org/article/{rec.get('source')}/{rec.get('id')}"
                if rec.get("id")
                else "",
                cited_by=rec.get("citedByCount"),
            )
        next_cursor = data.get("nextCursorMark")
        cursor = next_cursor if next_cursor and next_cursor != cursor else None


def search_semanticscholar(client: HttpClient, query: str, max_results: int, mailto: str) -> Iterator[Paper]:
    """Semantic Scholar bulk search; supports quoted phrases. API key optional."""
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        client.session.headers["x-api-key"] = api_key
    token = None
    fetched = 0
    while fetched < max_results:
        params = {
            "q": query,
            "fields": "title,year,authors,externalIds,venue,url,citationCount",
        }
        if token:
            params["token"] = token
        data = client.get_json(
            "semanticscholar",
            "https://api.semanticscholar.org/graph/v1/paper/search/bulk",
            params,
        )
        if not data:
            return
        for rec in data.get("data", []):
            fetched += 1
            if fetched > max_results:
                return
            ext = rec.get("externalIds") or {}
            yield Paper(
                source="semanticscholar",
                title=rec.get("title") or "",
                year=rec.get("year"),
                authors="; ".join(a.get("name", "") for a in (rec.get("authors") or [])[:10]),
                venue=rec.get("venue") or "",
                doi=ext.get("DOI") or "",
                url=rec.get("url") or "",
                cited_by=rec.get("citationCount"),
            )
        token = data.get("token")
        if not token:
            return


def search_crossref(client: HttpClient, query: str, max_results: int, mailto: str) -> Iterator[Paper]:
    """Crossref metadata search (titles + deposited abstracts)."""
    cursor = "*"
    fetched = 0
    # Crossref treats quotes literally in `query.bibliographic`; strip them.
    plain = query.replace('"', "")
    while cursor and fetched < max_results:
        params = {
            "query.bibliographic": plain,
            "rows": min(100, max_results - fetched),
            "cursor": cursor,
            "select": "DOI,title,author,issued,container-title,is-referenced-by-count,URL",
        }
        if mailto:
            params["mailto"] = mailto
        data = client.get_json("crossref", "https://api.crossref.org/works", params)
        if not data:
            return
        msg = data.get("message") or {}
        items = msg.get("items", [])
        if not items:
            return
        for rec in items:
            fetched += 1
            title = " ".join(rec.get("title") or [])
            # Crossref relevance search returns loose matches; keep only rows
            # that actually contain the key terms.
            if not _title_matches(title, plain):
                continue
            authors = "; ".join(
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in (rec.get("author") or [])[:10]
            )
            issued = ((rec.get("issued") or {}).get("date-parts") or [[None]])[0][0]
            yield Paper(
                source="crossref",
                title=title,
                year=issued,
                authors=authors,
                venue=" ".join(rec.get("container-title") or []),
                doi=rec.get("DOI") or "",
                url=rec.get("URL") or "",
                cited_by=rec.get("is-referenced-by-count"),
            )
        cursor = msg.get("next-cursor")


def search_serpapi(client: HttpClient, query: str, max_results: int, mailto: str) -> Iterator[Paper]:
    """Real Google Scholar results via SerpAPI (paid; needs SERPAPI_KEY).

    SerpAPI operates its own Scholar access and returns structured JSON, so
    your own IP/account never touches Scholar's bot detection.
    """
    api_key = os.environ.get("SERPAPI_KEY")
    if not api_key:
        print("  [serpapi] SERPAPI_KEY not set, skipping Google Scholar source", file=sys.stderr)
        return
    start = 0
    fetched = 0
    while fetched < max_results:
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": api_key,
            "start": start,
            "num": 20,
        }
        data = client.get_json("serpapi", "https://serpapi.com/search.json", params)
        if not data:
            return
        results = data.get("organic_results") or []
        if not results:
            return
        for rec in results:
            fetched += 1
            summary = (rec.get("publication_info") or {}).get("summary", "")
            cited = ((rec.get("inline_links") or {}).get("cited_by") or {}).get("total")
            year_match = re.search(r"\b(19|20)\d{2}\b", summary)
            yield Paper(
                source="google_scholar(serpapi)",
                title=rec.get("title") or "",
                year=int(year_match.group()) if year_match else None,
                authors=summary.split(" - ")[0] if " - " in summary else "",
                venue="",
                doi="",
                url=rec.get("link") or "",
                snippet=rec.get("snippet") or "",
                cited_by=cited,
            )
        start += 20


SOURCES: dict[str, Callable[..., Iterator[Paper]]] = {
    "openalex": search_openalex,
    "europepmc": search_europepmc,
    "semanticscholar": search_semanticscholar,
    "crossref": search_crossref,
    "serpapi": search_serpapi,
}


# ---------------------------------------------------------------------------
# De-duplication and output
# ---------------------------------------------------------------------------

def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _title_matches(title: str, query: str) -> bool:
    words = [w.lower() for w in re.findall(r"\w+", query)]
    lowered = title.lower()
    return all(w in lowered for w in words if len(w) > 3)


def dedupe(papers: list[Paper]) -> list[Paper]:
    by_key: dict[str, Paper] = {}
    for p in papers:
        key = p.doi.lower() if p.doi else _norm_title(p.title)
        if not key:
            continue
        if key in by_key:
            kept = by_key[key]
            if p.source not in kept.also_found_in and p.source != kept.source:
                kept.also_found_in.append(p.source)
            # Prefer whichever record has more filled-in fields.
            for attr in ("doi", "venue", "authors", "snippet", "url"):
                if not getattr(kept, attr) and getattr(p, attr):
                    setattr(kept, attr, getattr(p, attr))
            if kept.year is None:
                kept.year = p.year
        else:
            by_key[key] = p
    return sorted(by_key.values(), key=lambda p: (-(p.year or 0), p.title))


def write_outputs(papers: list[Paper], out_prefix: str) -> None:
    json_path = f"{out_prefix}.json"
    csv_path = f"{out_prefix}.csv"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump([asdict(p) for p in papers], fh, indent=2, ensure_ascii=False)
    fields = ["title", "year", "authors", "venue", "doi", "url", "cited_by", "source", "also_found_in", "snippet"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for p in papers:
            row = asdict(p)
            row["also_found_in"] = ",".join(row["also_found_in"])
            writer.writerow({k: row[k] for k in fields})
    print(f"\nWrote {len(papers)} unique records to {csv_path} and {json_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find papers mentioning the Telluride Neuromorphic Workshop via scholarly APIs."
    )
    parser.add_argument("--query", default=DEFAULT_QUERY, help=f"search phrase (default: {DEFAULT_QUERY})")
    parser.add_argument(
        "--sources",
        default="openalex,europepmc,semanticscholar,crossref",
        help="comma-separated subset of: " + ",".join(SOURCES),
    )
    parser.add_argument("--max-results", type=int, default=1000, help="cap per source (default 1000)")
    parser.add_argument(
        "--mailto",
        default=os.environ.get("SCHOLAR_SEARCH_MAILTO", ""),
        help="your email; joins the OpenAlex/Crossref 'polite pools' for faster, more reliable service",
    )
    parser.add_argument("--out", default="telluride_mentions", help="output file prefix")
    args = parser.parse_args()

    wanted = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in SOURCES]
    if unknown:
        parser.error(f"unknown source(s): {unknown}; choose from {list(SOURCES)}")

    client = HttpClient()
    all_papers: list[Paper] = []
    for name in wanted:
        print(f"Searching {name} for {args.query} ...")
        found = list(SOURCES[name](client, args.query, args.max_results, args.mailto))
        print(f"  {name}: {len(found)} results")
        all_papers.extend(found)

    unique = dedupe(all_papers)
    write_outputs(unique, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
