# Telluride Neuromorphic Workshop — literature mention search

A small CLI tool that finds papers mentioning the **Telluride Neuromorphic
Workshop** (a.k.a. the Telluride Neuromorphic Cognition Engineering Workshop)
and exports them to CSV/JSON.

## Why this doesn't scrape Google Scholar

Google Scholar has no public API, and its terms of service prohibit automated
querying. Its bot detection (CAPTCHAs, IP blocks) exists to enforce that, so
"avoiding the limits" by rotating proxies or spoofing browsers is both fragile
and against the ToS. The robust way around the limits is to not hit Scholar
at all:

| Source | Coverage | Key needed | What it searches |
|---|---|---|---|
| [OpenAlex](https://docs.openalex.org) | ~250M works (superset of MAG) | no | title + abstract + full text where indexed |
| [Europe PMC](https://europepmc.org/RestfulWebService) | life sciences + open access | no | **true full text** of OA papers — catches mentions in methods/acknowledgements |
| [Semantic Scholar](https://api.semanticscholar.org) | ~220M papers | optional | title + abstract |
| [Crossref](https://api.crossref.org) | publisher-deposited metadata | no | titles + deposited abstracts |
| Google Scholar via [SerpAPI](https://serpapi.com/google-scholar-api) | actual Scholar results | paid key | Scholar's own index, incl. citations-only snippets |

Together the free sources cover the large majority of what Scholar indexes.
If you specifically need Scholar's numbers (e.g. its citation counts or its
grey-literature coverage), the last row is the legitimate route: SerpAPI and
similar services (Scale SERP, SearchAPI) serve Scholar results through their
own infrastructure, so your IP and account are never involved.

## Usage

```bash
# All free sources, default phrase query
python telluride_search.py --mailto you@example.com

# Different phrasing of the workshop name
python telluride_search.py --query '"Telluride Neuromorphic Cognition Engineering Workshop"'

# Only full-text-capable sources
python telluride_search.py --sources openalex,europepmc

# Add real Google Scholar results via SerpAPI
export SERPAPI_KEY=...
python telluride_search.py --sources openalex,europepmc,serpapi
```

Outputs `telluride_mentions.csv` and `telluride_mentions.json` (override with
`--out`), de-duplicated across sources by DOI and normalised title, sorted by
year. The `also_found_in` column shows which other sources returned the same
paper.

Only dependency: `requests` (`pip install requests`).

## Staying inside rate limits

The tool never needs proxy rotation or user-agent spoofing because every
source it uses *wants* programmatic access. It stays reliable by:

- **Pacing**: a per-source minimum delay between requests (see `POLITE_DELAY`),
  well under each service's published limit.
- **Backoff**: on HTTP 429/5xx it honours `Retry-After` or backs off
  exponentially, up to 5 attempts.
- **Polite pools**: pass `--mailto` (or set `SCHOLAR_SEARCH_MAILTO`) to get
  OpenAlex's and Crossref's faster, more reliable service tiers.
- **API keys where offered**: set `SEMANTIC_SCHOLAR_API_KEY` (free on request)
  to move off Semantic Scholar's shared anonymous pool.
- **Cursor pagination**: uses each API's cursor mechanism rather than deep
  offsets, so large result sets don't degrade.

## Extending

Each source is one generator function in `telluride_search.py` registered in
the `SOURCES` dict — add a new backend (e.g. CORE, arXiv, dblp) by writing one
function that yields `Paper` objects. Useful query variants to sweep:
`"Telluride Neuromorphic"`, `"Neuromorphic Cognition Engineering Workshop"`,
`"Telluride workshop" neuromorphic`.
