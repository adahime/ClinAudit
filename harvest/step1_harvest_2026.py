#!/usr/bin/env python3
"""
Step 1: Unbiased, Reproducible 2026 Cross-Journal Harvest (2026-01-01 to 2026-11-25)
Targets:
  - PLOS Medicine (ISSN 1549-1676): ALL 2026 articles
  - npj Digital Medicine (ISSN 2398-6352): ALL 2026 articles
  - Nature Communications (ISSN 2041-1723): Pre-specified clinical ML/prediction query
    (merged across Europe PMC + Crossref to eliminate PMC indexing lag for Nov 2026 articles)

Outputs frozen raw JSON files with SHA-256 checksums + harvest_manifest.json.
"""
import datetime
import hashlib
import json
import pathlib
import re
import time
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "harvest" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "ClinAudit2026Study/1.0 (mailto:takuru1048596@gmail.com)"}
DATE_START = "2026-01-01"
DATE_END = "2026-11-25"


def fetch_json(url: str, timeout: int = 45, retries: int = 4):
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            last_err = e
            wait = 2 * (attempt + 1)
            print(f"  [retry {attempt+1}/{retries}] {type(e).__name__}: {e} -> sleep {wait}s", flush=True)
            time.sleep(wait)
    raise last_err


def harvest_epmc(query: str, page_size: int = 200, max_records: int = 5000):
    records = []
    cursor = "*"
    total_hits = None
    while len(records) < max_records:
        params = {
            "query": query,
            "resultType": "core",
            "pageSize": str(page_size),
            "cursorMark": cursor,
            "format": "json",
        }
        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode(params)
        data = fetch_json(url, timeout=50)
        if total_hits is None:
            total_hits = data.get("hitCount", 0)
            print(f"  EPMC hitCount={total_hits}", flush=True)
        batch = (data.get("resultList") or {}).get("result") or []
        if not batch:
            break
        for item in batch:
            pub_types = ((item.get("pubTypeList") or {}).get("pubType")) or []
            full_urls = []
            for u in ((item.get("fullTextUrlList") or {}).get("fullTextUrl")) or []:
                if u.get("url"):
                    full_urls.append(u["url"])
            records.append({
                "source": "europepmc",
                "doi": (item.get("doi") or "").strip().lower(),
                "pmid": item.get("pmid"),
                "pmcid": item.get("pmcid"),
                "title": (item.get("title") or "").strip(),
                "abstract": (item.get("abstractText") or "").strip(),
                "pub_date": item.get("firstPublicationDate") or "",
                "pub_types": pub_types,
                "is_oa": item.get("isOpenAccess"),
                "in_epmc": item.get("inEPMC"),
                "in_pmc": item.get("inPMC"),
                "cited_by_count": item.get("citedByCount", 0),
                "journal_title": (item.get("journalInfo") or {}).get("journal", {}).get("title", ""),
                "full_text_urls": full_urls,
            })
        next_cursor = data.get("nextCursorMark")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        time.sleep(0.25)
    return total_hits or 0, records


def harvest_crossref(issn: str, query_bibliographic: str = None, rows_per_page: int = 200, max_records: int = 3000):
    records = []
    cursor = "*"
    total_results = None
    while len(records) < max_records:
        params = {
            "filter": f"from-pub-date:{DATE_START},until-pub-date:{DATE_END},type:journal-article",
            "rows": str(rows_per_page),
            "cursor": cursor,
            "select": "DOI,title,abstract,published,published-online,published-print,created,is-referenced-by-count,update-to,link",
        }
        if query_bibliographic:
            params["query.bibliographic"] = query_bibliographic
        url = f"https://api.crossref.org/journals/{issn}/works?" + urllib.parse.urlencode(params)
        data = fetch_json(url, timeout=45)
        msg = data.get("message") or {}
        if total_results is None:
            total_results = msg.get("total-results", 0)
            print(f"  Crossref total-results={total_results}", flush=True)
        items = msg.get("items") or []
        if not items:
            break
        for it in items:
            # Extract earliest 2026 publication date
            pub_date = ""
            for dk in ("published-online", "published", "published-print", "created"):
                parts = ((it.get(dk) or {}).get("date-parts") or [[]])[0]
                if parts:
                    y = parts[0]
                    m = parts[1] if len(parts) > 1 else 1
                    d = parts[2] if len(parts) > 2 else 1
                    pub_date = f"{y:04d}-{m:02d}-{d:02d}"
                    break
            abs_raw = it.get("abstract") or ""
            abs_clean = re.sub(r"<[^>]+>", " ", abs_raw)
            abs_clean = re.sub(r"\s+", " ", abs_clean).strip()
            records.append({
                "source": "crossref",
                "doi": (it.get("DOI") or "").strip().lower(),
                "title": ((it.get("title") or [""])[0] or "").strip(),
                "abstract": abs_clean,
                "pub_date": pub_date,
                "cited_by_count": it.get("is-referenced-by-count", 0),
                "has_update_to": bool(it.get("update-to")),
            })
        next_cursor = msg.get("next-cursor")
        if not next_cursor or next_cursor == cursor or len(items) < rows_per_page:
            break
        cursor = next_cursor
        time.sleep(0.2)
    return total_results or 0, records


def save_frozen(filename: str, payload: dict):
    path = RAW_DIR / filename
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(text, encoding="utf-8")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"  Saved {filename} ({len(text):,} B, SHA256={sha[:16]}...)", flush=True)
    return {"file": filename, "bytes": len(text), "sha256": sha}


def main():
    start_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest = {
        "protocol_version": "1.0-frozen",
        "harvest_utc": start_utc,
        "date_window": [DATE_START, DATE_END],
        "journals": {},
        "files": [],
    }

    # 1. PLOS Medicine (ALL 2026 articles)
    print("\n[1/3] PLOS Medicine (ISSN 1549-1676)...", flush=True)
    cr_n, cr_rows = harvest_crossref("1549-1676", rows_per_page=200, max_records=1000)
    ep_n, ep_rows = harvest_epmc(f'ISSN:"1549-1676" AND FIRST_PDATE:[{DATE_START} TO {DATE_END}]', page_size=200, max_records=1000)
    manifest["journals"]["plos_medicine"] = {"issn": "1549-1676", "crossref_total": cr_n, "crossref_fetched": len(cr_rows), "epmc_total": ep_n, "epmc_fetched": len(ep_rows)}
    manifest["files"].append(save_frozen("plos_medicine_crossref_2026.json", {"harvest_utc": start_utc, "total": cr_n, "records": cr_rows}))
    manifest["files"].append(save_frozen("plos_medicine_epmc_2026.json", {"harvest_utc": start_utc, "total": ep_n, "records": ep_rows}))

    # 2. npj Digital Medicine (ALL 2026 articles)
    print("\n[2/3] npj Digital Medicine (ISSN 2398-6352)...", flush=True)
    cr_n, cr_rows = harvest_crossref("2398-6352", rows_per_page=200, max_records=2000)
    ep_n, ep_rows = harvest_epmc(f'ISSN:"2398-6352" AND FIRST_PDATE:[{DATE_START} TO {DATE_END}]', page_size=200, max_records=2000)
    manifest["journals"]["npj_digital_medicine"] = {"issn": "2398-6352", "crossref_total": cr_n, "crossref_fetched": len(cr_rows), "epmc_total": ep_n, "epmc_fetched": len(ep_rows)}
    manifest["files"].append(save_frozen("npj_digital_medicine_crossref_2026.json", {"harvest_utc": start_utc, "total": cr_n, "records": cr_rows}))
    manifest["files"].append(save_frozen("npj_digital_medicine_epmc_2026.json", {"harvest_utc": start_utc, "total": ep_n, "records": ep_rows}))

    # 3. Nature Communications (ISSN 2041-1723) — Pre-specified clinical ML search in EPMC + Oct-Nov Crossref lag supplement
    print("\n[3/3] Nature Communications (ISSN 2041-1723)...", flush=True)
    # Get authoritative Crossref total count (1-row query)
    cr_meta = fetch_json(f"https://api.crossref.org/journals/2041-1723/works?filter=from-pub-date:{DATE_START},until-pub-date:{DATE_END},type:journal-article&rows=1")
    nc_cr_total = cr_meta["message"]["total-results"]
    print(f"  Nature Communications Crossref 2026 total articles: {nc_cr_total}", flush=True)

    # Pre-specified clean EPMC query for clinical/medical ML prediction studies
    nc_epmc_query = (
        f'ISSN:"2041-1723" AND FIRST_PDATE:[{DATE_START} TO {DATE_END}] '
        'AND ("machine learning" OR "deep learning" OR "foundation model" OR "prediction model" OR "prognostic model") '
        'AND (patient OR patients OR clinical OR cohort OR pathology OR oncology)'
    )
    ep_n, ep_rows = harvest_epmc(nc_epmc_query, page_size=200, max_records=2500)

    # Also harvest Oct 15 - Nov 25 2026 Crossref Nature Communications articles to cover EPMC indexing lag (Article in Press)
    print("  Fetching Nature Communications Oct 15 - Nov 25 Crossref lag supplement...", flush=True)
    nc_recent = []
    cursor = "*"
    while len(nc_recent) < 2500:
        params = {
            "filter": f"from-pub-date:2026-10-15,until-pub-date:{DATE_END},type:journal-article",
            "rows": "200",
            "cursor": cursor,
            "select": "DOI,title,abstract,published,published-online,created,is-referenced-by-count,update-to",
        }
        url = "https://api.crossref.org/journals/2041-1723/works?" + urllib.parse.urlencode(params)
        d = fetch_json(url, timeout=45)
        msg = d.get("message") or {}
        items = msg.get("items") or []
        if not items:
            break
        for it in items:
            pub_date = ""
            for dk in ("published-online", "published", "created"):
                parts = ((it.get(dk) or {}).get("date-parts") or [[]])[0]
                if parts:
                    y = parts[0]
                    m = parts[1] if len(parts) > 1 else 1
                    day = parts[2] if len(parts) > 2 else 1
                    pub_date = f"{y:04d}-{m:02d}-{day:02d}"
                    break
            abs_clean = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", it.get("abstract") or "")).strip()
            nc_recent.append({
                "source": "crossref_recent_lag_supplement",
                "doi": (it.get("DOI") or "").strip().lower(),
                "title": ((it.get("title") or [""])[0] or "").strip(),
                "abstract": abs_clean,
                "pub_date": pub_date,
                "cited_by_count": it.get("is-referenced-by-count", 0),
                "has_update_to": bool(it.get("update-to")),
            })
        next_cursor = msg.get("next-cursor")
        if not next_cursor or next_cursor == cursor or len(items) < 200:
            break
        cursor = next_cursor
        time.sleep(0.2)
    print(f"  Nature Communications Oct 15 - Nov 25 Crossref fetched: {len(nc_recent)}", flush=True)

    manifest["journals"]["nature_communications"] = {
        "issn": "2041-1723",
        "crossref_total_all_disciplines": nc_cr_total,
        "epmc_clinical_ml_query": nc_epmc_query,
        "epmc_clinical_ml_total": ep_n,
        "epmc_clinical_ml_fetched": len(ep_rows),
        "crossref_oct15_nov25_lag_fetched": len(nc_recent),
    }
    manifest["files"].append(save_frozen("nature_communications_epmc_clinical_ml_2026.json", {"harvest_utc": start_utc, "query": nc_epmc_query, "total": ep_n, "records": ep_rows}))
    manifest["files"].append(save_frozen("nature_communications_crossref_oct15_nov25_2026.json", {"harvest_utc": start_utc, "total": len(nc_recent), "records": nc_recent}))

    save_frozen("harvest_manifest.json", manifest)
    print("\nALL HARVESTS COMPLETED SUCCESSFULLY!", flush=True)


if __name__ == "__main__":
    main()
