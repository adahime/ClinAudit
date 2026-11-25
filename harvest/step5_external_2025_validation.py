#!/usr/bin/env python3
"""
Step 5: Strictly Out-of-Time External Temporal Validation Cohort (2025-11-01 to 2025-12-31)
for FROZEN `ClinAudit v0.2.0` — zero peeking, zero rule tuning after this run!

Applies the exact same pre-specified eligibility rule (ML + clinical + quantitative metric)
across PLOS Medicine, npj Digital Medicine, and Nature Communications (clinical ML subset),
fetches full-text XML, downloads auditable code/tabular files, and evaluates frozen v0.2.0.
"""
import concurrent.futures
import hashlib
import json
import pathlib
import sys
import time
import urllib.parse
import urllib.request
import warnings
from collections import Counter

warnings.filterwarnings("ignore", category=SyntaxWarning)

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harvest"))

import step2a_screen_corpus as s2a
import step2b_extract_code_data_links as s2b
import step2d_verify_20_codeload as s2d
import step3_fetch_n3_artifacts as s3
from clinaudit import __version__
from clinaudit.scanner import audit_directory

EXT_DIR = ROOT / "harvest" / "external_2025"
EXT_ART = EXT_DIR / "artifacts"
EXT_DIR.mkdir(parents=True, exist_ok=True)
EXT_ART.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "ClinAudit2025ExtVal/1.0 (mailto:takuru1048596@gmail.com)"}
D_START, D_END = "2025-11-01", "2025-12-31"


def fetch_epmc(query: str, max_records: int = 800):
    records = []
    cursor = "*"
    while len(records) < max_records:
        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode({
            "query": query, "resultType": "core", "pageSize": "200", "cursorMark": cursor, "format": "json"
        })
        data = None
        last_err = None
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=45) as r:
                    data = json.loads(r.read().decode("utf-8"))
                break
            except Exception as exc:
                last_err = exc
                time.sleep(1.5 * (2 ** attempt))
        if data is None:
            raise RuntimeError(f"Europe PMC query failed after 5 attempts: {last_err}")
        batch = (data.get("resultList") or {}).get("result") or []
        if not batch:
            break
        for it in batch:
            records.append({
                "doi": (it.get("doi") or "").strip().lower(),
                "pmid": it.get("pmid"),
                "pmcid": it.get("pmcid"),
                "title": (it.get("title") or "").strip(),
                "abstract": (it.get("abstractText") or "").strip(),
                "pub_date": it.get("firstPublicationDate") or "",
                "pub_types": ((it.get("pubTypeList") or {}).get("pubType")) or [],
                "has_update_to": False,
            })
        nxt = data.get("nextCursorMark")
        if not nxt or nxt == cursor:
            break
        cursor = nxt
        time.sleep(0.35)
    return records


def screen_2025(rec: dict):
    """Exact same pre-specified eligibility logic as 2026, with window 2025-11-01..2025-12-31."""
    title = rec["title"] or ""
    abstract = rec["abstract"] or ""
    text = f"{title}\n{abstract}"
    pt_l = {p.lower() for p in (rec.get("pub_types") or [])}
    pdate = rec.get("pub_date") or ""
    if pdate and not (D_START <= pdate[:10] <= D_END):
        return False, "outside_window"
    if s2a.NON_ORIGINAL_TITLE_RE.search(title) or s2a.REVIEW_IN_TITLE_RE.search(title):
        return False, "non_original_title"
    if pt_l & s2a.NON_ORIGINAL_PUBTYPES:
        return False, "non_original_pubtype"
    if s2a.LLM_NON_PRED_RE.search(title) and not s2a.re.search(r"\b(auc|auroc|c-index|concordance|prognostic model|risk prediction|survival prediction)\b", text, s2a.re.I):
        return False, "llm_qa_benchmark"
    if len(abstract.strip()) < 40:
        return False, "missing_abstract"
    if not s2a.ML_TERM_RE.search(text):
        return False, "no_ml_term"
    if not s2a.CLINICAL_TERM_RE.search(text):
        return False, "no_clinical_term"
    if not s2a.QUANT_PRED_RE.search(text):
        return False, "no_quant_metric"
    return True, "eligible"


def fetch_ext_artifacts(rec: dict):
    doi = rec["doi"]
    dest = EXT_ART / s3.safe_dir_name(doi)
    dest.mkdir(parents=True, exist_ok=True)
    gh_slugs = rec.get("github_repos") or []
    gh_res = s3.fetch_github_code(gh_slugs[0], dest) if gh_slugs else None
    esm_res = s3.fetch_esm_files(rec.get("esm_tabular_urls") or [], dest) if rec.get("esm_tabular_urls") else []
    files = sorted(p.name for p in dest.iterdir() if p.is_file() and p.stat().st_size > 0)
    return {"doi": doi, "journal_key": rec["journal_key"], "dir": dest.name, "n_files": len(files), "gh_fetch": gh_res, "esm_fetch": esm_res}


def main():
    print(f"=== OUT-OF-TIME EXTERNAL VALIDATION ({D_START} to {D_END}) | Frozen ClinAudit v{__version__} ===", flush=True)
    queries = [
        ("plos_medicine", f'ISSN:"1549-1676" AND FIRST_PDATE:[{D_START} TO {D_END}]'),
        ("npj_digital_medicine", f'ISSN:"2398-6352" AND FIRST_PDATE:[{D_START} TO {D_END}]'),
        ("nature_communications", f'ISSN:"2041-1723" AND FIRST_PDATE:[{D_START} TO {D_END}] AND ("machine learning" OR "deep learning" OR "foundation model" OR "prediction model" OR "prognostic model") AND (patient OR patients OR clinical OR cohort OR pathology OR oncology)'),
    ]
    all_raw = []
    eligible = []
    seen_dois = set()
    for jkey, q in queries:
        rows = fetch_epmc(q)
        print(f"  Harvested {jkey}: {len(rows)} raw records", flush=True)
        for r in rows:
            if not r["doi"] or r["doi"] in seen_dois:
                continue
            seen_dois.add(r["doi"])
            r["journal_key"] = jkey
            all_raw.append(r)
            ok, reason = screen_2025(r)
            r["eligible"] = ok
            r["screen_reason"] = reason
            if ok:
                eligible.append(r)

    print(f"Total screened N0={len(all_raw)} -> Eligible N1={len(eligible)}", flush=True)

    # Extract full-text links & live availability (parallel 8 threads)
    enriched = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for res in ex.map(s2b.process_one, eligible):
            enriched.append(res)
    auditable = [r for r in enriched if r.get("verifiably_auditable_any") or r.get("github_repos") or r.get("esm_tabular_urls")]
    print(f"Fulltext fetched={sum(1 for r in enriched if r['fulltext_fetched'])}/{len(enriched)} | Candidate auditable={len(auditable)}", flush=True)

    # Download lightweight artifacts
    dl_manifest = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for m in ex.map(fetch_ext_artifacts, auditable):
            dl_manifest.append(m)
    scanned_targets = [m for m in dl_manifest if m["n_files"] > 0]
    total_files = sum(m["n_files"] for m in scanned_targets)
    print(f"Downloaded non-empty artifacts: {len(scanned_targets)} papers ({total_files} files)", flush=True)

    # Run FROZEN ClinAudit v0.2.0 strictly out-of-time!
    elig_map = {r["doi"]: r for r in enriched}
    scan_rows = []
    t0 = time.time()
    for rec in scanned_targets:
        pdir = EXT_ART / rec["dir"]
        tp0 = time.perf_counter()
        err = None
        alerts = []
        try:
            rep = audit_directory(str(pdir))
            alerts = [a.to_dict() for a in rep.alerts]
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        rt_ms = round((time.perf_counter() - tp0) * 1000.0, 2)
        scan_rows.append({
            "doi": rec["doi"],
            "journal_key": rec["journal_key"],
            "pub_date": elig_map[rec["doi"]].get("pub_date", ""),
            "pmcid": elig_map[rec["doi"]].get("pmcid"),
            "title": elig_map[rec["doi"]].get("title", ""),
            "github_repos": elig_map[rec["doi"]].get("github_repos") or [],
            "esm_tabular_urls": elig_map[rec["doi"]].get("esm_tabular_urls") or [],
            "dir": rec["dir"],
            "n_files": rec["n_files"],
            "runtime_ms": rt_ms,
            "error": err,
            "n_alerts": len(alerts),
            "rules": sorted(set(a["rule_id"] for a in alerts)),
            "modules": sorted(set(a["module"] for a in alerts)),
            "alerts": alerts,
        })

    rts = sorted(r["runtime_ms"] for r in scan_rows)
    flagged = [r for r in scan_rows if r["n_alerts"] > 0]
    summary = {
        "window": [D_START, D_END],
        "clinaudit_version": __version__,
        "N0_screened": len(all_raw),
        "N1_eligible": len(eligible),
        "N_candidate_auditable": len(auditable),
        "N_scanned": len(scan_rows),
        "total_files": total_files,
        "n_errors": sum(1 for r in scan_rows if r["error"]),
        "median_runtime_ms": rts[len(rts) // 2] if rts else 0.0,
        "papers_flagged": len(flagged),
        "flag_rate_pct": round(100.0 * len(flagged) / len(scan_rows), 2) if scan_rows else 0.0,
        "flagged_summary": [
            {"doi": x["doi"], "journal": x["journal_key"], "n_alerts": x["n_alerts"], "rules": x["rules"], "title": x["title"][:90]}
            for x in flagged
        ],
    }
    (EXT_DIR / "external_2025_validation_results.json").write_text(
        json.dumps({"summary": summary, "papers": scan_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n=== OUT-OF-TIME LATE-2025 EXTERNAL VALIDATION SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
