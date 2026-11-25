#!/usr/bin/env python3
"""
Step 5b: Expand strictly out-of-time external temporal validation window from
Nov-Dec 2025 (2 months) to Sep-Dec 2025 (4 full months: 2025-09-01 to 2025-12-31)
using FROZEN ClinAudit v0.2.0 (zero rule tuning!).

Fetches Sep 1 - Oct 31, 2025 records across PLOS Medicine, npj Digital Medicine,
and Nature Communications (exact same pre-specified query & eligibility rules),
merges with existing Nov-Dec 2025 scan results without re-downloading Nov-Dec,
and saves the unified 4-month out-of-time cohort report.
"""
import concurrent.futures
import json
import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harvest"))
sys.path.insert(0, str(ROOT / "harvest" / "external_2025"))

import step2a_screen_corpus as s2a
import step2b_extract_code_data_links as s2b
import step3_fetch_n3_artifacts as s3
import step5_external_2025_validation as s5
from clinaudit import __version__
from clinaudit.scanner import audit_directory

EXT_DIR = ROOT / "harvest" / "external_2025"
EXT_ART = EXT_DIR / "artifacts"
D_START, D_END = "2025-09-01", "2025-10-31"


def screen_sep_oct(rec: dict):
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
    if s2a.LLM_NON_PRED_RE.search(title) and not s2a.re.search(
        r"\b(auc|auroc|c-index|concordance|prognostic model|risk prediction|survival prediction)\b",
        text, s2a.re.I
    ):
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


def main():
    print(f"=== EXPANDING OUT-OF-TIME EXTERNAL VALIDATION: ADDING {D_START} to {D_END} | Frozen v{__version__} ===", flush=True)
    prev = json.loads((EXT_DIR / "external_2025_validation_results.json").read_text(encoding="utf-8"))
    prev_summary = prev["summary"]
    prev_papers = prev["papers"]
    existing_dois = {r["doi"] for r in prev_papers}

    queries = [
        ("plos_medicine", f'ISSN:"1549-1676" AND FIRST_PDATE:[{D_START} TO {D_END}]'),
        ("npj_digital_medicine", f'ISSN:"2398-6352" AND FIRST_PDATE:[{D_START} TO {D_END}]'),
        ("nature_communications", f'ISSN:"2041-1723" AND FIRST_PDATE:[{D_START} TO {D_END}] AND ("machine learning" OR "deep learning" OR "foundation model" OR "prediction model" OR "prognostic model") AND (patient OR patients OR clinical OR cohort OR pathology OR oncology)'),
    ]
    raw_new = []
    elig_new = []
    seen = set()
    for jkey, q in queries:
        rows = s5.fetch_epmc(q)
        print(f"  Harvested {jkey} (Sep-Oct): {len(rows)} raw records", flush=True)
        for r in rows:
            doi = r["doi"]
            if not doi or doi in seen or doi in existing_dois:
                continue
            seen.add(doi)
            r["journal_key"] = jkey
            raw_new.append(r)
            ok, reason = screen_sep_oct(r)
            r["eligible"] = ok
            r["screen_reason"] = reason
            if ok:
                elig_new.append(r)

    print(f"Sep-Oct screened N0={len(raw_new)} -> Eligible N1={len(elig_new)}", flush=True)

    enriched = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for res in ex.map(s2b.process_one, elig_new):
            enriched.append(res)
    ft_ok = sum(1 for r in enriched if r["fulltext_fetched"])
    auditable = [r for r in enriched if r.get("verifiably_auditable_any") or r.get("github_repos") or r.get("esm_tabular_urls")]
    print(f"Sep-Oct fulltext fetched={ft_ok}/{len(enriched)} | Candidate auditable={len(auditable)}", flush=True)

    dl_manifest = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for m in ex.map(s5.fetch_ext_artifacts, auditable):
            dl_manifest.append(m)
    scanned_targets = [m for m in dl_manifest if m["n_files"] > 0]
    new_files = sum(m["n_files"] for m in scanned_targets)
    print(f"Sep-Oct downloaded non-empty artifacts: {len(scanned_targets)} papers ({new_files} files)", flush=True)

    elig_map = {r["doi"]: r for r in enriched}
    new_scan_rows = []
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
        new_scan_rows.append({
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

    # Merge Sep-Oct + Nov-Dec into unified 4-month strictly out-of-time cohort
    merged_papers = new_scan_rows + prev_papers
    merged_papers.sort(key=lambda x: (x.get("pub_date") or "", x["doi"]))
    rts = sorted(r["runtime_ms"] for r in merged_papers)
    flagged = [r for r in merged_papers if r["n_alerts"] > 0]
    new_flagged = [r for r in new_scan_rows if r["n_alerts"] > 0]

    merged_summary = {
        "window": ["2025-09-01", "2025-12-31"],
        "clinaudit_version": __version__,
        "N0_screened": prev_summary["N0_screened"] + len(raw_new),
        "N1_eligible": prev_summary["N1_eligible"] + len(elig_new),
        "N_candidate_auditable": prev_summary["N_candidate_auditable"] + len(auditable),
        "N_scanned": len(merged_papers),
        "total_files": sum(r["n_files"] for r in merged_papers),
        "n_errors": sum(1 for r in merged_papers if r["error"]),
        "median_runtime_ms": rts[len(rts) // 2] if rts else 0.0,
        "papers_flagged": len(flagged),
        "flag_rate_pct": round(100.0 * len(flagged) / len(merged_papers), 2) if merged_papers else 0.0,
        "breakdown": {
            "sep_oct_2025": {
                "N0": len(raw_new), "N1": len(elig_new), "N_fulltext": ft_ok,
                "N_auditable": len(auditable), "N_scanned": len(new_scan_rows),
                "total_files": new_files, "flagged": len(new_flagged),
            },
            "nov_dec_2025": {
                "N0": prev_summary["N0_screened"], "N1": prev_summary["N1_eligible"],
                "N_auditable": prev_summary["N_candidate_auditable"], "N_scanned": prev_summary["N_scanned"],
                "total_files": prev_summary["total_files"], "flagged": prev_summary["papers_flagged"],
            },
        },
        "flagged_summary": [
            {"doi": x["doi"], "journal": x["journal_key"], "pub_date": x.get("pub_date", ""), "pmcid": x.get("pmcid"), "n_alerts": x["n_alerts"], "rules": x["rules"], "title": x["title"][:95]}
            for x in flagged
        ],
    }
    out_path = EXT_DIR / "external_2025_4mo_validation_results.json"
    out_path.write_text(
        json.dumps({"summary": merged_summary, "papers": merged_papers}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n=== UNIFIED 4-MONTH OUT-OF-TIME (SEP-DEC 2025) VALIDATION SUMMARY ===")
    print(json.dumps(merged_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
