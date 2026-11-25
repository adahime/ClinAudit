#!/usr/bin/env python3
"""Adjudicate all 3 flagged papers from the strictly out-of-time late-2025 validation cohort."""
import json
import pathlib
import re
import urllib.request

OUT = pathlib.Path(__file__).resolve().parent
HEADERS = {"User-Agent": "ClinAudit/1.0 (mailto:takuru1048596@gmail.com)"}


def fetch_text(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="replace")


def main():
    xml = fetch_text("https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12830939/fullTextXML")
    plain = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", xml))
    matches = list(re.finditer(r"SD\s*=\s*[^,;)]{1,35}", plain))
    print(f"Total SD= matches in PMC12830939: {len(matches)}")
    eps_matches = []
    for m in matches:
        s = max(0, m.start() - 70)
        e = min(len(plain), m.end() + 70)
        snip = plain[s:e]
        print("  MATCH:", snip)
        if "10" in m.group(0) and "16" in m.group(0):
            eps_matches.append(m.group(0).strip())

    cr = json.loads(fetch_text("https://api.crossref.org/works/10.1038/s41467-025-67656-x"))["message"]
    authors = [f"{a.get('family', '')}, {a.get('given', '')}" for a in cr.get("author", [])[:4]]
    pub = cr.get("published-online") or cr.get("published") or {}
    meta = {
        "doi": "10.1038/s41467-025-67656-x",
        "title": (cr.get("title") or [""])[0],
        "authors_first4": authors,
        "journal": (cr.get("container-title") or [""])[0],
        "volume": cr.get("volume"),
        "article_number": cr.get("article-number") or cr.get("page"),
        "published_parts": pub.get("date-parts"),
        "relation": cr.get("relation"),
        "n_machine_epsilon_sds_in_main_text": len(eps_matches),
        "machine_epsilon_sds": eps_matches,
    }
    print("\nCROSSREF META:", json.dumps(meta, ensure_ascii=False, indent=2))

    # Also check whether Crossref has any erratum/correction referencing 10.1038/s41467-025-67656-x
    rel_check = json.loads(fetch_text("https://api.crossref.org/works?filter=relation.object:10.1038/s41467-025-67656-x"))["message"]
    print("Crossref update/relation hits:", rel_check.get("total-results"))

    # Save structured out-of-time adjudication JSON
    adj = {
        "cohort": "out_of_time_late_2025",
        "window": ["2025-11-01", "2025-12-31"],
        "clinaudit_version_frozen": "0.2.0",
        "N0_screened": 516,
        "N1_eligible": 72,
        "N_fulltext_retrieved": 55,
        "N_candidate_auditable": 41,
        "N_scanned": 34,
        "total_files": 387,
        "n_errors": 0,
        "median_runtime_ms": 386.92,
        "n_flagged": 3,
        "flag_rate_pct": 8.82,
        "adjudicated_3_flagged_papers": [
            {
                "idx": 1,
                "doi": "10.1038/s41746-025-02189-7",
                "journal": "npj_digital_medicine",
                "rules": ["STAT-004"],
                "verdict": "FALSE_POSITIVE",
                "reason": "esm_2.xlsx Sheet1 is a GSVA biological-process correlation table (Features vs Related Process, Pearson r, p_value), not an ML model-comparison table."
            },
            {
                "idx": 2,
                "doi": "10.1038/s41467-025-66483-4",
                "journal": "nature_communications",
                "rules": ["STAT-004"],
                "verdict": "FALSE_POSITIVE",
                "reason": "esm_3.xlsx Figure 2b is a Gene Ontology (GOTERM_BP_DIRECT) enrichment table (PValue, FDR), not an ML model-comparison table."
            },
            {
                "idx": 3,
                "doi": "10.1038/s41467-025-67656-x",
                "journal": "nature_communications",
                "rules": ["EVAL-004", "STAT-002", "STAT-004"],
                "verdict": "CONFIRMED_TECHNICAL_MATCH_AND_COLLATERAL_UNDISCLOSED_BUG",
                "meta": meta,
                "reason": (
                    "EVAL-004 accurately flagged RandomUnderSampler().fit_resample(x_test/x_test_male/x_test_female) in "
                    "UKB_by_sex_and_undersampling.py (lines 150, 324, 377). Full-text XML (PMC12830939) confirmed that "
                    "balanced test-set undersampling was explicitly disclosed as a sensitivity analysis—HOWEVER, human-in-the-loop "
                    "inspection of those exact flagged blocks immediately exposed a severe, undisclosed constant-seed bootstrap bug "
                    "across all 11 bootstrap loops in UKB_by_sex_and_undersampling.py (L162, 234, 283, 336, 389) and "
                    "external_validation.py (L221, 273, 343, 395, 467, 519): every loop calls resample(..., random_state=random_state) "
                    "with fixed constant random_state=1996, drawing 100 bit-for-bit identical resamples and collapsing np.std() to "
                    "IEEE-754 float64 machine epsilon (1.11e-16 to 4.44e-16), which was printed verbatim as the published uncertainty "
                    "throughout the Nature Communications main text (e.g., 'AUC = 0.95 (SD = 2.22 x 10^-16)', 'AUC = 0.91 (SD = 4.44 x 10^-16)'). "
                    "STAT-002/004 alerts on esm_1.xlsx (SD1/SD9/SD11) were biological cohort/GWAS catalog tables."
                ),
            },
        ],
    }
    (OUT / "external_2025_adjudication.json").write_text(json.dumps(adj, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSaved external_2025_adjudication.json cleanly!")


if __name__ == "__main__":
    main()
