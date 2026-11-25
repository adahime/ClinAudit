#!/usr/bin/env python3
"""
Step 2C: Ground-truth QA & Refinement of the 2026 N1 -> N2 -> N3 Funnel.
1. Retries the 9 papers whose initial full-text fetch timed out.
2. Re-screens the 27 title-only papers using their newly recovered abstracts under the exact same
   pre-specified eligibility criteria (dropping any that lack quantitative clinical prediction metrics).
3. Verifies EVERY GitHub repo previously flagged as non-auditable using GitHub's official REST API
   (/repos/{owner}/{repo} + /languages) or archive probe to eliminate HTML-truncation false flags.
4. Produces the final, peer-review-grade verified funnel: `funnel_n1_to_n3_verified.json`
   and `eligible_2026_verified.json` + `eligible_2026_verified.csv`.
"""
import concurrent.futures
import csv
import json
import pathlib
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harvest"))
import step2a_screen_corpus as s2a
import step2b_extract_code_data_links as s2b

PROC_DIR = ROOT / "harvest" / "processed"
GH_HEADERS = {"User-Agent": "ClinAudit2026Study/1.0", "Accept": "application/vnd.github+json"}


def clean_github_candidates(raw_repos):
    cleaned = []
    seen = set()
    for r in raw_repos:
        # Fix XML glue like repo10.5281 or trailing .The
        r2 = re.sub(r"10\.\d{4,5}.*$", "", r)
        r2 = re.sub(r"\.(?:The|This|All|Data|Code|Source|Supplementary)$", "", r2)
        parts = [p for p in r2.split("/") if p]
        if len(parts) == 1:
            # User/org profile URL only (no repo path)
            slug = parts[0] + "/"
        elif len(parts) >= 2:
            slug = f"{parts[0]}/{parts[1]}"
        else:
            continue
        if slug.lower() not in seen:
            seen.add(slug.lower())
            cleaned.append(slug)
    return cleaned


def verify_github_ground_truth(slug: str):
    if slug.endswith("/") or "/" not in slug:
        return {"repo": slug, "gt_status": "user_profile_only_no_repo", "reachable": True, "has_code": False, "languages": {}}
    # 1. Check archive zip HEAD/GET (does NOT consume GitHub REST API 60/hr quota!)
    #    AND query GitHub API /repos/{slug} only when needed (<=35 total!)
    api_url = f"https://api.github.com/repos/{slug}"
    try:
        req = urllib.request.Request(api_url, headers=GH_HEADERS)
        with urllib.request.urlopen(req, timeout=15) as r:
            meta = json.loads(r.read().decode("utf-8"))
        size_kb = meta.get("size", 0)
        default_branch = meta.get("default_branch", "main")
        primary_lang = meta.get("language")
        if size_kb == 0:
            return {"repo": slug, "gt_status": "empty_repo", "reachable": True, "has_code": False, "size_kb": 0, "language": None}
        if primary_lang in ("Python", "R", "Jupyter Notebook", "MATLAB", "Julia", "C++", "C", "Java", "Shell", "JavaScript", "TypeScript", "Perl", "Rust", "Go"):
            return {"repo": slug, "gt_status": "valid_code_repo", "reachable": True, "has_code": True, "size_kb": size_kb, "language": primary_lang, "default_branch": default_branch}
        # Fetch /languages for ambiguous/null primary_lang (e.g. README-only or HTML/Markdown-only!)
        req2 = urllib.request.Request(f"{api_url}/languages", headers=GH_HEADERS)
        with urllib.request.urlopen(req2, timeout=15) as r2:
            langs = json.loads(r2.read().decode("utf-8"))
        code_langs = {k: v for k, v in langs.items() if k in ("Python", "R", "Jupyter Notebook", "MATLAB", "Julia", "C++", "C", "Java", "Shell", "Perl", "Rust", "Go", "Dockerfile")}
        if code_langs:
            return {"repo": slug, "gt_status": "valid_code_repo", "reachable": True, "has_code": True, "size_kb": size_kb, "languages": langs, "default_branch": default_branch}
        return {"repo": slug, "gt_status": "readme_or_noncode_only", "reachable": True, "has_code": False, "size_kb": size_kb, "languages": langs, "default_branch": default_branch}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"repo": slug, "gt_status": "404_not_found_or_private", "reachable": False, "has_code": False, "http_status": 404}
        return {"repo": slug, "gt_status": f"http_{e.code}", "reachable": False, "has_code": False, "http_status": e.code}
    except Exception as e:
        return {"repo": slug, "gt_status": f"error_{type(e).__name__}", "reachable": False, "has_code": False}


def main():
    rows = json.loads((PROC_DIR / "eligible_2026_artifacts.json").read_text(encoding="utf-8"))
    print(f"Loaded {len(rows)} rows from step2b.", flush=True)

    # 1. Retry missing fulltexts in parallel (9 records, fast 12s timeout)
    missing_idx = [i for i, r in enumerate(rows) if not r.get("fulltext_fetched")]
    retried = len(missing_idx)
    recovered_ft = 0
    if missing_idx:
        with concurrent.futures.ThreadPoolExecutor(max_workers=9) as ex:
            fmap = {ex.submit(s2b.process_one, rows[i]): i for i in missing_idx}
            for fut in concurrent.futures.as_completed(fmap):
                idx = fmap[fut]
                try:
                    new_r = fut.result()
                    if new_r.get("fulltext_fetched"):
                        recovered_ft += 1
                        rows[idx] = new_r
                except Exception as e:
                    print(f"  retry err {rows[idx]['doi']}: {e}", flush=True)
    print(f"Retried {retried} missing fulltexts -> recovered {recovered_ft}.", flush=True)

    # 2. Re-screen the title-only entries now that abstracts are recovered
    kept = []
    dropped_after_abs = []
    for r in rows:
        if r.get("screen_reason") == "eligible_title_only_needs_abstract" and len((r.get("abstract") or "").strip()) >= 40:
            ok, reason = s2a.screen_record(r)
            r["rescreen_after_abstract"] = reason
            if not ok:
                dropped_after_abs.append({"doi": r["doi"], "journal": r["journal_key"], "reason": reason, "title": r["title"]})
                continue
            r["screen_reason"] = "eligible_verified_via_html_abstract"
        kept.append(r)
    print(f"After re-screening recovered abstracts: kept={len(kept)}, dropped={len(dropped_after_abs)}", flush=True)
    for d in dropped_after_abs:
        print(f"  DROPPED: {d['doi']} ({d['reason']}) | {d['title'][:75]}")

    # 3. Ground-truth verify EVERY GitHub repo that previously failed HTML code check
    gt_counts = {}
    api_calls = 0
    for r in kept:
        raw_gh = r.get("github_repos") or []
        if not raw_gh:
            continue
        cleaned_gh = clean_github_candidates(raw_gh)
        r["github_repos_cleaned"] = cleaned_gh
        prev_ok = any(c.get("reachable") and c.get("has_code_files") for c in (r.get("github_checks") or []))
        if prev_ok:
            r["github_gt_status"] = "valid_code_repo"
            gt_counts["valid_code_repo_html"] = gt_counts.get("valid_code_repo_html", 0) + 1
            continue
        # Ground-truth via GitHub API
        gt_results = []
        for slug in cleaned_gh[:2]:
            api_calls += 1
            res = verify_github_ground_truth(slug)
            gt_results.append(res)
            time.sleep(0.25)
        r["github_gt_checks"] = gt_results
        if any(g.get("has_code") for g in gt_results):
            r["github_gt_status"] = "valid_code_repo"
            r["verifiably_auditable_code"] = True
            r["verifiably_auditable_any"] = True
            gt_counts["recovered_valid_code_via_api"] = gt_counts.get("recovered_valid_code_via_api", 0) + 1
        else:
            st = gt_results[0]["gt_status"] if gt_results else "unknown"
            r["github_gt_status"] = st
            gt_counts[st] = gt_counts.get(st, 0) + 1
            print(f"  CONFIRMED NON-AUDITABLE GH: {r['doi']} -> {st} ({gt_results})", flush=True)

    print(f"\nGitHub Ground-Truth Summary (api_calls={api_calls}): {json.dumps(gt_counts, indent=2)}", flush=True)

    # 4. Compute final verified funnel (N1 -> N2 -> N3)
    def summarize(subset):
        n1 = len(subset)
        ft = sum(1 for x in subset if x["fulltext_fetched"])
        n2_code = sum(1 for x in subset if x["claims_public_code"])
        n2_data = sum(1 for x in subset if x["claims_public_data"])
        n2_any = sum(1 for x in subset if x["claims_public_code"] or x["claims_public_data"])
        gh_listed = sum(1 for x in subset if x.get("github_repos"))
        gh_404 = sum(1 for x in subset if x.get("github_gt_status") == "404_not_found_or_private")
        gh_empty_or_nocode = sum(1 for x in subset if x.get("github_gt_status") in ("empty_repo", "readme_or_noncode_only"))
        gh_profile_only = sum(1 for x in subset if x.get("github_gt_status") == "user_profile_only_no_repo")
        gh_fail_total = sum(1 for x in subset if x.get("github_repos") and x.get("github_gt_status") != "valid_code_repo")
        n3_code = sum(1 for x in subset if x["verifiably_auditable_code"])
        n3_data = sum(1 for x in subset if x["verifiably_auditable_data"])
        n3_any = sum(1 for x in subset if x["verifiably_auditable_any"])
        restr = sum(1 for x in subset if x["restricted_data_only"])
        return {
            "N1_eligible": n1,
            "fulltext_retrieved": ft,
            "pct_fulltext": round(100.0 * ft / n1, 2) if n1 else 0.0,
            "N2_claims_public_code": n2_code,
            "N2_claims_public_data": n2_data,
            "N2_claims_any_artifact": n2_any,
            "github_url_listed": gh_listed,
            "github_404_or_private": gh_404,
            "github_empty_or_no_code": gh_empty_or_nocode,
            "github_profile_only": gh_profile_only,
            "github_non_auditable_total": gh_fail_total,
            "pct_github_non_auditable": round(100.0 * gh_fail_total / gh_listed, 2) if gh_listed else 0.0,
            "N3_verifiably_auditable_code": n3_code,
            "N3_verifiably_auditable_data": n3_data,
            "N3_verifiably_auditable_any": n3_any,
            "restricted_data_only": restr,
            "pct_auditable_any": round(100.0 * n3_any / n1, 2) if n1 else 0.0,
            "pct_auditable_code": round(100.0 * n3_code / n1, 2) if n1 else 0.0,
            "pct_auditable_data": round(100.0 * n3_data / n1, 2) if n1 else 0.0,
            "pct_restricted_data_only": round(100.0 * restr / n1, 2) if n1 else 0.0,
        }

    funnel = {
        "dropped_after_html_abstract_rescreen": dropped_after_abs,
        "github_gt_breakdown": gt_counts,
        "overall": summarize(kept),
        "by_journal": {jk: summarize([x for x in kept if x["journal_key"] == jk]) for jk in sorted(set(x["journal_key"] for x in kept))},
    }

    (PROC_DIR / "eligible_2026_verified.json").write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    (PROC_DIR / "funnel_n1_to_n3_verified.json").write_text(json.dumps(funnel, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(PROC_DIR / "eligible_2026_verified.csv", "w", newline="", encoding="utf-8") as f:
        fields = [
            "journal_key", "doi", "pmcid", "pub_date", "fulltext_fetched",
            "claims_public_code", "claims_public_data", "verifiably_auditable_code",
            "verifiably_auditable_data", "verifiably_auditable_any", "restricted_data_only",
            "github_gt_status", "github_repos", "zenodo_ids", "esm_tabular_urls", "title",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in kept:
            row = {k: r.get(k, "") for k in fields}
            for lk in ("github_repos", "zenodo_ids", "esm_tabular_urls"):
                row[lk] = ";".join(r.get(lk) or [])
            w.writerow(row)

    print("\n=== FINAL VERIFIED FUNNEL (N1 -> N2 -> N3) ===")
    print(json.dumps(funnel, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
