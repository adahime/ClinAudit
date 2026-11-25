#!/usr/bin/env python3
"""
Step 2B: Full-Text Artifact Extraction & Live Reachability Audit across ALL Eligible 2026
Clinical Prediction/Prognostic ML Original Research Articles (N_1 -> N_2 -> N_3).

For every eligible paper in `harvest/processed/eligible_2026_corpus.json`:
  1. Fetches official full-text XML (Europe PMC fullTextXML if PMCID available, PLOS XML for PLOS Medicine,
     or Springer Nature article HTML for brand-new Article-in-Press papers lacking PMCID).
  2. Parses Data Availability, Code Availability, Methods, and Supplementary Information links:
     - Code repos: GitHub, GitLab, Bitbucket, Code Ocean, Hugging Face
     - Data/archive repos: Zenodo, Figshare, Dryad, OSF, Mendeley Data, Synapse
     - Publisher source data / supplementary tabular files: Springer ESM `.xlsx`/`.csv`/`.zip`, PLOS supplementary `.xlsx`/`.csv`/`.zip`
  3. Verifies LIVE reachability & non-emptiness:
     - GitHub repos: checks live GitHub tree/archive or HTML to confirm 200 OK & presence of executable code (`.py`, `.R`, `.ipynb`, `.m`)
     - Publisher ESM / Zenodo / Figshare links: HTTP HEAD/GET check for 200 OK & content-type/size
  4. Saves complete per-paper audit metadata to `harvest/processed/eligible_2026_artifacts.json`
     and `harvest/processed/eligible_2026_artifacts.csv`, plus funnel summary `harvest/processed/funnel_n1_to_n3.json`.
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
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "harvest" / "processed"
XML_CACHE = ROOT / "harvest" / "xml_cache"
XML_CACHE.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "ClinAudit2026Study/1.0 (mailto:takuru1048596@gmail.com)"}

GITHUB_RE = re.compile(r"https?://(?:www\.)?github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", re.I)
GITLAB_RE = re.compile(r"https?://(?:www\.)?gitlab\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", re.I)
ZENODO_RE = re.compile(r"https?://(?:doi\.org/10\.5281/zenodo\.|zenodo\.org/(?:records?|doi/10\.5281/zenodo\.))(\d+)", re.I)
FIGSHARE_RE = re.compile(r"https?://(?:doi\.org/10\.6084/m9\.figshare\.|figshare\.com/\S+?/)(\d+)", re.I)
DRYAD_RE = re.compile(r"https?://(?:doi\.org/10\.5061/dryad\.|datadryad\.org/\S+?dryad\.)([A-Za-z0-9_.-]+)", re.I)
OSF_RE = re.compile(r"https?://(?:www\.)?osf\.io/([A-Za-z0-9]{4,10})", re.I)
HF_RE = re.compile(r"https?://(?:www\.)?huggingface\.co/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", re.I)
SPRINGER_ESM_RE = re.compile(r"https?://static-content\.springer\.com/esm/[^\s\"'<>]+?\.(?:xlsx|xls|csv|zip|tsv|gz)", re.I)
PLOS_SUPP_RE = re.compile(r"(?:10\.1371/journal\.pmed\.\d+\.s\d+)", re.I)
REQUEST_ONLY_RE = re.compile(
    r"(upon reasonable request|on reasonable request|available from the corresponding author|"
    r"not publicly available|cannot be shared publicly|restricted due to privacy|"
    r"controlled access|data access committee|available upon request)",
    re.I,
)
# Exclude generic boilerplate github links (e.g. tools cited in methods like pytorch, shap, lifelines)
BOILERPLATE_GH_OWNERS = {
    "slundberg", "shap", "dmlc", "microsoft", "pytorch", "tensorflow", "scikit-learn",
    "huggingface", "openai", "google", "google-research", "facebookresearch", "meta-llama",
    "mahmoodlab", "owkin", "camdavidsonpilon", "scipy", "numpy", "pandas-dev", "matplotlib",
    "tidyverse", "rstudio", "cran", "bioconductor", "satijalab", "theislab", "scverse",
    "monai", "project-monai", "ultralytics", "timm", "rwightman", "albumentations-team",
    "optuna", "ray-project", "wandb", "mlflow", "keras-team", "catboost", "lightgbm",
}


def http_get_text(url: str, timeout: int = 25, max_bytes: int = 4_000_000):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(max_bytes).decode("utf-8", errors="ignore")


def http_check_url(url: str, timeout: int = 12):
    for method in ("HEAD", "GET"):
        try:
            req = urllib.request.Request(url, headers=HEADERS, method=method)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                cl = r.headers.get("Content-Length")
                ct = r.headers.get("Content-Type", "")
                if method == "GET":
                    sample = r.read(4096)
                else:
                    sample = b""
                return {
                    "ok": 200 <= r.status < 400,
                    "status": r.status,
                    "content_length": int(cl) if (cl and cl.isdigit()) else len(sample),
                    "content_type": ct[:80],
                }
        except urllib.error.HTTPError as e:
            if e.code == 405 and method == "HEAD":
                continue
            return {"ok": False, "status": e.code, "content_length": 0, "content_type": ""}
        except Exception as e:
            if method == "HEAD":
                continue
            return {"ok": False, "status": 0, "error": type(e).__name__, "content_length": 0}
    return {"ok": False, "status": 0, "content_length": 0}


def clean_repo_slug(slug: str) -> str:
    slug = re.sub(r"[).,;:\]\"'<>]+$", "", slug.strip())
    slug = re.sub(r"\.git$", "", slug, flags=re.I)
    parts = slug.split("/")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return slug


def check_github_repo(slug: str):
    """Check GitHub repo live via HTML page (avoids GitHub 60/hr unauthenticated REST API rate limit!)."""
    url = f"https://github.com/{slug}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (ClinAudit2026)"})
        with urllib.request.urlopen(req, timeout=15) as r:
            html = r.read(300_000).decode("utf-8", errors="ignore")
            status = r.status
    except urllib.error.HTTPError as e:
        return {"repo": slug, "reachable": False, "status": e.code, "has_code_files": False, "is_empty": False}
    except Exception as e:
        return {"repo": slug, "reachable": False, "status": 0, "error": type(e).__name__, "has_code_files": False, "is_empty": False}

    is_empty = bool(re.search(r"This repository is empty|Quick setup\s*[—-]\s*if you", html, re.I))
    # Look for language breakdown or file extensions in repo page JSON/HTML
    has_code = bool(re.search(
        r'(\.py\b|\.ipynb\b|\.r\b|\.rmd\b|\.m\b|\.jl\b|"Python"|"Jupyter Notebook"|"R"|"MATLAB")',
        html,
        re.I,
    ))
    has_data = bool(re.search(r'(\.csv\b|\.xlsx\b|\.tsv\b|\.rds\b|\.rdata\b|\.parquet\b)', html, re.I))
    return {
        "repo": slug,
        "reachable": (status == 200),
        "status": status,
        "is_empty": is_empty,
        "has_code_files": (not is_empty) and has_code,
        "has_data_files": (not is_empty) and has_data,
    }


def fetch_article_text(rec: dict):
    doi = rec["doi"]
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", doi)
    cache_path = XML_CACHE / f"{safe_name}.txt"
    if cache_path.exists() and cache_path.stat().st_size > 200:
        return cache_path.read_text(encoding="utf-8", errors="ignore"), "cache"

    pmcid = rec.get("pmcid")
    jkey = rec["journal_key"]
    text = ""
    src = "none"

    # 1. Europe PMC FullText XML (cleanest structured JATS XML!)
    if pmcid:
        try:
            text = http_get_text(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML", timeout=25)
            if len(text) > 500:
                src = "epmc_xml"
        except Exception:
            text = ""

    # 2. PLOS direct XML fallback
    if not text and jkey == "plos_medicine":
        try:
            text = http_get_text(f"https://journals.plos.org/plosmedicine/article/file?id={doi}&type=manuscript", timeout=25)
            if len(text) > 500:
                src = "plos_xml"
        except Exception:
            text = ""

    # 3. Springer Nature article HTML fallback (especially for brand-new Oct-Nov 2026 Article-in-Press)
    if not text and doi.startswith("10.1038/"):
        suffix = doi.split("/", 1)[1]
        try:
            text = http_get_text(f"https://www.nature.com/articles/{suffix}", timeout=25)
            if len(text) > 500:
                src = "nature_html"
        except Exception:
            text = ""

    if text:
        cache_path.write_text(text, encoding="utf-8", errors="ignore")
    return text, src


def extract_availability_sections(full_text: str, src: str):
    """Extract dedicated Data Availability, Code Availability, and Supplementary text blocks."""
    avail_chunks = []
    if src in ("epmc_xml", "plos_xml", "cache") and full_text.lstrip().startswith("<"):
        try:
            root = ET.fromstring(full_text)
            for el in root.iter():
                tag = el.tag.split("}")[-1].lower()
                if tag in ("sec", "notes", "custom-meta", "supplementary-material"):
                    sec_type = (el.attrib.get("sec-type") or el.attrib.get("notes-type") or "").lower()
                    title_txt = ""
                    t_el = el.find("title")
                    if t_el is not None:
                        title_txt = "".join(t_el.itertext()).strip().lower()
                    if (
                        any(k in sec_type for k in ("data", "code", "avail", "supp"))
                        or any(k in title_txt for k in ("data avail", "code avail", "supplementary", "source data", "software"))
                        or tag == "supplementary-material"
                    ):
                        avail_chunks.append(" ".join("".join(el.itertext()).split()))
        except Exception:
            pass
    # Regex window fallback for HTML or malformed XML
    for pat in (
        r"(?:Data\s+[Aa]vailability|Code\s+[Aa]vailability|Source\s+[Dd]ata|Supplementary\s+[Ii]nformation).{0,3000}",
    ):
        for m in re.finditer(pat, full_text, re.S):
            chunk = re.sub(r"<[^>]+>", " ", m.group(0))
            chunk = re.sub(r"\s+", " ", chunk)
            avail_chunks.append(chunk)
    return "\n".join(avail_chunks)


def process_one(rec: dict):
    doi = rec["doi"]
    text, src = fetch_article_text(rec)
    if not text:
        return {
            **rec,
            "fulltext_fetched": False,
            "fulltext_source": src,
            "github_repos": [],
            "gitlab_repos": [],
            "zenodo_ids": [],
            "figshare_ids": [],
            "dryad_ids": [],
            "osf_ids": [],
            "hf_repos": [],
            "esm_tabular_urls": [],
            "has_source_data_mention": False,
            "claims_public_code": False,
            "claims_public_data": False,
            "restricted_data_only": False,
            "github_checks": [],
            "esm_checks": [],
            "verifiably_auditable_code": False,
            "verifiably_auditable_data": False,
            "verifiably_auditable_any": False,
        }

    # Extract abstract if previously missing (for title-only Oct-Nov Nature HTML records!)
    if len((rec.get("abstract") or "").strip()) < 40 and src == "nature_html":
        m_abs = re.search(r'<div[^>]+id="Abs1-content"[^>]*>(.*?)</div>', text, re.S | re.I)
        if m_abs:
            rec["abstract"] = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", m_abs.group(1))).strip()

    avail_text = extract_availability_sections(text, src)
    # Search BOTH availability section (priority) and full text for repos, filtering out boilerplate libraries
    gh_avail = [clean_repo_slug(m) for m in GITHUB_RE.findall(avail_text)]
    gh_all = [clean_repo_slug(m) for m in GITHUB_RE.findall(text)]

    def filter_gh(slugs):
        out = []
        seen = set()
        for s in slugs:
            if "/" not in s:
                continue
            owner = s.split("/")[0].lower()
            if owner in BOILERPLATE_GH_OWNERS:
                continue
            sl = s.lower()
            if sl not in seen:
                seen.add(sl)
                out.append(s)
        return out

    gh_repos = filter_gh(gh_avail) or filter_gh(gh_all)[:3]
    gl_repos = sorted(set(clean_repo_slug(m) for m in GITLAB_RE.findall(avail_text or text)))[:3]
    zen_ids = sorted(set(ZENODO_RE.findall(avail_text or text)))[:5]
    fig_ids = sorted(set(FIGSHARE_RE.findall(avail_text or text)))[:5]
    dry_ids = sorted(set(DRYAD_RE.findall(avail_text or text)))[:5]
    osf_ids = sorted(set(OSF_RE.findall(avail_text or text)))[:5]
    hf_repos = sorted(set(clean_repo_slug(m) for m in HF_RE.findall(avail_text or text)))[:3]

    # Publisher tabular/archive supplements (Springer ESM .xlsx/.csv/.zip, or PLOS S*_Data)
    esm_urls = sorted(set(SPRINGER_ESM_RE.findall(text)))
    # Also construct Springer Source Data / MOESM URLs from XML `MOESM*_ESM.xlsx` references!
    if doi.startswith("10.1038/"):
        moesms = set(re.findall(r"\b\d+_\d+_\d+_MOESM\d+_ESM\.(?:xlsx|xls|csv|zip)\b", text, re.I))
        enc_doi = urllib.parse.quote(doi, safe="")
        for fn in sorted(moesms):
            u = f"https://static-content.springer.com/esm/art%3A{enc_doi}/MediaObjects/{fn}"
            if u not in esm_urls:
                esm_urls.append(u)
    elif rec["journal_key"] == "plos_medicine":
        # Check for S*_Data / xlsx / csv mentions alongside PLOS supplementary IDs
        supp_ids = sorted(set(PLOS_SUPP_RE.findall(text)))
        for sid in supp_ids:
            # Check nearby text for xlsx/csv/data/code
            idx = text.find(sid)
            window = text[max(0, idx - 300): idx + 400]
            if re.search(r"(xlsx|xls|csv|zip|data|code|table|dataset|s\d+\s*data)", window, re.I):
                esm_urls.append(f"https://journals.plos.org/plosmedicine/article/file?id={sid}&type=supplementary")

    esm_urls = sorted(set(esm_urls))[:8]
    has_source_data = bool(re.search(r"Source\s+Data|MOESM\d+_ESM\.xlsx|S1_Data", text, re.I))

    claims_code = bool(gh_repos or gl_repos or hf_repos or re.search(r"(code\s+is\s+available|custom\s+code.*available|scripts?\s+(?:are|is)\s+available|github\.com|zenodo\.org)", avail_text, re.I))
    claims_data = bool(esm_urls or zen_ids or fig_ids or dry_ids or osf_ids or has_source_data)
    restricted_only = bool(REQUEST_ONLY_RE.search(avail_text)) and not (esm_urls or zen_ids or fig_ids or dry_ids or osf_ids)

    # Live check up to 2 GitHub repos & up to 2 ESM/Zenodo data URLs
    gh_checks = [check_github_repo(slug) for slug in gh_repos[:2]]
    esm_checks = []
    for u in esm_urls[:2]:
        chk = http_check_url(u, timeout=12)
        chk["url"] = u
        esm_checks.append(chk)
    if not esm_checks and zen_ids:
        zu = f"https://zenodo.org/api/records/{zen_ids[0]}"
        chk = http_check_url(zu, timeout=12)
        chk["url"] = zu
        esm_checks.append(chk)

    auditable_code = any(c.get("reachable") and c.get("has_code_files") for c in gh_checks) or bool(gl_repos)
    auditable_data = (
        any(c.get("ok") and c.get("content_length", 0) > 100 for c in esm_checks)
        or any(c.get("reachable") and c.get("has_data_files") for c in gh_checks)
    )

    return {
        **rec,
        "fulltext_fetched": True,
        "fulltext_source": src,
        "github_repos": gh_repos,
        "gitlab_repos": gl_repos,
        "zenodo_ids": zen_ids,
        "figshare_ids": fig_ids,
        "dryad_ids": dry_ids,
        "osf_ids": osf_ids,
        "hf_repos": hf_repos,
        "esm_tabular_urls": esm_urls,
        "has_source_data_mention": has_source_data,
        "claims_public_code": claims_code,
        "claims_public_data": claims_data,
        "restricted_data_only": restricted_only,
        "github_checks": gh_checks,
        "esm_checks": esm_checks,
        "verifiably_auditable_code": auditable_code,
        "verifiably_auditable_data": auditable_data,
        "verifiably_auditable_any": (auditable_code or auditable_data),
    }


def main():
    eligible = json.loads((PROC_DIR / "eligible_2026_corpus.json").read_text(encoding="utf-8"))
    print(f"Processing {len(eligible)} eligible 2026 clinical ML papers with 8 workers...", flush=True)
    results = []
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(process_one, rec) for rec in eligible]
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            res = fut.result()
            results.append(res)
            if i % 40 == 0 or i == len(eligible):
                dt = time.time() - t0
                ok_ft = sum(1 for r in results if r["fulltext_fetched"])
                aud_any = sum(1 for r in results if r["verifiably_auditable_any"])
                print(f"  [{i}/{len(eligible)}] {dt:.1f}s | fulltext={ok_ft} | auditable_any={aud_any}", flush=True)

    # Sort deterministically by journal_key, pub_date, doi
    results.sort(key=lambda r: (r["journal_key"], r.get("pub_date", ""), r["doi"]))

    # Compute funnel summary by journal and overall
    funnel = {"overall": {}, "by_journal": {}}

    def summarize(rows):
        n1 = len(rows)
        ft = sum(1 for r in rows if r["fulltext_fetched"])
        n2_code = sum(1 for r in rows if r["claims_public_code"])
        n2_data = sum(1 for r in rows if r["claims_public_data"])
        n2_any = sum(1 for r in rows if r["claims_public_code"] or r["claims_public_data"])
        gh_listed = sum(1 for r in rows if r["github_repos"])
        gh_broken_or_empty = sum(
            1 for r in rows
            if r["github_repos"] and not any(c.get("reachable") and c.get("has_code_files") for c in r["github_checks"])
        )
        n3_code = sum(1 for r in rows if r["verifiably_auditable_code"])
        n3_data = sum(1 for r in rows if r["verifiably_auditable_data"])
        n3_any = sum(1 for r in rows if r["verifiably_auditable_any"])
        restr = sum(1 for r in rows if r["restricted_data_only"])
        return {
            "N1_eligible": n1,
            "fulltext_retrieved": ft,
            "N2_claims_public_code": n2_code,
            "N2_claims_public_data": n2_data,
            "N2_claims_any_artifact": n2_any,
            "github_url_listed": gh_listed,
            "github_broken_empty_or_no_code": gh_broken_or_empty,
            "N3_verifiably_auditable_code": n3_code,
            "N3_verifiably_auditable_data": n3_data,
            "N3_verifiably_auditable_any": n3_any,
            "restricted_data_only": restr,
            "pct_auditable_any": round(100.0 * n3_any / n1, 2) if n1 else 0.0,
            "pct_auditable_code": round(100.0 * n3_code / n1, 2) if n1 else 0.0,
            "pct_auditable_data": round(100.0 * n3_data / n1, 2) if n1 else 0.0,
            "pct_github_broken_or_empty": round(100.0 * gh_broken_or_empty / gh_listed, 2) if gh_listed else 0.0,
        }

    funnel["overall"] = summarize(results)
    for jk in sorted(set(r["journal_key"] for r in results)):
        funnel["by_journal"][jk] = summarize([r for r in results if r["journal_key"] == jk])

    (PROC_DIR / "eligible_2026_artifacts.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (PROC_DIR / "funnel_n1_to_n3.json").write_text(json.dumps(funnel, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(PROC_DIR / "eligible_2026_artifacts.csv", "w", newline="", encoding="utf-8") as f:
        fields = [
            "journal_key", "doi", "pmcid", "pub_date", "fulltext_fetched", "fulltext_source",
            "claims_public_code", "claims_public_data", "verifiably_auditable_code",
            "verifiably_auditable_data", "verifiably_auditable_any", "restricted_data_only",
            "github_repos", "zenodo_ids", "esm_tabular_urls", "title",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fields}
            for lk in ("github_repos", "zenodo_ids", "esm_tabular_urls"):
                row[lk] = ";".join(r.get(lk) or [])
            w.writerow(row)

    print("\n=== FUNNEL SUMMARY (N1 -> N2 -> N3) ===")
    print(json.dumps(funnel, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
