#!/usr/bin/env python3
"""
Step 3: Download lightweight auditable code (.py/.R/.ipynb/.Rmd) and tabular Source Data
(.xlsx/.csv/.tsv) for all N3=199 verifiably auditable 2026 clinical ML papers.

Skips giant binary weights/images inside GitHub ZIPs so disk usage stays small & fast.
Logs exact download success/bytes/SHA-256 per paper to `harvest/processed/n3_downloaded_manifest.json`.
"""
import concurrent.futures
import hashlib
import io
import json
import pathlib
import re
import time
import urllib.error
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "harvest" / "processed"
ART_DIR = ROOT / "harvest" / "artifacts_n3"
ART_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "ClinAudit2026Study/1.0 (mailto:takuru1048596@gmail.com)"}
KEEP_EXTS = {".py", ".r", ".ipynb", ".rmd", ".xlsx", ".csv", ".tsv", ".rds"}
MAX_ZIP_BYTES = 25_000_000   # 25 MB cap for GitHub ZIP archive stream
MAX_FILE_BYTES = 12_000_000  # 12 MB cap per extracted tabular/code file
MAX_FILES_PER_REPO = 40


def safe_dir_name(doi: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", doi)


def download_bytes(url: str, max_bytes: int = MAX_ZIP_BYTES, timeout: int = 25):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read(max_bytes + 1)
        truncated = len(data) > max_bytes
        return data[:max_bytes], truncated


def fetch_github_code(slug: str, dest: pathlib.Path):
    """Download GitHub main/master ZIP and extract only code & compact tabular files."""
    if "/" not in slug or slug.endswith("/"):
        return {"status": "skip_invalid_slug", "extracted": 0}
    for branch in ("main", "master"):
        url = f"https://codeload.github.com/{slug}/zip/refs/heads/{branch}"
        try:
            raw, trunc = download_bytes(url, max_bytes=MAX_ZIP_BYTES, timeout=25)
            if trunc:
                return {"status": "repo_zip_gt_25MB", "branch": branch, "extracted": 0}
            extracted = []
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                infos = [zi for zi in zf.infolist() if not zi.is_dir()]
                # Prioritize code files first, then tabular files
                def prio(zi):
                    ext = pathlib.PurePosixPath(zi.filename).suffix.lower()
                    if ext in (".py", ".r", ".ipynb", ".rmd"):
                        return (0, zi.file_size)
                    if ext in (".xlsx", ".csv", ".tsv", ".rds"):
                        return (1, zi.file_size)
                    return (9, zi.file_size)
                infos.sort(key=prio)
                for zi in infos:
                    if len(extracted) >= MAX_FILES_PER_REPO:
                        break
                    ext = pathlib.PurePosixPath(zi.filename).suffix.lower()
                    if ext not in KEEP_EXTS or zi.file_size > MAX_FILE_BYTES or zi.file_size == 0:
                        continue
                    rel_parts = pathlib.PurePosixPath(zi.filename).parts[1:]  # strip repo-branch root
                    if not rel_parts:
                        continue
                    safe_rel = "__".join(re.sub(r"[^A-Za-z0-9._-]", "_", p) for p in rel_parts)[-110:]
                    out_p = dest / f"gh_{safe_rel}"
                    out_p.write_bytes(zf.read(zi.filename))
                    extracted.append(out_p.name)
            return {
                "status": "ok",
                "branch": branch,
                "zip_sha256": hashlib.sha256(raw).hexdigest(),
                "zip_bytes": len(raw),
                "extracted": len(extracted),
                "files": extracted[:15],
            }
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            return {"status": f"http_{e.code}", "extracted": 0}
        except Exception as e:
            return {"status": f"err_{type(e).__name__}", "extracted": 0}
    return {"status": "branch_not_found", "extracted": 0}


def fetch_esm_files(urls: list, dest: pathlib.Path):
    fetched = []
    for i, u in enumerate(urls[:3]):  # up to 3 tabular/zip ESM files per paper
        try:
            raw, trunc = download_bytes(u, max_bytes=MAX_FILE_BYTES, timeout=25)
            if trunc or len(raw) < 100:
                fetched.append({"url": u, "status": "too_large_or_empty", "bytes": len(raw)})
                continue
            # Guess extension
            m_ext = re.search(r"\.(xlsx|xls|csv|tsv|zip)(?:$|\?)", u, re.I)
            ext = "." + m_ext.group(1).lower() if m_ext else (".xlsx" if raw[:2] == b"PK" else ".bin")
            if ext == ".bin":
                continue
            out_p = dest / f"esm_{i+1}{ext}"
            out_p.write_bytes(raw)
            fetched.append({"url": u, "status": "ok", "file": out_p.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        except Exception as e:
            fetched.append({"url": u, "status": f"err_{type(e).__name__}"})
    return fetched


def fetch_one(rec: dict):
    doi = rec["doi"]
    dest = ART_DIR / safe_dir_name(doi)
    dest.mkdir(parents=True, exist_ok=True)

    # Skip if already fetched non-empty
    existing = [p.name for p in dest.iterdir() if p.is_file() and p.stat().st_size > 0]
    if existing:
        return {"doi": doi, "journal_key": rec["journal_key"], "dir": dest.name, "cached": True, "n_files": len(existing)}

    gh_res = None
    gh_slugs = rec.get("github_repos_cleaned") or rec.get("github_repos") or []
    if rec.get("verifiably_auditable_code") and gh_slugs:
        gh_res = fetch_github_code(gh_slugs[0], dest)

    esm_res = []
    if rec.get("esm_tabular_urls"):
        esm_res = fetch_esm_files(rec["esm_tabular_urls"], dest)

    files_now = sorted(p.name for p in dest.iterdir() if p.is_file() and p.stat().st_size > 0)
    return {
        "doi": doi,
        "journal_key": rec["journal_key"],
        "dir": dest.name,
        "cached": False,
        "n_files": len(files_now),
        "gh_fetch": gh_res,
        "esm_fetch": esm_res,
    }


def main():
    rows = json.loads((PROC_DIR / "eligible_2026_verified.json").read_text(encoding="utf-8"))
    n3 = [r for r in rows if r.get("verifiably_auditable_any")]
    print(f"Downloading lightweight code & tabular Source Data for N3={len(n3)} auditable 2026 papers...", flush=True)
    t0 = time.time()
    manifest = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(fetch_one, r) for r in n3]
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            manifest.append(fut.result())
            if i % 30 == 0 or i == len(n3):
                got = sum(1 for m in manifest if m["n_files"] > 0)
                tot_f = sum(m["n_files"] for m in manifest)
                print(f"  [{i}/{len(n3)}] {time.time()-t0:.1f}s | papers_with_files={got} | total_files={tot_f}", flush=True)

    manifest.sort(key=lambda x: x["doi"])
    out = {
        "n3_target": len(n3),
        "papers_downloaded_nonempty": sum(1 for m in manifest if m["n_files"] > 0),
        "total_extracted_files": sum(m["n_files"] for m in manifest),
        "records": manifest,
    }
    (PROC_DIR / "n3_downloaded_manifest.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDONE! Downloaded non-empty artifacts for {out['papers_downloaded_nonempty']}/{len(n3)} papers ({out['total_extracted_files']} total code/tabular files).", flush=True)


if __name__ == "__main__":
    main()
