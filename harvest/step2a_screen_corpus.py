#!/usr/bin/env python3
"""
Step 2A: Deduplicate Crossref + Europe PMC 2026 raw harvests and apply
pre-specified, deterministic eligibility screening for Clinical Prediction /
Prognostic Machine Learning Original Research Articles (2026-01-01 to 2026-11-25).
"""
import csv
import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "harvest" / "raw"
PROC_DIR = ROOT / "harvest" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Pre-specified exclusion title patterns (non-original research)
NON_ORIGINAL_TITLE_RE = re.compile(
    r"^(author correction|publisher correction|correction:|retraction|retracted|erratum|"
    r"matters arising|reply to:|comment on|correspondence|editorial|perspective:|"
    r"study protocol|protocol for|a systematic review|systematic review|scoping review|"
    r"umbrella review|narrative review|meta-analysis|consensus statement|reporting guideline|"
    r"tripod|consort|prisma|spirit|stard)",
    re.I,
)
REVIEW_IN_TITLE_RE = re.compile(
    r"\b(systematic review|scoping review|meta-analysis|literature review|umbrella review|"
    r"narrative review|rapid review|mini-review|study protocol|protocol for a|editorial|"
    r"viewpoint|perspective|commentary|correspondence|matters arising|author correction|"
    r"publisher correction|retraction note)\b",
    re.I,
)
NON_ORIGINAL_PUBTYPES = {
    "review", "systematic review", "meta-analysis", "editorial", "comment",
    "letter", "published erratum", "retraction of publication", "retracted publication",
    "news", "interview", "biography", "consensus development conference", "practice guideline",
}

# Pre-specified Inclusion Criteria for Clinical Prediction / Prognostic ML Original Research
ML_TERM_RE = re.compile(
    r"\b(machine learning|deep learning|neural network|convolutional neural|transformer|"
    r"foundation model|random forest|xgboost|lightgbm|catboost|gradient boosting|"
    r"support vector machine|lasso|elastic net|artificial intelligence|ensemble model|"
    r"ensemble learning|prediction model|predictive model|prognostic model|risk prediction|"
    r"risk stratification|risk score|aging index|nomogram|multimodal fusion|multi-omics fusion|"
    r"radiomics|pathomics|survival prediction|supervised learning|graph neural)\b",
    re.I,
)

CLINICAL_TERM_RE = re.compile(
    r"\b(patient|patients|clinical|cohort|hospital|mortality|survival|cancer|carcinoma|"
    r"tumor|tumour|oncology|stroke|atrial fibrillation|cardiovascular|heart failure|"
    r"myocardial|sepsis|icu|intensive care|kidney|renal|ckd|ckm|diabetes|alzheimer|"
    r"dementia|depression|psychiatr\w*|pediatric|paediatric|pregnant|maternal|neonatal|"
    r"surgery|surgical|postoperative|histolog\w*|pathology|biopsy|radiograph|ct scan|mri|"
    r"ecg|electrocardiogram|echocardiogram|electronic health record|ehr|emr|nhanes|"
    r"uk biobank|tcga|mimic|diagnosis|diagnostic|prognosis|prognostic|treatment response|"
    r"drug response|chemotherapy|immunotherapy|disease risk|syndrome|readmission|triage)\b",
    re.I,
)

QUANT_PRED_RE = re.compile(
    r"\b(auc|auroc|auprc|roc|c-index|c-statistic|concordance|sensitivity|specificity|"
    r"f1[\s-]score|brier|calibration|decision curve|net benefit|hazard ratio|odds ratio|"
    r"external validation|cross-validation|held-out|test set|validation cohort|"
    r"predict(?:ing|ion|ive|ed)?\s+(?:risk|mortality|survival|outcome|outcomes|response|stroke|"
    r"onset|progression|recurrence|diagnosis|subtype|mutation|biomarker|molecular))\b",
    re.I,
)

# Exclude pure LLM benchmark / chatbot / exam / qualitative / policy / survey studies
LLM_NON_PRED_RE = re.compile(
    r"\b(large language model|chatgpt|gpt-4|gpt-3|claude|gemini|llm|llms|chatbot|"
    r"conversational agent|medical licensing exam|usmle|question answering|summarization|"
    r"clinical documentation|patient education|medical advice|prompt engineering)\b",
    re.I,
)


def load_json(name: str):
    return json.loads((RAW_DIR / name).read_text(encoding="utf-8"))


def merge_journal(journal_key: str, cr_records: list, ep_records: list):
    merged = {}
    for rec in ep_records:
        doi = (rec.get("doi") or "").strip().lower()
        if not doi:
            continue
        merged[doi] = {
            "journal_key": journal_key,
            "doi": doi,
            "pmid": rec.get("pmid"),
            "pmcid": rec.get("pmcid"),
            "title": rec.get("title") or "",
            "abstract": rec.get("abstract") or "",
            "pub_date": rec.get("pub_date") or "",
            "pub_types": rec.get("pub_types") or [],
            "cited_by_count": rec.get("cited_by_count", 0),
            "in_epmc": True,
            "in_crossref": False,
            "has_update_to": False,
        }
    for rec in cr_records:
        doi = (rec.get("doi") or "").strip().lower()
        if not doi:
            continue
        if doi in merged:
            m = merged[doi]
            m["in_crossref"] = True
            if not m["abstract"] and rec.get("abstract"):
                m["abstract"] = rec["abstract"]
            if not m["title"] and rec.get("title"):
                m["title"] = rec["title"]
            if not m["pub_date"] and rec.get("pub_date"):
                m["pub_date"] = rec["pub_date"]
            if rec.get("has_update_to"):
                m["has_update_to"] = True
        else:
            merged[doi] = {
                "journal_key": journal_key,
                "doi": doi,
                "pmid": None,
                "pmcid": None,
                "title": rec.get("title") or "",
                "abstract": rec.get("abstract") or "",
                "pub_date": rec.get("pub_date") or "",
                "pub_types": [],
                "cited_by_count": rec.get("cited_by_count", 0),
                "in_epmc": False,
                "in_crossref": True,
                "has_update_to": bool(rec.get("has_update_to")),
            }
    return list(merged.values())


def screen_record(rec: dict):
    title = rec["title"] or ""
    abstract = rec["abstract"] or ""
    text = f"{title}\n{abstract}"
    pub_types_l = {pt.lower() for pt in (rec.get("pub_types") or [])}

    # Check date window strictly (keep 2026-01-01..2026-11-25)
    pdate = rec.get("pub_date") or ""
    if pdate and not ("2026-01-01" <= pdate[:10] <= "2026-11-25"):
        return False, "outside_date_window"

    if rec.get("has_update_to") or NON_ORIGINAL_TITLE_RE.search(title) or REVIEW_IN_TITLE_RE.search(title):
        return False, "non_original_title_or_update"

    if pub_types_l & NON_ORIGINAL_PUBTYPES:
        return False, "non_original_pubtype"

    # If abstract missing, screen on title + flag needs_abstract_enrichment
    has_abs = len(abstract.strip()) >= 40
    has_ml = bool(ML_TERM_RE.search(text))
    has_clin = bool(CLINICAL_TERM_RE.search(text))
    has_quant = bool(QUANT_PRED_RE.search(text))

    # Exclude pure LLM chatbot/exam/QA papers unless they also explicitly build a patient prediction/prognostic model with AUC/C-index
    if LLM_NON_PRED_RE.search(title) and not re.search(r"\b(auc|auroc|c-index|concordance|prognostic model|risk prediction|survival prediction)\b", text, re.I):
        return False, "llm_chatbot_or_qa_benchmark"

    if not has_abs:
        # Title-only tentative keep if ML + clinical keywords appear in title (we will fetch abstract next!)
        if has_ml and has_clin:
            return True, "eligible_title_only_needs_abstract"
        return False, "missing_abstract_no_title_match"

    if not has_ml:
        return False, "no_ml_or_prediction_term"
    if not has_clin:
        return False, "no_clinical_patient_term"
    if not has_quant:
        return False, "no_quantitative_prediction_metric"

    return True, "eligible"


def main():
    plos_cr = load_json("plos_medicine_crossref_2026.json")["records"]
    plos_ep = load_json("plos_medicine_epmc_2026.json")["records"]

    npj_cr = load_json("npj_digital_medicine_crossref_2026.json")["records"]
    npj_ep = load_json("npj_digital_medicine_epmc_2026.json")["records"]

    nc_ep = load_json("nature_communications_epmc_clinical_ml_2026.json")["records"]
    nc_cr_recent = load_json("nature_communications_crossref_oct15_nov25_2026.json")["records"]

    datasets = {
        "plos_medicine": merge_journal("plos_medicine", plos_cr, plos_ep),
        "npj_digital_medicine": merge_journal("npj_digital_medicine", npj_cr, npj_ep),
        "nature_communications": merge_journal("nature_communications", nc_cr_recent, nc_ep),
    }

    summary = {"journals": {}, "total_merged": 0, "total_eligible": 0, "missing_abstract_total": 0}
    all_eligible = []
    all_screened = []

    known_4 = {
        "10.1038/s41467-026-78947-2": "Case3_NatComm_PathLUPI",
        "10.1038/s41746-026-02991-x": "Case1_npjDM_02991x",
        "10.1038/s41746-026-02470-3": "Case2_npjDM_02470",
        "10.1371/journal.pmed.1005078": "Case4_PLOSMed_CKMAI",
    }
    known_status = {}

    for jkey, recs in datasets.items():
        reasons = {}
        eligible_j = []
        missing_abs = 0
        for r in recs:
            if len((r["abstract"] or "").strip()) < 40:
                missing_abs += 1
            ok, reason = screen_record(r)
            reasons[reason] = reasons.get(reason, 0) + 1
            r["eligible"] = ok
            r["screen_reason"] = reason
            all_screened.append(r)
            if ok:
                eligible_j.append(r)
                all_eligible.append(r)
            if r["doi"] in known_4:
                known_status[r["doi"]] = {
                    "label": known_4[r["doi"]],
                    "journal": jkey,
                    "eligible": ok,
                    "reason": reason,
                    "has_abstract": len((r["abstract"] or "").strip()) >= 40,
                    "pmcid": r["pmcid"],
                    "pub_date": r["pub_date"],
                }

        summary["journals"][jkey] = {
            "merged_unique_dois": len(recs),
            "missing_abstract_count": missing_abs,
            "eligible_count": len(eligible_j),
            "reason_breakdown": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        }
        summary["total_merged"] += len(recs)
        summary["total_eligible"] += len(eligible_j)
        summary["missing_abstract_total"] += missing_abs

    summary["known_4_audit_check"] = known_status

    # Write outputs
    (PROC_DIR / "screen_summary_step2a.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (PROC_DIR / "eligible_2026_corpus.json").write_text(json.dumps(all_eligible, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(PROC_DIR / "all_screened_2026.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["journal_key", "doi", "pmcid", "pmid", "pub_date", "in_epmc", "in_crossref", "eligible", "screen_reason", "title"])
        w.writeheader()
        for r in all_screened:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
