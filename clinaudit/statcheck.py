"""Module 3: Statistical Inference, Confidence Interval & P-Value Sanity Auditor (v0.3.1)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from .io_utils import TableData, read_tables
from .models import AuditAlert, Priority

# Regex patterns for inline CI cells
_INLINE_CI_RE = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?)\s*"
    r"[\(\[]\s*(?:95%?\s*CI\s*[:=]?\s*)?"
    r"([+-]?\d+(?:\.\d+)?)\s*(?:[-–—,~]|to)\s*([+-]?\d+(?:\.\d+)?)\s*[\)\]]\s*$",
    re.IGNORECASE,
)

# Detect European decimal-comma percentage like "7 (4,2)" inside n (%) count columns
_EURO_PCT_COUNT_RE = re.compile(r"^\s*\d+\s*\(\s*\d+,\d{1,2}\s*\)\s*$")
# Detect thousands-formatted single count in parentheses like "0.9247 (20,484)" or "0.071 (1,581)"
_THOUSANDS_COUNT_PAREN_RE = re.compile(r"[\(\[]\s*\d{1,3},\d{3}\s*[\)\]]\s*$")

_INLINE_PM_RE = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?)\s*(?:±|\+/-|\+-)\s*(\d+(?:\.\d+)?)\s*$"
)

_INTERVAL_ONLY_RE = re.compile(
    r"^\s*[\(\[]?\s*([+-]?\d+(?:\.\d+)?)\s*(?:[-–—,~]|to)\s*([+-]?\d+(?:\.\d+)?)\s*[\)\]]?\s*$",
    re.IGNORECASE,
)

_PVAL_SCI_RE = re.compile(
    r"^\s*([<>≤≥]?)\s*([+-]?\d+(?:\.\d+)?)\s*(?:[x×\*]\s*10\s*\^?\s*([+-]?\d+)|[eE]([+-]?\d+))?\s*$"
)

_EXCEL_FORMULA_ERRORS = frozenset({"#NUM!", "#REF!", "#DIV/0!", "#VALUE!", "#NAME?", "#NULL!"})

_ERR_OFFSET_COL_RE = re.compile(
    r"(?:^|[_\s\-])(?:yerr|xerr|err_?lo|err_?hi|error_?bar|sem|std_?err|sd_?lo|sd_?hi|"
    r"bar_?lo|bar_?hi|ymin_?err|ymax_?err|ci_?width|ci_?len|"
    r"ci[\s_\-]*(?:lo(?:w|wer)?|up(?:per)?|hi(?:gh)?)[\s_\-]*(?:diff(?:erence)?|err(?:or)?|offset|bar|len))(?:$|[_\s\-])",
    re.IGNORECASE,
)
_ASSOC_REG_COL_RE = re.compile(
    r"^(?:beta(?:[\s_\-]+est(?:imate)?)?|exp_?coef|or|hr|odds[\s_\-]+ratio|hazard[\s_\-]+ratio|"
    r"estimate|std[\s_\-.]?err(?:or)?|z[\s_\-.]?val(?:ue)?|t[\s_\-.]?stat|"
    r"corr(?:[\s_\-]?coef)?|abs[\s_\-]?corr(?:[\s_\-]?coef)?|pearson|spearman|"
    r"fold[\s_\-]+change|log2?[\s_\-]*fold[\s_\-]*change|log2?fc|avg[\s_\-]?log2?fc|"
    r"slope\d*[\s_\-]+(?:est(?:imate)?|se|t)|"
    r"metabolite.*|assay_?value|protein.*|aptamer.*|entrez.*|feature|gene)$",
    re.IGNORECASE,
)
_LO_COL_RE = re.compile(
    r"(?:^|[_\s\-])(?:95%?[\s_\-]*(?:ci|credible[\s_\-]*interval|confidence[\s_\-]*interval)[\s_\-]*(?:lo(?:w|wer)?|min)(?:[\s_\-]*bound)?|"
    r"ci[\s_\-]*(?:lo(?:w|wer)?|min)|"
    r"lo(?:w|wer)[\s_\-]*(?:95%?[\s_\-]*)?ci(?![\s_\-]*(?:up(?:per)?|hi(?:gh)?|max))|"
    r"lci|lower[\s_\-]*95%?[\s_\-]*(?:ci|conf(?:idence)?[\s_\-]*bound)?|"
    r"lower[\s_\-]*bound|low[\s_\-]*95|ci95\.low|lower)(?:$|[_\s\-])",
    re.IGNORECASE,
)
_HI_COL_RE = re.compile(
    r"(?:^|[_\s\-])(?:95%?[\s_\-]*(?:ci|credible[\s_\-]*interval|confidence[\s_\-]*interval)[\s_\-]*(?:hi(?:gh|gher)?|up(?:per)?|max)(?:[\s_\-]*bound)?|"
    r"ci[\s_\-]*(?:hi(?:gh|gher)?|up(?:per)?|max)|"
    r"(?:hi(?:gh|gher)?|up(?:per)?)[\s_\-]*(?:95%?[\s_\-]*)?ci(?![\s_\-]*(?:lo(?:w|wer)?|min))|"
    r"uci|upper[\s_\-]*95%?[\s_\-]*(?:ci|conf(?:idence)?[\s_\-]*bound)?|"
    r"upper[\s_\-]*bound|high[\s_\-]*95|ci95\.up|upper)(?:$|[_\s\-])",
    re.IGNORECASE,
)
_EXPLICIT_LO_SUFFIX_RE = re.compile(
    r"(?:ci[\s_\-]*(?:lo(?:w|wer)?|min)|(?:lo(?:w|wer)|lci|ll)(?:_\d+)?)$",
    re.IGNORECASE,
)
_EXPLICIT_HI_SUFFIX_RE = re.compile(
    r"(?:ci[\s_\-]*(?:hi(?:gh|gher)?|up(?:per)?|max)|(?:up(?:per)|uci|ul)(?:_\d+)?)$",
    re.IGNORECASE,
)
_NON_CI_BOUND_COL_RE = re.compile(
    r"(?:^|[_\s\-])(?:lower|upper)[\s_\-]+(?:respiratory|body|plane|panel|left|right|"
    r"jaw|limb|lobe|lip|incisor|arch|tooth|teeth|layer|rhombic|cutoff|threshold|"
    r"location|site|region|extremity|gi|tract|error|pretilt|xtilt|ytilt|tilt|face|delta)(?:$|[_\s\-])|"
    r"(?:^|[_\s\-])(?:location|count|site|region|panel|plane|body)[\s_\-]+(?:lower|upper)(?:$|[_\s\-])",
    re.IGNORECASE,
)
_EXPLICIT_CI_COL_TOKEN_RE = re.compile(
    r"(?:95%?\s*ci|\bci\b|ci[\s_\-]|[\s_\-]ci\b|\blci\b|\buci\b|conf(?:idence)?|credible)",
    re.IGNORECASE,
)
_PM_COL_RE = re.compile(
    r"(?:^|[_\s\-])(?:ci[\s_\-]*half|half[\s_\-]*width|margin|ci[\s_\-]*margin|"
    r"pm|plus[\s_\-]*minus|ci[\s_\-]*delta)(?:$|[_\s\-])",
    re.IGNORECASE,
)
_CI_STR_COL_RE = re.compile(r"(?:95%?\s*ci|^ci$|conf.*int)", re.IGNORECASE)

_NON_CI_INLINE_COL_RE = re.compile(
    r"(?:n\s*\(\s*%\s*\)|count\s*\(\s*%\s*\)|cases?\s*\(\s*%\s*\)|events?\s*\(\s*%\s*\)|"
    r"median\s*[\(\[]\s*(?:iqr|q1|q25|25%|range|min)|mean\s*[\(\[]\s*(?:sd|std|min|range)|"
    r"\biqr\b|interquartile|\brange\b|\bmin[\s/\-]+max\b|"
    r"\bstage\b|\bgrade\b|\btnm\b|^(?:c|p)?[tnm](?:[\s_\-]|$)|\bage\b|\bfollow[\s_\-]*up\b|"
    r"\bduration\b|\byears?\b|\bmonths?\b|\bdays?\b|\bnumber\s+of\b|\blikert\b)",
    re.IGNORECASE,
)
_DESCRIPTIVE_RANGE_CAPTION_RE = re.compile(
    r"(?:median\s*\(\s*(?:iqr|interquartile|range|min|q1|25)|"
    r"median\s+and\s+(?:interquartile\s+range|iqr)|"
    r"mean\s*\(\s*(?:sd|standard\s+deviation|range|min)|"
    r"values?\s+(?:are|shown|presented|expressed)\s+as\s+(?:median|mean|n\s*\(%))",
    re.IGNORECASE,
)
_EXPLICIT_CI_CAPTION_RE = re.compile(r"(?:95%?\s*ci|confidence\s+interval|credible\s+interval)", re.IGNORECASE)

_PVAL_COL_RE = re.compile(
    r"^(?:p|p[-_.\s]?val(?:ue)?s?|pval|p[-_.\s]?values?|"
    r"adj(?:usted)?[-_.\s]?p(?:[-_.\s]?val(?:ue)?)?|p[-_.\s]?adj(?:usted)?|padj|q[-_.\s]?val(?:ue)?|"
    r"fdr|bh[-_.\s]?fdr|perm[-_.\s]?fdr|fdr[-_.\s]?p(?:[-_.\s]?val(?:ue)?)?|"
    r"raw[-_.\s]?p(?:[-_.\s]?val(?:ue)?)?|"
    r"wilcoxon[_\s]?p(?:[_\s-]?val(?:ue)?)?|delong[_\s]?p(?:[_\s-]?val(?:ue)?)?|"
    r"mcnemar[_\s]?p(?:[_\s-]?val(?:ue)?)?|paired[-_\s]?p|p[-_\s]?diff|p[-_\s]?comp(?:arison)?)(?:_\d+)?$",
    re.IGNORECASE,
)
_PDIFF_COL_RE = re.compile(
    r"(?:(?:p[-_.\s]?val(?:ue)?|pval).*?(?:diff(?:erence)?|minimal|comp(?:arison)?|delong|lrt|change|nested)|"
    r"(?:delong|lrt|diff(?:erence)?)[-_\s]*p(?:[-_.\s]?val(?:ue)?)?)",
    re.IGNORECASE,
)
_PVAL_EXCLUDE_DOMAIN_RE = re.compile(
    r"(?:log\d*|ln\b|neglog|mlog|score|rank|%|pct|percent|ratio|fold|fc\b)",
    re.IGNORECASE,
)
_RAW_INTENSITY_COL_RE = re.compile(
    r"(?:log2?area|area_?mean|intensity|abundance|expression|counts|tpm|fpkm|rpkm|rfu|mfi|signal)",
    re.IGNORECASE,
)
_N_COL_RE = re.compile(
    r"^(?:n|n[-_\s]?total|n[-_\s]?test|n[-_\s]?samples?|n[-_\s]?patients?|n[-_\s]?donors?|"
    r"n[-_\s]?subjects?|n[-_\s]?individuals?|n[-_\s]?cases?|n[-_\s]?min[-_\s]?max|"
    r"sample[-_\s]?size|cohort[-_\s]?size|total[-_\s]?n|total|#\s*(?:total\s*)?samples?)$",
    re.IGNORECASE,
)
_DELTA_COL_RE = re.compile(
    r"^(?:delta|diff|difference|delta_?auc|auc_?diff|delta_?metric|effect_?size|improvement|abs_?diff|"
    r"difference\s+of\s+model.*)$",
    re.IGNORECASE,
)
_UNIT_BOUNDED_METRIC_RE = re.compile(
    r"(?:^|[_\s\-\(\[/])(?:auc|auroc|auprc|roc_?auc|pr_?auc|c_?index|concordance|"
    r"acc(?:uracy)?|sens(?:itivity)?|spec(?:ificity)?|ppv|npv|f1(?:_?score)?|"
    r"recall|precision|brier|r2|r_?squared)(?:$|[_\s\-\)\]/%])",
    re.IGNORECASE,
)
_STRICT_ML_ROW_LABEL_RE = re.compile(
    r"^(?:auc|auroc|auprc|roc[-_\s]?auc|pr[-_\s]?auc|c[-_\s]?index|concordance|"
    r"accuracy|sensitivity|specificity|ppv|npv|f1(?:[-_\s]?score)?|"
    r"recall|precision|brier|rmse|mae|r2|r\^?2)(?:\s*\(%\))?$",
    re.IGNORECASE,
)
_NON_NEGATIVE_UNIT_METRIC_RE = re.compile(
    r"(?:^|[_\s\-\(\[/])(?:auc|auroc|auprc|roc_?auc|pr_?auc|c_?index|concordance|"
    r"acc(?:uracy)?|sens(?:itivity)?|spec(?:ificity)?|ppv|npv|f1(?:_?score)?|"
    r"recall|precision|brier|survival|prob(?:ability)?)(?:$|[_\s\-\)\]/%])",
    re.IGNORECASE,
)
_DIFF_OR_SHIFT_NAME_RE = re.compile(
    r"(?:delta|diff|change|gain|improv|shift|bias|error|resid|log|logit|slope|coef|beta|mean_?diff)",
    re.IGNORECASE,
)
_NOT_ESTIMATE_COL_RE = re.compile(
    r"^(?:n|n[-_\s]?total|n[-_\s]?samples?|count|counts|cases|events|sample[-_\s]?size|id|index|idx|rank|"
    r"fold|k[-_\s]?fold|p|p[-_.\s]?val(?:ue)?s?|pval|adj(?:usted)?[-_.\s]?p(?:[-_.\s]?val(?:ue)?)?|"
    r"fdr|padj|q[-_.\s]?val(?:ue)?s?|se|sd|sem|std[-_.\s]?err(?:or)?|std[-_.\s]?dev|stddev|ci[-_\s]?level|"
    r"z|z[-_.\s]?val(?:ue)?|z[-_.\s]?stat|t|t[-_.\s]?stat|chi2|chisq|df|width|ci[-_\s]?width|err|weight|order)$",
    re.IGNORECASE,
)
_NOT_ESTIMATE_TOKEN_RE = re.compile(
    r"(?:(?:^|[_\s\-])(?:se|sd|std|stddev|std_?dev|sem|std_?err(?:or)?|stderr|"
    r"err|var|variance|width|ci_?width|ci_?level|conf_?level|"
    r"pval|p_?val(?:ue)?|fdr|padj|qval|q_?val|ll|ul|lci|uci)(?:$|[_\s\-])|"
    r"^(?:abs|max|min)[_\s\-])",
    re.IGNORECASE,
)
_GSEA_PATHWAY_HDR_RE = re.compile(
    r"^(?:gene[\s_\-]?set|geneset|pathway|go[\s_\-]?id|go[\s_\-]?term|nes|es|enrichment[\s_\-]?score)$",
    re.IGNORECASE,
)
_REF_ROW_LABEL_RE = re.compile(
    r"(?:^|[\s_\-\(\[/])(?:ref(?:erence)?|baseline|control|normal|unexposed|q1|"
    r"quartile[\s_\-]*1|tertile[\s_\-]*1|quintile[\s_\-]*1|decile[\s_\-]*1|"
    r"low[\s_\-]*risk|lowest|medium|moderate|none|no|0\s*\(ref\))(?:$|[\s_\-\)\]/])",
    re.IGNORECASE,
)
_RATIO_METRIC_RE = re.compile(
    r"(?:^|[_\s\-\(\[/])(?:hr|or|irr|rr|hazard|odds|ratio)(?:$|[_\s\-\(\)\]/])",
    re.IGNORECASE,
)
_CURVE_HEADER_RE = re.compile(
    r"^(?:x|y|fpr|tpr|time|days|months|years|surv|recall|precision|"
    r"threshold|cutoff|spec|sens|1-spec|curve|group|model)(?:$|[_\s\-])",
    re.IGNORECASE,
)
_CURVE_SHEET_RE = re.compile(r"(?:roc|pr_?curve|km|kaplan|surv|curve|fig)", re.IGNORECASE)

# Algebraic self-consistency column patterns (supporting prefixed regression columns like 'slope1 SE', 'slope1 t')
_SE_COL_RE = re.compile(
    r"(?:^|[_\s.])(?:se|s\.e\.|std[-_.\s]?err(?:or)?|stderr|lfcse|se[-_.\s]?beta|beta[-_.\s]?se|"
    r"se[-_.\s]?log|log[-_.\s]?se|se[-_.\s]?coef|se[-_.\s]?est(?:imate)?)$",
    re.IGNORECASE,
)
_BETA_COL_RE = re.compile(
    r"(?:^|[_\s.])(?:beta(?:[\s_\-]+est(?:imate)?)?|b|coef(?:ficient)?|estimate|est|"
    r"log2?[-_\s]?f(?:old)?[-_\s]?c(?:hange)?|log[-_.\s]?fc|avg[-_.\s]?log2?fc|"
    r"slope\d*[\s_\-]+est(?:imate)?|slope|effect(?:[-_.\s]?size)?|log[-_.\s]?or|log[-_.\s]?hr|diff(?:erence)?)$",
    re.IGNORECASE,
)
_RATIO_EST_COL_RE = re.compile(
    r"^(?:or|hr|irr|rr|st\.?\s*or|st\.?\s*hr|odds[-_.\s]?ratio|hazard[-_.\s]?ratio|exp[-_.\s]?coef|fold[-_\s]?change)$",
    re.IGNORECASE,
)
_Z_COL_RE = re.compile(
    r"(?:^|[_\s.])(?:z|z[-_.\s]?val(?:ue)?|z[-_.\s]?stat(?:istic)?|z[-_.\s]?score|wald[-_.\s]?z|stat)$",
    re.IGNORECASE,
)
_T_COL_RE = re.compile(r"(?:^|[_\s.])(?:t|t[-_.\s]?val(?:ue)?|t[-_.\s]?stat(?:istic)?)$", re.IGNORECASE)
_DF_COL_RE = re.compile(r"^(?:df|dof|deg(?:rees)?[-_.\s]?freedom)$", re.IGNORECASE)
_CHI2_COL_RE = re.compile(
    r"^(?:chi2|chisq|chi[-_.\s]?sq(?:uared)?|wald[-_.\s]?chi2|lrt[-_.\s]?stat|f[-_.\s]?stat(?:istic)?)$",
    re.IGNORECASE,
)
_CORR_COL_RE = re.compile(
    r"^(?:r|rho|cor(?:r(?:elation)?)?|corr[-_\s]?coef|pearson[-_.\s]?r|spearman[-_.\s]?rho)$",
    re.IGNORECASE,
)

# Omics / pathway / scRNA exploratory triage patterns (routes generic STAT-004 to P2; NEVER disables P0 STAT-006)
_OMICS_HEADER_RE = re.compile(
    r"(?:^|[_\s\-.])(?:gene|genes|gene_?id|gene_?name|gene_?symbol|ensembl|entrez|entrezgenesymbol|symbol|"
    r"transcript|probe|cpg|snp|rsid|variant|locus|chr|chromosome|"
    r"pathway|go_?id|go_?term|kegg|reactome|hallmark|gsea|nes|leading_?edge|geneset|"
    r"term|description|overlap|odds_?ratio|combined_?score|"
    r"cluster|pct\.1|pct\.2|avg_?log2?fc|log2?[-_\s]?fold[-_\s]?change|log2?foldchange|logfc|basemean|lfcse|"
    r"aptamer|aptamer_?name|seqid|uniprot|protein|peptide|metabolite|lipid|otu|asv|taxon|species|genus|phylum|"
    r"cell_?type|celltype|chem_?id)(?:$|[_\s\-.])",
    re.IGNORECASE,
)
_OMICS_SHEET_RE = re.compile(
    r"(?:gsea|go_|go\b|kegg|reactome|enrich|pathway|kea3|deg\b|degs|dep\b|deps|deseq|edger|limma|"
    r"marker|seurat|scrna|snrna|cluster|gwas|ewas|eqtl|pqtl|mqtl|proteom|metabolom|"
    r"aptamer|olink|soma|lipid|microb|diff_?exp|mwu\b|volcano|"
    r"t\s+and\s+nk|b\s+cells|stromal|tumor\s+cells|overlapped\s+genes|expressed\s+proteins|"
    r"immune_?checkpoint|cytokine|cm_|ec_|piecewise|brown[\s\-]?forsythe)",
    re.IGNORECASE,
)
_ML_PERF_HEADER_RE = re.compile(
    r"(?:^|[_\s\-\(\[/])(?:auc|auroc|auprc|roc_?auc|pr_?auc|c_?index|concordance|acc(?:uracy)?|"
    r"sens(?:itivity)?|spec(?:ificity)?|ppv|npv|f1(?:_?score)?|brier|rmse|mae|"
    r"delong|mcnemar|method1|method2|finetuning|k_?folds?|n_?folds?)(?:$|[_\s\-\)\]/%])",
    re.IGNORECASE,
)

# Semantic mismatch tokens for STAT-003 cross-table guard
_HORIZON_RE = re.compile(
    r"(?:^|[\s_\-\(\[/])(\d+)\s*[-_\s]?(y(?:ear|r)?s?|m(?:onth|o)?s?|d(?:ay)?s?)(?:$|[\s_\-\)\]/])",
    re.IGNORECASE,
)
_COHORT_SPLIT_TOKENS = {
    "train": re.compile(r"\b(?:train|training|dev|development|discovery)\b", re.I),
    "val": re.compile(r"\b(?:val|validation|test|testing|external|holdout|replication)\b", re.I),
}
_SEX_TOKENS = {
    "male": re.compile(r"\b(?:male|males|men)\b", re.I),
    "female": re.compile(r"\b(?:female|females|women)\b", re.I),
}
_ENDPOINT_TOKENS = {
    "os": re.compile(r"\b(?:os|overall[\s_\-]*survival)\b", re.I),
    "pfs": re.compile(r"\b(?:pfs|progression[\s_\-]*free)\b", re.I),
    "dfs": re.compile(r"\b(?:dfs|disease[\s_\-]*free|rfs|recurrence[\s_\-]*free)\b", re.I),
    "css": re.compile(r"\b(?:css|dss|cancer[\s_\-]*specific|disease[\s_\-]*specific)\b", re.I),
}
_ASSAY_TOKENS = {
    "snp": re.compile(r"\b(?:snp|prs|pgs|gwas|genomic|genotype)\b", re.I),
    "methyl": re.compile(r"\b(?:methyl|cpg|dnam|epigen)\b", re.I),
    "protein": re.compile(r"\b(?:protein|proteom|olink|soma|aptamer)\b", re.I),
    "metabol": re.compile(r"\b(?:metabol|lipid)\b", re.I),
    "antibody": re.compile(r"\b(?:antibod|igg|igm|iga|serolog|neutraliz)\b", re.I),
    "tcell": re.compile(r"\b(?:t[\s_\-]*cell|cd4|cd8|elispot)\b", re.I),
}

_BOOTSTRAP_CANDIDATES = (50, 100, 200, 250, 500, 1000, 2000, 5000, 10000)
_Z_95 = 1.959963984540054


@dataclass(slots=True)
class CIRecord:
    """Single parsed confidence interval observation from a table row."""

    table_name: str
    source_file: str | None
    row_index: int
    row_key: str
    metric_name: str
    estimate: float | None
    lo: float
    hi: float
    decimals: int
    n_sample: float | None = None
    is_unit_bounded: bool = False
    is_pm_format: bool = False
    sheet_base_name: str | None = None
    row_label_raw: str = ""
    row_has_numeric_p: bool = False
    row_p_val: float | None = None
    source_strategy: int = 1
    is_percent_scale: bool = False

    @property
    def width(self) -> float:
        return self.hi - self.lo

    @property
    def half_width(self) -> float:
        return 0.5 * (self.hi - self.lo)


def _clean_ws(s: str) -> str:
    if "\xa0" not in s and "\u202f" not in s and "\u2009" not in s and "  " not in s and "\t" not in s and "\n" not in s:
        return s.strip()
    return re.sub(r"[\s\xa0\u202f\u2009]+", " ", s).strip()


def _count_decimals(token: str) -> int:
    s = _clean_ws(token)
    if "e" in s.lower():
        return 4
    if "." in s:
        return len(s.split(".", 1)[1])
    return 0


def _normal_two_sided_p_log10(z: float) -> tuple[float, float]:
    """Exact two-sided standard normal P-value and log10(P) without float64 underflow."""
    az = abs(float(z))
    if not math.isfinite(az):
        return 0.0, -math.inf
    if az == 0.0:
        return 1.0, 0.0
    if az <= 35.0:
        p = math.erfc(az / math.sqrt(2.0))
        if p > 0.0:
            return min(1.0, p), math.log10(p)
    # Asymptotic Mill's ratio expansion: 2 * Phi(-z) ~ sqrt(2/pi) * exp(-z^2/2) / z * (1 - 1/z^2 + 3/z^4)
    inv_z2 = 1.0 / (az * az)
    series = max(1e-12, 1.0 - inv_z2 + 3.0 * inv_z2 * inv_z2)
    log_p = 0.5 * math.log(2.0 / math.pi) - 0.5 * az * az - math.log(az) + math.log(series)
    log10_p = log_p / math.log(10.0)
    p = math.exp(log_p) if log_p > -745.0 else 0.0
    return min(1.0, p), log10_p


def _student_t_two_sided_log10_p(t_val: float, df: float) -> float:
    """Accurate log10 two-sided Student-t tail via Wallace's z-transformation (never underflows)."""
    at = abs(float(t_val))
    if not math.isfinite(at) or df <= 0:
        return 0.0
    if at == 0.0:
        return 0.0
    # Wallace (1959) approximation: z = ((df - 0.5)/(df + 0.25)) * sqrt(df * ln(1 + t^2/df))
    # Accurate across all df >= 3 and matches exact t tail within <0.4 log10 units even at t = 100.
    z_eq = math.sqrt(df * math.log1p((at * at) / df))
    if df >= 2.0:
        z_eq *= (df - 0.125) / (df + 0.125)
    _, log10_p = _normal_two_sided_p_log10(z_eq)
    return log10_p


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def exact_binomial_test(
    k: int,
    n: int,
    p0: float = 0.5,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Exact binomial test computed in log-space using stdlib math.lgamma."""
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"Invalid binomial parameters k={k}, n={n}")
    if not (0.0 <= p0 <= 1.0):
        raise ValueError(f"p0 must be in [0, 1], got {p0}")
    if n == 0:
        return {"k": k, "n": n, "p0": p0, "p_value": 1.0, "log10_p_value": 0.0}

    def _log_pmf(j: int) -> float:
        if p0 == 0.0:
            return 0.0 if j == 0 else -math.inf
        if p0 == 1.0:
            return 0.0 if j == n else -math.inf
        return _log_comb(n, j) + j * math.log(p0) + (n - j) * math.log1p(-p0)

    def _logsumexp(vals: list[float]) -> float:
        finite = [v for v in vals if math.isfinite(v)]
        if not finite:
            return -math.inf
        m = max(finite)
        return m + math.log(sum(math.exp(v - m) for v in finite))

    if alternative == "greater":
        log_p = _logsumexp([_log_pmf(j) for j in range(k, n + 1)])
    elif alternative == "less":
        log_p = _logsumexp([_log_pmf(j) for j in range(0, k + 1)])
    elif alternative == "two-sided":
        if abs(p0 - 0.5) < 1e-12:
            tail_k = max(k, n - k)
            log_tail = _logsumexp([_log_pmf(j) for j in range(tail_k, n + 1)])
            log_p = min(0.0, log_tail + math.log(2.0))
        else:
            lp_k = _log_pmf(k)
            log_p = _logsumexp([_log_pmf(j) for j in range(0, n + 1) if _log_pmf(j) <= lp_k + 1e-12])
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    log_p = min(0.0, log_p)
    p_val = math.exp(log_p) if log_p > -745.0 else 0.0
    return {
        "k": k,
        "n": n,
        "p0": p0,
        "alternative": alternative,
        "p_value": min(1.0, p_val),
        "log10_p_value": log_p / math.log(10.0) if math.isfinite(log_p) else -math.inf,
    }


def exact_mcnemar_test(b: int, c: int) -> dict[str, Any]:
    """Exact two-sided McNemar binomial test plus Edwards continuity-corrected chi2."""
    if b < 0 or c < 0:
        raise ValueError(f"Discordant counts must be non-negative: b={b}, c={c}")
    d = b + c
    if d == 0:
        return {
            "b": b,
            "c": c,
            "n_discordant": 0,
            "p_value": 1.0,
            "log10_p_value": 0.0,
            "chi2_edwards": 0.0,
        }
    res = exact_binomial_test(max(b, c), d, p0=0.5, alternative="two-sided")
    chi2_cc = (max(0.0, abs(b - c) - 1.0) ** 2) / float(d)
    return {
        "b": b,
        "c": c,
        "n_discordant": d,
        "p_value": res["p_value"],
        "log10_p_value": res["log10_p_value"],
        "chi2_edwards": chi2_cc,
    }


def min_mcnemar_p_for_net_diff(net_diff_count: int) -> tuple[float, float]:
    """Smallest mathematically possible two-sided McNemar P-value for net difference |b - c|."""
    d = max(0, int(net_diff_count))
    if d == 0:
        return 1.0, 0.0
    # Achieved when all discordant pairs go in one direction: b = d, c = 0 -> P = min(1, 2 * 2^-d)
    log10_p = min(0.0, (1 - d) * math.log10(2.0))
    p = min(1.0, 10.0**log10_p) if log10_p > -323.0 else 0.0
    return p, log10_p


def paired_z_sensitivity_check(
    delta: float,
    se1: float | None = None,
    se2: float | None = None,
    se_diff: float | None = None,
    p_reported: float | None = None,
    rhos: tuple[float, ...] = (0.0, 0.5, 0.8, 0.9, 0.95),
) -> dict[str, Any]:
    """Normal-approximation paired-Z sensitivity grid across correlations rho."""
    abs_d = abs(float(delta))
    grid: dict[str, dict[str, float]] = {}

    if se_diff is not None and se_diff > 0.0:
        z = abs_d / float(se_diff)
        p, log10_p = _normal_two_sided_p_log10(z)
        grid["direct"] = {
            "se_diff": float(se_diff),
            "z": z,
            "p_value": p,
            "log10_p": log10_p,
        }
    elif se1 is not None and se2 is not None and se1 > 0.0 and se2 > 0.0:
        for rho in rhos:
            var_d = max(1e-24, se1 * se1 + se2 * se2 - 2.0 * rho * se1 * se2)
            sed = math.sqrt(var_d)
            z = abs_d / sed
            p, log10_p = _normal_two_sided_p_log10(z)
            grid[f"rho={rho:.2f}"] = {
                "rho": rho,
                "se_diff": sed,
                "z": z,
                "p_value": p,
                "log10_p": log10_p,
            }

    inconsistent = False
    max_rho_key = f"rho={max(rhos):.2f}" if rhos else "direct"
    ref_entry = grid.get("direct") or grid.get(max_rho_key)
    if p_reported is not None and ref_entry is not None and p_reported > 0.0:
        log10_rep = math.log10(p_reported)
        ref_p = ref_entry["p_value"]
        ref_log10_p = ref_entry["log10_p"]
        if p_reported < 1e-4 and ((ref_p > 0.05) or (log10_rep < ref_log10_p - 5.0)):
            inconsistent = True
    elif p_reported == 0.0 and ref_entry is not None and ref_entry["p_value"] > 1e-4:
        inconsistent = True

    return {
        "delta": abs_d,
        "p_reported": p_reported,
        "sensitivity_grid": grid,
        "contradicts_reported_p": inconsistent,
    }


def _parse_p_value_cell_unbounded(val: Any, raw_str: str = "") -> tuple[float | None, bool]:
    """Parse numeric or string P-value cell WITHOUT clipping to [0, 1] (used for STAT-006 domain checks)."""
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        f = float(val)
        return (f, False) if math.isfinite(f) else (None, False)
    s = _clean_ws(raw_str or (str(val) if val is not None else ""))
    if not s:
        return None, False
    m = _PVAL_SCI_RE.match(s)
    if not m:
        return None, False
    op, mant_s, exp1, exp2 = m.group(1), m.group(2), m.group(3), m.group(4)
    try:
        mant = float(mant_s)
        exp = int(exp1) if exp1 is not None else (int(exp2) if exp2 is not None else 0)
        p = mant * (10.0**exp)
    except (ValueError, OverflowError):
        return None, False
    if not math.isfinite(p):
        return None, False
    is_lt = op in ("<", "≤")
    return p, is_lt


def _parse_p_value_cell(val: Any, raw_str: str = "") -> tuple[float | None, bool]:
    """Parse numeric or string P-value cell into (p_float, is_inequality_less_than) within [0, 1]."""
    p, is_lt = _parse_p_value_cell_unbounded(val, raw_str=raw_str)
    if p is None or not (0.0 <= p <= 1.0):
        return None, False
    return p, is_lt


def _raw_cell(table: TableData, r_idx: int, c_idx: int) -> str:
    if r_idx < len(table.raw_rows) and c_idx < len(table.raw_rows[r_idx]):
        return str(table.raw_rows[r_idx][c_idx])
    if r_idx < len(table.rows) and c_idx < len(table.rows[r_idx]) and table.rows[r_idx][c_idx] is not None:
        return str(table.rows[r_idx][c_idx])
    return ""


def _to_float_or_none(val: Any) -> float | None:
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        f = float(val)
        return f if math.isfinite(f) else None
    try:
        f = float(_clean_ws(str(val)))
        return f if math.isfinite(f) else None
    except ValueError:
        return None


def _parse_n_cell(val: Any) -> float | None:
    """Parse sample-size cell including '450-460' ranges (returning the upper bound)."""
    f = _to_float_or_none(val)
    if f is not None and f > 0:
        return f
    if isinstance(val, str):
        m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)\s*$", val.strip())
        if m:
            hi = float(m.group(2))
            return hi if hi > 0 else None
    return None


def _normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _clean_ws(text).lower()).strip("_")


def _detect_unaligned_roc_columns(table: TableData, lo_j: int, hi_j: int) -> bool:
    """Detect ROC/PR sheets storing separate unaligned (x_min, y_min) and (x_max, y_max) polylines."""
    if table.n_rows < 30:
        return False
    headers = table.headers
    h_lo = _clean_ws(headers[lo_j]).lower()
    h_hi = _clean_ws(headers[hi_j]).lower()
    if not (
        re.search(r"\b(?:tpr|fpr|sens|spec|recall|precision)\b", h_lo)
        and re.search(r"\b(?:tpr|fpr|sens|spec|recall|precision)\b", h_hi)
    ):
        return False
    # Check if each has its own paired coordinate column immediately to the left/right with different non-empty lengths or x values
    x_lo_j = lo_j - 1 if lo_j > 0 else None
    x_hi_j = hi_j - 1 if hi_j > 0 else None
    if x_lo_j is not None and x_hi_j is not None and x_lo_j != x_hi_j:
        hx_lo = _clean_ws(headers[x_lo_j]).lower()
        hx_hi = _clean_ws(headers[x_hi_j]).lower()
        if re.search(r"\b(?:fpr|tpr|1-spec|recall|x)\b", hx_lo) and re.search(r"\b(?:fpr|tpr|1-spec|recall|x)\b", hx_hi):
            n_diff_x = 0
            n_both = 0
            n_lo_only = 0
            n_hi_only = 0
            for row in table.rows:
                v_lo = _to_float_or_none(row[x_lo_j]) if x_lo_j < len(row) else None
                v_hi = _to_float_or_none(row[x_hi_j]) if x_hi_j < len(row) else None
                if v_lo is not None and v_hi is not None:
                    n_both += 1
                    if abs(v_lo - v_hi) > 1e-4:
                        n_diff_x += 1
                elif v_lo is not None:
                    n_lo_only += 1
                elif v_hi is not None:
                    n_hi_only += 1
            if n_both >= 20 and (n_diff_x / n_both >= 0.20 or abs(n_lo_only - n_hi_only) >= 5):
                return True
    return False


def _is_descriptive_non_ci_table(table: TableData) -> bool:
    """Return True if table captions explicitly describe Median (IQR) / Mean (SD) / n (%) with NO CI mention."""
    cap = " ".join(table.caption_lines)
    if not cap:
        return False
    if _EXPLICIT_CI_CAPTION_RE.search(cap):
        return False
    return bool(_DESCRIPTIVE_RANGE_CAPTION_RE.search(cap))


def _is_omics_exploratory_table(table: TableData, has_ml_perf_context: bool) -> bool:
    """Identify exploratory omics / enrichment / scRNA / differential-expression tables for P2 triage."""
    if has_ml_perf_context:
        return False
    clean_headers = [_clean_ws(h) for h in table.headers]
    omics_hdr_hits = sum(1 for h in clean_headers if _OMICS_HEADER_RE.search(h))
    sheet_text = f"{table.sheet_base_name or table.name} {' '.join(table.caption_lines[:2])}"
    omics_sheet_hit = bool(_OMICS_SHEET_RE.search(sheet_text))
    if omics_hdr_hits >= 2:
        return True
    if omics_hdr_hits >= 1 and (omics_sheet_hit or table.n_rows >= 80):
        return True
    if omics_sheet_hit and table.n_rows >= 15:
        return True
    return False


def extract_ci_records(table: TableData) -> list[CIRecord]:
    """Extract structured CI records from inline formatted cells or paired low/high columns."""
    if not table.headers or not table.rows:
        return []

    n_cols = len(table.headers)
    headers = [_clean_ws(h) for h in table.headers]

    # Identify sample-size column, P-value columns, and key columns
    n_col_idx: int | None = None
    for j, h in enumerate(headers):
        if _N_COL_RE.match(h):
            n_col_idx = j
            break

    pval_cols = [j for j, h in enumerate(headers) if _PVAL_COL_RE.match(h)]

    key_col_indices: list[int] = []
    for j, h in enumerate(headers):
        hl = h.lower()
        if (
            j == n_col_idx
            or _UNIT_BOUNDED_METRIC_RE.search(h)
            or _LO_COL_RE.search(h)
            or _HI_COL_RE.search(h)
            or _PM_COL_RE.search(h)
            or _CI_STR_COL_RE.search(h)
            or _PVAL_COL_RE.match(h)
            or _DELTA_COL_RE.match(h)
        ):
            continue
        sample_cells = [_clean_ws(_raw_cell(table, r, j)) for r in range(min(5, len(table.rows)))]
        if any(_INLINE_CI_RE.match(s) or _INLINE_PM_RE.match(s) for s in sample_cells if s):
            continue
        if any(
            k in hl
            for k in (
                "model",
                "method",
                "task",
                "cohort",
                "dataset",
                "endpoint",
                "subgroup",
                "name",
                "metric",
                "outcome",
                "algorithm",
                "comparison",
                "feature",
                "marker",
                "variable",
                "diagnosis",
                "cancer",
                "curve",
                "geneset",
            )
        ):
            key_col_indices.append(j)
    if not key_col_indices:
        for j in range(min(2, n_cols)):
            sample_cells = [_clean_ws(_raw_cell(table, r, j)) for r in range(min(5, len(table.rows)))]
            if any(_INLINE_CI_RE.match(s) or _INLINE_PM_RE.match(s) for s in sample_cells if s):
                continue
            str_hits = sum(
                1 for row in table.rows[:20] if j < len(row) and isinstance(row[j], str)
            )
            if str_hits >= max(1, min(len(table.rows), 20) // 2):
                key_col_indices.append(j)
                break

    def _row_key_and_label(r_idx: int) -> tuple[str, str]:
        parts: list[str] = []
        raw_parts: list[str] = []
        row = table.rows[r_idx]
        for j in key_col_indices:
            if j < len(row) and row[j] is not None and str(row[j]).strip():
                s = _clean_ws(str(row[j]))
                parts.append(_normalize_key(s))
                raw_parts.append(s)
        if not raw_parts:
            # Capture first 2 string cells for Ref-row detection even when key_col_indices was empty
            for j in range(min(3, len(row))):
                if isinstance(row[j], str) and row[j].strip():
                    raw_parts.append(_clean_ws(row[j]))
        return ("|".join(parts) if parts else f"row_{r_idx}"), " ".join(raw_parts)

    def _row_n(r_idx: int) -> float | None:
        if n_col_idx is None or n_col_idx >= len(table.rows[r_idx]):
            return None
        return _parse_n_cell(table.rows[r_idx][n_col_idx])

    def _row_p_info(r_idx: int, col_j: int | None = None) -> tuple[bool, float | None]:
        if not pval_cols:
            return False, None
        row = table.rows[r_idx]
        # Prefer the nearest P-value column immediately to the right of col_j (e.g., HR_1, p, HR_2, p_1)
        ordered = pval_cols
        if col_j is not None and len(pval_cols) > 1:
            right = [pj for pj in pval_cols if pj > col_j]
            left = [pj for pj in pval_cols if pj <= col_j]
            ordered = right + left
        for pj in ordered:
            if pj < len(row):
                pv, _ = _parse_p_value_cell(row[pj], _raw_cell(table, r_idx, pj))
                if pv is not None:
                    return True, pv
        return False, None

    records: list[CIRecord] = []
    used_cols: set[int] = set()
    base_sheet = table.sheet_base_name or table.name

    def _is_valid_est_header(h: str) -> bool:
        if _NOT_ESTIMATE_COL_RE.match(h) or _NOT_ESTIMATE_TOKEN_RE.search(h):
            return False
        if _LO_COL_RE.search(h) or _HI_COL_RE.search(h) or _PVAL_COL_RE.match(h):
            return False
        return True

    # Strategy 1: Paired low/high columns (supporting both [est, lo, hi] and [est, hi, lo] orders)
    lo_cols = [
        j
        for j, h in enumerate(headers)
        if _LO_COL_RE.search(h)
        and not _ERR_OFFSET_COL_RE.search(h)
        and not (_NON_CI_BOUND_COL_RE.search(h) and not _EXPLICIT_CI_COL_TOKEN_RE.search(h))
        and not (_EXPLICIT_HI_SUFFIX_RE.search(h) and not _EXPLICIT_LO_SUFFIX_RE.search(h))
    ]
    hi_cols = [
        j
        for j, h in enumerate(headers)
        if _HI_COL_RE.search(h)
        and not _ERR_OFFSET_COL_RE.search(h)
        and not (_NON_CI_BOUND_COL_RE.search(h) and not _EXPLICIT_CI_COL_TOKEN_RE.search(h))
        and not (_EXPLICIT_LO_SUFFIX_RE.search(h) and not _EXPLICIT_HI_SUFFIX_RE.search(h))
    ]
    matched_pairs: list[tuple[int | None, int, int, str]] = []

    used_hi: set[int] = set()
    for lo_j in lo_cols:
        lo_prefix = _normalize_key(_LO_COL_RE.sub("", headers[lo_j]))
        lo_has_ci_tok = bool(_EXPLICIT_CI_COL_TOKEN_RE.search(headers[lo_j]))
        best_hi: int | None = None
        prefix_matched = False
        for hi_j in hi_cols:
            if hi_j in used_hi or hi_j == lo_j:
                continue
            hi_prefix = _normalize_key(_HI_COL_RE.sub("", headers[hi_j]))
            if lo_prefix and lo_prefix == hi_prefix:
                best_hi = hi_j
                prefix_matched = True
                break
        if best_hi is None:
            # Prefer immediately following unmatched hi_j, else immediately preceding hi_j ([est, hi, lo])
            after_cands = [h_j for h_j in hi_cols if h_j not in used_hi and h_j > lo_j]
            if after_cands:
                cand_hi = min(after_cands)
                hi_has_ci_tok = bool(_EXPLICIT_CI_COL_TOKEN_RE.search(headers[cand_hi]))
                # Bare lower/upper columns without explicit CI token must sit near each other or share prefix
                if lo_has_ci_tok or hi_has_ci_tok or (cand_hi - lo_j) <= 3:
                    best_hi = cand_hi
            else:
                before_cands = [h_j for h_j in hi_cols if h_j not in used_hi and h_j == lo_j - 1]
                if before_cands:
                    best_hi = before_cands[0]

        if best_hi is not None:
            used_hi.add(best_hi)
            # Skip unaligned ROC/PR curve polyline pairs (handled separately via trapezoidal AUC)
            if _detect_unaligned_roc_columns(table, lo_j, best_hi):
                used_cols.update({lo_j, best_hi})
                continue

            # Locate associated point estimate column (never pick lo_j, best_hi, or SE/SD/err/width/p cols)
            est_j: int | None = None
            metric_name = _LO_COL_RE.sub("", headers[lo_j]).strip("_ -") or headers[lo_j]
            if lo_prefix:
                exact_cand: int | None = None
                suffixed_cand: int | None = None
                for k_j, kh in enumerate(headers):
                    if k_j in (lo_j, best_hi) or k_j in used_cols or not _is_valid_est_header(kh):
                        continue
                    nk = _normalize_key(kh)
                    if nk in (f"{lo_prefix}_b", f"{lo_prefix}_mean", f"{lo_prefix}_est", f"{lo_prefix}_beta"):
                        suffixed_cand = k_j
                        break
                    if nk == lo_prefix and exact_cand is None:
                        exact_cand = k_j
                picked = suffixed_cand if suffixed_cand is not None else exact_cand
                if picked is not None:
                    est_j = picked
                    metric_name = headers[picked]
            if est_j is None:
                # Check immediately preceding column; if it's a rejected dispersion col (SE/SD/abs/max), look up to 2 cols left
                base_left = min(lo_j, best_hi) - 1
                for step_back in (0, 1, 2):
                    cand_j = base_left - step_back
                    if (
                        cand_j < 0
                        or cand_j in used_cols
                        or cand_j in (lo_j, best_hi)
                        or cand_j in key_col_indices
                        or cand_j == n_col_idx
                    ):
                        continue
                    ch = headers[cand_j]
                    if not _is_valid_est_header(ch):
                        continue
                    if step_back > 0:
                        # Only accept non-adjacent leftward column if it shares prefix or is an explicit estimate/mean/b col
                        nch = _normalize_key(ch)
                        if not (
                            (lo_prefix and (nch == lo_prefix or nch.startswith(f"{lo_prefix}_") or lo_prefix in nch))
                            or re.search(r"(?:^|_)(?:mean|est|estimate|b|beta|coef|auc|auroc|auprc|hr|or|score|rate)$", nch)
                        ):
                            continue
                    num_hits = sum(
                        1
                        for row in table.rows[:20]
                        if cand_j < len(row) and _to_float_or_none(row[cand_j]) is not None
                    )
                    if num_hits >= max(1, min(len(table.rows), 20) // 2):
                        est_j = cand_j
                        metric_name = ch
                        break
                    break

            matched_pairs.append((est_j, lo_j, best_hi, metric_name))
            used_cols.update({lo_j, best_hi})
            if est_j is not None:
                used_cols.add(est_j)

    for est_j, lo_j, hi_j, metric_name in matched_pairs:
        is_unit = bool(_UNIT_BOUNDED_METRIC_RE.search(metric_name))
        is_pct = "%" in re.sub(r"\b(?:90|95|99)%", "", metric_name)
        for r_idx, row in enumerate(table.rows):
            lo_v = _to_float_or_none(row[lo_j] if lo_j < len(row) else None)
            hi_v = _to_float_or_none(row[hi_j] if hi_j < len(row) else None)
            if lo_v is None or hi_v is None:
                continue
            est_v = _to_float_or_none(row[est_j]) if (est_j is not None and est_j < len(row)) else None
            dec = max(
                _count_decimals(_raw_cell(table, r_idx, lo_j)),
                _count_decimals(_raw_cell(table, r_idx, hi_j)),
            )
            rkey, rlbl = _row_key_and_label(r_idx)
            has_p, p_v = _row_p_info(r_idx, col_j=max(lo_j, hi_j))
            records.append(
                CIRecord(
                    table_name=table.name,
                    source_file=table.source_file,
                    row_index=r_idx,
                    row_key=rkey,
                    metric_name=metric_name,
                    estimate=est_v,
                    lo=lo_v,
                    hi=hi_v,
                    decimals=dec,
                    n_sample=_row_n(r_idx),
                    is_unit_bounded=is_unit,
                    sheet_base_name=base_sheet,
                    row_label_raw=rlbl,
                    row_has_numeric_p=has_p,
                    row_p_val=p_v,
                    source_strategy=1,
                    is_percent_scale=is_pct,
                )
            )

    # Strategy 2: Estimate + margin/half-width columns
    pm_cols = [j for j, h in enumerate(headers) if j not in used_cols and _PM_COL_RE.search(h)]
    for pm_j in pm_cols:
        est_j = (
            pm_j - 1
            if pm_j > 0 and (pm_j - 1) not in used_cols and (pm_j - 1) not in key_col_indices
            else None
        )
        if est_j is None:
            continue
        metric_name = headers[est_j]
        is_unit = bool(_UNIT_BOUNDED_METRIC_RE.search(metric_name))
        is_pct = "%" in metric_name
        used_cols.update({est_j, pm_j})
        for r_idx, row in enumerate(table.rows):
            est_v = _to_float_or_none(row[est_j] if est_j < len(row) else None)
            hw_v = _to_float_or_none(row[pm_j] if pm_j < len(row) else None)
            if est_v is None or hw_v is None or hw_v < 0:
                continue
            dec = _count_decimals(_raw_cell(table, r_idx, pm_j))
            rkey, rlbl = _row_key_and_label(r_idx)
            has_p, p_v = _row_p_info(r_idx, col_j=pm_j)
            records.append(
                CIRecord(
                    table_name=table.name,
                    source_file=table.source_file,
                    row_index=r_idx,
                    row_key=rkey,
                    metric_name=metric_name,
                    estimate=est_v,
                    lo=est_v - hw_v,
                    hi=est_v + hw_v,
                    decimals=dec,
                    n_sample=_row_n(r_idx),
                    is_unit_bounded=is_unit,
                    is_pm_format=True,
                    sheet_base_name=base_sheet,
                    row_label_raw=rlbl,
                    row_has_numeric_p=has_p,
                    row_p_val=p_v,
                    source_strategy=2,
                    is_percent_scale=is_pct,
                )
            )

    # Strategy 3: Inline formatted string columns: "0.842 (0.810-0.874)", "0.842 ± 0.015", or "(0.810, 0.874)"
    is_desc_table = _is_descriptive_non_ci_table(table)
    for j, h in enumerate(headers):
        if j in used_cols or j in key_col_indices or j == n_col_idx:
            continue
        has_explicit_ci_hdr = bool(_CI_STR_COL_RE.search(h))
        if not has_explicit_ci_hdr:
            if _NON_CI_INLINE_COL_RE.search(h):
                continue
            if is_desc_table:
                continue

        is_unit = bool(_UNIT_BOUNDED_METRIC_RE.search(h))
        is_pct = "%" in h
        # Fast pre-check: if a large column has zero string cells with brackets or +/- in first 100 rows, skip O(N) regex scan
        if len(table.rows) > 200:
            has_cand_str = False
            for r_chk in table.rows[:100]:
                if j < len(r_chk) and isinstance(r_chk[j], str):
                    sc = r_chk[j]
                    if "(" in sc or "[" in sc or "±" in sc or "+-" in sc or "+/-" in sc or (has_explicit_ci_hdr and ("-" in sc or "," in sc)):
                        has_cand_str = True
                        break
            if not has_cand_str:
                continue
        for r_idx, row in enumerate(table.rows):
            if j >= len(row):
                continue
            cell_v = row[j]
            if not isinstance(cell_v, str):
                continue
            s = _clean_ws(cell_v)
            if not s:
                continue
            # Guard against European decimal comma "7 (4,2)" or thousands-formatted count "0.92 (20,484)"
            if not has_explicit_ci_hdr and (_EURO_PCT_COUNT_RE.match(s) or _THOUSANDS_COUNT_PAREN_RE.search(s)):
                continue
            m_ci = _INLINE_CI_RE.match(s)
            if m_ci:
                est_v = float(m_ci.group(1))
                lo_v = float(m_ci.group(2))
                hi_v = float(m_ci.group(3))
                dec = max(_count_decimals(m_ci.group(2)), _count_decimals(m_ci.group(3)))
                rkey, rlbl = _row_key_and_label(r_idx)
                has_p, p_v = _row_p_info(r_idx, col_j=j)
                records.append(
                    CIRecord(
                        table_name=table.name,
                        source_file=table.source_file,
                        row_index=r_idx,
                        row_key=rkey,
                        metric_name=h,
                        estimate=est_v,
                        lo=lo_v,
                        hi=hi_v,
                        decimals=dec,
                        n_sample=_row_n(r_idx),
                        is_unit_bounded=is_unit,
                        sheet_base_name=base_sheet,
                        row_label_raw=rlbl,
                        row_has_numeric_p=has_p,
                        row_p_val=p_v,
                        source_strategy=3,
                        is_percent_scale=is_pct,
                    )
                )
                continue
            m_pm = _INLINE_PM_RE.match(s)
            if m_pm:
                est_v = float(m_pm.group(1))
                hw_v = float(m_pm.group(2))
                dec = _count_decimals(m_pm.group(2))
                rkey, rlbl = _row_key_and_label(r_idx)
                has_p, p_v = _row_p_info(r_idx, col_j=j)
                records.append(
                    CIRecord(
                        table_name=table.name,
                        source_file=table.source_file,
                        row_index=r_idx,
                        row_key=rkey,
                        metric_name=h,
                        estimate=est_v,
                        lo=est_v - hw_v,
                        hi=est_v + hw_v,
                        decimals=dec,
                        n_sample=_row_n(r_idx),
                        is_unit_bounded=is_unit,
                        is_pm_format=True,
                        sheet_base_name=base_sheet,
                        row_label_raw=rlbl,
                        row_has_numeric_p=has_p,
                        row_p_val=p_v,
                        source_strategy=3,
                        is_percent_scale=is_pct,
                    )
                )
                continue
            if has_explicit_ci_hdr:
                m_int = _INTERVAL_ONLY_RE.match(s)
                if m_int:
                    lo_v = float(m_int.group(1))
                    hi_v = float(m_int.group(2))
                    can_bind_left = (
                        j > 0
                        and (j - 1) not in key_col_indices
                        and (j - 1) != n_col_idx
                        and _is_valid_est_header(headers[j - 1])
                    )
                    est_v = _to_float_or_none(row[j - 1]) if can_bind_left else None
                    m_name = headers[j - 1] if est_v is not None else h
                    dec = max(_count_decimals(m_int.group(1)), _count_decimals(m_int.group(2)))
                    rkey, rlbl = _row_key_and_label(r_idx)
                    has_p, p_v = _row_p_info(r_idx)
                    records.append(
                        CIRecord(
                            table_name=table.name,
                            source_file=table.source_file,
                            row_index=r_idx,
                            row_key=rkey,
                            metric_name=m_name,
                            estimate=est_v,
                            lo=lo_v,
                            hi=hi_v,
                            decimals=dec,
                            n_sample=_row_n(r_idx),
                            is_unit_bounded=bool(_UNIT_BOUNDED_METRIC_RE.search(m_name)),
                            sheet_base_name=base_sheet,
                            row_label_raw=rlbl,
                            row_has_numeric_p=has_p,
                            row_p_val=p_v,
                            source_strategy=3,
                            is_percent_scale="%" in m_name,
                        )
                    )

    return records


def _is_legitimate_zero_width(
    rec: CIRecord,
    is_curve_sheet: bool,
    table_has_pval_col: bool,
) -> bool:
    """Return True ONLY for mathematically/conventionally legitimate zero-width CI rows."""
    est = rec.estimate
    # If point estimate is strictly outside [lo, hi], NEVER legitimate (e.g., 100.0 [0, 0] or 0.67 [0, 0])
    if est is not None and (est < rec.lo - 1e-6 or est > rec.hi + 1e-6):
        return False

    # 1. Explicit Reference Category Row: exact 1 [1, 1] with Ref/Baseline/Q1 label OR blank P-value
    if est is not None and abs(est - 1.0) <= 1e-6 and abs(rec.lo - 1.0) <= 1e-6:
        if _REF_ROW_LABEL_RE.search(rec.row_label_raw):
            return True
        if table_has_pval_col and not rec.row_has_numeric_p:
            return True
        # Continuous 2-3 decimal quantized HR/OR = 1.00 (1.00, 1.00) with non-significant P > 0.05 or per-unit scale
        if rec.decimals <= 3 and _RATIO_METRIC_RE.search(rec.metric_name):
            if (rec.row_p_val is not None and rec.row_p_val > 0.05) or ("per " in rec.metric_name.lower()):
                return True
        # Univariate/Multivariate regression table Reference row ("Univariate analysis" = 1.0 [1.0, 1.0])
        if re.search(r"\b(?:univariate|multivariate|multivariable|adjusted|crude)\b", rec.metric_name, re.I):
            if not rec.row_has_numeric_p or (rec.row_p_val is not None and rec.row_p_val > 0.10):
                return True

    # 2. Curve boundary endpoints (0 [0, 0] or 1 [1, 1] on ROC/PR/KM curve sheets with >= 20 rows)
    if is_curve_sheet and est is not None and (abs(est) <= 1e-6 or abs(est - 1.0) <= 1e-6):
        return True

    # 3. Large-N 2-decimal rounding quantization (N >= 5000, decimals <= 2, est == lo == hi)
    if rec.n_sample is not None and rec.n_sample >= 5000 and rec.decimals <= 2:
        return True

    # 4. Tiny-N degenerate boundary (N <= 3 and est at 0 / 1 / 100)
    if (
        rec.n_sample is not None
        and rec.n_sample <= 3
        and est is not None
        and (abs(est) <= 1e-6 or abs(est - 1.0) <= 1e-6 or abs(est - 100.0) <= 1e-6)
    ):
        return True

    return False


def _algebraic_p_consistent(
    row: list[Any],
    eff_p: float,
    row_n: float | None,
    se_col: int | None,
    beta_col: int | None,
    ratio_col: int | None,
    z_col: int | None,
    t_col: int | None,
    df_col: int | None,
    chi2_col: int | None,
    corr_col: int | None,
    row_cis: list[CIRecord],
) -> bool:
    """Return True if summary stats in the same row algebraically reproduce the reported P-value."""
    expected_log10_ps: list[float] = []

    # (a) Direct z-score / Wald z / DESeq2 stat
    if z_col is not None and z_col < len(row):
        zv = _to_float_or_none(row[z_col])
        if zv is not None and abs(zv) >= 4.0:
            _, lp = _normal_two_sided_p_log10(zv)
            expected_log10_ps.append(lp)

    # (b) Direct Student-t statistic
    if t_col is not None and t_col < len(row):
        tv = _to_float_or_none(row[t_col])
        if tv is not None and abs(tv) >= 4.0:
            df_v = _to_float_or_none(row[df_col]) if (df_col is not None and df_col < len(row)) else None
            if df_v is None and row_n is not None and row_n > 3:
                df_v = row_n - 2.0
            if df_v is not None and df_v >= 3.0:
                expected_log10_ps.append(_student_t_two_sided_log10_p(tv, df_v))
            elif abs(tv) >= 8.0:
                _, lp = _normal_two_sided_p_log10(tv)
                expected_log10_ps.append(lp)

    # (c) 1-df chi2 / F-stat
    if chi2_col is not None and chi2_col < len(row):
        c2 = _to_float_or_none(row[chi2_col])
        if c2 is not None and c2 >= 16.0:
            _, lp = _normal_two_sided_p_log10(math.sqrt(c2))
            expected_log10_ps.append(lp)

    # (d) (Beta / logFC, SE) or (log(OR/HR), SE)
    if se_col is not None and se_col < len(row):
        se_v = _to_float_or_none(row[se_col])
        if se_v is not None and se_v > 0.0:
            if beta_col is not None and beta_col < len(row):
                bv = _to_float_or_none(row[beta_col])
                if bv is not None:
                    _, lp = _normal_two_sided_p_log10(abs(bv) / se_v)
                    expected_log10_ps.append(lp)
            if ratio_col is not None and ratio_col < len(row):
                rv = _to_float_or_none(row[ratio_col])
                if rv is not None and rv > 0.0:
                    _, lp = _normal_two_sided_p_log10(abs(math.log(rv)) / se_v)
                    expected_log10_ps.append(lp)

    # (e) Correlation (r / rho, N)
    if corr_col is not None and corr_col < len(row) and row_n is not None and row_n >= 15:
        rv = _to_float_or_none(row[corr_col])
        if rv is not None and 0.0 < abs(rv) < 0.999999:
            df_v = row_n - 2.0
            t_eq = abs(rv) * math.sqrt(df_v / max(1e-12, 1.0 - rv * rv))
            expected_log10_ps.append(_student_t_two_sided_log10_p(t_eq, df_v))

    # (f) Same-row regression CI for Beta or OR/HR
    for rec in row_cis:
        if rec.width <= 1e-9 or rec.estimate is None:
            continue
        if _RATIO_METRIC_RE.search(rec.metric_name) and rec.lo > 0 and rec.hi > rec.lo and rec.estimate > 0:
            se_log = (math.log(rec.hi) - math.log(rec.lo)) / (2.0 * _Z_95)
            if se_log > 1e-12:
                _, lp = _normal_two_sided_p_log10(abs(math.log(rec.estimate)) / se_log)
                expected_log10_ps.append(lp)
        elif _BETA_COL_RE.match(_clean_ws(rec.metric_name)):
            se_est = rec.half_width / _Z_95
            if se_est > 1e-12:
                _, lp = _normal_two_sided_p_log10(abs(rec.estimate) / se_est)
                expected_log10_ps.append(lp)

    if not expected_log10_ps:
        return False

    best_exp = min(expected_log10_ps)  # Most extreme algebraically supported log10(P)
    if eff_p == 0.0:
        return best_exp <= -260.0
    log10_rep = math.log10(eff_p)
    # Consistent if within 3.5 orders of magnitude OR reported P is LESS extreme than algebraic expectation
    return (abs(log10_rep - best_exp) <= 3.5) or (best_exp <= -10.0 and log10_rep >= best_exp - 4.0)


def _column_is_fixed_decimal_rounded_p(table: TableData, col_idx: int) -> bool:
    """Detect columns where all non-zero P-values are fixed-decimal rounded (<= 5 dec) with rare 0.00000."""
    n_nonzero = 0
    n_zero = 0
    n_short_fixed = 0
    rows_slice = table.rows[:5000]
    for r_idx, row in enumerate(rows_slice):
        if col_idx >= len(row):
            continue
        cell_v = row[col_idx]
        if cell_v is None:
            continue
        raw = _raw_cell(table, r_idx, col_idx)
        pv, is_lt = _parse_p_value_cell(cell_v, raw)
        if pv is None or is_lt:
            continue
        if pv == 0.0:
            n_zero += 1
        else:
            n_nonzero += 1
            if "e" not in raw.lower() and pv >= 1e-5 and _count_decimals(raw) <= 5:
                n_short_fixed += 1
    if n_nonzero < 40 or n_zero == 0:
        return False
    return (n_short_fixed / float(n_nonzero) >= 0.90) and (n_zero <= max(3, int(0.02 * n_nonzero)))


def audit_table_stats(
    table: TableData,
    methods_context: dict[str, Any] | None = None,
) -> tuple[list[AuditAlert], dict[str, Any]]:
    """Run single-table Module 3 checks (domain invariants, CI axioms, P-values, paired-Z)."""
    alerts: list[AuditAlert] = []
    ctx = methods_context or {}
    clean_headers = [_clean_ws(h) for h in table.headers]
    ci_records = extract_ci_records(table)

    # Identify P-value and N columns early (shared across STAT-006, STAT-002, STAT-004, STAT-005)
    pval_cols = [j for j, h in enumerate(clean_headers) if _PVAL_COL_RE.match(h)]
    pdiff_cols = [
        j
        for j, h in enumerate(clean_headers)
        if _PDIFF_COL_RE.search(h) and not _PVAL_EXCLUDE_DOMAIN_RE.search(h)
    ]
    all_p_domain_cols = sorted(set(pval_cols) | set(pdiff_cols))
    n_col_idx = next((j for j, h in enumerate(clean_headers) if _N_COL_RE.match(h)), None)
    delta_col_idx = next((j for j, h in enumerate(clean_headers) if _DELTA_COL_RE.match(h)), None)
    method_col_idx = next(
        (j for j, h in enumerate(clean_headers) if re.search(r"^(?:test|stat_?test|method|p_?method)$", h, re.I)),
        None,
    )
    folds_col_idx = next(
        (j for j, h in enumerate(clean_headers) if re.search(r"^(?:k_?folds?|n_?folds?|folds)$", h, re.I)),
        None,
    )

    # =========================================================================
    # Rule STAT-006 (ALWAYS ON even on omics sheets — P0 CRITICAL):
    #   (a) P-value / FDR strictly outside [0, 1] (e.g., negative FDR = -0.01)
    #   (b) Unresolved Excel formula error tokens (#NUM!, #REF!, #DIV/0!, #VALUE!)
    #   (c) Exact P == 0 on tiny finite sample size (2 <= N <= 12)
    # =========================================================================
    domain_p_violations: list[dict[str, Any]] = []
    formula_err_violations: list[dict[str, Any]] = []
    tiny_n_zero_p_violations: list[dict[str, Any]] = []

    for pj in all_p_domain_cols:
        if _PVAL_EXCLUDE_DOMAIN_RE.search(clean_headers[pj]):
            continue
        n_valid_unit = 0
        n_numeric = 0
        bad_rows_col: list[dict[str, Any]] = []
        for r_idx, row in enumerate(table.rows):
            if pj >= len(row):
                continue
            pv, _ = _parse_p_value_cell_unbounded(row[pj], _raw_cell(table, r_idx, pj))
            if pv is None:
                continue
            n_numeric += 1
            if -1e-9 <= pv <= 1.0 + 1e-6:
                n_valid_unit += 1
            else:
                # Ignore bare footnote superscript integers (e.g. 1..9) on sub-header rows 0-1 where all other cells are strings
                if (
                    r_idx <= 1
                    and 1.0 <= pv <= 9.0
                    and abs(pv - round(pv)) < 1e-9
                    and all(
                        (c is None or isinstance(c, str))
                        for c_i, c in enumerate(row)
                        if c_i != pj
                    )
                ):
                    n_numeric -= 1
                    continue
                if len(bad_rows_col) < 20:
                    bad_rows_col.append({"row": r_idx, "col": clean_headers[pj], "value": pv})
        # Require >= 50% of numeric values in [0, 1] so non-probability columns named "p" never FP
        if bad_rows_col and n_numeric >= 2 and (n_valid_unit / float(n_numeric)) >= 0.50:
            domain_p_violations.extend(bad_rows_col)

    # Scan statistical inference & evaluation columns for Excel formula errors (skip raw omics intensity cols)
    stat_or_num_cols: list[int] = []
    for j, h in enumerate(clean_headers):
        if _RAW_INTENSITY_COL_RE.search(h):
            continue
        if (
            j in all_p_domain_cols
            or _LO_COL_RE.search(h)
            or _HI_COL_RE.search(h)
            or _UNIT_BOUNDED_METRIC_RE.search(h)
            or _RATIO_EST_COL_RE.match(h)
            or _BETA_COL_RE.search(h)
            or _SE_COL_RE.search(h)
            or _Z_COL_RE.search(h)
            or _T_COL_RE.search(h)
            or re.search(r"^(?:mean|median|score|ci|95%?\s*ci)$", h, re.I)
        ):
            stat_or_num_cols.append(j)

    for j in stat_or_num_cols:
        for r_idx, row in enumerate(table.rows):
            if j >= len(row):
                continue
            cell_v = row[j]
            if not isinstance(cell_v, str):
                continue
            if "#" not in cell_v:
                continue
            raw_tok = cell_v.strip().upper()
            if raw_tok in _EXCEL_FORMULA_ERRORS:
                formula_err_violations.append({"row": r_idx, "col": clean_headers[j], "token": raw_tok})
                if len(formula_err_violations) >= 20:
                    break
        if len(formula_err_violations) >= 20:
            break

    # Exact P == 0 on tiny sample size (2 <= N <= 12); skip GSEA/pathway enrichment where N is gene-set size
    is_gsea_pathway_n = any(_GSEA_PATHWAY_HDR_RE.match(h) for h in clean_headers)
    if (n_col_idx is not None or ctx.get("n") is not None) and not is_gsea_pathway_n:
        for r_idx, row in enumerate(table.rows):
            row_n = _parse_n_cell(row[n_col_idx]) if (n_col_idx is not None and n_col_idx < len(row)) else ctx.get("n")
            if row_n is None or not (2.0 <= row_n <= 12.0):
                continue
            for pj in all_p_domain_cols:
                if pj >= len(row):
                    continue
                pv, is_lt = _parse_p_value_cell(row[pj], _raw_cell(table, r_idx, pj))
                if pv == 0.0 and not is_lt:
                    tiny_n_zero_p_violations.append(
                        {"row": r_idx, "col": clean_headers[pj], "n": row_n, "p_reported": 0.0}
                    )
                    if len(tiny_n_zero_p_violations) >= 20:
                        break
            if len(tiny_n_zero_p_violations) >= 20:
                break

    if domain_p_violations:
        ex = domain_p_violations[0]
        alerts.append(
            AuditAlert(
                module="statcheck",
                rule_id="STAT-006",
                severity="CRITICAL",
                priority="P0",
                title="Out-of-domain P-value / FDR outside [0, 1]",
                message=(
                    f"Found {len(domain_p_violations)} probability cell(s) outside [0, 1] in '{table.name}' "
                    f"(e.g., row {ex['row']} col '{ex['col']}' = {ex['value']:g})."
                ),
                file=table.source_file,
                location=f"sheet '{table.name}' col '{ex['col']}'",
                evidence={"violations": domain_p_violations[:10]},
            )
        )

    if formula_err_violations:
        ex = formula_err_violations[0]
        alerts.append(
            AuditAlert(
                module="statcheck",
                rule_id="STAT-006",
                severity="CRITICAL",
                priority="P0",
                title="Unresolved Excel formula error token in statistical table",
                message=(
                    f"Found {len(formula_err_violations)} unresolved spreadsheet error token(s) in '{table.name}' "
                    f"(e.g., row {ex['row']} col '{ex['col']}' = '{ex['token']}')."
                ),
                file=table.source_file,
                location=f"sheet '{table.name}' col '{ex['col']}'",
                evidence={"violations": formula_err_violations[:10]},
            )
        )

    if tiny_n_zero_p_violations:
        ex = tiny_n_zero_p_violations[0]
        alerts.append(
            AuditAlert(
                module="statcheck",
                rule_id="STAT-006",
                severity="CRITICAL",
                priority="P0",
                title="Exact zero P-value on tiny finite sample size (N <= 12)",
                message=(
                    f"Found {len(tiny_n_zero_p_violations)} exact P=0 cell(s) with N={ex['n']:g} in '{table.name}' "
                    f"(col '{ex['col']}'), mathematically impossible for finite-sample inference."
                ),
                file=table.source_file,
                location=f"sheet '{table.name}' col '{ex['col']}'",
                evidence={"violations": tiny_n_zero_p_violations[:10]},
            )
        )

    # =========================================================================
    # Rule STAT-001 & STAT-002: Confidence Interval Axioms
    # =========================================================================
    by_metric: dict[str, list[CIRecord]] = {}
    for rec in ci_records:
        by_metric.setdefault(rec.metric_name, []).append(rec)

    is_curve_sheet = (
        table.n_rows >= 20
        and (
            bool(_CURVE_SHEET_RE.search(table.sheet_base_name or table.name))
            or sum(1 for h in clean_headers if _CURVE_HEADER_RE.match(h)) >= 2
        )
    )
    table_has_pval = len(pval_cols) > 0

    zero_width_interior: list[CIRecord] = []
    zero_width_boundary: list[CIRecord] = []
    invalid_order_rows: list[int] = []
    est_outside_violations: list[dict[str, Any]] = []
    out_of_domain_ci_violations: list[dict[str, Any]] = []

    for metric_name, recs in by_metric.items():
        # Pre-check Strategy-1 column scale alignment to guard against unrelated adjacent numeric columns
        strat1_scale_mismatch = False
        s1_with_est = [r for r in recs if r.source_strategy == 1 and r.estimate is not None and r.hi >= r.lo - 1e-9]
        if len(s1_with_est) >= 4:
            pos_w = [r for r in s1_with_est if r.width > 1e-9]
            if len(pos_w) >= 4:
                n_out = sum(
                    1
                    for r in pos_w
                    if r.estimate < r.lo - max(1e-4, 0.05 * r.width)
                    or r.estimate > r.hi + max(1e-4, 0.05 * r.width)
                )
                med_est = float(np.median([abs(r.estimate) for r in pos_w]))  # type: ignore[arg-type]
                med_mid = float(np.median([abs(0.5 * (r.lo + r.hi)) for r in pos_w]))
                if (n_out / float(len(pos_w)) >= 0.80) and (
                    (med_mid > 1e-6 and (med_est > 2.5 * med_mid or med_est < 0.4 * med_mid))
                    or (med_mid <= 1e-6 and med_est > 1.0)
                ):
                    strat1_scale_mismatch = True

        # Infer whether this metric column is on [0, 1] or [0, 100]% scale
        metric_ests = [r.estimate for r in recs if r.estimate is not None]
        max_est_col = max(metric_ests) if metric_ests else 0.0
        col_is_pct = (max_est_col > 1.5 and max_est_col <= 100.5) and (
            any(r.is_percent_scale for r in recs) or max_est_col > 10.0
        )
        col_is_diff = bool(_DIFF_OR_SHIFT_NAME_RE.search(metric_name))
        col_is_ratio = bool(_RATIO_METRIC_RE.search(metric_name))
        col_is_r2 = bool(re.search(r"\b(?:r2|r_?squared)\b", metric_name, re.I))
        col_is_nonneg_unit = bool(_NON_NEGATIVE_UNIT_METRIC_RE.search(metric_name)) and not col_is_diff and not col_is_ratio
        col_has_explicit_ci = bool(_EXPLICIT_CI_COL_TOKEN_RE.search(metric_name))

        for rec in recs:
            # Detect unit-scale mix where estimate is in [0, 1] while CI is in [0, 100]% (e.g., PPV=1.0, CI=(96.5-100))
            is_mixed_unit_pct = (
                rec.estimate is not None
                and 0.0 <= rec.estimate <= 1.001
                and rec.lo >= 0.0
                and rec.hi <= 100.0 + 1e-6
                and rec.hi > 1.5
                and (rec.lo - 0.5 <= 100.0 * rec.estimate <= rec.hi + 0.5)
            )
            # 1A: Inverted CI bounds (hi < lo)
            if rec.hi < rec.lo - 1e-9:
                invalid_order_rows.append(rec.row_index)
            else:
                # 1B: Point estimate strictly outside its own [lo, hi] CI
                if rec.estimate is not None and not rec.is_pm_format and not is_mixed_unit_pct:
                    if not (rec.source_strategy == 1 and strat1_scale_mismatch and rec.width > 1e-9):
                        step = (10.0 ** (-rec.decimals)) if rec.decimals <= 6 else 1e-6
                        base_tol = max(1e-4, 0.55 * step)
                        # Require meaningful margin beyond rounding ULP to avoid 1-ULP bootstrap ties
                        if rec.width <= 1e-9:
                            gap = max(rec.lo - rec.estimate, rec.estimate - rec.hi)
                            if gap > max(1e-3, 1.1 * step):
                                est_outside_violations.append(
                                    {
                                        "row": rec.row_index,
                                        "metric": rec.metric_name,
                                        "estimate": rec.estimate,
                                        "lo": rec.lo,
                                        "hi": rec.hi,
                                        "gap": round(gap, 6),
                                    }
                                )
                        else:
                            gap = max(rec.lo - rec.estimate, rec.estimate - rec.hi)
                            min_gap = max(base_tol, 0.0015 if (not col_is_pct and abs(rec.estimate) <= 1.5) else 0.01)
                            if gap > min_gap and (gap > 0.02 * rec.width or gap > 0.005):
                                est_outside_violations.append(
                                    {
                                        "row": rec.row_index,
                                        "metric": rec.metric_name,
                                        "estimate": rec.estimate,
                                        "lo": rec.lo,
                                        "hi": rec.hi,
                                        "gap": round(gap, 6),
                                    }
                                )

            # 1C: Out-of-domain [0, 1] or [0, 100]% CI bounds on explicitly bounded metrics
            # Skip unlabelled mean ± SD (where mean+SD > 1.0 or mean-SD < 0 is mathematically normal) and mixed [0,1]/[0,100]% scales
            if (
                (col_is_nonneg_unit or (rec.is_unit_bounded and not col_is_diff and not col_is_ratio))
                and not is_mixed_unit_pct
                and not (rec.is_pm_format and not col_has_explicit_ci)
            ):
                if not col_is_pct and max_est_col <= 1.05:
                    if rec.hi > 1.0 + 1e-3 or (not col_is_r2 and rec.lo < -1e-3):
                        out_of_domain_ci_violations.append(
                            {
                                "row": rec.row_index,
                                "metric": rec.metric_name,
                                "estimate": rec.estimate,
                                "lo": rec.lo,
                                "hi": rec.hi,
                                "domain": "[0, 1]",
                            }
                        )
                elif col_is_pct and max_est_col <= 100.5:
                    if rec.hi > 100.0 + 1e-2 or (not col_is_r2 and rec.lo < -1e-2):
                        out_of_domain_ci_violations.append(
                            {
                                "row": rec.row_index,
                                "metric": rec.metric_name,
                                "estimate": rec.estimate,
                                "lo": rec.lo,
                                "hi": rec.hi,
                                "domain": "[0, 100]%",
                            }
                        )

            # 2: Zero-width CIs (lo == hi) with targeted legitimate-case filtering
            if abs(rec.width) <= 1e-12:
                if _is_legitimate_zero_width(rec, is_curve_sheet=is_curve_sheet, table_has_pval_col=table_has_pval):
                    continue
                if rec.estimate is not None and (
                    abs(rec.estimate) <= 1e-6
                    or abs(rec.estimate - 1.0) <= 1e-6
                    or abs(rec.estimate - 100.0) <= 1e-6
                ) and abs(rec.lo - rec.estimate) <= 1e-6:
                    zero_width_boundary.append(rec)
                else:
                    zero_width_interior.append(rec)

    # Emit STAT-001 (P0 CRITICAL hard CI axiom violations)
    if invalid_order_rows:
        alerts.append(
            AuditAlert(
                module="statcheck",
                rule_id="STAT-001",
                severity="CRITICAL",
                priority="P0",
                title="Inverted confidence interval bounds (hi < lo)",
                message=f"Found {len(invalid_order_rows)} row(s) with upper CI bound strictly less than lower bound.",
                file=table.source_file,
                location=f"sheet '{table.name}'",
                evidence={"rows": invalid_order_rows[:10]},
            )
        )

    if est_outside_violations:
        ex = est_outside_violations[0]
        alerts.append(
            AuditAlert(
                module="statcheck",
                rule_id="STAT-001",
                severity="CRITICAL",
                priority="P0",
                title="Point estimate strictly outside its own confidence interval",
                message=(
                    f"Found {len(est_outside_violations)} row(s) in '{table.name}' where the point estimate lies "
                    f"strictly outside [lo, hi] (e.g., '{ex['metric']}' = {ex['estimate']} outside "
                    f"[{ex['lo']}, {ex['hi']}])."
                ),
                file=table.source_file,
                location=f"sheet '{table.name}' col '{ex['metric']}'",
                evidence={"violations": est_outside_violations[:10]},
            )
        )

    if out_of_domain_ci_violations:
        ex = out_of_domain_ci_violations[0]
        alerts.append(
            AuditAlert(
                module="statcheck",
                rule_id="STAT-001",
                severity="CRITICAL",
                priority="P0",
                title="Confidence interval bound exceeds mathematical metric domain",
                message=(
                    f"Found {len(out_of_domain_ci_violations)} row(s) in '{table.name}' where bounded metric "
                    f"'{ex['metric']}' has CI [{ex['lo']}, {ex['hi']}] outside valid {ex['domain']} domain."
                ),
                file=table.source_file,
                location=f"sheet '{table.name}' col '{ex['metric']}'",
                evidence={"violations": out_of_domain_ci_violations[:10]},
            )
        )

    # Emit STAT-002 (Zero-width CIs)
    if len(zero_width_interior) >= 1 or (len(zero_width_interior) + len(zero_width_boundary)) >= 2:
        n_tot = len(zero_width_interior) + len(zero_width_boundary)
        sev = "CRITICAL" if len(zero_width_interior) >= 1 else "HIGH"
        sample_rec = zero_width_interior[0] if zero_width_interior else zero_width_boundary[0]
        zw_prio: Priority = (
            "P0"
            if (
                len(zero_width_interior) >= 1
                and (
                    sample_rec.decimals >= 3
                    or (sample_rec.estimate is not None and abs(sample_rec.estimate - sample_rec.lo) > 1e-4)
                )
            )
            else "P1"
        )
        alerts.append(
            AuditAlert(
                module="statcheck",
                rule_id="STAT-002",
                severity=sev,
                priority=zw_prio,
                title="Zero-width confidence interval (lo == hi) on finite-sample metric",
                message=(
                    f"Detected {n_tot} zero-width CI(s) ({len(zero_width_interior)} interior) in "
                    f"'{table.name}' (e.g., metric '{sample_rec.metric_name}' = {sample_rec.estimate}, "
                    f"CI [{sample_rec.lo}, {sample_rec.hi}])."
                ),
                file=table.source_file,
                location=f"sheet '{table.name}'",
                evidence={
                    "n_zero_width_interior": len(zero_width_interior),
                    "n_zero_width_boundary": len(zero_width_boundary),
                    "example_rows": [r.row_index for r in (zero_width_interior + zero_width_boundary)[:10]],
                },
            )
        )

    # Rule STAT-001 (Fixed-width / zero-variance placeholder CIs -> P1)
    has_std_reg_table = any(_BETA_COL_RE.match(h) or _SE_COL_RE.match(h) for h in clean_headers)
    for metric_name, recs in by_metric.items():
        pos_recs = [r for r in recs if r.width > 1e-12]
        if len(pos_recs) < 5:
            continue
        med_dec = int(np.median([r.decimals for r in pos_recs]))
        round_dp = max(med_dec + 1, 4) if med_dec >= 3 else 3
        hw_vals = [round(r.half_width, round_dp) for r in pos_recs]
        counts: dict[float, int] = {}
        for hw in hw_vals:
            counts[hw] = counts.get(hw, 0) + 1
        mode_hw, mode_cnt = max(counts.items(), key=lambda kv: kv[1])
        mode_frac = mode_cnt / float(len(pos_recs))

        ests = [r.estimate for r in pos_recs if r.estimate is not None]
        est_span = (max(ests) - min(ests)) if len(ests) >= 2 else 0.0
        n_uniq_est = len({round(e, 4) for e in ests})

        ns = [r.n_sample for r in pos_recs if r.n_sample is not None and r.n_sample > 0]
        n_ratio = (max(ns) / min(ns)) if len(ns) >= 2 else 1.0

        sym_flags = [
            abs((r.hi - r.estimate) - (r.estimate - r.lo)) <= 1e-6
            for r in pos_recs
            if r.estimate is not None and round(r.half_width, round_dp) == mode_hw
        ]
        all_symmetric = bool(sym_flags) and all(sym_flags)
        any_pm = any(r.is_pm_format for r in pos_recs)

        # Guard: z-scored standardized fixed-N regression table at <= 2 decimals legitimately shares SE
        if (
            med_dec <= 2
            and all_symmetric
            and n_ratio <= 1.05
            and has_std_reg_table
            and not any(r.is_unit_bounded for r in pos_recs)
        ):
            continue

        flagged_fixed = False
        if med_dec >= 3 or any_pm:
            if mode_frac >= 0.80 and len(pos_recs) >= 5 and (
                (n_uniq_est >= 4 and est_span >= 0.03)
                or n_ratio >= 1.5
                or len(pos_recs) >= 8
            ):
                flagged_fixed = True
        else:
            if mode_frac >= 0.85 and len(pos_recs) >= 8 and (
                n_ratio >= 2.0
                or (est_span >= 0.12 and n_uniq_est >= 5 and all_symmetric)
            ):
                flagged_fixed = True

        if flagged_fixed:
            sev = "CRITICAL" if (mode_frac >= 0.95 and len(pos_recs) >= 6) or n_ratio >= 2.0 else "HIGH"
            alerts.append(
                AuditAlert(
                    module="statcheck",
                    rule_id="STAT-001",
                    severity=sev,
                    priority="P1",
                    title="Fixed-width / zero-variance placeholder confidence intervals",
                    message=(
                        f"{mode_cnt}/{len(pos_recs)} ({mode_frac:.0%}) rows for '{metric_name}' in "
                        f"'{table.name}' share the exact same CI half-width (±{mode_hw:g}) despite "
                        f"estimate span={est_span:.3f} and cohort N ratio={n_ratio:.2f}x."
                    ),
                    file=table.source_file,
                    location=f"sheet '{table.name}' col '{metric_name}'",
                    evidence={
                        "metric": metric_name,
                        "modal_half_width": mode_hw,
                        "modal_count": mode_cnt,
                        "total_rows": len(pos_recs),
                        "modal_fraction": round(mode_frac, 4),
                        "median_decimals": med_dec,
                        "estimate_span": round(est_span, 5),
                        "n_ratio": round(n_ratio, 3),
                    },
                )
            )

    # =========================================================================
    # Rule STAT-004 & STAT-005: P-value Forensics & Paired-Z
    # =========================================================================
    has_assoc_reg_headers = any(_ASSOC_REG_COL_RE.match(h) for h in clean_headers)
    has_strict_ml_perf_headers = any(_ML_PERF_HEADER_RE.search(h) for h in clean_headers)
    has_ml_discrim_headers = has_strict_ml_perf_headers or any(
        re.search(r"\b(?:wilcoxon|delong|mcnemar|model|algorithm)\b", h, re.I) for h in clean_headers
    )
    has_ml_perf_rows = any(
        any(isinstance(c, str) and len(c) <= 28 and _STRICT_ML_ROW_LABEL_RE.match(_clean_ws(c)) for c in row[:4])
        for row in table.rows[:25]
    )
    is_biomarker_assoc_table = has_assoc_reg_headers and not (
        has_strict_ml_perf_headers
        or has_ml_perf_rows
        or any(re.search(r"\b(?:wilcoxon|delong|mcnemar)\b", h, re.I) for h in clean_headers)
    )
    has_perf_context = (not is_biomarker_assoc_table) and (
        has_ml_discrim_headers
        or has_ml_perf_rows
        or (delta_col_idx is not None)
        or bool(ctx.get("test") or ctx.get("n_folds"))
        or (bool(ci_records) and all(r.estimate is None or (0.0 <= r.estimate <= 1.0) for r in ci_records[:20]))
    )

    # Determine if table is an exploratory omics/pathway/scRNA sheet for P2 routing of generic STAT-004
    has_strict_ml_context = (
        has_strict_ml_perf_headers
        or has_ml_perf_rows
        or bool(ctx.get("test") or ctx.get("n_folds"))
        or (
            any(re.search(r"^(?:model|method|algorithm)$", h, re.I) for h in clean_headers)
            and any(re.search(r"^(?:metric|c|auc|score|cohort|se)$", h, re.I) for h in clean_headers)
        )
    )
    is_omics_exploratory = _is_omics_exploratory_table(table, has_ml_perf_context=has_strict_ml_context)

    # Algebraic self-consistency column indices (allow prefixed columns like 'slope1 SE', 'slope1 t')
    se_col = next((j for j, h in enumerate(clean_headers) if _SE_COL_RE.search(h)), None)
    beta_col = next((j for j, h in enumerate(clean_headers) if _BETA_COL_RE.search(h)), None)
    ratio_col = next((j for j, h in enumerate(clean_headers) if _RATIO_EST_COL_RE.match(h)), None)
    z_col = next((j for j, h in enumerate(clean_headers) if _Z_COL_RE.search(h)), None)
    t_col = next((j for j, h in enumerate(clean_headers) if _T_COL_RE.search(h)), None)
    df_col = next((j for j, h in enumerate(clean_headers) if _DF_COL_RE.match(h)), None)
    chi2_col = next((j for j, h in enumerate(clean_headers) if _CHI2_COL_RE.match(h)), None)
    corr_col = next((j for j, h in enumerate(clean_headers) if _CORR_COL_RE.match(h)), None)

    fixed_dec_zero_p_cols = {pj for pj in pval_cols if _column_is_fixed_decimal_rounded_p(table, pj)}

    ci_by_row: dict[int, list[CIRecord]] = {}
    for rec in ci_records:
        ci_by_row.setdefault(rec.row_index, []).append(rec)

    # 4D: Nested-model / Predictor-OR vs Model-Comparison P_diff contradiction (P0 CRITICAL)
    pred_p_only_cols = [j for j in pval_cols if j not in set(pdiff_cols)]
    has_single_marker_col = any(
        re.search(r"(?:marker|biomarker|feature|variable|predictor|protein|gene|metabolite)", h, re.I)
        for h in clean_headers
    )
    has_reg_or_auc_col = any(
        _RATIO_EST_COL_RE.match(h) or _BETA_COL_RE.match(h) or _UNIT_BOUNDED_METRIC_RE.search(h)
        for h in clean_headers
    )
    nested_contradictions: list[dict[str, Any]] = []
    if pdiff_cols and pred_p_only_cols and has_single_marker_col and has_reg_or_auc_col:
        for r_idx, row in enumerate(table.rows):
            row_n = _parse_n_cell(row[n_col_idx]) if (n_col_idx is not None and n_col_idx < len(row)) else ctx.get("n")
            row_txt = " ".join(str(c) for c in row[:4] if isinstance(c, str)).lower()
            is_comp_row = bool(re.search(r"(?:minimal|diff|comp|vs\b|extended|model\s*2)", row_txt))
            for dj in pdiff_cols:
                if dj >= len(row):
                    continue
                p_diff, is_lt_d = _parse_p_value_cell(row[dj], _raw_cell(table, r_idx, dj))
                if p_diff is None:
                    continue
                eff_pd = p_diff * 0.5 if is_lt_d else p_diff
                if eff_pd >= 1e-6:
                    continue
                # Gather candidate predictor P-values from same row and (if Minimal/Comparison row) row above
                cand_preds: list[float] = []
                for pj in pred_p_only_cols:
                    if pj < len(row):
                        pp, _ = _parse_p_value_cell(row[pj], _raw_cell(table, r_idx, pj))
                        if pp is not None and pp > 0.0:
                            cand_preds.append(pp)
                    if is_comp_row and r_idx > 0 and pj < len(table.rows[r_idx - 1]):
                        pp_prev, _ = _parse_p_value_cell(
                            table.rows[r_idx - 1][pj], _raw_cell(table, r_idx - 1, pj)
                        )
                        if pp_prev is not None and pp_prev > 0.0:
                            cand_preds.append(pp_prev)
                if not cand_preds:
                    continue
                p_best = min(cand_preds)
                contradict = False
                if p_best > 0.05 and eff_pd < 1e-6:
                    contradict = True
                elif (
                    p_best > 1e-12
                    and eff_pd < 1e-20
                    and (row_n is None or row_n <= 3000)
                    and (eff_pd == 0.0 or math.log10(eff_pd) < math.log10(p_best) - 12.0)
                ):
                    contradict = True
                if contradict:
                    nested_contradictions.append(
                        {
                            "row": r_idx,
                            "p_predictor": p_best,
                            "p_model_diff": p_diff,
                            "col_diff": clean_headers[dj],
                            "n": row_n,
                        }
                    )

    wilcoxon_violations: list[dict[str, Any]] = []
    pseudorep_violations: list[dict[str, Any]] = []
    paired_z_violations: list[dict[str, Any]] = []

    for r_idx, row in enumerate(table.rows):
        row_n = _parse_n_cell(row[n_col_idx]) if (n_col_idx is not None and n_col_idx < len(row)) else ctx.get("n")
        row_delta = (
            abs(_to_float_or_none(row[delta_col_idx]))  # type: ignore[arg-type]
            if (delta_col_idx is not None and delta_col_idx < len(row) and _to_float_or_none(row[delta_col_idx]) is not None)
            else None
        )
        row_method = (
            str(row[method_col_idx]).lower()
            if (method_col_idx is not None and method_col_idx < len(row) and row[method_col_idx] is not None)
            else str(ctx.get("test", "")).lower()
        )
        row_folds_val = (
            _to_float_or_none(row[folds_col_idx])
            if (folds_col_idx is not None and folds_col_idx < len(row))
            else ctx.get("n_folds")
        )
        row_folds = int(row_folds_val) if row_folds_val is not None and row_folds_val > 0 else None

        if row_folds is None:
            m_kf = re.search(r"(\d+)\s*[- ]?fold", f"{row_method} {table.name}", re.I)
            if m_kf:
                row_folds = int(m_kf.group(1))
        is_wilcoxon = "wilcoxon" in f"{row_method} {table.name} {' '.join(clean_headers)}".lower()

        row_cis_all = ci_by_row.get(r_idx, [])

        for pj in pval_cols:
            if pj >= len(row):
                continue
            p_val, is_lt = _parse_p_value_cell(row[pj], _raw_cell(table, r_idx, pj))
            if p_val is None:
                continue
            eff_p = p_val * 0.5 if is_lt else p_val

            # 4A: Exact K-fold paired Wilcoxon minimum P-value violation (ALWAYS P0 CRITICAL)
            if is_wilcoxon and row_folds is not None and 2 <= row_folds <= 12:
                min_one_sided = 2.0 ** (-row_folds)
                if eff_p < min_one_sided - 1e-9:
                    wilcoxon_violations.append(
                        {
                            "row": r_idx,
                            "col": clean_headers[pj],
                            "p_reported": p_val,
                            "n_folds": row_folds,
                            "min_possible_one_sided": min_one_sided,
                            "min_possible_two_sided": 2.0 * min_one_sided,
                        }
                    )
                    continue

            if is_biomarker_assoc_table or not has_perf_context:
                continue

            # 4B: Bootstrap pseudoreplication / impossible microscopic P-value
            if eff_p < 1e-10:
                # Skip fixed-decimal rounding truncation P=0.00000 in large rounded tables
                skip_4b = (eff_p == 0.0 and pj in fixed_dec_zero_p_cols) or _algebraic_p_consistent(
                    row=row,
                    eff_p=eff_p,
                    row_n=row_n,
                    se_col=se_col,
                    beta_col=beta_col,
                    ratio_col=ratio_col,
                    z_col=z_col,
                    t_col=t_col,
                    df_col=df_col,
                    chi2_col=chi2_col,
                    corr_col=corr_col,
                    row_cis=row_cis_all,
                )
                if not skip_4b:
                    if row_n is not None and row_delta is not None and 0.0 < row_delta < 1.0:
                        net_disc = max(1, int(round(row_delta * row_n)))
                        p_min_mc, log10_p_mc = min_mcnemar_p_for_net_diff(net_disc)
                        if net_disc < 35 or (eff_p > 0 and math.log10(eff_p) < log10_p_mc - 2.0) or eff_p == 0.0:
                            pseudorep_violations.append(
                                {
                                    "row": r_idx,
                                    "p_reported": p_val,
                                    "n": row_n,
                                    "delta": row_delta,
                                    "net_discordant_approx": net_disc,
                                    "min_possible_mcnemar_p": p_min_mc,
                                }
                            )
                    elif row_n is not None and row_n <= 2000 and (row_delta is None or row_delta < 0.05):
                        pseudorep_violations.append(
                            {
                                "row": r_idx,
                                "p_reported": p_val,
                                "n": row_n,
                                "delta": row_delta,
                                "reason": "microscopic_p_on_modest_N",
                            }
                        )
                    elif row_n is None and row_delta is not None and row_delta < 0.02:
                        pseudorep_violations.append(
                            {
                                "row": r_idx,
                                "p_reported": p_val,
                                "delta": row_delta,
                                "reason": "microscopic_p_on_tiny_delta",
                            }
                        )
                    elif row_n is None and row_delta is None and has_perf_context and eff_p < 1e-15:
                        pseudorep_violations.append(
                            {
                                "row": r_idx,
                                "p_reported": p_val,
                                "reason": "extreme_microscopic_p_in_metric_table",
                            }
                        )

            # 4C / STAT-005: Paired-Z sensitivity check against two comparable performance CIs in same row
            row_cis = [r for r in row_cis_all if r.width > 1e-9 and r.estimate is not None]
            if len(row_cis) >= 2:
                c1, c2 = row_cis[0], row_cis[1]
                if (c1.is_unit_bounded and c2.is_unit_bounded) or (pj in set(pdiff_cols)):
                    se1 = c1.half_width / _Z_95
                    se2 = c2.half_width / _Z_95
                    d_val = row_delta if row_delta is not None else abs(c1.estimate - c2.estimate)  # type: ignore[operator]
                    pz = paired_z_sensitivity_check(d_val, se1=se1, se2=se2, p_reported=eff_p)
                    if pz["contradicts_reported_p"]:
                        paired_z_violations.append(
                            {
                                "row": r_idx,
                                "p_reported": p_val,
                                "delta": round(d_val, 6),
                                "se1": round(se1, 6),
                                "se2": round(se2, 6),
                                "p_rho_095": pz["sensitivity_grid"]["rho=0.95"]["p_value"],
                            }
                        )

    if wilcoxon_violations:
        ex = wilcoxon_violations[0]
        alerts.append(
            AuditAlert(
                module="statcheck",
                rule_id="STAT-004",
                severity="CRITICAL",
                priority="P0",
                title="Mathematically impossible K-fold paired Wilcoxon P-value",
                message=(
                    f"{len(wilcoxon_violations)} row(s) report P-values below the exact theoretical "
                    f"minimum for {ex['n_folds']}-fold paired Wilcoxon (reported P={ex['p_reported']:g} < "
                    f"min one-sided {ex['min_possible_one_sided']:g}, two-sided {ex['min_possible_two_sided']:g})."
                ),
                file=table.source_file,
                location=f"sheet '{table.name}'",
                evidence={"violations": wilcoxon_violations[:10]},
            )
        )

    if nested_contradictions:
        ex = nested_contradictions[0]
        alerts.append(
            AuditAlert(
                module="statcheck",
                rule_id="STAT-004",
                severity="CRITICAL",
                priority="P0",
                title="Predictor P-value severely contradicts nested-model comparison P-value",
                message=(
                    f"{len(nested_contradictions)} row(s) in '{table.name}' report single-predictor "
                    f"P={ex['p_predictor']:g} alongside nested-model '{ex['col_diff']}' P={ex['p_model_diff']:g} "
                    "(incompatible with nested Likelihood Ratio / DeLong inference)."
                ),
                file=table.source_file,
                location=f"sheet '{table.name}' col '{ex['col_diff']}'",
                evidence={"violations": nested_contradictions[:10]},
            )
        )

    if pseudorep_violations:
        ex = pseudorep_violations[0]
        if is_omics_exploratory:
            sev = "MEDIUM"
            prio: Priority = "P2"
        else:
            sev = "CRITICAL" if ("min_possible_mcnemar_p" in ex or ex["p_reported"] < 1e-15) else "HIGH"
            prio = "P1"
        alerts.append(
            AuditAlert(
                module="statcheck",
                rule_id="STAT-004",
                severity=sev,
                priority=prio,
                title=(
                    "Exploratory omics/enrichment microscopic P-value (P2 triage)"
                    if is_omics_exploratory
                    else "Bootstrap-pseudoreplication / implausibly microscopic P-value alert"
                ),
                message=(
                    f"{len(pseudorep_violations)} row(s) in '{table.name}' report extreme P-values "
                    f"(e.g., P={ex['p_reported']:g}) incompatible with sample size/effect size "
                    + (
                        "(exploratory omics/enrichment table routed to P2)."
                        if is_omics_exploratory
                        else "(classic signature of running paired tests across bootstrap resamples)."
                    )
                ),
                file=table.source_file,
                location=f"sheet '{table.name}'",
                evidence={"violations": pseudorep_violations[:10], "omics_triage": is_omics_exploratory},
            )
        )

    if paired_z_violations:
        ex = paired_z_violations[0]
        alerts.append(
            AuditAlert(
                module="statcheck",
                rule_id="STAT-005",
                severity="CRITICAL",
                priority="P0",
                title="Reported P-value severely contradicts reported confidence intervals (paired-Z check)",
                message=(
                    f"{len(paired_z_violations)} row(s) in '{table.name}' report P={ex['p_reported']:g} "
                    f"for |delta|={ex['delta']:.4f}, whereas even assuming inter-model correlation rho=0.95 "
                    f"the paired-Z P-value is {ex['p_rho_095']:.4g}."
                ),
                file=table.source_file,
                location=f"sheet '{table.name}'",
                evidence={"violations": paired_z_violations[:10]},
            )
        )

    diagnostics = {
        "table_name": table.name,
        "n_ci_records": len(ci_records),
        "n_zero_width_interior": len(zero_width_interior),
        "n_est_outside_ci": len(est_outside_violations),
        "n_out_of_domain_ci": len(out_of_domain_ci_violations),
        "n_domain_p_violations": len(domain_p_violations),
        "n_formula_err_violations": len(formula_err_violations),
        "n_tiny_n_zero_p_violations": len(tiny_n_zero_p_violations),
        "n_nested_contradictions": len(nested_contradictions),
        "n_wilcoxon_violations": len(wilcoxon_violations),
        "n_pseudorep_violations": len(pseudorep_violations),
        "n_paired_z_violations": len(paired_z_violations),
        "is_omics_exploratory": is_omics_exploratory,
    }
    return alerts, diagnostics


def _tables_semantically_mismatched(meta_a: str, meta_b: str) -> bool:
    """Return True if two table names/captions clearly refer to distinct horizons, splits, sexes, endpoints, or assays."""
    # 1. Distinct explicit time horizons (e.g., 3y vs 8y, 30d vs 90d)
    ha = {(int(m.group(1)), m.group(2)[0].lower()) for m in _HORIZON_RE.finditer(meta_a)}
    hb = {(int(m.group(1)), m.group(2)[0].lower()) for m in _HORIZON_RE.finditer(meta_b)}
    if ha and hb and ha.isdisjoint(hb):
        return True

    # 2. Disjoint domain tokens across cohort split, sex, survival endpoint, or biological assay
    for group in (_COHORT_SPLIT_TOKENS, _SEX_TOKENS, _ENDPOINT_TOKENS, _ASSAY_TOKENS):
        ta = {k for k, rx in group.items() if rx.search(meta_a)}
        tb = {k for k, rx in group.items() if rx.search(meta_b)}
        if ta and tb and ta.isdisjoint(tb):
            return True
    return False


def audit_cross_table_ci_consistency(
    tables: list[TableData],
    ratio_threshold: float = 5.0,
) -> tuple[list[AuditAlert], dict[str, Any]]:
    """Detect cross-sheet / cross-table CI width contradictions (>5x or sqrt(B) SE-of-bootstrap bug)."""
    all_recs: list[CIRecord] = []
    table_meta: dict[tuple[str | None, str], str] = {}
    for tbl in tables:
        cap = " ".join(tbl.caption_lines[:1])
        table_meta[(tbl.source_file, tbl.name)] = f"{tbl.sheet_base_name or tbl.name} {cap}"
        all_recs.extend(extract_ci_records(tbl))

    # Group positive-width records by normalized (row_key, metric_name), skipping positional fallback keys
    buckets: dict[tuple[str, str], dict[tuple[str | None, str], list[CIRecord]]] = {}
    for rec in all_recs:
        if rec.width <= 1e-12:
            continue
        if rec.row_key.startswith("row_"):
            continue
        k = (rec.row_key, _normalize_key(rec.metric_name))
        tk = (rec.source_file, rec.table_name)
        buckets.setdefault(k, {}).setdefault(tk, []).append(rec)

    contradictions: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str, str, str]] = set()

    for (rkey, mkey), by_tbl in buckets.items():
        if len(by_tbl) < 2:
            continue
        tbl_keys = list(by_tbl.keys())
        for i in range(len(tbl_keys)):
            for j in range(i + 1, len(tbl_keys)):
                tk_a, tk_b = tbl_keys[i], tbl_keys[j]
                recs_a = by_tbl[tk_a]
                recs_b = by_tbl[tk_b]
                base_a = recs_a[0].sheet_base_name or recs_a[0].table_name
                base_b = recs_b[0].sheet_base_name or recs_b[0].table_name
                # Never cross-compare stacked sub-blocks from the exact same worksheet
                if tk_a[0] == tk_b[0] and base_a == base_b:
                    continue
                if _tables_semantically_mismatched(
                    table_meta.get(tk_a, tk_a[1]),
                    table_meta.get(tk_b, tk_b[1]),
                ):
                    continue

                # Subsample very long identical-key curve tables to keep O(1) runtime
                samp_a = recs_a[:: max(1, len(recs_a) // 40)][:40]
                samp_b = recs_b[:: max(1, len(recs_b) // 40)][:40]

                pair_id = tuple(sorted([f"{tk_a[1]}:{rkey}:{mkey}", f"{tk_b[1]}:{rkey}:{mkey}"]))
                sig = (pair_id[0], pair_id[1], rkey, mkey)
                if sig in seen_pairs:
                    continue

                best_match: tuple[float, CIRecord, CIRecord] | None = None
                for a in samp_a:
                    for b in samp_b:
                        if a.estimate is not None and b.estimate is not None:
                            if abs(a.estimate - b.estimate) > 0.015:
                                continue
                        if (
                            a.n_sample is not None
                            and b.n_sample is not None
                            and min(a.n_sample, b.n_sample) > 0
                            and max(a.n_sample, b.n_sample) / min(a.n_sample, b.n_sample) >= 2.0
                        ):
                            continue
                        w_lo = min(a.width, b.width)
                        w_hi = max(a.width, b.width)
                        ratio = w_hi / max(w_lo, 1e-12)
                        if ratio <= ratio_threshold:
                            continue
                        if best_match is None or ratio > best_match[0]:
                            best_match = (ratio, a, b)

                if best_match is None:
                    continue
                seen_pairs.add(sig)
                ratio, a, b = best_match

                matched_b: int | None = None
                best_rel_err = 1.0
                for b_cand in _BOOTSTRAP_CANDIDATES:
                    sq_b = math.sqrt(b_cand)
                    rel_err = abs(ratio - sq_b) / sq_b
                    if rel_err <= 0.18 and rel_err < best_rel_err:
                        best_rel_err = rel_err
                        matched_b = b_cand

                contradictions.append(
                    {
                        "row_key": rkey,
                        "metric": mkey,
                        "table_a": a.table_name,
                        "table_b": b.table_name,
                        "width_a": round(a.width, 6),
                        "width_b": round(b.width, 6),
                        "width_ratio": round(ratio, 3),
                        "estimate_a": a.estimate,
                        "estimate_b": b.estimate,
                        "matched_sqrt_B": matched_b,
                    }
                )

    alerts: list[AuditAlert] = []
    if contradictions:
        contradictions.sort(key=lambda d: -d["width_ratio"])
        top = contradictions[0]
        any_sqrt_b = any(c["matched_sqrt_B"] is not None for c in contradictions)
        sev = "CRITICAL" if (top["width_ratio"] >= 10.0 or any_sqrt_b) else "HIGH"
        sqrt_note = (
            f" (ratio {top['width_ratio']:.1f}x matches sqrt({top['matched_sqrt_B']})={math.sqrt(top['matched_sqrt_B']):.1f}x, "
            "diagnostic of erroneous SE-of-bootstrap-mean division by sqrt(B))"
            if top["matched_sqrt_B"] is not None
            else ""
        )
        src_file = tables[0].source_file if tables else None
        alerts.append(
            AuditAlert(
                module="statcheck",
                rule_id="STAT-003",
                severity=sev,
                priority="P1",
                title="Cross-sheet/cross-table confidence interval width contradiction",
                message=(
                    f"Matched task '{top['row_key']}' ({top['metric']}) exhibits a {top['width_ratio']:.1f}x "
                    f"CI width contradiction between '{top['table_a']}' (width={top['width_a']:.5f}) and "
                    f"'{top['table_b']}' (width={top['width_b']:.5f}){sqrt_note}."
                ),
                file=src_file,
                location=f"'{top['table_a']}' vs '{top['table_b']}'",
                evidence={
                    "n_contradictions": len(contradictions),
                    "max_ratio": top["width_ratio"],
                    "contradictions": contradictions[:10],
                },
            )
        )

    return alerts, {"n_contradictions": len(contradictions), "contradictions": contradictions}


def audit_tables_suite(
    tables: list[TableData],
    methods_context: dict[str, Any] | None = None,
) -> tuple[list[AuditAlert], dict[str, Any]]:
    """Run all Module 3 single-table and cross-table statistical checks on a list of tables."""
    alerts: list[AuditAlert] = []
    per_table_diag: list[dict[str, Any]] = []
    for tbl in tables:
        t_alerts, t_diag = audit_table_stats(tbl, methods_context=methods_context)
        alerts.extend(t_alerts)
        per_table_diag.append(t_diag)

    cross_diag: dict[str, Any] = {"n_contradictions": 0}
    if len(tables) >= 2:
        c_alerts, cross_diag = audit_cross_table_ci_consistency(tables)
        alerts.extend(c_alerts)

    return alerts, {"per_table": per_table_diag, "cross_table": cross_diag}


def audit_stat_file(
    path: str | Path,
    methods_context: dict[str, Any] | None = None,
) -> tuple[list[AuditAlert], dict[str, Any]]:
    """Load an .xlsx/.csv/.tsv file and run all Module 3 statistical sanity checks."""
    tables = read_tables(path)
    return audit_tables_suite(tables, methods_context=methods_context)
