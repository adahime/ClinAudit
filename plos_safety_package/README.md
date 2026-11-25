# PLOS Medicine (`10.1371/journal.pmed.1005078`) Clinical Safety Verification Archive

**Author:** Xingyu Zhang (Independent Researcher; ORCID: `0009-0005-6181-7465`; Email: `takuru1048596@gmail.com`)  
**Scope:** Standalone, deterministic verification archive documenting two implementation issues in `CKMAIcalc::predict_risks()` and `CKMAI` (`code.R`, line 648) for Zhu et al., *"A machine learning-derived aging index for risk stratification and mortality prediction in cardiovascular-kidney-metabolic syndrome: A retrospective cohort study"*, *PLOS Medicine* (October 6, 2026; DOI: `10.1371/journal.pmed.1005078`).

## Included Files
- `verify_ckmaicalc_safety.py`: Self-contained Python verification script (Python standard library + NumPy only). Downloads the four public artifacts directly from PLOS Medicine, Zenodo, and GitHub, verifies their exact SHA-256 hashes, and deterministically reproduces both the cross-model `Sex`-encoding contradiction (`858 / 3,293 = 26.06%` female participants receiving predicted 10-year CVD mortality exceeding 10-year all-cause mortality; `2.84×` female 10-year CVD risk inflation) and the `weights <- abs(coef_vec)` sign inversion (`72.51%` loss in age+sex-adjusted LRT $\chi^2$).
- `results/ckmaicalc_safety_summary.json`: Derived summary metrics and verified official artifact SHA-256 hashes.
- `results/ckmai_feature_sign_audit.csv`: Feature-by-feature comparison of published `CKMAI_params.rds` weights versus Cox PH standardized log-hazard coefficients refitted on `S1_Data.xlsx` ($n = 6{,}896$).
- `SHA256SUMS.txt`: SHA-256 integrity manifest for all files in this archive.
- `LICENSE`: MIT License (no third-party copyrighted publisher files are redistributed).

## Reproduction

```bash
python verify_ckmaicalc_safety.py
```

Requires Python >= 3.10 and `numpy`. Official public files are fetched into a local cache directory (`--cache-dir`, default `./_official_cache_do_not_redistribute`) and verified against recorded SHA-256 digests before computing derived results.
