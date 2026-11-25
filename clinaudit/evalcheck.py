"""Module 2: Clinical Evaluation & Baseline Integrity Scanner.

Static AST/regex analyzer for .py, .ipynb, and .R clinical ML pipelines plus
tabular Decision Curve Analysis (DCA) mathematical sanity auditor.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re
from typing import Any

import numpy as np

from .io_utils import CodeDocument, TableData, read_code_file
from .models import AuditAlert

# Established continuous/ordinal clinical risk score identifiers
_CLINICAL_SCORE_RE = re.compile(
    r"(?:cha2ds2(?:_?vasc)?|chads2?|chadsvasc|meld(?:_?na)?|sofa|qsofa|"
    r"apache(?:_?ii|_?2)?|framingham|grace|timi|wells|curb_?65|has_?bled|"
    r"ascvd|score2|euroscore|child_?pugh|albi|fib_?4|apri|news2?|mews|"
    r"nihss|gcs|ranson|bisap|glasgow|rockall|aims65|smart_?cop|abcd2|"
    r"heart_score|s?pesi|caprini|padua|charlson|cci_score|"
    r"clinical_?score|risk_?score|baseline_?score|prognostic_?score)",
    re.IGNORECASE,
)

# Discrimination & probabilistic calibration functions that require continuous scores/probs
_CONTINUOUS_METRIC_FUNCS = {
    "roc_auc_score",
    "roc_curve",
    "average_precision_score",
    "precision_recall_curve",
    "brier_score_loss",
    "calibration_curve",
    "log_loss",
    "delong_roc_test",
    "delong_test",
    "delong",
    "concordance_index",
    "c_index",
    "expected_calibration_error",
    "ece_score",
    "from_predictions",
}

_PROB_CALIBRATION_FUNCS = {
    "brier_score_loss",
    "calibration_curve",
    "log_loss",
    "expected_calibration_error",
    "ece_score",
    "from_predictions",
}

_SKLEARN_CLASSIFIERS = {
    "LogisticRegression",
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "GradientBoostingClassifier",
    "HistGradientBoostingClassifier",
    "AdaBoostClassifier",
    "BaggingClassifier",
    "DecisionTreeClassifier",
    "SVC",
    "LinearSVC",
    "KNeighborsClassifier",
    "MLPClassifier",
    "GaussianNB",
    "MultinomialNB",
    "BernoulliNB",
    "XGBClassifier",
    "LGBMClassifier",
    "CatBoostClassifier",
    "CalibratedClassifierCV",
    "VotingClassifier",
    "StackingClassifier",
    "GridSearchCV",
    "RandomizedSearchCV",
    "Pipeline",
    "ImbPipeline",
}

_NON_SKLEARN_PROB_PREDICTORS = {
    "Logit",
    "GLM",
    "Probit",
    "MNLogit",
    "Booster",
    "Sequential",
    "Model",
    "Functional",
}

_LEAKY_PREPROCESSORS = {
    "SMOTE",
    "ADASYN",
    "BorderlineSMOTE",
    "SVMSMOTE",
    "SMOTENC",
    "SMOTEN",
    "SMOTETomek",
    "SMOTEENN",
    "RandomOverSampler",
    "RandomUnderSampler",
    "TomekLinks",
    "EditedNearestNeighbours",
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "MaxAbsScaler",
    "QuantileTransformer",
    "PowerTransformer",
    "Normalizer",
    "SimpleImputer",
    "KNNImputer",
    "IterativeImputer",
    "PCA",
    "TruncatedSVD",
    "SelectKBest",
    "SelectPercentile",
    "SelectFromModel",
    "RFE",
    "RFECV",
    "LassoCV",
    "VarianceThreshold",
    "KBinsDiscretizer",
}

_SPLIT_FUNCS = {
    "train_test_split",
    "StratifiedKFold",
    "KFold",
    "GroupKFold",
    "StratifiedGroupKFold",
    "RepeatedStratifiedKFold",
    "RepeatedKFold",
    "ShuffleSplit",
    "StratifiedShuffleSplit",
    "cross_val_score",
    "cross_validate",
    "cross_val_predict",
}

_SURVIVAL_FUNCS = {
    "KaplanMeierFitter",
    "CoxPHFitter",
    "logrank_test",
    "multivariate_logrank_test",
    "pairwise_logrank_test",
    "kaplan_meier_estimator",
    "CoxPHSurvivalAnalysis",
    "RandomSurvivalForest",
}

_FUTURE_EXPOSURE_RE = re.compile(
    r"(?:^|_)(?:post_?(?:op|discharge|baseline|index|diagnosis|admission|treatment|tx)|"
    r"after_?(?:baseline|diagnosis|discharge|index|admission)|"
    r"ever_?treated|received_?(?:chemo|statin|treatment|tx|therapy|drug|surgery|transplant|adjuvant)|"
    r"adjuvant_?(?:chemo|therapy|tx|treatment)?|subsequent_?|follow_?up_?(?:tx|treatment|exposure|drug|visit)|"
    r"(?<!have_)(?<!has_)(?<!is_)(?<!use_)future_?(?:tx|treatment|exposure|drug|use|noac|doac|medication|therapy)|"
    r"noac_?\d+[ym]|doac_?\d+[ym]|statin_?\d+[ym]|"
    r"completed_?(?:cycles|chemo|tx|treatment|therapy)|"
    r"responded|responder(?:_status)?|achieved_?(?:remission|response|cr|pr)|"
    r"eventual_?|any_?(?:chemo|statin|tx|treatment)|"
    r"length_?of_?stay|hospital_?stay|icu_?stay|readmission|rehospitalization|"
    r"cumulative_?dose|total_?cycles|cycles_?completed|days_?on_?treatment|"
    r"treatment_?duration|time_?to_?treatment)(?:$|_)",
    re.IGNORECASE,
)

_TEST_VAR_RE = re.compile(r"(?:^|_)(?:x_?test|y_?test|test_?(?:x|y|df|data|set|loader|dl|idx|auc|acc|f1|loss|score|prob|pred|preds)|dtest)(?:$|_)", re.IGNORECASE)
_BEST_VAR_RE = re.compile(r"(?:^|_)(?:best_?(?:score|auc|acc|accuracy|loss|f1|metric|model|params|weights|ckpt|checkpoint|epoch|val)|opt_?params)(?:$|_)", re.IGNORECASE)


def _get_call_name(node: ast.Call) -> str:
    fn = node.func
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return ""


def _get_full_attr_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _get_full_attr_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _collect_names_and_strings(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            names.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            names.add(sub.attr)
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            names.add(sub.value)
    return names


def _is_binarizing_expr(node: ast.AST) -> bool:
    """Return True if an AST expression binarizes a continuous score/probability."""
    if isinstance(node, ast.Compare):
        return True
    if isinstance(node, ast.Call):
        cname = _get_call_name(node)
        if cname in ("astype", "int", "round", "rint", "argmax", "predict_classes"):
            if cname == "astype" and isinstance(node.func, ast.Attribute):
                if _is_binarizing_expr(node.func.value):
                    return True
                # e.g. arr.astype(int) on comparison or bool
                for arg in node.args:
                    if isinstance(arg, ast.Name) and arg.id in ("int", "bool"):
                        if isinstance(node.func.value, (ast.Compare, ast.BoolOp)):
                            return True
            if cname in ("round", "rint", "argmax", "predict_classes"):
                return True
        if cname == "where" and len(node.args) >= 3:
            # np.where(cond, 1, 0)
            c1 = node.args[1]
            c2 = node.args[2]
            if (
                isinstance(c1, ast.Constant)
                and isinstance(c2, ast.Constant)
                and {c1.value, c2.value} <= {0, 1, 0.0, 1.0, True, False}
            ):
                return True
        if cname == "cut":
            for kw in node.keywords:
                if kw.arg == "labels" and isinstance(kw.value, (ast.List, ast.Tuple)):
                    if len(kw.value.elts) == 2:
                        return True
        if cname in ("fit_transform", "transform") and isinstance(node.func, ast.Attribute):
            recv = _get_full_attr_name(node.func.value)
            if "binarizer" in recv.lower():
                return True
        if cname == "apply" and node.args and isinstance(node.args[0], ast.Lambda):
            body = node.args[0].body
            if isinstance(body, (ast.IfExp, ast.Compare)):
                return True
        # e.g. np.array([1 if x >= 2 else 0 for x in score])
        if node.args and any(_is_binarizing_expr(a) for a in node.args):
            if cname in ("array", "asarray", "Series", "int64", "int32", "float64"):
                return True
    if isinstance(node, ast.ListComp) and isinstance(node.elt, (ast.IfExp, ast.Compare)):
        return True
    if isinstance(node, ast.IfExp) and isinstance(node.test, ast.Compare):
        if (
            isinstance(node.body, ast.Constant)
            and isinstance(node.orelse, ast.Constant)
            and {node.body.value, node.orelse.value} <= {0, 1, 0.0, 1.0, True, False}
        ):
            return True
    return False


def _expr_references_clinical_score(node: ast.AST, clinical_vars: set[str]) -> str | None:
    for token in _collect_names_and_strings(node):
        if token in clinical_vars or _CLINICAL_SCORE_RE.search(token):
            return token
    return None


def _is_inside_pipeline_call(parent_map: dict[ast.AST, ast.AST], node: ast.AST) -> bool:
    cur = parent_map.get(node)
    while cur is not None:
        if isinstance(cur, ast.Call):
            cname = _get_call_name(cur)
            if cname in ("Pipeline", "make_pipeline", "ImbPipeline", "ColumnTransformer", "FeatureUnion"):
                return True
        cur = parent_map.get(cur)
    return False


def _scan_python_ast(doc: CodeDocument) -> list[AuditAlert]:
    """Static AST audit of Python source code for all 6 Module 2 evaluation defects."""
    try:
        tree = ast.parse(doc.source)
    except SyntaxError:
        return _scan_regex_fallback(doc)

    parent_map: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[child] = parent

    alerts: list[AuditAlert] = []
    seen_keys: set[tuple[str, int]] = set()

    def _emit(
        rule_id: str,
        severity: str,
        title: str,
        message: str,
        lineno: int,
        evidence: dict[str, Any],
    ) -> None:
        key = (rule_id, lineno)
        if key in seen_keys:
            return
        seen_keys.add(key)
        alerts.append(
            AuditAlert(
                module="evalcheck",
                rule_id=rule_id,
                severity=severity,  # type: ignore[arg-type]
                title=title,
                message=message,
                file=doc.path,
                location=doc.format_location(lineno),
                evidence=evidence,
            )
        )

    # Pass 1: Track variable provenance across module/functions
    clinical_score_vars: set[str] = set()
    binarized_score_vars: dict[str, tuple[str, int]] = {}  # var -> (score_name, line)
    sklearn_clf_vars: set[str] = set()
    non_sklearn_prob_vars: set[str] = set()
    binary_pred_vars: dict[str, tuple[str, int]] = {}  # var -> (reason, line)
    preprocessor_vars: dict[str, str] = {}  # var -> class name
    pre_split_transformed_vars: dict[str, tuple[str, int]] = {}  # var -> (transformer, line)
    has_time_varying_or_landmark = bool(
        re.search(
            r"\b(?:CoxTimeVaryingFitter|landmark\w*|tmerge|time_varying|counting_process)\b",
            doc.source,
            re.IGNORECASE,
        )
    )
    has_survival_context = False

    # Inspect imports/calls for survival context and class bindings
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if any(k in node.module for k in ("lifelines", "sksurv")):
                has_survival_context = True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if any(k in alias.name for k in ("lifelines", "sksurv")):
                    has_survival_context = True
        elif isinstance(node, ast.Call):
            cname = _get_call_name(node)
            if cname in _SURVIVAL_FUNCS:
                has_survival_context = True

    def _inspect_block(stmts: list[ast.stmt]) -> None:
        split_seen_line: int | None = None
        local_pre_transformed: dict[str, tuple[str, int]] = dict(pre_split_transformed_vars)

        for stmt in stmts:
            # Recurse into function definitions independently
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check (b): Net-benefit function hardcoding a fixed scalar decision threshold
                if re.search(r"(?:net_?benefit|^nb_|_dca\b|dca_)", stmt.name, re.I):
                    arg_names = {a.arg.lower() for a in stmt.args.args}
                    takes_thresh = any(re.search(r"thresh|p_?t|cutoff|prob_?t", a) for a in arg_names)
                    if not takes_thresh:
                        for b_stmt in ast.walk(stmt):
                            if isinstance(b_stmt, ast.Assign) and isinstance(b_stmt.value, ast.Constant) and isinstance(b_stmt.value.value, (int, float)):
                                v = float(b_stmt.value.value)
                                if 0.0 < v < 1.0:
                                    for bt in b_stmt.targets:
                                        if isinstance(bt, ast.Name) and re.search(r"^(?:thresh(?:old)?|p_?t|cutoff)$", bt.id, re.I):
                                            _emit(
                                                "EVAL-002",
                                                "CRITICAL",
                                                "Hardcoded constant threshold in Decision Curve Analysis function",
                                                f"Function `{stmt.name}()` hardcodes constant `{bt.id} = {v}` instead of varying threshold probability across the decision curve.",
                                                getattr(b_stmt, "lineno", stmt.lineno),
                                                {"function": stmt.name, "threshold_var": bt.id, "hardcoded_value": v},
                                            )
                _inspect_block(stmt.body)
                continue
            if isinstance(stmt, ast.ClassDef):
                _inspect_block(stmt.body)
                continue

            lineno = getattr(stmt, "lineno", 1)

            # Check assignments
            if isinstance(stmt, ast.Assign):
                target_names: list[str] = []
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        target_names.append(t.id)
                    elif isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str):
                        target_names.append(t.slice.value)
                    elif isinstance(t, (ast.Tuple, ast.List)):
                        for elt in t.elts:
                            if isinstance(elt, ast.Name):
                                target_names.append(elt.id)

                val = stmt.value
                # Track clinical score variables
                for tname in target_names:
                    if _CLINICAL_SCORE_RE.search(tname):
                        clinical_score_vars.add(tname)

                # Track classifier / preprocessor instantiation
                if isinstance(val, ast.Call):
                    cname = _get_call_name(val)
                    if cname in _SKLEARN_CLASSIFIERS:
                        for tname in target_names:
                            sklearn_clf_vars.add(tname)
                    if cname in _NON_SKLEARN_PROB_PREDICTORS:
                        for tname in target_names:
                            non_sklearn_prob_vars.add(tname)
                    if cname in _LEAKY_PREPROCESSORS and not _is_inside_pipeline_call(parent_map, val):
                        for tname in target_names:
                            preprocessor_vars[tname] = cname

                    # Check statsmodels fit: res = sm.Logit(y, X).fit()
                    if cname == "fit" and isinstance(val.func, ast.Attribute):
                        if isinstance(val.func.value, ast.Call):
                            inner = _get_call_name(val.func.value)
                            if inner in _NON_SKLEARN_PROB_PREDICTORS:
                                for tname in target_names:
                                    non_sklearn_prob_vars.add(tname)
                            elif inner in _SKLEARN_CLASSIFIERS:
                                for tname in target_names:
                                    sklearn_clf_vars.add(tname)
                        elif isinstance(val.func.value, ast.Name):
                            recv = val.func.value.id
                            if recv in non_sklearn_prob_vars:
                                for tname in target_names:
                                    non_sklearn_prob_vars.add(tname)
                            elif recv in sklearn_clf_vars:
                                for tname in target_names:
                                    sklearn_clf_vars.add(tname)

                    # Track binary predictions from sklearn clf.predict(...)
                    if cname == "predict" and isinstance(val.func, ast.Attribute):
                        recv = _get_full_attr_name(val.func.value)
                        recv_root = recv.split(".")[0]
                        if recv_root not in non_sklearn_prob_vars and (
                            recv_root in sklearn_clf_vars
                            or re.search(r"(?:clf|classifier|rf|xgb|lgb|svm|svc|knn|dt|et|gb|gbdt|lr|pipe)", recv_root, re.I)
                        ):
                            for tname in target_names:
                                binary_pred_vars[tname] = (f"{recv}.predict()", lineno)

                    # Track pre-split fit_resample / fit_transform
                    if cname in ("fit_resample", "fit_transform") and not _is_inside_pipeline_call(parent_map, val):
                        transformer_name = ""
                        if isinstance(val.func, ast.Attribute):
                            if isinstance(val.func.value, ast.Call):
                                inner = _get_call_name(val.func.value)
                                if inner in _LEAKY_PREPROCESSORS:
                                    transformer_name = inner
                            elif isinstance(val.func.value, ast.Name):
                                transformer_name = preprocessor_vars.get(val.func.value.id, "")
                        if transformer_name:
                            # Check if fitted directly on test set!
                            arg_names = _collect_names_and_strings(val.args[0]) if val.args else set()
                            if any(_TEST_VAR_RE.search(a) for a in arg_names):
                                _emit(
                                    "EVAL-004",
                                    "CRITICAL",
                                    "Preprocessor fitted directly on test set",
                                    f"`{transformer_name}.{cname}()` is called directly on test data "
                                    f"({', '.join(sorted(arg_names))}), leaking test distribution parameters.",
                                    lineno,
                                    {"transformer": transformer_name, "method": cname, "args": sorted(arg_names)},
                                )
                            elif split_seen_line is None:
                                for tname in target_names:
                                    local_pre_transformed[tname] = (transformer_name, lineno)
                                    pre_split_transformed_vars[tname] = (transformer_name, lineno)

                # Track binarized clinical score or binarized prediction
                if _is_binarizing_expr(val):
                    score_ref = _expr_references_clinical_score(val, clinical_score_vars)
                    if score_ref is not None:
                        for tname in target_names:
                            binarized_score_vars[tname] = (score_ref, lineno)
                    else:
                        for tname in target_names:
                            binary_pred_vars[tname] = ("binarized_threshold_expression", lineno)

                # Track tiled/repeated constant Net Benefit (EVAL-002)
                if any(re.search(r"(?:^|_)(?:nb|net_?benefit|dca)", t, re.I) for t in target_names):
                    if isinstance(val, ast.Call) and _get_call_name(val) in ("full", "full_like", "tile", "repeat"):
                        _emit(
                            "EVAL-002",
                            "CRITICAL",
                            "Tiled constant Net Benefit across DCA thresholds",
                            f"Net Benefit array `{target_names[0]}` is constructed via `{_get_call_name(val)}()`, "
                            "producing an invalid constant Decision Curve across thresholds.",
                            lineno,
                            {"target": target_names[0], "func": _get_call_name(val)},
                        )
                    elif isinstance(val, ast.BinOp) and isinstance(val.op, ast.Mult):
                        if isinstance(val.left, ast.List) and len(val.left.elts) == 1:
                            _emit(
                                "EVAL-002",
                                "CRITICAL",
                                "Repeated scalar Net Benefit across DCA thresholds",
                                f"Net Benefit list `{target_names[0]}` repeats a single scalar value via list multiplication.",
                                lineno,
                                {"target": target_names[0]},
                            )

            # Check standalone calls within this statement for splits, metrics, early stopping, etc.
            for sub in ast.walk(stmt):
                if not isinstance(sub, ast.Call):
                    continue
                cname = _get_call_name(sub)
                sub_line = getattr(sub, "lineno", lineno)

                # (d) Pre-split leakage check when hitting train_test_split / CV
                if cname in _SPLIT_FUNCS or (
                    cname == "split" and isinstance(sub.func, ast.Attribute)
                ):
                    if split_seen_line is None:
                        split_seen_line = sub_line
                    used_names = set()
                    for a in sub.args:
                        used_names |= _collect_names_and_strings(a)
                    for kw in sub.keywords:
                        used_names |= _collect_names_and_strings(kw.value)
                    leaked_vars = [
                        (v, local_pre_transformed[v])
                        for v in sorted(used_names)
                        if v in local_pre_transformed
                    ]
                    if leaked_vars:
                        vname, (tname, tline) = leaked_vars[0]
                        _emit(
                            "EVAL-004",
                            "CRITICAL",
                            "Pre-split oversampling/preprocessing data leakage",
                            f"`{tname}` was fitted/applied to `{vname}` at {doc.format_location(tline)} "
                            f"BEFORE `{cname}()` at {doc.format_location(sub_line)}, leaking test information into training.",
                            sub_line,
                            {
                                "transformer": tname,
                                "variable": vname,
                                "transform_line": tline,
                                "split_func": cname,
                                "split_line": sub_line,
                            },
                        )

                # (a) Binarized clinical risk score passed to continuous/rank metric
                if cname in _CONTINUOUS_METRIC_FUNCS:
                    pred_args: list[ast.AST] = list(sub.args[1:])
                    for kw in sub.keywords:
                        if kw.arg in ("y_score", "y_prob", "y_pred", "y_proba", "score", "scores", "pred"):
                            pred_args.append(kw.value)

                    for pa in pred_args:
                        # Direct inline binarization of a clinical risk score
                        if _is_binarizing_expr(pa):
                            sref = _expr_references_clinical_score(pa, clinical_score_vars)
                            if sref is not None:
                                _emit(
                                    "EVAL-001",
                                    "CRITICAL",
                                    "Binarized clinical risk score passed to discrimination/calibration metric",
                                    f"Clinical score `{sref}` is binarized inline inside `{cname}()`, artificially "
                                    "degrading baseline comparator discrimination/calibration.",
                                    sub_line,
                                    {"metric": cname, "clinical_score": sref, "inline": True},
                                )
                                continue
                        # Variable previously binarized from clinical risk score
                        if isinstance(pa, ast.Name) and pa.id in binarized_score_vars:
                            sref, bline = binarized_score_vars[pa.id]
                            _emit(
                                "EVAL-001",
                                "CRITICAL",
                                "Binarized clinical risk score passed to discrimination/calibration metric",
                                f"Variable `{pa.id}` (binarized from clinical score `{sref}` at "
                                f"{doc.format_location(bline)}) is passed to `{cname}()`.",
                                sub_line,
                                {"metric": cname, "variable": pa.id, "clinical_score": sref, "binarize_line": bline},
                            )
                            continue

                # (c) Binary {0,1} predictions passed to probabilistic calibration / Brier score
                if cname in _PROB_CALIBRATION_FUNCS:
                    pred_args = []
                    if len(sub.args) >= 2:
                        pred_args.append(sub.args[1])
                    for kw in sub.keywords:
                        if kw.arg in ("y_prob", "y_proba", "y_pred", "prob", "y_score"):
                            pred_args.append(kw.value)
                    for pa in pred_args:
                        # Skip if already caught as EVAL-001 clinical score binarization
                        if isinstance(pa, ast.Name) and pa.id in binarized_score_vars:
                            continue
                        if _is_binarizing_expr(pa):
                            if _expr_references_clinical_score(pa, clinical_score_vars) is None:
                                _emit(
                                    "EVAL-003",
                                    "HIGH",
                                    "Binary predictions passed to probabilistic calibration/Brier metric",
                                    f"Thresholded/binary expression is passed to `{cname}()`, which requires "
                                    "continuous predicted probabilities in [0, 1].",
                                    sub_line,
                                    {"metric": cname, "reason": "inline_binarized_prediction"},
                                )
                        elif isinstance(pa, ast.Call) and _get_call_name(pa) == "predict":
                            if isinstance(pa.func, ast.Attribute):
                                recv = _get_full_attr_name(pa.func.value)
                                recv_root = recv.split(".")[0]
                                if recv_root not in non_sklearn_prob_vars and (
                                    recv_root in sklearn_clf_vars
                                    or re.search(r"(?:clf|classifier|rf|xgb|lgb|svm|svc|knn|dt|et|gb|lr|pipe|model)", recv_root, re.I)
                                ):
                                    _emit(
                                        "EVAL-003",
                                        "HIGH",
                                        "Hard class `.predict()` passed to probabilistic calibration/Brier metric",
                                        f"`{recv}.predict()` outputs discrete {{0,1}} labels and is passed directly "
                                        f"into `{cname}()` instead of `.predict_proba()[:, 1]`.",
                                        sub_line,
                                        {"metric": cname, "receiver": recv},
                                    )
                        elif isinstance(pa, ast.Name) and pa.id in binary_pred_vars:
                            reason, bline = binary_pred_vars[pa.id]
                            _emit(
                                "EVAL-003",
                                "HIGH",
                                "Binary {0,1} predictions passed to probabilistic calibration/Brier metric",
                                f"Binary variable `{pa.id}` ({reason} at {doc.format_location(bline)}) "
                                f"is passed to `{cname}()` instead of continuous probabilities.",
                                sub_line,
                                {"metric": cname, "variable": pa.id, "origin": reason, "origin_line": bline},
                            )

                # (f) Early stopping / validation_data / GridSearchCV directly on test set
                for kw in sub.keywords:
                    if kw.arg in ("eval_set", "validation_data", "valid_sets", "eval_data"):
                        kw_names = _collect_names_and_strings(kw.value)
                        test_hits = sorted(n for n in kw_names if _TEST_VAR_RE.search(n))
                        if test_hits:
                            _emit(
                                "EVAL-006",
                                "CRITICAL",
                                "Test set used for early stopping / validation monitoring",
                                f"`{kw.arg}` in `{cname}()` references test-set variable(s) "
                                f"{', '.join(test_hits)}, causing test-set selection leakage.",
                                sub_line,
                                {"func": cname, "keyword": kw.arg, "test_vars": test_hits},
                            )
                if cname == "fit" and isinstance(sub.func, ast.Attribute):
                    recv = _get_full_attr_name(sub.func.value)
                    if any(k in recv.lower() for k in ("grid", "random", "search", "tuner", "optuna")):
                        arg_names = set()
                        for a in sub.args:
                            arg_names |= _collect_names_and_strings(a)
                        test_hits = sorted(n for n in arg_names if _TEST_VAR_RE.search(n))
                        if test_hits:
                            _emit(
                                "EVAL-006",
                                "CRITICAL",
                                "Hyperparameter search fitted directly on test set",
                                f"`{recv}.fit()` receives test-set variable(s) {', '.join(test_hits)}.",
                                sub_line,
                                {"receiver": recv, "test_vars": test_hits},
                            )

            # Inspect For / While loops for (b) DCA hardcoded threshold, (c) constant NRI, (f) test checkpoint selection
            if isinstance(stmt, (ast.For, ast.While)):
                _inspect_loop(stmt, binary_pred_vars, _emit, doc)
                # Also recurse into loop body for nested checks
                _inspect_block(stmt.body)
            elif isinstance(stmt, ast.If):
                _inspect_block(stmt.body)
                _inspect_block(stmt.orelse)
            elif isinstance(stmt, ast.With):
                _inspect_block(stmt.body)
            elif isinstance(stmt, ast.Try):
                _inspect_block(stmt.body)
                for h in stmt.handlers:
                    _inspect_block(h.body)

    def _inspect_loop(
        loop_node: ast.For | ast.While,
        bin_preds: dict[str, tuple[str, int]],
        emit_fn: Any,
        code_doc: CodeDocument,
    ) -> None:
        loop_line = getattr(loop_node, "lineno", 1)
        loop_names = _collect_names_and_strings(loop_node)
        loop_var: str | None = None
        if isinstance(loop_node, ast.For) and isinstance(loop_node.target, ast.Name):
            loop_var = loop_node.target.id

        # Check (b) DCA loop using fixed binary prediction or hardcoded threshold
        is_dca_loop = (
            any(re.search(r"(?:net_?benefit|dca|^nb$|nb_list|net_benefits)", n, re.I) for n in loop_names)
            and isinstance(loop_node, ast.For)
            and any(re.search(r"(?:thresh|pt|cutoff|prob_?grid|p_?t)", n, re.I) for n in _collect_names_and_strings(loop_node.iter) | ({loop_var} if loop_var else set()))
        )
        if is_dca_loop and loop_var:
            # Does the loop body compare probabilities against loop_var?
            rethresholds_with_loop_var = False
            hardcoded_thresh_compare = False
            for sub in ast.walk(loop_node):
                if isinstance(sub, ast.Compare):
                    comp_names = _collect_names_and_strings(sub)
                    if loop_var in comp_names:
                        rethresholds_with_loop_var = True
                    elif any(
                        isinstance(c, ast.Constant)
                        and isinstance(c.value, (int, float))
                        and 0.0 < float(c.value) < 1.0
                        for c in sub.comparators
                    ):
                        # Ignore comparisons like `1 - t` or `t > 0`
                        if loop_var not in comp_names:
                            hardcoded_thresh_compare = True
            uses_outer_binary = any(n in bin_preds for n in loop_names)
            if not rethresholds_with_loop_var and (uses_outer_binary or hardcoded_thresh_compare):
                emit_fn(
                    "EVAL-002",
                    "CRITICAL",
                    "Hardcoded single-threshold Decision Curve Analysis (DCA)",
                    f"DCA threshold loop over `{loop_var}` does not re-threshold predicted probabilities "
                    f"at `{loop_var}` (reuses fixed binary predictions or a hardcoded cutpoint).",
                    loop_line,
                    {
                        "loop_var": loop_var,
                        "uses_outer_binary": uses_outer_binary,
                        "hardcoded_thresh_compare": hardcoded_thresh_compare,
                    },
                )

        # Check (c) Constant-across-thresholds NRI loop
        is_nri_loop = (
            any(re.search(r"(?:^|_)nri(?:$|_|s|list)", n, re.I) for n in loop_names)
            and isinstance(loop_node, ast.For)
            and any(re.search(r"(?:thresh|cutoff|pt)", n, re.I) for n in _collect_names_and_strings(loop_node.iter) | ({loop_var} if loop_var else set()))
        )
        if is_nri_loop and loop_var:
            uses_loop_in_cmp = any(
                isinstance(sub, ast.Compare) and loop_var in _collect_names_and_strings(sub)
                for sub in ast.walk(loop_node)
            )
            uses_loop_in_call = any(
                isinstance(sub, ast.Call)
                and any(
                    isinstance(a, ast.Name) and a.id == loop_var
                    for a in list(sub.args) + [kw.value for kw in sub.keywords]
                )
                for sub in ast.walk(loop_node)
            )
            if not uses_loop_in_cmp and not uses_loop_in_call:
                emit_fn(
                    "EVAL-003",
                    "HIGH",
                    "Constant-across-thresholds Net Reclassification Index (NRI)",
                    f"NRI threshold loop over `{loop_var}` never uses `{loop_var}` to re-stratify predictions, "
                    "producing identical constant NRI across all cutoffs.",
                    loop_line,
                    {"loop_var": loop_var},
                )

        # Check (f) Test-set checkpoint / hyperparameter selection inside loop
        has_test_eval = any(_TEST_VAR_RE.search(n) for n in loop_names)
        if has_test_eval:
            for sub in ast.walk(loop_node):
                if isinstance(sub, ast.If):
                    cond_names = _collect_names_and_strings(sub.test)
                    body_names = _collect_names_and_strings(ast.Module(body=sub.body, type_ignores=[]))
                    cond_uses_test = any(_TEST_VAR_RE.search(n) for n in cond_names)
                    cond_uses_best = any(_BEST_VAR_RE.search(n) for n in cond_names)
                    cond_is_ordering_cmp = any(
                        isinstance(n, ast.Compare)
                        and any(isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for op in n.ops)
                        for n in ast.walk(sub.test)
                    )
                    body_saves = any(
                        n in ("save", "save_weights", "dump", "save_model", "deepcopy")
                        or _BEST_VAR_RE.search(n)
                        for n in body_names
                    )
                    if cond_uses_test and (cond_uses_best or (cond_is_ordering_cmp and body_saves)):
                        emit_fn(
                            "EVAL-006",
                            "CRITICAL",
                            "Test-set checkpoint or hyperparameter selection",
                            "Model checkpoint/hyperparameter selection condition inside loop directly compares "
                            "test-set performance against best metric without an independent validation split.",
                            getattr(sub, "lineno", loop_line),
                            {
                                "condition_vars": sorted(cond_names),
                                "loop_line": loop_line,
                            },
                        )

    _inspect_block(tree.body)

    # Pass 2: (e) Immortal-time bias in Python survival analysis
    if has_survival_context and not has_time_varying_or_landmark:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                cname = _get_call_name(node)
                if cname in ("fit", "logrank_test", "multivariate_logrank_test", "pairwise_logrank_test"):
                    # Check if this call is on a survival fitter or survival test
                    is_surv_call = cname != "fit"
                    if cname == "fit" and isinstance(node.func, ast.Attribute):
                        recv = _get_full_attr_name(node.func.value)
                        if re.search(r"(?:kmf?|cox|cph|surv|rsf)", recv, re.I):
                            is_surv_call = True
                    if is_surv_call:
                        # Check parent block / call arguments for future exposure variable names
                        tokens = _collect_names_and_strings(node)
                        # Also check enclosing For/If/Assign if KM fit is inside group filter
                        cur = parent_map.get(node)
                        for _ in range(3):
                            if cur is None:
                                break
                            if isinstance(cur, (ast.For, ast.If, ast.With)):
                                tokens |= _collect_names_and_strings(
                                    cur.iter if isinstance(cur, ast.For) else (cur.test if isinstance(cur, ast.If) else cur)
                                )
                            cur = parent_map.get(cur)
                        # Also scan script-level dataframe subsetting around survival calls
                        future_hits = sorted(t for t in tokens if _FUTURE_EXPOSURE_RE.search(t))
                        if future_hits:
                            _emit(
                                "EVAL-005",
                                "HIGH",
                                "Potential immortal-time bias in static survival analysis",
                                f"Static survival analysis (`{cname}()`) stratifies or adjusts on post-baseline "
                                f"exposure variable(s) {', '.join(future_hits)} without time-varying Cox or landmarking.",
                                getattr(node, "lineno", 1),
                                {"func": cname, "future_exposure_vars": future_hits},
                            )
        # Also catch DataFrame filtering on a future exposure variable right before kmf.fit()
        lines = doc.source.splitlines()
        for idx, line in enumerate(lines, start=1):
            if re.search(r"(?:kmf|cox|cph|logrank|kaplan)", line, re.I):
                window = "\n".join(lines[max(0, idx - 6) : min(len(lines), idx + 3)])
                # Extract identifiers/quoted strings in window
                cand_tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", window))
                future_hits = sorted(t for t in cand_tokens if _FUTURE_EXPOSURE_RE.search(t))
                if future_hits and not any(a.rule_id == "EVAL-005" for a in alerts):
                    _emit(
                        "EVAL-005",
                        "HIGH",
                        "Potential immortal-time bias in static survival analysis",
                        f"Survival stratification near {doc.format_location(idx)} references post-baseline "
                        f"variable(s) {', '.join(future_hits)} without time-dependent modeling or landmarking.",
                        idx,
                        {"future_exposure_vars": future_hits},
                    )

    return alerts


def _scan_r_code(doc: CodeDocument) -> list[AuditAlert]:
    """Static regex/structural audit of R clinical evaluation scripts."""
    alerts: list[AuditAlert] = []
    seen_keys: set[tuple[str, int]] = set()
    lines = doc.source.splitlines()

    def _emit(
        rule_id: str,
        severity: str,
        title: str,
        message: str,
        lineno: int,
        evidence: dict[str, Any],
    ) -> None:
        key = (rule_id, lineno)
        if key in seen_keys:
            return
        seen_keys.add(key)
        alerts.append(
            AuditAlert(
                module="evalcheck",
                rule_id=rule_id,
                severity=severity,  # type: ignore[arg-type]
                title=title,
                message=message,
                file=doc.path,
                location=doc.format_location(lineno),
                evidence=evidence,
            )
        )

    binarized_score_vars: dict[str, tuple[str, int]] = {}
    binary_pred_vars: dict[str, int] = {}
    pre_split_vars: dict[str, tuple[str, int]] = {}
    split_seen = False

    has_time_varying = bool(
        re.search(
            r"(?:Surv\s*\(\s*[^,)]+,\s*[^,)]+,\s*[^,)]+\)|tmerge\s*\(|landmark\w*|time_varying|tt\s*\()",
            doc.source,
            re.IGNORECASE,
        )
    )

    for lineno, raw_line in enumerate(lines, start=1):
        # Strip R comments
        line = raw_line.split("#", 1)[0]
        if not line.strip():
            continue

        # Track assignments: var <- expr or var = expr
        m_assign = re.match(r"^\s*([A-Za-z_.][A-Za-z0-9_.]*)\s*(?:<-|=)\s*(.+)$", line)
        if m_assign:
            lhs, rhs = m_assign.group(1), m_assign.group(2)

            # (a) Binarization of clinical risk score (require binary output literals for ifelse, excluding outlier-to-NA trimming)
            is_r_bin = (
                (re.search(r"(?:as\.(?:numeric|integer)\s*\(|cut\s*\()", rhs) and re.search(r"[<>]=?", rhs))
                or (
                    re.search(r"ifelse\s*\(", rhs)
                    and re.search(r"[<>]=?", rhs)
                    and re.search(r",\s*(?:0|1|TRUE|FALSE|T|F)\s*,\s*(?:0|1|TRUE|FALSE|T|F)\s*\)", rhs)
                )
            )
            if is_r_bin:
                m_sc = _CLINICAL_SCORE_RE.search(rhs)
                if m_sc:
                    binarized_score_vars[lhs] = (m_sc.group(0), lineno)
                else:
                    binary_pred_vars[lhs] = lineno
            elif re.search(r"predict\s*\([^)]*type\s*=\s*['\"]class['\"]", rhs):
                binary_pred_vars[lhs] = lineno

            # (b) Tiled constant Net Benefit in R
            if re.search(r"(?:^|_|\.)(?:nb|net_?benefit|dca)", lhs, re.I) and re.search(
                r"\brep\s*\(", rhs
            ):
                _emit(
                    "EVAL-002",
                    "CRITICAL",
                    "Tiled constant Net Benefit across DCA thresholds (R)",
                    f"Net Benefit vector `{lhs}` is created with `rep()`, producing a flat constant DCA curve.",
                    lineno,
                    {"target": lhs},
                )

            # (d) Pre-split oversampling / scaling in R
            if not split_seen:
                m_pre = re.search(r"\b(SMOTE|ovun\.sample|ROSE|preProcess|scale)\s*\(", rhs)
                if m_pre:
                    pre_split_vars[lhs] = (m_pre.group(1), lineno)

        # Check R split calls
        if re.search(r"\b(createDataPartition|initial_split|sample\.split|createFolds)\s*\(", line):
            split_seen = True
            for vname, (tname, tline) in pre_split_vars.items():
                if re.search(rf"\b{re.escape(vname)}\b", line):
                    _emit(
                        "EVAL-004",
                        "CRITICAL",
                        "Pre-split oversampling/preprocessing data leakage (R)",
                        f"`{tname}()` produced `{vname}` at line {tline} BEFORE data splitting at line {lineno}.",
                        lineno,
                        {"transformer": tname, "variable": vname, "transform_line": tline},
                    )

        # (a) Binarized clinical risk score passed into pROC::roc / roc.test / auc / Brier
        if re.search(r"\b(?:pROC::)?(?:roc|roc\.test|auc|val\.prob|Brier|Score)\s*\(", line):
            # Inline binarization
            if re.search(r"(?:ifelse|as\.numeric|as\.integer)\s*\([^)]*[<>]=?", line):
                m_sc = _CLINICAL_SCORE_RE.search(line)
                if m_sc:
                    _emit(
                        "EVAL-001",
                        "CRITICAL",
                        "Binarized clinical risk score passed to R ROC/calibration metric",
                        f"Clinical risk score `{m_sc.group(0)}` is binarized inline inside ROC/calibration call.",
                        lineno,
                        {"clinical_score": m_sc.group(0)},
                    )
            for vname, (sref, bline) in binarized_score_vars.items():
                if re.search(rf"\b{re.escape(vname)}\b", line):
                    _emit(
                        "EVAL-001",
                        "CRITICAL",
                        "Binarized clinical risk score passed to R ROC/calibration metric",
                        f"Binarized score `{vname}` (from `{sref}` at line {bline}) passed to ROC/calibration test.",
                        lineno,
                        {"variable": vname, "clinical_score": sref, "binarize_line": bline},
                    )

        # (c) Binary prediction passed into R Brier / val.prob / hoslem.test
        if re.search(r"\b(?:Brier|brier_score|val\.prob|hoslem\.test|calibrate)\s*\(", line):
            for vname, bline in binary_pred_vars.items():
                if re.search(rf"\b{re.escape(vname)}\b", line):
                    _emit(
                        "EVAL-003",
                        "HIGH",
                        "Binary predictions passed to R probabilistic calibration/Brier function",
                        f"Binary variable `{vname}` (line {bline}) passed to calibration/Brier evaluation.",
                        lineno,
                        {"variable": vname, "origin_line": bline},
                    )

        # (e) Immortal-time bias in R survival models
        if not has_time_varying and re.search(r"\b(?:survfit|coxph|survdiff|ggsurvplot)\s*\(", line):
            tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", line))
            future_hits = sorted(t for t in tokens if _FUTURE_EXPOSURE_RE.search(t))
            if future_hits:
                _emit(
                    "EVAL-005",
                    "HIGH",
                    "Potential immortal-time bias in R survival model",
                    f"Static survival formula uses post-baseline variable(s) {', '.join(future_hits)} "
                    "without counting-process `Surv(tstart, tstop, event)` or landmarking.",
                    lineno,
                    {"future_exposure_vars": future_hits},
                )

    return alerts


def _scan_regex_fallback(doc: CodeDocument) -> list[AuditAlert]:
    """Fallback scanner when Python code has syntax errors."""
    alerts: list[AuditAlert] = []
    for lineno, line in enumerate(doc.source.splitlines(), start=1):
        if re.search(r"roc_auc_score\s*\(.*[<>]=?", line) and _CLINICAL_SCORE_RE.search(line):
            alerts.append(
                AuditAlert(
                    module="evalcheck",
                    rule_id="EVAL-001",
                    severity="CRITICAL",
                    title="Binarized clinical risk score in roc_auc_score",
                    message="Inline binarization of a clinical risk score detected in roc_auc_score.",
                    file=doc.path,
                    location=doc.format_location(lineno),
                    evidence={"fallback": True},
                )
            )
    return alerts


def scan_code_document(doc: CodeDocument) -> list[AuditAlert]:
    """Scan a CodeDocument (.py, .ipynb, .R) for Module 2 evaluation & baseline defects."""
    if doc.language == "python":
        return _scan_python_ast(doc)
    if doc.language == "r":
        return _scan_r_code(doc)
    return []


def scan_code_file(path: str | Path) -> list[AuditAlert]:
    """Read and scan a .py, .ipynb, or .R file with Module 2."""
    doc = read_code_file(path)
    return scan_code_document(doc)


def check_dca_curve(
    thresholds: np.ndarray | list[float],
    net_benefits: np.ndarray | list[float],
    prevalence: float | None = None,
    treat_all: np.ndarray | list[float] | None = None,
    file_label: str | None = None,
    location_label: str | None = None,
) -> tuple[list[AuditAlert], dict[str, Any]]:
    """Audit a tabular Decision Curve Analysis (DCA) curve for mathematical impossibilities.

    Detects:
      - Constant / flat non-zero Net Benefit curves (tiled scalar NB bug)
      - Exact linear-in-odds single-threshold fixed-binary DCA curves
      - Impossible bounds (NB > 1, NB > prevalence, NB < -pt/(1-pt))
      - Treat-all monotonicity violations or sustained impossible upward positive NB climb
    """
    pt = np.asarray(thresholds, dtype=np.float64).ravel()
    nb = np.asarray(net_benefits, dtype=np.float64).ravel()

    if pt.size != nb.size or pt.size < 3:
        return [], {"skipped": True, "reason": "Insufficient or mismatched DCA points"}

    finite = np.isfinite(pt) & np.isfinite(nb)
    ta_arr: np.ndarray | None = None
    if treat_all is not None:
        ta_raw = np.asarray(treat_all, dtype=np.float64).ravel()
        if ta_raw.size == pt.size:
            finite = finite & np.isfinite(ta_raw)
            ta_arr = ta_raw[finite]

    pt = pt[finite]
    nb = nb[finite]
    if pt.size < 3:
        return [], {"skipped": True, "reason": "Fewer than 3 finite DCA points"}

    # Auto-normalize percentage thresholds in [0, 100] -> [0, 1]
    if float(np.max(pt)) > 1.0 and float(np.max(pt)) <= 100.0:
        pt = pt / 100.0

    # Filter strictly interior thresholds (0 < pt < 1) for odds calculation
    order = np.argsort(pt)
    pt = pt[order]
    nb = nb[order]
    if ta_arr is not None:
        ta_arr = ta_arr[order]

    interior = (pt > 1e-6) & (pt < 1.0 - 1e-6)
    pt_i = pt[interior]
    nb_i = nb[interior]
    ta_i = ta_arr[interior] if ta_arr is not None else None
    if pt_i.size < 3:
        return [], {"skipped": True, "reason": "Fewer than 3 interior thresholds in (0, 1)"}

    odds = pt_i / (1.0 - pt_i)
    span = float(pt_i[-1] - pt_i[0])
    nb_ptp = float(np.ptp(nb_i))
    nb_mean = float(np.mean(nb_i))

    # Infer prevalence from treat_all if not explicitly supplied
    inferred_prev = prevalence
    if inferred_prev is None and ta_i is not None and ta_i.size >= 2:
        # Exact algebra: NB_all(pt) = pi - (1 - pi)*w => pi = (NB_all + w) / (1 + w)
        pi_est = (ta_i + odds) / (1.0 + odds)
        if float(np.std(pi_est)) < 0.02:
            cand = float(np.median(pi_est))
            if 0.0 < cand <= 1.0:
                inferred_prev = cand

    alerts: list[AuditAlert] = []
    diagnostics: dict[str, Any] = {
        "n_points": int(pt_i.size),
        "threshold_range": [round(float(pt_i[0]), 6), round(float(pt_i[-1]), 6)],
        "nb_min": round(float(np.min(nb_i)), 6),
        "nb_max": round(float(np.max(nb_i)), 6),
        "nb_ptp": round(nb_ptp, 8),
        "inferred_prevalence": round(inferred_prev, 6) if inferred_prev is not None else None,
    }

    # Check 1: Constant / flat non-zero Net Benefit across thresholds
    n_distinct_pt = int(np.unique(np.round(pt_i, 6)).size)
    is_perfect_low_band = (
        inferred_prev is not None
        and abs(nb_mean - inferred_prev) <= 1e-3
        and float(pt_i[-1]) < 0.5
    )
    if n_distinct_pt >= 4 and span >= 0.10 and nb_ptp <= 1e-8 and abs(nb_mean) > 1e-4 and not is_perfect_low_band:
        alerts.append(
            AuditAlert(
                module="evalcheck",
                rule_id="EVAL-002",
                severity="CRITICAL",
                title="Constant/flat non-zero Net Benefit curve across DCA thresholds",
                message=(
                    f"Net Benefit is identically constant ({nb_mean:.6f}, range={nb_ptp:.2e}) "
                    f"across {n_distinct_pt} thresholds spanning [{pt_i[0]:.2f}, {pt_i[-1]:.2f}], "
                    "indicating a tiled scalar or hardcoded-threshold DCA bug."
                ),
                file=file_label,
                location=location_label,
                evidence=diagnostics,
            )
        )

    # Check 2: Exact single-threshold fixed-binary DCA line: NB(pt) = a - b * w
    # Exclude exact treat-all curve and degenerate zero curves.
    if n_distinct_pt >= 8 and span >= 0.30 and float(pt_i[-1]) >= 0.55 and nb_ptp > 1e-5:
        A = np.column_stack([np.ones_like(odds), -odds])
        coef, _, _, _ = np.linalg.lstsq(A, nb_i, rcond=None)
        a_hat, b_hat = float(coef[0]), float(coef[1])
        pred_line = a_hat - b_hat * odds
        max_resid = float(np.max(np.abs(nb_i - pred_line)))
        is_treat_all = False
        if ta_i is not None and float(np.max(np.abs(nb_i - ta_i))) < 1e-4:
            is_treat_all = True
        elif inferred_prev is not None and abs(a_hat - inferred_prev) < 2e-3 and abs(b_hat - (1.0 - inferred_prev)) < 2e-3:
            is_treat_all = True
        elif abs((a_hat + b_hat) - 1.0) < 1e-3:
            # a = TP/N, b = FP/N => a + b = 1 means 100% treated (treat-all!)
            is_treat_all = True

        diagnostics["linear_in_odds_max_resid"] = max_resid
        diagnostics["linear_in_odds_params"] = [round(a_hat, 6), round(b_hat, 6)]
        if max_resid < 1e-7 and b_hat > 1e-3 and a_hat > 1e-3 and not is_treat_all:
            alerts.append(
                AuditAlert(
                    module="evalcheck",
                    rule_id="EVAL-002",
                    severity="HIGH",
                    title="Single-threshold fixed-binary DCA curve detected",
                    message=(
                        f"Net Benefit follows an exact linear function of threshold odds "
                        f"(TP/N={a_hat:.4f}, FP/N={b_hat:.4f}, max residual={max_resid:.2e}) "
                        f"across {n_distinct_pt} thresholds, proving predictions were never re-thresholded."
                    ),
                    file=file_label,
                    location=location_label,
                    evidence=diagnostics,
                )
            )

    # Check 3: Theoretical bounds & treat-all monotonicity violations
    bound_violations: list[str] = []
    if float(np.max(nb_i)) > 1.0 + 1e-4:
        bound_violations.append(f"max(NB)={float(np.max(nb_i)):.4f} > 1.0")
    if inferred_prev is not None and float(np.max(nb_i)) > inferred_prev + 2e-3:
        bound_violations.append(
            f"max(NB)={float(np.max(nb_i)):.4f} exceeds prevalence={inferred_prev:.4f}"
        )
    floor_excess = -odds - nb_i
    if float(np.max(floor_excess)) > 2e-3:
        worst_idx = int(np.argmax(floor_excess))
        bound_violations.append(
            f"NB({pt_i[worst_idx]:.2f})={nb_i[worst_idx]:.4f} < theoretical floor {-odds[worst_idx]:.4f}"
        )

    if ta_i is not None and ta_i.size >= 2:
        ta_diffs = np.diff(ta_i)
        if float(np.max(ta_diffs)) > 1e-4:
            bound_violations.append(
                f"treat-all curve increases by +{float(np.max(ta_diffs)):.4f} as threshold increases"
            )

    # Impossible sustained positive climb across high thresholds
    if pt_i.size >= 5 and float(pt_i[-1]) >= 0.5:
        diffs = np.diff(nb_i)
        inc_frac = float(np.mean(diffs > 1e-6))
        if inc_frac >= 0.8 and (float(nb_i[-1] - nb_i[0]) > 0.08) and float(nb_i[-1]) > 0.05:
            bound_violations.append(
                f"NB climbs monotonically upward ({nb_i[0]:.4f} -> {nb_i[-1]:.4f}) into high thresholds"
            )

    if bound_violations:
        diagnostics["bound_violations"] = bound_violations
        alerts.append(
            AuditAlert(
                module="evalcheck",
                rule_id="EVAL-002",
                severity="CRITICAL",
                title="Mathematically impossible Decision Curve Analysis (DCA) values",
                message="; ".join(bound_violations) + ".",
                file=file_label,
                location=location_label,
                evidence=diagnostics,
            )
        )

    return alerts, diagnostics


def audit_dca_table(table: TableData) -> list[AuditAlert]:
    """Detect DCA columns in a TableData and run check_dca_curve on each model NB column."""
    if not table.headers or len(table.rows) < 4:
        return []
    thresh_col: int | None = None
    for j, h in enumerate(table.headers):
        if re.search(r"^(?:p_?t|thresh(?:old)?|threshold_?prob(?:ability)?|cutoff|risk_?threshold)$", h.strip(), re.I):
            thresh_col = j
            break
    if thresh_col is None:
        return []

    def _col_floats(idx: int) -> np.ndarray:
        vals = []
        for row in table.rows:
            v = row[idx] if idx < len(row) else None
            try:
                vals.append(float(v) if v is not None else np.nan)
            except (TypeError, ValueError):
                vals.append(np.nan)
        return np.asarray(vals, dtype=np.float64)

    pt = _col_floats(thresh_col)
    if np.sum(np.isfinite(pt)) < 4:
        return []

    treat_all_arr: np.ndarray | None = None
    nb_cols: list[tuple[int, str]] = []
    for j, h in enumerate(table.headers):
        if j == thresh_col:
            continue
        hl = h.strip().lower()
        if re.search(r"(?:treat_?all|all_?patients|nb_?all)", hl):
            treat_all_arr = _col_floats(j)
        elif re.search(r"(?:treat_?none|nb_?none|none)", hl):
            continue
        elif re.search(r"(?:net_?benefit|^nb\b|dca)", hl):
            nb_cols.append((j, h))

    alerts: list[AuditAlert] = []
    for j, h in nb_cols:
        nb = _col_floats(j)
        if np.sum(np.isfinite(nb)) < 4:
            continue
        col_alerts, _ = check_dca_curve(
            pt,
            nb,
            treat_all=treat_all_arr,
            file_label=table.source_file,
            location_label=f"sheet '{table.name}' col '{h}'",
        )
        alerts.extend(col_alerts)
    return alerts
