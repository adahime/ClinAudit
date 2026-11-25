"""Module 1: Split Integrity, Near-Exact Mirror-Clone & Confusion Matrix Bounds Detector."""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from .io_utils import TableData, extract_numeric_matrix, read_tables
from .models import AuditAlert

_TRAIN_SHEET_RE = re.compile(r"(?:^|[_\-\s])(train|training|dev|development|discovery)(?:$|[_\-\s])", re.I)
_TEST_SHEET_RE = re.compile(r"(?:^|[_\-\s])(test|testing|holdout|held_?out|external|val|validation)(?:$|[_\-\s])", re.I)


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def exact_binomial_upper_tail_log10(k: int, n: int, p: float) -> tuple[float, float]:
    """Compute exact P(Binomial(n, p) >= k) and its log10 without float64 underflow."""
    if n <= 0 or k <= 0:
        return 1.0, 0.0
    if k > n:
        return 0.0, -math.inf
    if p <= 0.0:
        return 0.0, -math.inf
    if p >= 1.0:
        return 1.0, 0.0

    log_p = math.log(p)
    log_q = math.log1p(-p)
    log_terms = [
        _log_comb(n, j) + j * log_p + (n - j) * log_q
        for j in range(k, n + 1)
    ]
    m = max(log_terms)
    log_sum = m + math.log(sum(math.exp(t - m) for t in log_terms))
    log_sum = min(0.0, log_sum)
    log10_p = log_sum / math.log(10.0)
    p_val = math.exp(log_sum) if log_sum > -745.0 else 0.0
    return min(1.0, p_val), log10_p


class _UnionFind:
    __slots__ = ("parent",)

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def _cluster_feature_rows(
    X_train: np.ndarray,
    X_test: np.ndarray,
    tol: float = 1e-10,
) -> tuple[list[dict[str, Any]], int]:
    """Group exact and near-exact (L_inf <= tol) feature rows across train and test.

    Returns (clusters, n_near_exact_merges). Each cluster dict stores:
      - train_indices: list[int]
      - test_indices: list[int]
      - near_exact: bool
    """
    n_tr = X_train.shape[0]
    n_te = X_test.shape[0]
    X_all = np.vstack([X_train, X_test]).astype(np.float64, copy=True)
    # Normalize -0.0 to +0.0 for deterministic byte hashing
    X_all[X_all == 0.0] = 0.0

    nan_mask = np.isnan(X_all)
    X_clean = np.where(nan_mask, 0.0, X_all)

    # Step 1: exact byte-level deduplication into unique representatives
    exact_map: dict[tuple[bytes, bytes], int] = {}
    rep_rows: list[np.ndarray] = []
    rep_masks: list[np.ndarray] = []
    rep_mask_keys: list[bytes] = []
    rep_train_idx: list[list[int]] = []
    rep_test_idx: list[list[int]] = []

    for i in range(n_tr + n_te):
        m_bytes = np.packbits(nan_mask[i]).tobytes()
        key = (m_bytes, X_clean[i].tobytes())
        cid = exact_map.get(key)
        if cid is None:
            cid = len(rep_rows)
            exact_map[key] = cid
            rep_rows.append(X_clean[i])
            rep_masks.append(nan_mask[i])
            rep_mask_keys.append(m_bytes)
            rep_train_idx.append([])
            rep_test_idx.append([])
        if i < n_tr:
            rep_train_idx[cid].append(i)
        else:
            rep_test_idx[cid].append(i - n_tr)

    n_unique = len(rep_rows)
    uf = _UnionFind(n_unique)
    near_merges = 0

    # Step 2: near-exact L_inf <= tol merging using 1D L1-normalized projection sweep
    if tol > 0.0 and n_unique > 1:
        # Only rows sharing the exact same NaN pattern can match
        mask_groups: dict[bytes, list[int]] = {}
        for cid, m_key in enumerate(rep_mask_keys):
            mask_groups.setdefault(m_key, []).append(cid)

        for _, cids in mask_groups.items():
            if len(cids) < 2:
                continue
            finite_cols = ~rep_masks[cids[0]]
            if not np.any(finite_cols):
                # All-NaN rows all match each other
                for other in cids[1:]:
                    if uf.find(cids[0]) != uf.find(other):
                        uf.union(cids[0], other)
                        near_merges += 1
                continue

            M = np.vstack([rep_rows[c][finite_cols] for c in cids])
            med = np.median(M, axis=0)
            Mc = M - med
            d_eff = Mc.shape[1]
            # Fixed deterministic strictly-positive L1-unit projection weights
            w = np.sin(np.arange(1, d_eff + 1, dtype=np.float64) * 1.618033988749895) + 2.0
            w /= np.sum(w)

            proj = Mc @ w
            # Safe numerical slack for float64 projection dot-product roundoff
            row_slack = tol + 1e-12 * (1.0 + (np.abs(Mc) @ w))
            order = np.argsort(proj, kind="mergesort")
            s_sorted = proj[order]
            slack_sorted = row_slack[order]
            k_len = len(order)

            for pos_i in range(k_len):
                idx_i = int(order[pos_i])
                s_i = s_sorted[pos_i]
                sl_i = slack_sorted[pos_i]
                row_i = M[idx_i]
                for pos_j in range(pos_i + 1, k_len):
                    if s_sorted[pos_j] - s_i > sl_i + slack_sorted[pos_j]:
                        break
                    idx_j = int(order[pos_j])
                    if np.all(np.abs(row_i - M[idx_j]) <= tol):
                        ca, cb = cids[idx_i], cids[idx_j]
                        if uf.find(ca) != uf.find(cb):
                            uf.union(ca, cb)
                            near_merges += 1

    # Collect final merged clusters
    grouped: dict[int, dict[str, Any]] = {}
    for cid in range(n_unique):
        root = uf.find(cid)
        entry = grouped.get(root)
        if entry is None:
            entry = {
                "train_indices": [],
                "test_indices": [],
                "member_reps": 0,
            }
            grouped[root] = entry
        entry["train_indices"].extend(rep_train_idx[cid])
        entry["test_indices"].extend(rep_test_idx[cid])
        entry["member_reps"] += 1

    clusters: list[dict[str, Any]] = []
    for entry in grouped.values():
        clusters.append(
            {
                "train_indices": entry["train_indices"],
                "test_indices": entry["test_indices"],
                "near_exact": entry["member_reps"] > 1,
            }
        )
    return clusters, near_merges


def _assess_feature_cardinality(X_all: np.ndarray, n_train: int) -> dict[str, Any]:
    """Assess whether feature space is continuous/high-cardinality vs coarse discrete."""
    n_rows, n_cols = X_all.shape
    n_continuous = 0
    log10_space = 0.0
    log_pair_collision = 0.0

    for j in range(n_cols):
        col = X_all[:, j]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            continue
        uniq, counts = np.unique(finite, return_counts=True)
        n_uniq = int(uniq.size)
        is_fractional = bool(np.any(np.abs(uniq - np.round(uniq)) > 1e-9))
        if is_fractional or n_uniq > 15:
            n_continuous += 1
        log10_space += math.log10(max(1, n_uniq))
        probs = counts.astype(np.float64) / float(finite.size)
        col_coll = float(np.sum(probs * probs))
        log_pair_collision += math.log(max(col_coll, 1e-300))

    pair_coll = math.exp(log_pair_collision) if log_pair_collision > -700.0 else 0.0
    if pair_coll <= 0.0:
        expected_hit_rate = 0.0
    elif pair_coll >= 1.0:
        expected_hit_rate = 1.0
    else:
        expected_hit_rate = -math.expm1(n_train * math.log1p(-pair_coll))

    is_low_cardinality = (
        n_continuous == 0
        and (
            log10_space < math.log10(max(n_rows, 2)) + 2.5
            or expected_hit_rate > 0.02
        )
    )
    return {
        "n_continuous_cols": n_continuous,
        "log10_state_space": round(log10_space, 3),
        "expected_random_hit_rate": round(expected_hit_rate, 6),
        "is_low_cardinality": is_low_cardinality,
    }


def confusion_matrix_unseen_bounds(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    n_leak_neg: int,
    n_leak_pos: int,
    leak_memorization_rate: float = 1.0,
) -> dict[str, Any]:
    """Derive exact model-free bounds on unseen specificity, sensitivity, and accuracy.

    Given published confusion matrix counts (TN, FP, FN, TP) on the test split and known
    counts of leaked test negatives (n_leak_neg) and leaked test positives (n_leak_pos):

      - Model-free upper bound (zero assumption on classifier accuracy on leaked rows):
          Spec_unseen <= min(TN, N_neg_unseen) / N_neg_unseen
          Sens_unseen <= min(TP, N_pos_unseen) / N_pos_unseen
      - Model-free lower bound (equivalently, exact bound under 100% memorization of
        leaked training clones, i.e., r = 1.0):
          Spec_unseen >= max(0, TN - n_leak_neg) / N_neg_unseen
          Sens_unseen >= max(0, TP - n_leak_pos) / N_pos_unseen
      - Parametric memorization-rate upper bound (assuming accuracy >= r on leaked rows):
          Spec_unseen <= clamp((TN - r * n_leak_neg) / N_neg_unseen, 0, 1)
      - Non-inferiority upper bound (assuming Spec_leak >= Spec_unseen):
          Spec_unseen <= TN / (TN + FP)
    """
    for name, val in (
        ("tn", tn),
        ("fp", fp),
        ("fn", fn),
        ("tp", tp),
        ("n_leak_neg", n_leak_neg),
        ("n_leak_pos", n_leak_pos),
    ):
        if val < 0:
            raise ValueError(f"{name} must be non-negative, got {val}")
    if not (0.0 <= leak_memorization_rate <= 1.0):
        raise ValueError(f"leak_memorization_rate must be in [0, 1], got {leak_memorization_rate}")

    n_neg = tn + fp
    n_pos = tp + fn
    if n_leak_neg > n_neg:
        raise ValueError(f"n_leak_neg ({n_leak_neg}) cannot exceed TN + FP ({n_neg})")
    if n_leak_pos > n_pos:
        raise ValueError(f"n_leak_pos ({n_leak_pos}) cannot exceed TP + FN ({n_pos})")

    unseen_neg = n_neg - n_leak_neg
    unseen_pos = n_pos - n_leak_pos
    n_total = n_neg + n_pos
    n_leak_total = n_leak_neg + n_leak_pos
    unseen_total = n_total - n_leak_total

    pub_spec = tn / n_neg if n_neg > 0 else None
    pub_sens = tp / n_pos if n_pos > 0 else None
    pub_acc = (tn + tp) / n_total if n_total > 0 else None

    def _metric_bounds(
        correct: int,
        total: int,
        leaked: int,
        unseen: int,
        pub: float | None,
    ) -> dict[str, Any]:
        if unseen == 0:
            return {
                "published": pub,
                "total": total,
                "leaked": leaked,
                "unseen": 0,
                "leak_fraction": leaked / total if total > 0 else 0.0,
                "identifiable": False,
                "model_free_upper_bound": None,
                "model_free_lower_bound": None,
                "memorization_upper_bound": None,
                "non_inferiority_upper_bound": pub,
                "max_unseen_correct": 0,
                "min_unseen_correct": 0,
            }
        max_unseen_corr = min(correct, unseen)
        min_unseen_corr = max(0, correct - leaked)
        mem_corr = min(float(unseen), max(0.0, correct - leak_memorization_rate * leaked))
        return {
            "published": pub,
            "total": total,
            "leaked": leaked,
            "unseen": unseen,
            "leak_fraction": leaked / total if total > 0 else 0.0,
            "identifiable": True,
            "model_free_upper_bound": max_unseen_corr / unseen,
            "model_free_lower_bound": min_unseen_corr / unseen,
            "memorization_upper_bound": mem_corr / unseen,
            "non_inferiority_upper_bound": pub,
            "max_unseen_correct": max_unseen_corr,
            "min_unseen_correct": min_unseen_corr,
            "max_possible_leaked_correct": min(correct, leaked),
        }

    spec_bounds = _metric_bounds(tn, n_neg, n_leak_neg, unseen_neg, pub_spec)
    sens_bounds = _metric_bounds(tp, n_pos, n_leak_pos, unseen_pos, pub_sens)
    acc_bounds = _metric_bounds(tn + tp, n_total, n_leak_total, unseen_total, pub_acc)

    return {
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "leak_memorization_rate": leak_memorization_rate,
        "specificity": spec_bounds,
        "sensitivity": sens_bounds,
        "accuracy": acc_bounds,
    }


def audit_split_leakage(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    tol: float = 1e-10,
    file_label: str | None = None,
    location_label: str | None = None,
    confusion_matrix: tuple[int, int, int, int] | None = None,
) -> tuple[list[AuditAlert], dict[str, Any]]:
    """Run Module 1 split integrity, mirror-clone parity, and label-conflict audit."""
    X_tr = np.asarray(X_train, dtype=np.float64)
    X_te = np.asarray(X_test, dtype=np.float64)
    if X_tr.ndim == 1:
        X_tr = X_tr.reshape(-1, 1)
    if X_te.ndim == 1:
        X_te = X_te.reshape(-1, 1)

    if X_tr.shape[0] == 0 or X_te.shape[0] == 0 or X_tr.shape[1] == 0:
        return [], {"skipped": True, "reason": "Empty train/test matrix"}
    if X_tr.shape[1] != X_te.shape[1]:
        return [], {
            "skipped": True,
            "reason": f"Feature dimension mismatch: train={X_tr.shape[1]}, test={X_te.shape[1]}",
        }

    n_tr = int(X_tr.shape[0])
    n_te = int(X_te.shape[0])
    n_total = n_tr + n_te

    y_tr_arr = np.asarray(y_train) if y_train is not None else None
    y_te_arr = np.asarray(y_test) if y_test is not None else None

    X_all = np.vstack([X_tr, X_te])
    card = _assess_feature_cardinality(X_all, n_tr)
    clusters, near_merges = _cluster_feature_rows(X_tr, X_te, tol=tol)

    leaked_test_indices: list[int] = []
    leaked_train_indices: list[int] = []
    shared_clusters = 0
    near_exact_shared_clusters = 0

    m2_total = 0
    m2_mirror = 0
    dup_total = 0
    dup_mirror = 0

    conflicting_profiles = 0
    cross_split_conflicts = 0
    conflicting_rows = 0

    for cl in clusters:
        k_tr = len(cl["train_indices"])
        k_te = len(cl["test_indices"])
        m = k_tr + k_te

        if k_tr > 0 and k_te > 0:
            shared_clusters += 1
            leaked_test_indices.extend(cl["test_indices"])
            leaked_train_indices.extend(cl["train_indices"])
            if cl["near_exact"]:
                near_exact_shared_clusters += 1

        if m >= 2:
            dup_total += 1
            if k_tr == k_te:
                dup_mirror += 1
            if m == 2:
                m2_total += 1
                if k_tr == 1 and k_te == 1:
                    m2_mirror += 1

        if y_tr_arr is not None or y_te_arr is not None:
            tr_lbls = (
                {
                    float(y_tr_arr[i])
                    for i in cl["train_indices"]
                    if np.isfinite(float(y_tr_arr[i]))
                }
                if (y_tr_arr is not None and k_tr > 0)
                else set()
            )
            te_lbls = (
                {
                    float(y_te_arr[i])
                    for i in cl["test_indices"]
                    if np.isfinite(float(y_te_arr[i]))
                }
                if (y_te_arr is not None and k_te > 0)
                else set()
            )
            all_lbls = tr_lbls | te_lbls
            if len(all_lbls) > 1:
                conflicting_profiles += 1
                conflicting_rows += m
                if tr_lbls and te_lbls and (tr_lbls != te_lbls or len(all_lbls) > 1):
                    cross_split_conflicts += 1

    n_test_leaked = len(leaked_test_indices)
    leak_fraction = n_test_leaked / max(n_te, 1)

    p_train = n_tr / max(n_total, 1)
    q2_null = 2.0 * p_train * (1.0 - p_train)
    p_m2, log10_p_m2 = exact_binomial_upper_tail_log10(m2_mirror, m2_total, q2_null)
    p_dup, log10_p_dup = exact_binomial_upper_tail_log10(dup_mirror, dup_total, q2_null)

    # Breakdown of leaked test negatives vs positives if y_test available
    n_leak_neg: int | None = None
    n_leak_pos: int | None = None
    cm_bounds: dict[str, Any] | None = None
    if y_te_arr is not None:
        leaked_y = y_te_arr[leaked_test_indices] if leaked_test_indices else np.array([])
        n_leak_neg = int(np.sum(leaked_y == 0))
        n_leak_pos = int(np.sum(leaked_y == 1))
        if confusion_matrix is not None:
            tn, fp, fn, tp = confusion_matrix
            cm_bounds = confusion_matrix_unseen_bounds(
                tn=tn,
                fp=fp,
                fn=fn,
                tp=tp,
                n_leak_neg=n_leak_neg,
                n_leak_pos=n_leak_pos,
            )

    diagnostics: dict[str, Any] = {
        "n_train": n_tr,
        "n_test": n_te,
        "n_features": int(X_tr.shape[1]),
        "tol": tol,
        "cardinality": card,
        "n_unique_profiles": len(clusters),
        "n_near_exact_merges": near_merges,
        "shared_profiles": shared_clusters,
        "near_exact_shared_profiles": near_exact_shared_clusters,
        "n_test_leaked": n_test_leaked,
        "test_leak_fraction": round(leak_fraction, 6),
        "n_leak_neg": n_leak_neg,
        "n_leak_pos": n_leak_pos,
        "multiplicity_parity": {
            "split_train_fraction": round(p_train, 6),
            "null_mirror_prob_q2": round(q2_null, 6),
            "m2_total": m2_total,
            "m2_mirror": m2_mirror,
            "m2_p_value": p_m2,
            "m2_log10_p": round(log10_p_m2, 4) if math.isfinite(log10_p_m2) else None,
            "dup_total": dup_total,
            "dup_mirror": dup_mirror,
            "dup_p_value": p_dup,
            "dup_log10_p": round(log10_p_dup, 4) if math.isfinite(log10_p_dup) else None,
        },
        "conflicting_profiles": conflicting_profiles,
        "cross_split_conflicts": cross_split_conflicts,
        "conflicting_rows": conflicting_rows,
        "confusion_matrix_bounds": cm_bounds,
    }

    alerts: list[AuditAlert] = []

    # Rule LEAK-001: Exact / near-exact train-test feature-vector leakage
    # Guarded against expected random collisions in coarse low-cardinality discrete tables.
    if not card["is_low_cardinality"]:
        if (n_test_leaked >= 3 and leak_fraction >= 0.02) or (
            n_test_leaked >= 10 and leak_fraction >= 0.005
        ):
            severity = "CRITICAL" if leak_fraction >= 0.15 or n_test_leaked >= 50 else "HIGH"
            near_note = (
                f" ({near_exact_shared_clusters} clusters matched within tol={tol:g})"
                if near_exact_shared_clusters > 0
                else ""
            )
            alerts.append(
                AuditAlert(
                    module="leakage",
                    rule_id="LEAK-001",
                    severity=severity,
                    title="Train-test feature-vector leakage detected",
                    message=(
                        f"{n_test_leaked}/{n_te} ({leak_fraction:.1%}) test rows match training "
                        f"feature vectors across {shared_clusters} shared profiles{near_note}."
                    ),
                    file=file_label,
                    location=location_label,
                    evidence={
                        "n_test_leaked": n_test_leaked,
                        "n_test": n_te,
                        "n_train": n_tr,
                        "test_leak_fraction": round(leak_fraction, 6),
                        "shared_profiles": shared_clusters,
                        "near_exact_shared_profiles": near_exact_shared_clusters,
                        "n_leak_neg": n_leak_neg,
                        "n_leak_pos": n_leak_pos,
                        "confusion_matrix_bounds": cm_bounds,
                    },
                )
            )

    # Rule LEAK-002: Improbable (k_train, k_test) 1:1 mirror-clone multiplicity parity
    parity_m2_flag = (
        m2_total >= 6
        and (m2_mirror / max(m2_total, 1)) > q2_null + 0.15
        and log10_p_m2 <= -4.0
    )
    parity_dup_flag = (
        dup_total >= 6
        and (dup_mirror / max(dup_total, 1)) > q2_null + 0.15
        and log10_p_dup <= -4.0
    )
    if parity_m2_flag or parity_dup_flag:
        best_log10_p = min(
            log10_p_m2 if parity_m2_flag else 0.0,
            log10_p_dup if parity_dup_flag else 0.0,
        )
        severity = "CRITICAL" if best_log10_p <= -6.0 else "HIGH"
        alerts.append(
            AuditAlert(
                module="leakage",
                rule_id="LEAK-002",
                severity=severity,
                title="Improbable 1:1 mirror-clone multiplicity parity across train/test splits",
                message=(
                    f"Exact 1:1 split symmetry observed in {m2_mirror}/{m2_total} multiplicity-2 "
                    f"profiles and {dup_mirror}/{dup_total} total duplicated profiles "
                    f"(null parity prob q2={q2_null:.3f}, log10(p)={best_log10_p:.2f})."
                ),
                file=file_label,
                location=location_label,
                evidence=diagnostics["multiplicity_parity"],
            )
        )

    # Rule LEAK-003: Conflicting-label identical continuous/high-cardinality profiles
    if not card["is_low_cardinality"] and conflicting_profiles >= 1:
        severity = "CRITICAL" if (cross_split_conflicts >= 1 or conflicting_profiles >= 3) else "HIGH"
        alerts.append(
            AuditAlert(
                module="leakage",
                rule_id="LEAK-003",
                severity=severity,
                title="Conflicting outcome labels on identical feature profiles",
                message=(
                    f"{conflicting_profiles} identical high-cardinality feature profiles "
                    f"({conflicting_rows} rows; {cross_split_conflicts} cross-split) carry "
                    f"contradictory class labels (both y=0 and y=1)."
                ),
                file=file_label,
                location=location_label,
                evidence={
                    "conflicting_profiles": conflicting_profiles,
                    "cross_split_conflicts": cross_split_conflicts,
                    "conflicting_rows": conflicting_rows,
                    "cardinality": card,
                },
            )
        )

    return alerts, diagnostics


def audit_tabular_split_tables(
    train_table: TableData,
    test_table: TableData,
    target_col: str | None = None,
    tol: float = 1e-10,
) -> tuple[list[AuditAlert], dict[str, Any]]:
    """Align shared numeric feature columns across two TableData objects and audit leakage."""
    X_tr, y_tr, f_tr, t_tr = extract_numeric_matrix(train_table, target_col=target_col)
    X_te, y_te, f_te, t_te = extract_numeric_matrix(test_table, target_col=target_col)

    if not f_tr or not f_te:
        return [], {"skipped": True, "reason": "No numeric feature columns extracted"}

    te_idx_map = {name.lower(): idx for idx, name in enumerate(f_te)}
    common_tr_idx: list[int] = []
    common_te_idx: list[int] = []
    for i, name in enumerate(f_tr):
        j = te_idx_map.get(name.lower())
        if j is not None:
            common_tr_idx.append(i)
            common_te_idx.append(j)

    if len(common_tr_idx) < 2:
        return [], {"skipped": True, "reason": "Fewer than 2 shared numeric feature columns"}

    X_tr_aligned = X_tr[:, common_tr_idx]
    X_te_aligned = X_te[:, common_te_idx]

    file_lbl = train_table.source_file or train_table.name
    loc_lbl = f"train='{train_table.name}' vs test='{test_table.name}'"
    if test_table.source_file and test_table.source_file != train_table.source_file:
        file_lbl = f"{train_table.source_file} | {test_table.source_file}"

    return audit_split_leakage(
        X_tr_aligned,
        X_te_aligned,
        y_train=y_tr,
        y_test=y_te,
        tol=tol,
        file_label=file_lbl,
        location_label=loc_lbl,
    )


def audit_tabular_split_files(
    train_path: str | Path,
    test_path: str | Path,
    target_col: str | None = None,
    tol: float = 1e-10,
) -> tuple[list[AuditAlert], dict[str, Any]]:
    """Load train and test CSV/TSV/XLSX files and run Module 1 leakage checks."""
    tr_tables = read_tables(train_path)
    te_tables = read_tables(test_path)
    if not tr_tables or not te_tables:
        return [], {"skipped": True, "reason": "Empty train or test file"}
    return audit_tabular_split_tables(tr_tables[0], te_tables[0], target_col=target_col, tol=tol)


def audit_xlsx_splits(
    xlsx_path: str | Path,
    target_col: str | None = None,
    tol: float = 1e-10,
) -> list[AuditAlert]:
    """Detect train/test sheet pairs inside a multi-sheet .xlsx file and audit leakage."""
    tables = read_tables(xlsx_path)
    if len(tables) < 2:
        return []
    train_sheets = [t for t in tables if _TRAIN_SHEET_RE.search(t.name)]
    test_sheets = [
        t for t in tables if _TEST_SHEET_RE.search(t.name) and t not in train_sheets
    ]
    alerts: list[AuditAlert] = []
    for tr in train_sheets:
        for te in test_sheets:
            pair_alerts, _ = audit_tabular_split_tables(tr, te, target_col=target_col, tol=tol)
            alerts.extend(pair_alerts)
    return alerts
