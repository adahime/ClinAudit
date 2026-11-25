"""clinaudit: Lightweight zero-heavy-dependency forensic auditor for clinical AI studies."""

from .deploycheck import (
    DEFAULT_CLINICAL_DIRECTIONS,
    audit_linear_coefficients,
    audit_risk_calculator,
    scan_deploy_code_document,
    scan_deploy_code_file,
)
from .evalcheck import (
    audit_dca_table,
    check_dca_curve,
    scan_code_document,
    scan_code_file,
)
from .io_utils import (
    CodeDocument,
    TableData,
    extract_numeric_matrix,
    read_code_file,
    read_delimited,
    read_tables,
    read_xlsx,
    write_xlsx,
)
from .leakage import (
    audit_split_leakage,
    audit_tabular_split_files,
    audit_tabular_split_tables,
    audit_xlsx_splits,
    confusion_matrix_unseen_bounds,
    exact_binomial_upper_tail_log10,
)
from .models import AuditAlert, AuditReport
from .scanner import VERSION as __version__, audit_directory, audit_files
from .statcheck import (
    CIRecord,
    audit_cross_table_ci_consistency,
    audit_stat_file,
    audit_table_stats,
    audit_tables_suite,
    exact_binomial_test,
    exact_mcnemar_test,
    extract_ci_records,
    min_mcnemar_p_for_net_diff,
    paired_z_sensitivity_check,
)

__all__ = [
    "__version__",
    "AuditAlert",
    "AuditReport",
    "CIRecord",
    "CodeDocument",
    "DEFAULT_CLINICAL_DIRECTIONS",
    "TableData",
    "audit_cross_table_ci_consistency",
    "audit_dca_table",
    "audit_directory",
    "audit_files",
    "audit_linear_coefficients",
    "audit_risk_calculator",
    "audit_split_leakage",
    "audit_stat_file",
    "audit_table_stats",
    "audit_tables_suite",
    "audit_tabular_split_files",
    "audit_tabular_split_tables",
    "audit_xlsx_splits",
    "check_dca_curve",
    "confusion_matrix_unseen_bounds",
    "exact_binomial_test",
    "exact_binomial_upper_tail_log10",
    "exact_mcnemar_test",
    "extract_ci_records",
    "extract_numeric_matrix",
    "min_mcnemar_p_for_net_diff",
    "paired_z_sensitivity_check",
    "read_code_file",
    "read_delimited",
    "read_tables",
    "read_xlsx",
    "scan_code_document",
    "scan_code_file",
    "scan_deploy_code_document",
    "scan_deploy_code_file",
    "write_xlsx",
]
