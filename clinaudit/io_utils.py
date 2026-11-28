"""Zero-dependency deterministic I/O utilities for spreadsheets, tables, and code files.

Uses only the Python standard library + NumPy (no pandas, openpyxl, or scipy).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import io
import json
from pathlib import Path
import re
from typing import Any
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape as xml_escape
import zipfile

import numpy as np

_CELL_REF_RE = re.compile(r"^([A-Za-z]+)(\d+)$")
_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")

_ID_COL_PATTERNS = re.compile(
    r"^(?:id|patient_?id|subject_?id|sample_?id|case_?id|record_?id|"
    r"row_?id|index|idx|unnamed:?\s*0|mrn|study_?id|pid|sid)$",
    re.IGNORECASE,
)

_TARGET_COL_PATTERNS = re.compile(
    r"^(?:y|label|target|outcome|class|status|event|death|mortality|"
    r"diagnosis|disease|positive|case_control|ground_?truth|y_?true)$",
    re.IGNORECASE,
)


@dataclass(slots=True)
class TableData:
    """Normalized 2D tabular structure extracted from CSV, TSV, or XLSX sheet."""

    name: str
    headers: list[str]
    rows: list[list[Any]]
    raw_rows: list[list[str]] = field(default_factory=list)
    source_file: str | None = None
    caption_lines: list[str] = field(default_factory=list)
    sheet_base_name: str | None = None

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return len(self.headers)


@dataclass(slots=True)
class CodeDocument:
    """Normalized source code extracted from .py, .R, or .ipynb files."""

    path: str
    language: str  # "python" or "r"
    source: str
    # Maps 1-based merged line number -> (0-based code cell index, 1-based line in cell)
    cell_line_map: dict[int, tuple[int, int]] = field(default_factory=dict)

    def format_location(self, line: int) -> str:
        if self.cell_line_map and line in self.cell_line_map:
            cell_idx, local_line = self.cell_line_map[line]
            return f"cell {cell_idx} line {local_line}"
        return f"line {line}"


def _local_name(tag: str) -> str:
    """Strip XML namespace URI prefix from an ElementTree tag."""
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def col_letters_to_index(letters: str) -> int:
    """Convert Excel column letters (e.g. 'A'->0, 'Z'->25, 'AA'->26) to 0-based index."""
    idx = 0
    for ch in letters.upper():
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def index_to_col_letters(idx: int) -> str:
    """Convert 0-based column index to Excel column letters."""
    if idx < 0:
        raise ValueError(f"Column index must be non-negative, got {idx}")
    letters: list[str] = []
    n = idx + 1
    while n > 0:
        n, rem = divmod(n - 1, 26)
        letters.append(chr(ord("A") + rem))
    return "".join(reversed(letters))


def parse_scalar(text: str | None) -> Any:
    """Deterministically parse a raw string cell into None, bool, int, float, or str."""
    if text is None:
        return None
    s = text.strip()
    if s == "" or s.lower() in {"na", "n/a", "nan", "null", "none", "."}:
        return None
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if _INT_RE.match(s):
        try:
            return int(s)
        except ValueError:
            pass
    if _FLOAT_RE.match(s):
        try:
            val = float(s)
            if val == val:  # not NaN
                return val
        except ValueError:
            pass
    return s


def _read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    names = set(zf.namelist())
    ss_path = None
    for candidate in ("xl/sharedStrings.xml", "xl/sharedstrings.xml"):
        if candidate in names:
            ss_path = candidate
            break
    if ss_path is None:
        return []
    root = ET.fromstring(zf.read(ss_path))
    strings: list[str] = []
    for elem in root:
        if _local_name(elem.tag) != "si":
            continue
        parts: list[str] = []
        for desc in elem.iter():
            if _local_name(desc.tag) == "t" and desc.text is not None:
                parts.append(desc.text)
        strings.append("".join(parts))
    return strings


def _resolve_sheet_targets(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
    """Return ordered list of (sheet_name, zip_member_path) from workbook XML."""
    names = set(zf.namelist())
    rels_map: dict[str, str] = {}
    if "xl/_rels/workbook.xml.rels" in names:
        rels_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        for rel in rels_root:
            if _local_name(rel.tag) != "Relationship":
                continue
            rid = rel.attrib.get("Id")
            target = rel.attrib.get("Target", "")
            if rid and target:
                norm = target.lstrip("/")
                if not norm.startswith("xl/"):
                    norm = f"xl/{norm}"
                # Normalize relative ../ segments if any
                parts: list[str] = []
                for p in norm.split("/"):
                    if p == "..":
                        if parts:
                            parts.pop()
                    elif p and p != ".":
                        parts.append(p)
                rels_map[rid] = "/".join(parts)

    sheets: list[tuple[str, str]] = []
    if "xl/workbook.xml" in names:
        wb_root = ET.fromstring(zf.read("xl/workbook.xml"))
        for elem in wb_root.iter():
            if _local_name(elem.tag) != "sheet":
                continue
            sheet_name = elem.attrib.get("name", f"Sheet{len(sheets) + 1}")
            rid = None
            for k, v in elem.attrib.items():
                if _local_name(k) == "id":
                    rid = v
                    break
            target = rels_map.get(rid or "")
            if target and target in names:
                sheets.append((sheet_name, target))
            else:
                # Fallback via sheetId or ordinal index
                sheet_id = elem.attrib.get("sheetId", str(len(sheets) + 1))
                guess = f"xl/worksheets/sheet{sheet_id}.xml"
                if guess in names:
                    sheets.append((sheet_name, guess))

    if not sheets:
        ws_files = sorted(
            n for n in names if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")
        )
        for i, ws in enumerate(ws_files, start=1):
            sheets.append((f"Sheet{i}", ws))
    return sheets


def _parse_xlsx_sheet_with_rownums(
    xml_bytes: bytes,
    shared_strings: list[str],
) -> tuple[list[list[Any]], list[list[str]], list[int]]:
    """Parse a single OOXML worksheet into rectangular typed rows, raw rows, and 1-based Excel row numbers."""
    root = ET.fromstring(xml_bytes)
    sheet_data = None
    for elem in root:
        if _local_name(elem.tag) == "sheetData":
            sheet_data = elem
            break
    if sheet_data is None:
        return [], [], []

    sparse_rows: list[tuple[int, dict[int, tuple[Any, str]]]] = []
    max_col = -1
    next_row_num = 1

    for row_elem in sheet_data:
        if _local_name(row_elem.tag) != "row":
            continue
        r_attr = row_elem.attrib.get("r")
        row_num = int(r_attr) if (r_attr and r_attr.isdigit()) else next_row_num
        next_row_num = row_num + 1

        row_cells: dict[int, tuple[Any, str]] = {}
        fallback_col = 0
        for c_elem in row_elem:
            if _local_name(c_elem.tag) != "c":
                continue
            ref = c_elem.attrib.get("r", "")
            m = _CELL_REF_RE.match(ref)
            if m:
                col_idx = col_letters_to_index(m.group(1))
            else:
                col_idx = fallback_col
            fallback_col = col_idx + 1

            ctype = c_elem.attrib.get("t", "")
            val: Any = None
            raw_str = ""

            if ctype == "inlineStr":
                parts: list[str] = []
                for desc in c_elem.iter():
                    if _local_name(desc.tag) == "t" and desc.text is not None:
                        parts.append(desc.text)
                raw_str = "".join(parts)
                val = raw_str if raw_str != "" else None
            else:
                v_elem = None
                for child in c_elem:
                    if _local_name(child.tag) == "v":
                        v_elem = child
                        break
                v_text = v_elem.text if (v_elem is not None and v_elem.text is not None) else ""
                if ctype == "s":
                    if v_text.isdigit():
                        idx = int(v_text)
                        if 0 <= idx < len(shared_strings):
                            raw_str = shared_strings[idx]
                            val = raw_str
                elif ctype == "b":
                    raw_str = "TRUE" if v_text == "1" else "FALSE"
                    val = v_text == "1"
                elif ctype in ("str", "e"):
                    raw_str = v_text
                    val = v_text if v_text != "" else None
                else:
                    # Numeric (default or t="n")
                    raw_str = v_text
                    if v_text != "":
                        if _INT_RE.match(v_text.strip()):
                            try:
                                val = int(v_text.strip())
                            except ValueError:
                                val = float(v_text.strip())
                        else:
                            try:
                                val = float(v_text.strip())
                            except ValueError:
                                val = v_text

            if val is not None or raw_str != "":
                row_cells[col_idx] = (val, raw_str)
                if col_idx > max_col:
                    max_col = col_idx

        if row_cells:
            sparse_rows.append((row_num, row_cells))

    if not sparse_rows or max_col < 0:
        return [], [], []

    width = max_col + 1
    typed_rows: list[list[Any]] = []
    raw_rows: list[list[str]] = []
    row_nums: list[int] = []
    for rnum, row_cells in sparse_rows:
        t_row = [None] * width
        r_row = [""] * width
        for c_idx, (v, r) in row_cells.items():
            t_row[c_idx] = v
            r_row[c_idx] = r
        typed_rows.append(t_row)
        raw_rows.append(r_row)
        row_nums.append(rnum)
    return typed_rows, raw_rows, row_nums


def _parse_xlsx_sheet(
    xml_bytes: bytes,
    shared_strings: list[str],
) -> tuple[list[list[Any]], list[list[str]]]:
    """Parse a single OOXML worksheet into rectangular typed rows and raw string rows."""
    t_rows, r_rows, _ = _parse_xlsx_sheet_with_rownums(xml_bytes, shared_strings)
    return t_rows, r_rows


def read_xlsx(path: str | Path) -> dict[str, list[list[Any]]]:
    """Read an .xlsx workbook into {sheet_name: 2D list of typed values}."""
    sheets = read_xlsx_with_raw(path)
    return {name: pair[0] for name, pair in sheets.items()}


def read_xlsx_with_raw(
    path: str | Path,
) -> dict[str, tuple[list[list[Any]], list[list[str]]]]:
    """Read an .xlsx workbook returning both typed and raw string grids per sheet."""
    p = Path(path)
    with zipfile.ZipFile(p, "r") as zf:
        shared_strings = _read_shared_strings(zf)
        targets = _resolve_sheet_targets(zf)
        out: dict[str, tuple[list[list[Any]], list[list[str]]]] = {}
        for sheet_name, member in targets:
            out[sheet_name] = _parse_xlsx_sheet(zf.read(member), shared_strings)
    return out


def read_xlsx_with_raw_and_rownums(
    path: str | Path,
) -> dict[str, tuple[list[list[Any]], list[list[str]], list[int]]]:
    """Read an .xlsx workbook returning typed rows, raw rows, and 1-based Excel row numbers."""
    p = Path(path)
    with zipfile.ZipFile(p, "r") as zf:
        shared_strings = _read_shared_strings(zf)
        targets = _resolve_sheet_targets(zf)
        out: dict[str, tuple[list[list[Any]], list[list[str]], list[int]]] = {}
        for sheet_name, member in targets:
            out[sheet_name] = _parse_xlsx_sheet_with_rownums(zf.read(member), shared_strings)
    return out


def write_xlsx(
    path: str | Path,
    sheets: dict[str, list[list[Any]]],
    use_inline_strings: bool = False,
) -> Path:
    """Write a minimal valid OOXML .xlsx workbook using only stdlib zipfile."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    shared_list: list[str] = []
    shared_idx: dict[str, int] = {}

    def _intern(s: str) -> int:
        if s not in shared_idx:
            shared_idx[s] = len(shared_list)
            shared_list.append(s)
        return shared_idx[s]

    sheet_names = list(sheets.keys())
    sheet_xmls: list[str] = []

    for sname in sheet_names:
        rows = sheets[sname]
        row_xml_parts: list[str] = []
        for r_i, row in enumerate(rows, start=1):
            cell_xml_parts: list[str] = []
            for c_i, val in enumerate(row):
                if val is None:
                    continue
                ref = f"{index_to_col_letters(c_i)}{r_i}"
                if isinstance(val, bool):
                    cell_xml_parts.append(
                        f'<c r="{ref}" t="b"><v>{"1" if val else "0"}</v></c>'
                    )
                elif isinstance(val, (int, float, np.integer, np.floating)):
                    fval = float(val)
                    if fval != fval:  # NaN
                        continue
                    cell_xml_parts.append(f'<c r="{ref}"><v>{val}</v></c>')
                else:
                    s = str(val)
                    if use_inline_strings:
                        cell_xml_parts.append(
                            f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{xml_escape(s)}</t></is></c>'
                        )
                    else:
                        idx = _intern(s)
                        cell_xml_parts.append(f'<c r="{ref}" t="s"><v>{idx}</v></c>')
            if cell_xml_parts:
                row_xml_parts.append(f'<row r="{r_i}">{"".join(cell_xml_parts)}</row>')
        ws_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<sheetData>{"".join(row_xml_parts)}</sheetData>'
            "</worksheet>"
        )
        sheet_xmls.append(ws_xml)

    # Workbook & rels
    wb_sheets_xml = "".join(
        f'<sheet name="{xml_escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
        for i, name in enumerate(sheet_names, start=1)
    )
    wb_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{wb_sheets_xml}</sheets></workbook>"
    )

    rels_items = [
        f'<Relationship Id="rId{i}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{i}.xml"/>'
        for i in range(1, len(sheet_names) + 1)
    ]
    if shared_list:
        rels_items.append(
            f'<Relationship Id="rId{len(sheet_names) + 1}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" '
            'Target="sharedStrings.xml"/>'
        )
    wb_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{"".join(rels_items)}</Relationships>'
    )

    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )

    ct_overrides = [
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
    ]
    for i in range(1, len(sheet_names) + 1):
        ct_overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    if shared_list:
        ct_overrides.append(
            '<Override PartName="/xl/sharedStrings.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
        )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f'{"".join(ct_overrides)}</Types>'
    )

    with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", root_rels_xml)
        zf.writestr("xl/workbook.xml", wb_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", wb_rels_xml)
        for i, ws_xml in enumerate(sheet_xmls, start=1):
            zf.writestr(f"xl/worksheets/sheet{i}.xml", ws_xml)
        if shared_list:
            ss_items = "".join(
                f'<si><t xml:space="preserve">{xml_escape(s)}</t></si>' for s in shared_list
            )
            ss_xml = (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                f'count="{len(shared_list)}" uniqueCount="{len(shared_list)}">'
                f"{ss_items}</sst>"
            )
            zf.writestr("xl/sharedStrings.xml", ss_xml)

    return p


def _detect_delimiter(sample_text: str, suffix: str) -> str:
    if suffix.lower() == ".tsv":
        return "\t"
    lines = [ln for ln in sample_text.splitlines() if ln.strip()][:10]
    if not lines:
        return ","
    first = lines[0]
    counts = {sep: first.count(sep) for sep in (",", "\t", ";")}
    best_sep = max(counts, key=lambda k: counts[k])
    if counts[best_sep] > 0:
        return best_sep
    try:
        dialect = csv.Sniffer().sniff("\n".join(lines), delimiters=",\t;")
        return dialect.delimiter
    except csv.Error:
        return ","


def read_delimited_with_raw(
    path: str | Path,
    delimiter: str | None = None,
) -> tuple[list[list[Any]], list[list[str]]]:
    """Read a CSV/TSV file into typed rows and raw string rows."""
    p = Path(path)
    raw_bytes = p.read_bytes()
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1")

    sep = delimiter or _detect_delimiter(text[:8192], p.suffix)
    reader = csv.reader(io.StringIO(text), delimiter=sep)
    raw_rows_unpadded = [row for row in reader if any(cell.strip() != "" for cell in row)]
    if not raw_rows_unpadded:
        return [], []
    width = max(len(r) for r in raw_rows_unpadded)
    typed_rows: list[list[Any]] = []
    raw_rows: list[list[str]] = []
    for row in raw_rows_unpadded:
        padded_raw = list(row) + [""] * (width - len(row))
        typed_rows.append([parse_scalar(c) for c in padded_raw])
        raw_rows.append(padded_raw)
    return typed_rows, raw_rows


def read_delimited(path: str | Path, delimiter: str | None = None) -> list[list[Any]]:
    """Read a CSV/TSV file into a 2D list of typed values."""
    typed_rows, _ = read_delimited_with_raw(path, delimiter=delimiter)
    return typed_rows


_STACKED_HDR_KW_RE = re.compile(
    r"\b(p[-_\s]?val|pval|fdr|q[-_\s]?val|ci|95%|lower|upper|min|max|auc|auroc|auprc|"
    r"sens|spec|sensitivity|specificity|acc|accuracy|ppv|npv|f1|hr|or|beta|se|sd|sem|iqr|"
    r"mean|median|model|method|metric|cohort|group|dataset|cancer|stage|grade|endpoint|"
    r"outcome|feature|gene|pathway|protein|variable|subgroup|class|type|sample|total|fold|"
    r"score|estimate|value|ratio|diff|delta|test|comparison|name|id)\b",
    re.IGNORECASE,
)


_PANEL_BANNER_RE = re.compile(
    r"^(?:fig(?:ure)?|ext(?:ended)?\s+data\s+fig(?:ure)?|supp(?:lementary)?\s+(?:fig(?:ure)?|table|data)|"
    r"table|panel)\s*\.?\s*[a-z0-9]{1,6}(?:[\s:.\-]|$)",
    re.IGNORECASE,
)


def _is_banner_row(row: list[Any]) -> bool:
    non_empty = [c for c in row if c is not None and str(c).strip() != ""]
    if len(non_empty) == 1 and isinstance(non_empty[0], str):
        s = non_empty[0].strip()
        return len(s) >= 4
    return False


def _is_panel_banner_row(row: list[Any]) -> bool:
    non_empty = [c for c in row if c is not None and str(c).strip() != ""]
    if len(non_empty) == 1 and isinstance(non_empty[0], str):
        s = non_empty[0].strip()
        return bool(_PANEL_BANNER_RE.match(s))
    return False


def _is_pure_string_header_row(row: list[Any]) -> bool:
    non_empty = [c for c in row if c is not None and str(c).strip() != ""]
    if len(non_empty) < 2:
        return False
    str_cells = [c.strip() for c in non_empty if isinstance(c, str) and c.strip()]
    return len(str_cells) >= max(2, int(0.80 * len(non_empty)))


def _is_header_like_row(row: list[Any]) -> bool:
    non_empty = [c for c in row if c is not None and str(c).strip() != ""]
    if len(non_empty) < 2:
        return False
    str_cells = [c.strip() for c in non_empty if isinstance(c, str) and c.strip()]
    if len(str_cells) < max(2, int(0.75 * len(non_empty))):
        return False
    return any(_STACKED_HDR_KW_RE.search(s) for s in str_cells)


def _block_has_data_rows(typed_rows: list[list[Any]], start: int, end: int) -> bool:
    """Return True if typed_rows[start:end] has a header-like row followed by >= 1 non-banner data row."""
    if end - start < 2:
        return False
    hdr_pos: int | None = None
    limit = min(end, start + 8)
    for i in range(start, limit):
        r = typed_rows[i]
        cnt = 0
        for c in r:
            if c is not None and str(c).strip() != "":
                cnt += 1
                if cnt >= 2:
                    hdr_pos = i
                    break
        if hdr_pos is not None:
            break
    if hdr_pos is None:
        return False
    for i in range(hdr_pos + 1, end):
        r = typed_rows[i]
        cnt = 0
        for c in r:
            if c is not None and str(c).strip() != "":
                cnt += 1
                if cnt >= 2:
                    return True
    return False


def _split_stacked_blocks(
    typed_rows: list[list[Any]],
    raw_rows: list[list[str]],
    row_nums: list[int],
) -> list[tuple[list[list[Any]], list[list[str]]]]:
    """Conservatively split vertically stacked sub-tables separated by blank Excel rows."""
    n = len(typed_rows)
    if n < 4 or len(row_nums) != n:
        return [(typed_rows, raw_rows)]

    split_indices: list[int] = [0]
    curr_start = 0
    for i in range(1, n):
        gap = row_nums[i] - row_nums[i - 1]
        row_i = typed_rows[i]
        is_explicit_panel = _is_panel_banner_row(row_i)
        if gap < 2 and not is_explicit_panel:
            continue
        if not _block_has_data_rows(typed_rows, curr_start, i):
            continue
        starts_new_block = False
        if is_explicit_panel:
            # Explicit Figure/Table/Panel banner followed within 2 rows by a string header row
            for k in range(i + 1, min(n, i + 4)):
                if _is_pure_string_header_row(typed_rows[k]):
                    starts_new_block = True
                    break
        elif _is_header_like_row(row_i):
            if i + 1 < n:
                starts_new_block = True
        elif _is_banner_row(row_i):
            for k in range(i + 1, min(n, i + 4)):
                if _is_header_like_row(typed_rows[k]):
                    starts_new_block = True
                    break
        if starts_new_block:
            split_indices.append(i)
            curr_start = i

    if len(split_indices) == 1:
        return [(typed_rows, raw_rows)]

    split_indices.append(n)
    blocks: list[tuple[list[list[Any]], list[list[str]]]] = []
    for s_idx in range(len(split_indices) - 1):
        a, b = split_indices[s_idx], split_indices[s_idx + 1]
        blocks.append((typed_rows[a:b], raw_rows[a:b] if raw_rows else []))
    return blocks


def _grid_to_table(
    name: str,
    typed_rows: list[list[Any]],
    raw_rows: list[list[str]],
    source_file: str | None = None,
    sheet_base_name: str | None = None,
    inherited_captions: list[str] | None = None,
) -> TableData:
    """Convert raw 2D grid into TableData by locating the most plausible header row."""
    base = sheet_base_name or name
    if not typed_rows:
        return TableData(
            name=name,
            headers=[],
            rows=[],
            raw_rows=[],
            source_file=source_file,
            caption_lines=list(inherited_captions or []),
            sheet_base_name=base,
        )

    # Locate header row among the first 8 non-empty rows:
    # Prefer the first row that has >= 2 non-empty cells and is predominantly string labels
    # (skipping single-cell title banners like "Supplementary Table 1: ...").
    header_idx = 0
    for idx in range(min(8, len(typed_rows))):
        row = typed_rows[idx]
        non_empty = [c for c in row if c is not None and str(c).strip() != ""]
        if len(non_empty) == 0:
            continue
        str_count = sum(1 for c in non_empty if isinstance(c, str))
        if len(non_empty) >= 2 and str_count >= max(1, len(non_empty) // 2):
            header_idx = idx
            break

    # Capture banner / title lines appearing before the header row
    caption_lines: list[str] = []
    for idx in range(header_idx):
        parts = [str(c).strip() for c in typed_rows[idx] if c is not None and str(c).strip() != ""]
        if parts:
            caption_lines.append(" ".join(parts))
    if not caption_lines and inherited_captions:
        caption_lines.extend(inherited_captions)

    hdr_row = typed_rows[header_idx]
    data_start = header_idx + 1

    # Generic 2-row merged multi-header support (common in journal Source Data spreadsheets,
    # e.g., Row r: ["Task", "Metric", "ModelA", None, None, "P value"]
    #       Row r+1: [None, None, "Mean", "95% CI lower", "95% CI upper", None])
    if data_start < len(typed_rows):
        sub_row = typed_rows[data_start]
        sub_non_empty = [c for c in sub_row if c is not None and str(c).strip() != ""]
        sub_str_count = sum(1 for c in sub_non_empty if isinstance(c, str))
        sub_has_stat_kw = any(
            isinstance(c, str)
            and re.search(
                r"\b(mean|median|ci|lower|upper|low|high|min|max|sd|se|auc|auroc|f1|acc|"
                r"sens|spec|sensitivity|specificity|p[- ]?val|std|estimate|hr|or|95%)\b",
                c,
                re.I,
            )
            for c in sub_non_empty
        )
        if len(sub_non_empty) >= 2 and sub_str_count == len(sub_non_empty) and sub_has_stat_kw:
            # Forward-fill parent group headers across empty spans where sub-headers exist
            width = max(len(hdr_row), len(sub_row))
            combined: list[str] = []
            last_group = ""
            for c_i in range(width):
                top = str(hdr_row[c_i]).strip() if (c_i < len(hdr_row) and hdr_row[c_i] is not None) else ""
                sub = str(sub_row[c_i]).strip() if (c_i < len(sub_row) and sub_row[c_i] is not None) else ""
                if top:
                    last_group = top
                if top and sub:
                    combined.append(f"{top}_{sub}")
                elif sub and last_group:
                    combined.append(f"{last_group}_{sub}")
                elif top:
                    combined.append(top)
                elif sub:
                    combined.append(sub)
                else:
                    combined.append(f"col_{c_i}")
            hdr_row = combined
            data_start = header_idx + 2

    headers: list[str] = []
    seen: dict[str, int] = {}
    for c_i, cell in enumerate(hdr_row):
        h = str(cell).strip() if cell is not None and str(cell).strip() != "" else f"col_{c_i}"
        if h in seen:
            seen[h] += 1
            h = f"{h}_{seen[h]}"
        else:
            seen[h] = 0
        headers.append(h)

    data_typed = typed_rows[data_start:]
    data_raw = raw_rows[data_start:] if raw_rows else []

    # Peel trailing single-cell footnote rows (common in journal supplementary tables) into caption_lines
    while len(data_typed) >= 2:
        last_row = data_typed[-1]
        non_empty = [c for c in last_row if c is not None and str(c).strip() != ""]
        if len(non_empty) == 1 and isinstance(non_empty[0], str):
            s = non_empty[0].strip()
            if len(s) >= 12 or s.startswith(("*", "†", "‡", "Note", "Abbreviations", "CI", "IQR")):
                caption_lines.append(s)
                data_typed = data_typed[:-1]
                if data_raw:
                    data_raw = data_raw[:-1]
                continue
        break

    return TableData(
        name=name,
        headers=headers,
        rows=data_typed,
        raw_rows=data_raw,
        source_file=source_file,
        caption_lines=caption_lines,
        sheet_base_name=base,
    )


def read_tables(path: str | Path) -> list[TableData]:
    """Load one (CSV/TSV) or multiple (XLSX) TableData objects from a file."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".xlsx":
        sheets = read_xlsx_with_raw_and_rownums(p)
        tables: list[TableData] = []
        for sheet_name, (t_rows, r_rows, r_nums) in sheets.items():
            blocks = _split_stacked_blocks(t_rows, r_rows, r_nums)
            if len(blocks) == 1:
                tables.append(
                    _grid_to_table(
                        sheet_name,
                        blocks[0][0],
                        blocks[0][1],
                        source_file=str(p),
                        sheet_base_name=sheet_name,
                    )
                )
            else:
                first_tbl = _grid_to_table(
                    f"{sheet_name} [block 1]",
                    blocks[0][0],
                    blocks[0][1],
                    source_file=str(p),
                    sheet_base_name=sheet_name,
                )
                tables.append(first_tbl)
                # Propagate sheet-level top banner to subsequent blocks that lack their own banner
                sheet_caps = [c for c in first_tbl.caption_lines if len(c) >= 6]
                for b_i, (bt, br) in enumerate(blocks[1:], start=2):
                    tables.append(
                        _grid_to_table(
                            f"{sheet_name} [block {b_i}]",
                            bt,
                            br,
                            source_file=str(p),
                            sheet_base_name=sheet_name,
                            inherited_captions=sheet_caps,
                        )
                    )
        return tables
    if ext in (".csv", ".tsv", ".txt"):
        t_rows, r_rows = read_delimited_with_raw(p)
        return [_grid_to_table(p.stem, t_rows, r_rows, source_file=str(p), sheet_base_name=p.stem)]
    raise ValueError(f"Unsupported tabular file extension: {ext}")


def _sanitize_notebook_cell_line(line: str) -> str:
    stripped = line.lstrip()
    if stripped.startswith(("%", "!", "?")) or stripped.endswith("?"):
        indent = line[: len(line) - len(stripped)]
        return f"{indent}# [jupyter-magic] {stripped}"
    return line


def read_code_file(path: str | Path) -> CodeDocument:
    """Read .py, .R/.r, or .ipynb into a normalized CodeDocument."""
    p = Path(path)
    ext = p.suffix.lower()
    raw_bytes = p.read_bytes()
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1")

    if ext == ".ipynb":
        nb = json.loads(text)
        meta = nb.get("metadata", {}) or {}
        ks_lang = str(
            (meta.get("kernelspec") or {}).get("language")
            or (meta.get("language_info") or {}).get("name")
            or "python"
        ).lower()
        language = "r" if ks_lang in ("r", "ir", "rlang") else "python"
        merged_lines: list[str] = []
        cell_line_map: dict[int, tuple[int, int]] = {}
        code_cell_idx = 0
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            src = cell.get("source", "")
            if isinstance(src, list):
                cell_text = "".join(src)
            else:
                cell_text = str(src)
            cell_lines = cell_text.splitlines()
            if language == "python":
                # If cell starts with %% cell magic, comment out whole cell
                first_nonempty = next((ln.strip() for ln in cell_lines if ln.strip()), "")
                if first_nonempty.startswith("%%"):
                    cell_lines = [f"# [cell-magic] {ln}" for ln in cell_lines]
                else:
                    cell_lines = [_sanitize_notebook_cell_line(ln) for ln in cell_lines]
            for local_no, ln in enumerate(cell_lines, start=1):
                merged_lines.append(ln)
                cell_line_map[len(merged_lines)] = (code_cell_idx, local_no)
            # Blank separator between cells
            merged_lines.append("")
            code_cell_idx += 1
        return CodeDocument(
            path=str(p),
            language=language,
            source="\n".join(merged_lines),
            cell_line_map=cell_line_map,
        )

    if ext == ".r":
        return CodeDocument(path=str(p), language="r", source=text)
    if ext == ".py":
        return CodeDocument(path=str(p), language="python", source=text)
    if ext in (".js", ".ts"):
        return CodeDocument(path=str(p), language="javascript", source=text)
    return CodeDocument(path=str(p), language="unknown", source=text)


def extract_numeric_matrix(
    table: TableData,
    target_col: str | None = None,
    exclude_cols: set[str] | None = None,
    min_numeric_fraction: float = 0.8,
) -> tuple[np.ndarray, np.ndarray | None, list[str], str | None]:
    """Extract numeric feature matrix X and optional target vector y from a TableData.

    Automatically excludes obvious ID columns and non-numeric text columns.
    """
    if not table.rows or not table.headers:
        return np.empty((0, 0), dtype=np.float64), None, [], None

    excluded = {c.lower() for c in (exclude_cols or set())}
    n_rows = len(table.rows)
    n_cols = len(table.headers)

    # Identify target column if not explicitly given
    resolved_target_idx: int | None = None
    if target_col is not None:
        for j, h in enumerate(table.headers):
            if h.lower() == target_col.lower():
                resolved_target_idx = j
                break
    else:
        for j, h in enumerate(table.headers):
            if _TARGET_COL_PATTERNS.match(h.strip()):
                resolved_target_idx = j
                break

    def _to_float_col(col_idx: int) -> tuple[np.ndarray, float]:
        arr = np.full(n_rows, np.nan, dtype=np.float64)
        valid = 0
        for i, row in enumerate(table.rows):
            if col_idx >= len(row):
                continue
            v = row[col_idx]
            if v is None:
                continue
            if isinstance(v, bool):
                arr[i] = 1.0 if v else 0.0
                valid += 1
            elif isinstance(v, (int, float)):
                if v == v:
                    arr[i] = float(v)
                    valid += 1
            elif isinstance(v, str):
                parsed = parse_scalar(v)
                if isinstance(parsed, bool):
                    arr[i] = 1.0 if parsed else 0.0
                    valid += 1
                elif isinstance(parsed, (int, float)) and parsed == parsed:
                    arr[i] = float(parsed)
                    valid += 1
        return arr, valid / max(n_rows, 1)

    y_vec: np.ndarray | None = None
    target_name: str | None = None
    if resolved_target_idx is not None:
        y_cand, frac = _to_float_col(resolved_target_idx)
        if frac >= min_numeric_fraction:
            y_vec = y_cand
            target_name = table.headers[resolved_target_idx]

    feat_cols: list[np.ndarray] = []
    feat_names: list[str] = []

    for j, h in enumerate(table.headers):
        if j == resolved_target_idx:
            continue
        if h.lower() in excluded or _ID_COL_PATTERNS.match(h.strip()):
            continue
        col_arr, frac = _to_float_col(j)
        if frac < min_numeric_fraction:
            continue
        # Skip strict sequential integer index columns (0..N-1 or 1..N) in column 0
        if j == 0 and np.all(np.isfinite(col_arr)) and n_rows >= 5:
            diffs = np.diff(col_arr)
            if np.all(diffs == 1.0) and col_arr[0] in (0.0, 1.0):
                continue
        feat_cols.append(col_arr)
        feat_names.append(h)

    if not feat_cols:
        return np.empty((n_rows, 0), dtype=np.float64), y_vec, [], target_name

    X = np.column_stack(feat_cols).astype(np.float64, copy=False)
    return X, y_vec, feat_names, target_name
