# Notice:
# Carter Saar. Copyright (C) 2025 State of Utah
# Licensed under Apache License, Version 2.0 (Apache v2). This program is distributed on an "AS IS" BASIS, WITHOUT ANY WARRANTY OR CONDITIONS OF ANY KIND, either express or implied. See the Apache License, Version 2.0 (Apache v2) for more details.

"""
Bounded loading and interpretation of risk-matrix workbooks.

Every path from uploaded bytes to a usable DataFrame goes through here, so the
limits below apply no matter which call site loads a matrix.

An .xlsx is a ZIP of XML, and neither pandas nor openpyxl limits how much of it
they will expand. Since Streamlit serves every session from one process, the
bounds are applied before parsing rather than after.

Matrix layout contract:
    column 0      risk category name
    column 1      human-readable risk description
    columns 2..N  risk levels, ordered lowest to highest
"""

import io
import logging
import re
import zipfile
from xml.etree import ElementTree

import pandas as pd

logger = logging.getLogger(__name__)

# Upload bounds. The shipped matrix is ~15 KB, so 2 MB is ~130x headroom while
# staying far below anything that could stress the parser.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024

# Archive bounds, checked before pandas is allowed to touch the file.
MAX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
MAX_COMPRESSION_RATIO = 100
MAX_ZIP_MEMBERS = 100

# Post-parse shape bounds. A risk matrix is a small human-authored document.
MAX_ROWS = 200
MAX_COLUMNS = 20
MAX_CELL_CHARS = 5000

# Category + description + at least one risk level.
MIN_COLUMNS = 3

# Cap on the sheet XML the coordinate scan will read, so the scan is itself
# bounded regardless of what the archive bounds allowed through.
MAX_SHEET_SCAN_BYTES = 8 * 1024 * 1024

# Coordinate guards, deliberately much looser than MAX_ROWS / MAX_COLUMNS
# because they answer a different question: not "is this a well-formed risk
# matrix" (that is _check_shape, on the parsed frame) but "is this cheap enough
# to parse at all". Real workbooks carry styled-but-empty cells well beyond
# their data, so these sit far above any plausible matrix.
MAX_SCAN_ROWS = 5000
MAX_SCAN_COLUMNS = 256

_CELL_REF_RE = re.compile(r"^([A-Z]+)([0-9]+)$")


# Fixed text for any parser failure, so no third-party exception content is
# echoed to the user. The detail is logged server-side instead.
_PARSE_FAILED = (
    "The workbook could not be parsed. Check that it is a valid .xlsx or .csv "
    "risk matrix."
)


class MatrixError(Exception):
    """
    A risk matrix was rejected.

    ``str(...)`` is safe to show to a user: every message is either fixed text
    or built from the app's own numeric limits. Third-party parser output is
    never interpolated.
    """


def _check_zip_bounds(data):
    """Check the archive's declared sizes before any XML is parsed."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = archive.infolist()
            if len(members) > MAX_ZIP_MEMBERS:
                raise MatrixError(
                    f"Workbook has {len(members)} internal parts (limit {MAX_ZIP_MEMBERS}). "
                    "This does not look like a risk matrix."
                )
            total_uncompressed = sum(m.file_size for m in members)
            total_compressed = sum(m.compress_size for m in members) or 1
            if total_uncompressed > MAX_UNCOMPRESSED_BYTES:
                raise MatrixError(
                    f"Workbook expands to {total_uncompressed / 1024 / 1024:.1f} MB, over the "
                    f"{MAX_UNCOMPRESSED_BYTES // 1024 // 1024} MB limit."
                )
            ratio = total_uncompressed / total_compressed
            if ratio > MAX_COMPRESSION_RATIO:
                raise MatrixError(
                    f"Workbook compression ratio is {ratio:.0f}:1, over the "
                    f"{MAX_COMPRESSION_RATIO}:1 limit. Refusing to expand it."
                )
    except zipfile.BadZipFile as exc:
        raise MatrixError("That file is not a readable .xlsx workbook.") from exc


def _column_index(letters):
    """Convert a spreadsheet column label (A, B, ... XFD) to a 1-based index."""
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index


def _parse_row_number(ref):
    """
    Row number as openpyxl would read it, or None if there is no r= at all.

    Fails closed. openpyxl's grammar is more permissive than this one, so a
    reference it would accept and this cannot parse is rejected rather than
    treated as in-bounds.
    """
    if ref is None:
        return None
    text = str(ref).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        value = float(text)
    except ValueError:
        raise MatrixError(
            "Risk matrix contains an unreadable row reference. Refusing to parse it."
        ) from None
    if value != int(value):
        raise MatrixError(
            "Risk matrix contains an unreadable row reference. Refusing to parse it."
        )
    return int(value)


def _check_row_ref(ref):
    row = _parse_row_number(ref)
    if row is None:
        return
    if row > MAX_SCAN_ROWS:
        raise MatrixError(
            f"Risk matrix references row {row}, far beyond the {MAX_ROWS}-row limit. "
            "Refusing to parse it."
        )


def _check_cell_ref(ref):
    if ref is None or not str(ref).strip():
        return
    match = _CELL_REF_RE.match(str(ref).strip().upper())
    if not match:
        raise MatrixError(
            "Risk matrix contains an unreadable cell reference. Refusing to parse it."
        )
    letters, digits = match.groups()
    if _column_index(letters) > MAX_SCAN_COLUMNS:
        raise MatrixError(
            f"Risk matrix references column {letters}, far beyond the "
            f"{MAX_COLUMNS}-column limit. Refusing to parse it."
        )
    _check_row_ref(digits)


def _scan_sheet(archive, name):
    """Stream one sheet's XML, failing fast on an out-of-range coordinate."""
    parser = ElementTree.XMLPullParser(["start"])
    read = 0
    # openpyxl positions a <row> or <c> with no r= by its own counters, so
    # element counts are tracked here too, not just explicit references.
    implicit_row = 0
    implicit_col = 0
    with archive.open(name) as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            read += len(chunk)
            if read > MAX_SHEET_SCAN_BYTES:
                raise MatrixError("The workbook's sheet data is too large to inspect.")
            parser.feed(chunk)
            for _, element in parser.read_events():
                tag = element.tag.rsplit("}", 1)[-1]
                if tag == "row":
                    explicit = _parse_row_number(element.get("r"))
                    implicit_row = explicit if explicit is not None else implicit_row + 1
                    implicit_col = 0
                    _check_row_ref(implicit_row)
                elif tag == "c":
                    ref = element.get("r")
                    if ref is None or not str(ref).strip():
                        implicit_col += 1
                        if implicit_col > MAX_SCAN_COLUMNS:
                            raise MatrixError(
                                f"Risk matrix row extends past column {MAX_SCAN_COLUMNS}. "
                                "Refusing to parse it."
                            )
                    else:
                        _check_cell_ref(ref)
                elif tag == "dimension":
                    for part in (element.get("ref") or "").split(":"):
                        _check_cell_ref(part)
                element.clear()


def _check_sheet_bounds(data):
    """
    Reject a workbook whose cells sit outside the allowed rectangle, before
    pandas materialises anything.

    The archive bounds are byte-oriented, but a sparse sheet is small on disk
    and enormous once expanded: openpyxl fills in the gaps between cells, so
    row and column extent has to be read from the sheet XML directly. Scanning
    is incremental and stops at the first out-of-range coordinate.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            # Workbook relationships decide what a sheet is, not the file
            # path, so every candidate XML part is checked.
            scanned = 0
            for name in archive.namelist():
                if not name.lower().endswith(".xml"):
                    continue
                if name.startswith("xl/") or name.startswith("worksheets/"):
                    _scan_sheet(archive, name)
                    scanned += 1
            if scanned == 0:
                raise MatrixError("That file is not a readable .xlsx workbook.")
    except MatrixError:
        raise
    except zipfile.BadZipFile as exc:
        raise MatrixError("That file is not a readable .xlsx workbook.") from exc
    except Exception as exc:
        raise MatrixError("The workbook could not be inspected.") from exc


def _check_shape(frame):
    """Enforce row, column and cell-size bounds on a parsed frame."""
    rows, cols = frame.shape
    if cols < MIN_COLUMNS:
        raise MatrixError(
            f"Risk matrix needs at least {MIN_COLUMNS} columns (category, description, and at "
            f"least one risk level) but has {cols}."
        )
    if rows > MAX_ROWS:
        raise MatrixError(f"Risk matrix has {rows} rows, over the {MAX_ROWS}-row limit.")
    if cols > MAX_COLUMNS:
        raise MatrixError(f"Risk matrix has {cols} columns, over the {MAX_COLUMNS}-column limit.")
    if rows == 0:
        raise MatrixError("Risk matrix contains no rows.")

    longest = frame.astype(str).map(len).to_numpy().max() if rows else 0
    if longest > MAX_CELL_CHARS:
        raise MatrixError(
            f"Risk matrix contains a cell of {longest} characters, over the "
            f"{MAX_CELL_CHARS}-character limit."
        )
    return frame


def load_matrix(data, filename):
    """
    Parse risk-matrix bytes into a bounded, shape-validated DataFrame.

    Args:
        data: the raw file bytes.
        filename: original name, used only to pick the parser.

    Returns:
        pandas.DataFrame

    Raises:
        MatrixError: with a user-facing reason, before any unbounded work.
    """
    if data is None:
        raise MatrixError("No risk matrix file was provided.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise MatrixError(
            f"Risk matrix is {len(data) / 1024 / 1024:.1f} MB, over the "
            f"{MAX_UPLOAD_BYTES // 1024 // 1024} MB limit."
        )

    is_csv = str(filename).lower().endswith(".csv")
    if not is_csv:
        _check_zip_bounds(data)
        _check_sheet_bounds(data)

    try:
        if is_csv:
            frame = pd.read_csv(
                io.BytesIO(data), nrows=MAX_ROWS + 1, usecols=range(MAX_COLUMNS + 1)
            )
        else:
            # Explicit, so a multi-sheet upload does not silently pick one.
            frame = pd.read_excel(io.BytesIO(data), engine="openpyxl", sheet_name=0)
    except MatrixError:
        raise
    except ValueError:
        # usecols wider than the file is fine; retry unbounded but still capped.
        try:
            frame = pd.read_csv(io.BytesIO(data), nrows=MAX_ROWS + 1)
        except Exception as exc:
            logger.exception("Risk matrix CSV parse failed")
            raise MatrixError(_PARSE_FAILED) from exc
    except Exception as exc:
        # Parser messages quote content from the file itself, so they are
        # logged rather than shown.
        logger.exception("Risk matrix parse failed")
        raise MatrixError(_PARSE_FAILED) from exc

    return _check_shape(frame)


def load_matrix_path(path):
    """Load a matrix from disk (used for the bundled default)."""
    with open(path, "rb") as handle:
        return load_matrix(handle.read(), str(path))


def risk_levels(frame):
    """
    Risk level names, lowest to highest.

    Columns 0 and 1 are the category and its description. Slicing from column 1
    offered "Risk Description" to the model as a selectable risk level and
    paired every level with the wrong explanation column.
    """
    return frame.columns.tolist()[2:]


def risk_categories(frame):
    """Risk category names, in matrix order."""
    return frame.iloc[:, 0].tolist()


def category_explanations(frame, row_index, levels):
    """Map each risk level to its explanation text for one category row."""
    return {level: str(frame.iloc[row_index, i + 2]) for i, level in enumerate(levels)}


def risk_level_color(risk_level, levels):
    """
    Map a risk level onto a Streamlit color, scaled to the matrix's level count.

    The lowest level is always green and the highest always red. Fixed
    percentage thresholds assumed a five-level matrix and rendered the middle of
    a three-level matrix blue, reading as "medium-low".
    """
    if risk_level not in levels:
        return "gray"
    if len(levels) == 1:
        return "green"
    idx = levels.index(risk_level)
    if idx == 0:
        return "green"
    if idx == len(levels) - 1:
        return "red"
    return "blue" if idx / (len(levels) - 1) < 0.5 else "orange"
