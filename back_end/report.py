# Notice:
# Carter Saar. Copyright (C) 2025 State of Utah
# Licensed under Apache License, Version 2.0 (Apache v2). This program is distributed on an "AS IS" BASIS, WITHOUT ANY WARRANTY OR CONDITIONS OF ANY KIND, either express or implied. See the Apache License, Version 2.0 (Apache v2) for more details.

"""
Sanitisation and report export for risk assessment results.

Assessment text is not authored by this application: risk levels, reasoning and
mitigations come from the model, and category names come from whatever matrix
was uploaded. Both destinations interpret what they are given -- Streamlit
renders Markdown, and a spreadsheet evaluates a leading "=" -- so everything is
neutralised here before it reaches either.
"""

import io
import re

import pandas as pd

# Leading characters a spreadsheet client may interpret as a formula.
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")

# Generous cap; long enough for real reasoning, short enough to bound a cell.
MAX_FIELD_CHARS = 8000

UNKNOWN_LEVEL = "Unknown"

# Any sequence a Markdown renderer or browser could turn into a destination.
# Ordered longest-prefix-first so http:// is consumed before a bare scheme:host.
def coerce_text(value, max_chars=MAX_FIELD_CHARS):
    """Force any model- or file-supplied value to a bounded string."""
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    if len(text) > max_chars:
        text = text[:max_chars] + " …[truncated]"
    return text


_URL_RE = re.compile(
    r"""(?ix)
    (?: [a-z][a-z0-9+.\-]* : //          # scheme://host
      | [a-z][a-z0-9+.\-]* : (?=[^\s/]) # scheme:host  (no slashes; browsers resolve it)
      | //[^\s/]                          # //host       (protocol-relative)
      | www\.                             # www.host     (GFM autolinks this)
    )
    \S*
    """
)

# CommonMark says a backslash before ASCII punctuation is a literal. Escaping
# every one of these means no construct -- inline, reference-style, autolink or
# otherwise -- can be parsed as markup, while the rendered text still reads
# normally because the backslashes are consumed by the renderer.
_MD_PUNCT_RE = re.compile(r"([!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~])")

_LINK_PLACEHOLDER = "[link removed]"

# Characters that cannot appear in sheet XML; openpyxl raises on them, which
# would break the download. Tab, newline and carriage return are legal and kept.
_ILLEGAL_XML_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def strip_control_characters(text):
    """Remove characters that cannot be represented in a spreadsheet cell."""
    return _ILLEGAL_XML_RE.sub("", text)


def strip_urls(text):
    """Remove anything that could act as a link destination."""
    return _URL_RE.sub(_LINK_PLACEHOLDER, text)


def escape_markdown(text):
    """
    Neutralise every Markdown construct by escaping ASCII punctuation.

    An allowlist rather than a denylist: nothing renders as markup unless we
    let it. Markdown has too many ways to express a link for pattern-matching
    them to be reliable.
    """
    return _MD_PUNCT_RE.sub(r"\\\1", text)


def plain_text(value, max_chars=MAX_FIELD_CHARS):
    """
    Bounded text with link destinations removed, but no Markdown escaping.

    For sinks that do not interpret Markdown -- spreadsheet cells, logs.
    """
    return strip_control_characters(strip_urls(coerce_text(value, max_chars)))


def markdown_text(value, max_chars=MAX_FIELD_CHARS):
    """
    Bounded text safe to hand to st.markdown / st.write / an expander label.

    Destinations are removed and every remaining punctuation mark is escaped, so
    the renderer can produce neither an anchor nor an image.
    """
    return escape_markdown(plain_text(value, max_chars))


def safe_label(value, max_chars=1000):
    """
    Bounded text for a NAME the user needs to read: a URL they typed, a filename,
    a risk category.

    Markup is neutralised but the text stays legible, because the reader needs
    to recognise it. Escaping alone stops it rendering as a link or an image;
    model-generated prose goes through markdown_text() instead, which also
    removes destinations.

    Escaping is not idempotent, so apply it once, where the value is
    interpolated -- not again when rendering.
    """
    return escape_markdown(coerce_text(value, max_chars))


# Retained under the original name: callers that render Markdown want the
# escaped form, which is the safe default.
def safe_display_text(value):
    """Bounded, markup-free text safe for any Streamlit text surface."""
    return markdown_text(value)


def validate_risk_level(risk_level, allowed_levels):
    """
    Constrain a model-supplied risk level to the matrix's real levels.

    Anything else -- an invented level, an injected formula, a Markdown payload
    -- collapses to "Unknown" rather than being rendered or exported verbatim.
    """
    text = coerce_text(risk_level, max_chars=200).strip()
    for level in allowed_levels or ():
        if text == str(level):
            # Matching is not the same as being safe: the levels come from the
            # uploaded matrix's header row, so callers still sanitise the result.
            return text
    return UNKNOWN_LEVEL


def sanitize_result(result, allowed_levels, for_markdown=True):
    """
    Normalise one assessment result.

    for_markdown=True also escapes punctuation, for surfaces that render
    Markdown. The workbook needs the unescaped form, since a spreadsheet cell
    does not interpret Markdown.
    """
    render = markdown_text if for_markdown else plain_text
    return {
        "category": render(result.get("category", "Unknown")),
        "risk_level": render(
            validate_risk_level(result.get("risk_level"), allowed_levels), 200
        ),
        "reasoning": render(result.get("reasoning", "No reasoning provided")),
        "mitigations": render(result.get("mitigations", "No mitigations suggested")),
    }


def _is_formula_like(value):
    return isinstance(value, str) and value.startswith(FORMULA_PREFIXES)


def build_report_workbook(results, allowed_levels=None):
    """
    Build the downloadable .xlsx with every cell stored as inert text.

    Cells are written with pandas, then any value a spreadsheet might evaluate
    is forced to data_type "s". Note that Cell.set_explicit_value does not exist
    in openpyxl 3.1.x; forcing data_type after assignment is the supported path.
    """
    rows = []
    for result in results or ():
        clean = sanitize_result(result, allowed_levels, for_markdown=False)
        rows.append(
            {
                "Risk Type": clean["category"],
                "Risk Level": clean["risk_level"],
                "Reasoning": clean["reasoning"],
                "Potential Mitigations": clean["mitigations"],
            }
        )

    frame = pd.DataFrame(
        rows, columns=["Risk Type", "Risk Level", "Reasoning", "Potential Mitigations"]
    )

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="Risk Assessment")
        worksheet = writer.sheets["Risk Assessment"]

        for row in worksheet.iter_rows():
            for cell in row:
                if _is_formula_like(cell.value):
                    cell.data_type = "s"

        # Column widths. The previous implementation used chr(65 + i), which
        # produces junk past column Z.
        from openpyxl.utils import get_column_letter

        for index, column in enumerate(frame.columns, start=1):
            if len(frame):
                longest = max(frame[column].astype(str).map(len).max(), len(column))
            else:
                longest = len(column)
            worksheet.column_dimensions[get_column_letter(index)].width = min(longest + 2, 100)

    buffer.seek(0)
    return buffer.getvalue()
