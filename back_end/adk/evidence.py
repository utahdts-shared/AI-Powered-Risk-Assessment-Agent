# Notice:
# Carter Saar. Copyright (C) 2025 State of Utah
# Licensed under Apache License, Version 2.0 (Apache v2). This program is distributed on an "AS IS" BASIS, WITHOUT ANY WARRANTY OR CONDITIONS OF ANY KIND, either express or implied. See the Apache License, Version 2.0 (Apache v2) for more details.

"""
Evidence extraction: turn user-supplied URLs and PDFs into bounded markdown.

URL retrieval deliberately does not go through ADK's url_context tool. Gemini's
url_context can fail to fetch a page and still answer from the model's own prior
knowledge, and the reply reads the same either way. For a risk assessment that
matters: the whole point is what the vendor's page actually says.

Retrieval is confirmed through `url_context_metadata` on the response, which ADK
does not surface. So this makes a direct google-genai call instead, and accepts
a page only when the API reports it was genuinely read.
"""

from __future__ import annotations

import io
import logging
import time

from google.genai import types
from pydantic import BaseModel


class ExtractionResult(BaseModel):
    """Outcome of extracting one piece of evidence (a URL or a PDF)."""

    source: str
    kind: str  # "url" | "pdf"
    ok: bool
    text: str = ""
    warning: str = ""

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.source}"


# Per-source cap, so one oversized document cannot dominate the
# evidence brief and from blowing out the context of every category agent.
MAX_EXTRACT_CHARS = 12000

# A PDF that yields less than this is almost certainly scanned or image-only.
# pypdf extracts embedded text only; it does not perform OCR.
MIN_USEFUL_PDF_CHARS = 200

# Bounds on PDF parsing work. A page count is its own dimension: a file can be
# small on disk and still hold an enormous page tree, and empty pages never trip
# a character limit.
MAX_PDF_PAGES = 500

# Wall-clock budget for one document, so a slow file cannot hold up the run.
MAX_PDF_SECONDS = 20.0

logger = logging.getLogger(__name__)

_URL_EXTRACTION_PROMPT = (
    "Summarise the factual content of {url} that is relevant to assessing the risk of "
    "adopting this technology: data handling, retention, training on customer data, "
    "subprocessors, hosting locations, security certifications, guardrails, and the "
    "intended user base.\n\n"
    "Report ONLY what the page itself states. Do not use prior knowledge about this "
    "company or product. If the page could not be read, say exactly: PAGE NOT READABLE."
)


def _retrieval_succeeded(response) -> bool:
    """
    True only when the API confirms it actually fetched the page.

    Missing metadata counts as failure, not as success by default.
    """
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return False
    metadata = getattr(candidates[0], "url_context_metadata", None)
    entries = getattr(metadata, "url_metadata", None) or []
    if not entries:
        return False
    for entry in entries:
        status = str(getattr(entry, "url_retrieval_status", ""))
        if not status.endswith("URL_RETRIEVAL_STATUS_SUCCESS"):
            return False
    return True


def extract_url(client, url: str, model: str) -> ExtractionResult:
    """Fetch one URL through Gemini's url_context tool and verify it was read."""
    url = (url or "").strip()
    if not url:
        return ExtractionResult(source=url, kind="url", ok=False, warning="Empty URL.")

    try:
        response = client.models.generate_content(
            model=model,
            contents=_URL_EXTRACTION_PROMPT.format(url=url),
            config=types.GenerateContentConfig(
                tools=[types.Tool(url_context=types.UrlContext())],
            ),
        )
    except Exception as exc:
        return ExtractionResult(
            source=url, kind="url", ok=False, warning=f"Request failed: {exc}"
        )

    if not _retrieval_succeeded(response):
        return ExtractionResult(
            source=url,
            kind="url",
            ok=False,
            warning=(
                "The page could not be retrieved, so it was excluded. Any answer the model "
                "produced would have come from prior knowledge rather than this page."
            ),
        )

    text = (getattr(response, "text", None) or "").strip()
    if not text or "PAGE NOT READABLE" in text:
        return ExtractionResult(
            source=url, kind="url", ok=False, warning="The page returned no usable content."
        )

    return ExtractionResult(source=url, kind="url", ok=True, text=text[:MAX_EXTRACT_CHARS])


def extract_pdf(name: str, content: bytes) -> ExtractionResult:
    """
    Extract embedded text from one PDF with pypdf.

    There is no OCR, so a scanned or image-only PDF yields little or nothing.
    That is reported rather than passing silently, which would be
    indistinguishable from a successful read.
    """
    if not content:
        return ExtractionResult(source=name, kind="pdf", ok=False, warning="File was empty.")

    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                return ExtractionResult(
                    source=name, kind="pdf", ok=False,
                    warning="The PDF is password protected, so no text could be read.",
                )
        # Reject an oversized page tree before decoding anything.
        page_count = len(reader.pages)
        if page_count > MAX_PDF_PAGES:
            return ExtractionResult(
                source=name, kind="pdf", ok=False,
                warning=(
                    f"The PDF has {page_count} pages, over the {MAX_PDF_PAGES}-page limit, "
                    "so it was not read."
                ),
            )

        pages = []
        total = 0
        deadline = time.monotonic() + MAX_PDF_SECONDS
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)
            total += len(page_text)
            if total > MAX_EXTRACT_CHARS:
                break
            if time.monotonic() > deadline:
                return ExtractionResult(
                    source=name, kind="pdf", ok=False,
                    warning=(
                        f"Reading the PDF exceeded the {MAX_PDF_SECONDS:.0f}-second limit, "
                        "so it was not used."
                    ),
                )
        text = "\n\n".join(p.strip() for p in pages if p.strip()).strip()
    except Exception:
        # Parser messages quote content from the file itself, so they are logged
        # rather than shown.
        logger.exception("PDF extraction failed for %s", name)
        return ExtractionResult(
            source=name, kind="pdf", ok=False,
            warning="The PDF could not be read. It may be corrupt or an unsupported format.",
        )

    if len(text) < MIN_USEFUL_PDF_CHARS:
        return ExtractionResult(
            source=name, kind="pdf", ok=False, text=text,
            warning=(
                f"Only {len(text)} characters of text could be extracted. This PDF is likely "
                "scanned or image-based; text extraction does not perform OCR, so it "
                "contributed no evidence."
            ),
        )

    return ExtractionResult(source=name, kind="pdf", ok=True, text=text[:MAX_EXTRACT_CHARS])


def build_evidence_brief(results: list[ExtractionResult]) -> str:
    """
    Fold every successful extraction into one bounded markdown brief.

    Building this once keeps a run at roughly (urls + pdfs + categories) model
    calls, rather than re-reading every source for every category, which matters
    on a metered free tier.
    """
    usable = [r for r in results if r.ok and r.text]
    failed = [r for r in results if not r.ok]

    sections = ["# Evidence Brief", ""]
    if not usable:
        sections += [
            "No external evidence was successfully gathered. Base the assessment solely on "
            "the project details supplied by the analyst.",
            "",
        ]
    for result in usable:
        sections += [f"## {result.kind.upper()}: {result.source}", "", result.text, ""]

    if failed:
        sections += ["## Sources that could not be read", ""]
        sections += [f"- `{r.source}` — {r.warning}" for r in failed]
        sections += [
            "",
            "Treat these as absent evidence. Do not infer their contents.",
            "",
        ]
    return "\n".join(sections)
