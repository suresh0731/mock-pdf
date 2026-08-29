"""Per-page native-PDF-text detection and extraction.

For a page that already carries a real, selectable text layer (a
"digital"/copyable-text page — as opposed to a "scanned" page, which has
no usable text layer and must go through OCR), re-OCRing a rasterized
image of it is both wasteful and strictly less accurate than reading the
text the PDF already stores. This module classifies a page from its
source ``fitz.Page`` and, when classified digital, extracts its words
directly — producing the same :class:`EnsembleWord` shape the OCR
ensemble produces, so every downstream consumer (field extraction,
custom-term/dictionary-scan matching, structural join, redaction) works
unmodified regardless of which path a given page took.

Mixed documents (e.g. a cover page with copyable text followed by
scanned pages) are handled for free: classification runs independently
per page, so the caller (``app.pipeline.redact._build_page_states``)
simply asks this module page-by-page and falls back to
``ensemble_ocr_page`` for any page not classified digital.
"""

import logging
from typing import Literal

from app.services.ocr.ensemble import align_word_boxes, merged_text_from_words
from app.services.ocr.ensemble_types import EnsembleWord

logger = logging.getLogger(__name__)

PageKind = Literal["digital", "scanned"]

# Treated as a single OCR "engine" so align_word_boxes/merged_text_from_words
# (already-tested reading-order + char-span logic) can be reused as-is —
# every native word lands in its own cluster since there's nothing else to
# cross-reference, giving the same EnsembleWord shape the OCR path produces.
_NATIVE_ENGINE_NAME = "native_pdf_text"
_NATIVE_CONFIDENCE = 1.0


def _points_to_pixels_scale(dpi: int) -> float:
    """PDF points (72/inch) -> pixels at ``dpi``, matching page.get_pixmap(dpi=dpi)."""
    return dpi / 72.0


def _text_coverage_ratio(fitz_page: object) -> float:
    """Fraction of the page area covered by native text blocks.

    Uses ``get_text("blocks")`` (block_type == 0 is text; 1 is image) per
    PyMuPDF's own documented layout-triage pattern. Returns 0.0 for a
    degenerate (zero-area) page rather than raising.
    """
    rect = fitz_page.rect
    page_area = rect.width * rect.height
    if page_area <= 0:
        return 0.0
    text_area = 0.0
    for block in fitz_page.get_text("blocks"):
        if len(block) < 7 or block[6] != 0:
            continue
        x0, y0, x1, y1 = block[0], block[1], block[2], block[3]
        text_area += max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return min(text_area / page_area, 1.0)


def classify_page(
    fitz_page: object,
    min_words: int,
    min_coverage_ratio: float,
) -> tuple[PageKind, int, float]:
    """Classify a page as "digital" or "scanned". Pure/side-effect-free.

    "digital" requires *both* a minimum word count and a minimum text-area
    coverage ratio — a coverage-only or count-only check misses the
    "hybrid" case where an already-scanned page carries a thin, possibly
    misaligned OCR text layer baked in by a prior scan-to-PDF step (a few
    words, near-zero coverage). Any extraction failure, or either signal
    falling short, is treated as "scanned" — the safe default, since that
    just means this page runs through the existing OCR path unchanged.

    Args:
        fitz_page: A ``fitz.Page``.
        min_words: Minimum extractable word count to qualify as digital.
        min_coverage_ratio: Minimum text-block area / page area to qualify.

    Returns:
        ``(kind, word_count, coverage_ratio)``. ``word_count``/
        ``coverage_ratio`` are 0 when classification itself failed.
    """
    try:
        words = fitz_page.get_text("words")
        word_count = len(words)
        coverage = _text_coverage_ratio(fitz_page)
    except Exception:
        logger.warning("native_text: classification failed, treating page as scanned")
        return "scanned", 0, 0.0
    if word_count >= min_words and coverage >= min_coverage_ratio:
        return "digital", word_count, coverage
    return "scanned", word_count, coverage


def _extract_word_dicts(fitz_page: object, scale: float) -> list[dict]:
    word_dicts: list[dict] = []
    for entry in fitz_page.get_text("words"):
        if len(entry) < 5:
            continue
        x0, y0, x1, y1, text = entry[0], entry[1], entry[2], entry[3], entry[4]
        stripped = (text or "").strip()
        if not stripped:
            continue
        x = x0 * scale
        y = y0 * scale
        w = (x1 - x0) * scale
        h = (y1 - y0) * scale
        if w <= 0 or h <= 0:
            continue
        word_dicts.append(
            {
                "text": stripped,
                "x": int(round(x)),
                "y": int(round(y)),
                "w": max(1, int(round(w))),
                "h": max(1, int(round(h))),
                "confidence": _NATIVE_CONFIDENCE,
            }
        )
    return word_dicts


def extract_native_words(
    fitz_page: object, dpi: int, page_index: int
) -> tuple[str, list[EnsembleWord]]:
    """Build reading-order EnsembleWords straight from the PDF text layer.

    Word boxes come back from ``fitz`` in PDF point space (72/inch); they
    are scaled by ``dpi / 72.0`` to land in the same pixel space
    ``canonical_image``/``original_image`` already use, so no further
    coordinate translation is needed by any downstream consumer.

    Returns:
        ``(merged_text, ensemble_words)``. Empty on any extraction failure
        — the caller treats that identically to "scanned".
    """
    scale = _points_to_pixels_scale(dpi)
    try:
        word_dicts = _extract_word_dicts(fitz_page, scale)
    except Exception:
        logger.warning("native_text: word extraction failed for page_index=%s", page_index)
        return "", []
    aligned = align_word_boxes([(_NATIVE_ENGINE_NAME, word_dicts)], page=page_index)
    merged = merged_text_from_words(aligned)
    return merged, aligned


def classify_and_extract(
    fitz_page: object,
    dpi: int,
    page_index: int,
    min_words: int,
    min_coverage_ratio: float,
) -> tuple[PageKind, str, list[EnsembleWord]]:
    """Classify a page and, only if digital, extract its native text.

    ``merged_text``/``ensemble_words`` are always ``("", [])`` for a
    "scanned" result — the caller (``_build_page_states``) falls back to
    ``ensemble_ocr_page`` for that page instead.

    Args:
        fitz_page: A ``fitz.Page`` for this page, or ``None`` (e.g. non-PDF
            input, or the pdf2image fallback path was used) — always
            classified "scanned".
        dpi: Render DPI, for point->pixel scaling of native word boxes.
        page_index: Zero-based page index, copied onto every word.
        min_words: Passed through to :func:`classify_page`.
        min_coverage_ratio: Passed through to :func:`classify_page`.

    Returns:
        ``(page_kind, merged_text, ensemble_words)``.
    """
    if fitz_page is None:
        return "scanned", "", []
    kind, word_count, coverage = classify_page(fitz_page, min_words, min_coverage_ratio)
    logger.info(
        "native_text: page_index=%s kind=%s word_count=%s coverage=%.4f",
        page_index,
        kind,
        word_count,
        coverage,
    )
    if kind != "digital":
        return "scanned", "", []
    merged, words = extract_native_words(fitz_page, dpi, page_index)
    if not words:
        # Passed classification but extraction came back empty — fail safe
        # to OCR rather than silently handing the pipeline an empty page.
        logger.warning(
            "native_text: page_index=%s classified digital but extraction "
            "produced no words; falling back to scanned",
            page_index,
        )
        return "scanned", "", []
    return "digital", merged, words
