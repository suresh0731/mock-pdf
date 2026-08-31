"""Vector-native redaction for digital (copyable-text) PDF pages.

A digital page already carries a real, selectable text layer and real
vector table lines — rasterizing it just to paint white boxes on top and
re-embed it as a JPEG (the scanned-page path) throws that fidelity away
for no reason. This module instead uses PyMuPDF's own redaction
primitives directly on the page: ``add_redact_annot`` marks a rect,
``apply_redactions`` genuinely deletes whatever text/graphics/image
content sits under it (not just an overlay), and the mock value is then
drawn back in as real vector text. The page stays a real PDF page
throughout — see ``app/services/redact/pdf_renderer.py``'s
``_append_vector_page``, which clones the source page before calling
into this module so the long-lived source document is never mutated.

Two ordered, independent phases, matching the plan's explicit "image
redaction is a separate final phase, run after text" requirement:

- :func:`redact_text_regions` — every non-brand (text/PII) region.
- :func:`redact_image_regions` — run strictly *after* the above, on the
  same page: every brand/picture-zone region (``assignment_source ==
  "brand"`` — logos, footer chrome, generic images Docling detected;
  see ``app/services/pii/brand_zones.py``). Uses
  ``PDF_REDACT_IMAGE_PIXELS`` (not the default ``PDF_REDACT_IMAGE_
  REMOVE``) so an image that extends beyond the detected zone only loses
  the covered pixels, not the whole image object.
- :func:`strip_signature_artifacts` — after both phases, delete leftover
  signature widgets and Stamp/Ink annotations whose appearance streams
  can survive ``apply_redactions``.

Both phases pass the same defensive ``images``/``graphics`` removal
options to ``apply_redactions`` — not only the image phase — so any
image or vector-drawn graphic that happens to sit under a *text* region
(e.g. text over a background graphic) is also genuinely stripped rather
than only visually covered.
"""

import logging
from collections.abc import Sequence
from typing import Protocol

import fitz

logger = logging.getLogger(__name__)

_REDACT_FILL = (1, 1, 1)
_TEXT_FILL = (0, 0, 0)
_FALLBACK_FONT_NAME = "helv"
# PyMuPDF's built-in "Times-Roman" alias — used instead of the Helvetica
# fallback whenever the source page's own body text is serif, so the
# inserted mock value reads as a natural continuation of the line rather
# than an obviously mismatched sans-serif patch (see _detect_page_font_name).
_SERIF_FONT_NAME = "tiro"
_SERIF_NAME_HINTS = (
    "times",
    "georgia",
    "garamond",
    "serif",
    "cambria",
    "palatino",
    "minion",
    "book antiqua",
    "century",
    "cardo",
)
_MIN_FONT_SIZE = 4
_MIN_BOX_HEIGHT = 4
# Same idea as the raster renderer's _page_font_size_cap: a page-wide
# outlier box (e.g. a large brand/footer zone) must not paint its label
# dramatically larger than every other, same-purpose redaction on the
# page — cap the font-height-driven starting size to a multiple of the
# page's typical (median) box height instead.
_FONT_CAP_SLACK = 1.3


class BBoxLike(Protocol):
    """Duck-typed bbox with integer pixel origin and size."""

    x: int
    y: int
    w: int
    h: int


class PaintedRegion(Protocol):
    """Duck-typed redaction region — same shape pdf_renderer.py consumes."""

    padded_bbox: BBoxLike
    mock_value: str


def _points_scale(dpi: int) -> float:
    """pixels -> PDF points (72/inch); inverse of native_text's
    ``_points_to_pixels_scale``, used to convert a ``padded_bbox``
    (pixel space, from ``RedactOptions.dpi`` rendering) back to the
    source page's own point space.
    """
    return 72.0 / dpi if dpi else 1.0


def _bbox_to_rect(bbox: BBoxLike, dpi: int) -> fitz.Rect:
    scale = _points_scale(dpi)
    return fitz.Rect(
        bbox.x * scale,
        bbox.y * scale,
        (bbox.x + bbox.w) * scale,
        (bbox.y + bbox.h) * scale,
    )


def _is_brand_region(region: PaintedRegion) -> bool:
    return getattr(region, "assignment_source", None) == "brand"


def _apply_redactions_safely(page: "fitz.Page") -> None:
    """One ``apply_redactions`` call with defensive image/graphics
    removal — used identically by both phases (see module docstring for
    why the text phase also needs this, not just the image phase).
    """
    page.apply_redactions(
        images=fitz.PDF_REDACT_IMAGE_PIXELS,
        graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,
        text=fitz.PDF_REDACT_TEXT_REMOVE,
    )


def _get_page_text_dict(page: "fitz.Page") -> dict:
    """``page.get_text("dict")``, or ``{}`` on failure — the single
    snapshot both :func:`_detect_page_font_name` and
    :func:`_region_font_and_size` read from. Must be captured *before*
    ``apply_redactions`` removes any text — both call sites in this
    module sample the page while it's still intact.
    """
    try:
        return page.get_text("dict")
    except Exception:
        return {}


def _font_family_from_name(font_name: str) -> str:
    """Map a raw PDF span font name (e.g. ``"TimesNewRomanPSMT"``) to
    the closest PyMuPDF built-in alias via the serif name-hint list.
    """
    if any(hint in font_name for hint in _SERIF_NAME_HINTS):
        return _SERIF_FONT_NAME
    return _FALLBACK_FONT_NAME


def _detect_page_font_name(page_dict: dict) -> str:
    """Guess whether the page's body text is serif or sans-serif and
    return the matching PyMuPDF built-in font alias — the page-wide
    fallback used whenever a region has no overlapping original span
    (see :func:`_region_font_and_size`).

    Picking the wrong family (e.g. always defaulting to Helvetica on a
    Times New Roman bank statement) is the main reason a vector-native
    mock replacement can look like an obviously pasted-on patch rather
    than a natural continuation of the line: the box/text deletion and
    reinsertion is genuine either way, but a jarring font mismatch makes
    it *look* fake. Falls back to Helvetica when nothing conclusive is
    found (matches the module's original default).
    """
    serif_chars = 0
    sans_chars = 0
    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                n = len(text.strip())
                if n == 0:
                    continue
                if _font_family_from_name(span.get("font", "").lower()) == _SERIF_FONT_NAME:
                    serif_chars += n
                else:
                    sans_chars += n
    if serif_chars > sans_chars:
        return _SERIF_FONT_NAME
    return _FALLBACK_FONT_NAME


def _overlap_area(a: fitz.Rect, b: fitz.Rect) -> float:
    """Plain rectangle-intersection area; 0 when disjoint. Computed from
    raw coordinates rather than fitz's ``&`` operator so this doesn't
    depend on how a given PyMuPDF version represents an empty result.
    """
    width = min(a.x1, b.x1) - max(a.x0, b.x0)
    height = min(a.y1, b.y1) - max(a.y0, b.y0)
    if width <= 0 or height <= 0:
        return 0.0
    return width * height


def _region_font_and_size(
    page_dict: dict, rect: fitz.Rect, fallback_font_name: str
) -> tuple[str, float | None]:
    """Original span (font family + point size) whose bbox overlaps
    ``rect`` the most, so the replacement text tracks what was actually
    printed at that spot instead of a single page-wide family plus a
    guess derived purely from the redaction box's own height — a table
    page routinely mixes a larger header font with small body-row text,
    and a page-wide/height-only guess flattens that difference away.

    Falls back to ``fallback_font_name`` and no preferred size (letting
    ``_fit_font_size`` derive one from the box height as before) when
    nothing overlaps — e.g. a table cell that was blank in the source.
    """
    best_area = 0.0
    best_span: dict | None = None
    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if not span.get("text", "").strip():
                    continue
                area = _overlap_area(fitz.Rect(span["bbox"]), rect)
                if area > best_area:
                    best_area = area
                    best_span = span
    if best_span is None:
        return fallback_font_name, None
    family = _font_family_from_name(best_span.get("font", "").lower())
    size = best_span.get("size")
    return family, float(size) if size else None


def _fit_font_size(
    text: str,
    box_w: float,
    box_h: float,
    max_size_cap: int | None = None,
    font_name: str = _FALLBACK_FONT_NAME,
    preferred_size: float | None = None,
) -> int | None:
    """Largest font size (>= ``_MIN_FONT_SIZE``) whose rendered width of
    ``text`` fits ``box_w``, starting from ``preferred_size`` (the
    original span's own point size — see ``_region_font_and_size``)
    when given, else the same box-height-driven guess the raster
    renderer's ``_font_for_box`` uses. ``None`` when the box is too
    small for any size to fit.

    ``max_size_cap``, when given, still clamps the starting size (see
    ``_page_font_size_cap``) even when ``preferred_size`` is used — an
    overlapping span picked by bbox overlap alone is an imperfect
    match, so a stray large span (e.g. a heading that only clips the
    edge of a body-row redaction box) can't blow past what every other
    redaction on the page renders at.
    """
    if box_h < _MIN_BOX_HEIGHT or box_w <= 2:
        return None
    if preferred_size is not None and preferred_size >= 1:
        size = int(round(preferred_size))
    else:
        size = max(1, int(0.7 * box_h))
    if max_size_cap is not None:
        size = min(size, max_size_cap)
    # A short box/small preferred size yields a guess below
    # _MIN_FONT_SIZE even though it may still be plenty *wide* enough
    # for the smallest readable size — always attempt _MIN_FONT_SIZE
    # rather than bailing out purely on that (real bank-statement rows
    # are routinely this short). max_size_cap is always >=
    # _MIN_FONT_SIZE (see _page_font_size_cap), so this never exceeds
    # the page-wide cap.
    size = max(size, _MIN_FONT_SIZE)
    max_width = box_w - 2
    while size >= _MIN_FONT_SIZE:
        width = fitz.get_text_length(text, fontname=font_name, fontsize=size)
        if width <= max_width:
            return size
        size -= 1
    return None


def _page_font_size_cap(regions: Sequence[PaintedRegion], dpi: int) -> int | None:
    """Median non-brand box height across the page (in PDF points),
    scaled the same way ``_fit_font_size`` derives a starting size from
    a single box's own height — mirrors the raster renderer's
    ``_page_font_size_cap`` exactly, just in point space instead of
    pixels. Excludes brand regions from the median itself (a footer/
    picture zone is usually much taller than a text redaction) but the
    resulting cap is applied to both phases, so a brand label is capped
    to the *text* boxes' typical size too.
    """
    scale = _points_scale(dpi)
    heights = [
        h
        for r in regions
        if not _is_brand_region(r) and (h := r.padded_bbox.h * scale) >= _MIN_BOX_HEIGHT
    ]
    if not heights:
        return None
    heights.sort()
    median_h = heights[len(heights) // 2]
    return max(_MIN_FONT_SIZE, int(0.7 * median_h * _FONT_CAP_SLACK))


def _insert_fitted_text(
    page: "fitz.Page",
    rect: fitz.Rect,
    mock_value: str,
    max_size_cap: int | None = None,
    font_name: str = _FALLBACK_FONT_NAME,
    preferred_size: float | None = None,
) -> None:
    """Draw ``mock_value`` centered in ``rect`` as real vector text, or
    skip silently if it can't be shrunk to fit — matching the raster
    renderer's ``_paint_mock_value`` skip behavior.
    """
    mock = (mock_value or "").strip()
    if not mock:
        return
    size = _fit_font_size(mock, rect.width, rect.height, max_size_cap, font_name, preferred_size)
    if size is None:
        return
    text_width = fitz.get_text_length(mock, fontname=font_name, fontsize=size)
    x = rect.x0 + (rect.width - text_width) / 2
    # insert_text's origin is the text baseline, not a bounding-box
    # corner; ``0.35 * size`` approximates half the cap-height so the
    # glyphs land roughly centered vertically in the box.
    y = rect.y0 + rect.height / 2 + 0.35 * size
    page.insert_text((x, y), mock, fontsize=size, fontname=font_name, color=_TEXT_FILL)


def redact_text_regions(page: "fitz.Page", regions: Sequence[PaintedRegion], dpi: int) -> None:
    """Phase 1: genuinely remove every non-brand region's underlying
    content and draw its mock value back in as vector text.

    No-op if ``regions`` has no non-brand entries (nothing to redact —
    also avoids an ``apply_redactions()`` call with zero annotations).
    """
    text_regions = [r for r in regions if not _is_brand_region(r)]
    if not text_regions:
        return
    page_dict = _get_page_text_dict(page)
    page_font_name = _detect_page_font_name(page_dict)
    font_cap = _page_font_size_cap(regions, dpi)
    rects = [_bbox_to_rect(r.padded_bbox, dpi) for r in text_regions]
    region_fonts = [_region_font_and_size(page_dict, rect, page_font_name) for rect in rects]
    for rect in rects:
        if rect.width <= 0 or rect.height <= 0:
            continue
        page.add_redact_annot(rect, fill=_REDACT_FILL)
    _apply_redactions_safely(page)
    for region, rect, (font_name, preferred_size) in zip(
        text_regions, rects, region_fonts, strict=True
    ):
        if rect.width <= 0 or rect.height <= 0:
            continue
        _insert_fitted_text(
            page,
            rect,
            getattr(region, "mock_value", "") or "",
            font_cap,
            font_name,
            preferred_size,
        )


def strip_signature_artifacts(page: "fitz.Page") -> None:
    """Drop leftover PDF signature widgets and Stamp/Ink annots.

    ``apply_redactions`` removes page-content under a rect but does not
    always delete a signature widget's appearance stream, so a digital
    page can still *look* signed after the visual ink is blanked. A
    redacted document also must not carry a still-valid-looking
    cryptographic signature widget. Failures are logged and skipped —
    never abort the page over one stubborn annot.
    """
    try:
        widgets = list(page.widgets() or [])
    except Exception:
        widgets = []
    for widget in widgets:
        if getattr(widget, "field_type", None) != fitz.PDF_WIDGET_TYPE_SIGNATURE:
            continue
        try:
            page.delete_widget(widget)
        except Exception:
            logger.warning("strip_signature_artifacts: widget delete failed", exc_info=True)
    try:
        annots = list(page.annots() or [])
    except Exception:
        annots = []
    for annot in annots:
        try:
            kind = annot.type[1]
        except (AttributeError, TypeError, IndexError):
            continue
        if kind not in ("Stamp", "Ink"):
            continue
        try:
            page.delete_annot(annot)
        except Exception:
            logger.warning("strip_signature_artifacts: annot delete failed", exc_info=True)


def redact_image_regions(page: "fitz.Page", regions: Sequence[PaintedRegion], dpi: int) -> None:
    """Phase 2 (separate, final): pixel-blank every brand/picture-zone
    region. Must run *after* :func:`redact_text_regions` on the same
    page — see module docstring.

    No mock text is drawn for these (matches the raster path: a brand
    zone with an empty ``mock_value`` — the common case for a generic
    "IMAGE" zone — just stays a plain white patch); a non-empty
    ``mock_value`` (e.g. "FOOTER") is still drawn for parity with the
    raster renderer's uniform "paint mock text if present" behavior.
    """
    brand_regions = [r for r in regions if _is_brand_region(r)]
    if not brand_regions:
        return
    page_dict = _get_page_text_dict(page)
    page_font_name = _detect_page_font_name(page_dict)
    font_cap = _page_font_size_cap(regions, dpi)
    rects = [_bbox_to_rect(r.padded_bbox, dpi) for r in brand_regions]
    region_fonts = [_region_font_and_size(page_dict, rect, page_font_name) for rect in rects]
    for rect in rects:
        if rect.width <= 0 or rect.height <= 0:
            continue
        page.add_redact_annot(rect, fill=_REDACT_FILL)
    _apply_redactions_safely(page)
    for region, rect, (font_name, preferred_size) in zip(
        brand_regions, rects, region_fonts, strict=True
    ):
        if rect.width <= 0 or rect.height <= 0:
            continue
        _insert_fitted_text(
            page,
            rect,
            getattr(region, "mock_value", "") or "",
            font_cap,
            font_name,
            preferred_size,
        )
