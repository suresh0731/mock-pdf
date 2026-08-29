"""Assemble the redacted output PDF: one real vector page for every
source page that was "digital" (see pdf_native_redactor.py), and a
painted-then-flattened raster page for every "scanned"/"blank" one.
"""

import hashlib
import io
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import fitz
from PIL import Image, ImageDraw, ImageFont

from app.services.redact.pdf_native_redactor import redact_image_regions, redact_text_regions

logger = logging.getLogger(__name__)

_MIN_FONT_SIZE = 4
_MIN_BOX_HEIGHT = 4
_FONT_CAP_SLACK = 1.3
_TTF_CANDIDATES = (
    "DejaVuSans.ttf",
    "arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    r"C:\Windows\Fonts\arial.ttf",
)


class BBoxLike(Protocol):
    """Duck-typed bbox with integer pixel origin and size."""

    x: int
    y: int
    w: int
    h: int


class PaintedRegion(Protocol):
    """Duck-typed redaction region. Optional mock_value / assignment_source."""

    page: int
    padded_bbox: BBoxLike


def _regions_in_draw_order(regions: Sequence[PaintedRegion]) -> list[PaintedRegion]:
    """Stable partition: non-brand first, assignment_source == 'brand' last."""
    non_brand = [r for r in regions if getattr(r, "assignment_source", None) != "brand"]
    brand = [r for r in regions if getattr(r, "assignment_source", None) == "brand"]
    return [*non_brand, *brand]


def _sanitize_filename(filename: str) -> str:
    """Basename only; strip NUL; max 255 chars; fallback 'document'."""
    name = Path(filename).name.replace("\x00", "").strip()
    if not name or name == ".":
        return "document"
    return name[:255]


def _try_truetype(size: int) -> ImageFont.FreeTypeFont | None:
    """Load the first available TTF at ``size``, or None if none load."""
    for candidate in _TTF_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return None


def _text_size(font: ImageFont.ImageFont, text: str) -> tuple[int, int]:
    """Return (width, height) of ``text`` in pixels for ``font``."""
    left, top, right, bottom = font.getbbox(text)
    return right - left, bottom - top


def _font_for_box(
    mock_value: str, box_w: int, box_h: int, max_size_cap: int | None = None
) -> tuple[ImageFont.ImageFont, int] | None:
    """Return (font, size) or None when text should be skipped.

    ``max_size_cap``, when given, clamps the box-height-driven starting
    size (see ``_page_font_size_cap``) so a single box whose own height
    came out an outlier (e.g. a multi-line OCR union) doesn't render its
    mock value dramatically larger than every other redaction on the same
    page — text still shrinks further to fit narrow boxes as before.
    """
    if box_h < _MIN_BOX_HEIGHT or box_w <= 2:
        return None
    size = max(1, int(0.7 * box_h))
    if max_size_cap is not None:
        size = min(size, max_size_cap)
    # A short box yields a height-driven guess below _MIN_FONT_SIZE even
    # though it may still be plenty *wide* enough for the smallest
    # readable size — always attempt _MIN_FONT_SIZE rather than bailing
    # out purely on a short box (real bank-statement rows are routinely
    # this short). max_size_cap is always >= _MIN_FONT_SIZE (see
    # _page_font_size_cap), so this never exceeds the page-wide cap.
    size = max(size, _MIN_FONT_SIZE)
    max_width = box_w - 2

    scalable = _try_truetype(size)
    if scalable is not None:
        while size >= _MIN_FONT_SIZE:
            font = _try_truetype(size)
            if font is None:
                return None
            text_w, _ = _text_size(font, mock_value)
            if text_w <= max_width:
                return font, size
            size -= 1
        return None

    try:
        default = ImageFont.load_default()
        text_w, _ = _text_size(default, mock_value)
    except (OSError, AttributeError, ValueError):
        return None
    if text_w > max_width:
        return None
    return default, size


_BOX_FILL = "white"
_TEXT_FILL = "black"


def _page_font_size_cap(regions: Sequence[PaintedRegion]) -> int | None:
    """Median painted-box height across a page, scaled the same way
    ``_font_for_box`` derives a starting size from a single box's own
    height. Bank-statement-style tables sometimes yield one box whose
    height is an outlier — e.g. a table cell's word union spanning more
    than one OCR-detected line — which would otherwise paint its mock
    value visibly larger than every other, same-purpose redaction on the
    page. Capping to (a multiple of) the page's typical box height keeps
    mock text a consistent, readable size throughout the document.
    """
    heights = [
        int(r.padded_bbox.h)
        for r in regions
        if int(r.padded_bbox.h) >= _MIN_BOX_HEIGHT
        and getattr(r, "assignment_source", None) != "brand"
    ]
    if not heights:
        return None
    heights.sort()
    median_h = heights[len(heights) // 2]
    return max(_MIN_FONT_SIZE, int(0.7 * median_h * _FONT_CAP_SLACK))


def _paint_mock_value(
    draw: ImageDraw.ImageDraw,
    mock_value: str,
    x: int,
    y: int,
    w: int,
    h: int,
    max_size_cap: int | None = None,
) -> None:
    """Skip if mock empty or box too small; else centered mock_value text."""
    mock = (mock_value or "").strip()
    if not mock:
        return
    fitted = _font_for_box(mock, w, h, max_size_cap)
    if fitted is None:
        return
    font, _size = fitted
    bbox = draw.textbbox((0, 0), mock, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    tx = x + (w - text_w) // 2 - bbox[0]
    ty = y + (h - text_h) // 2 - bbox[1]
    draw.text((tx, ty), mock, fill=_TEXT_FILL, font=font)


def _draw_on_image(image: Image.Image, regions: Sequence[PaintedRegion]) -> Image.Image:
    """Copy image (RGB). For each region: white rect, then centered black
    mock_value. No outline — a hairline border on a table cell sits on the
    gridline and makes the rule look broken; the white fill plus mock text
    is enough to mark the redaction, and cell-inset padding keeps the fill
    off the rule itself.
    """
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    font_cap = _page_font_size_cap(regions)
    for region in regions:
        box = region.padded_bbox
        x, y, w, h = int(box.x), int(box.y), int(box.w), int(box.h)
        if w <= 0 or h <= 0:
            continue
        draw.rectangle([x, y, x + w, y + h], fill=_BOX_FILL)
        mock = getattr(region, "mock_value", "") or ""
        if not str(mock).strip():
            continue
        _paint_mock_value(draw, str(mock), x, y, w, h, font_cap)
    return out


_VALID_IMAGE_FORMATS = ("jpeg", "png")


@dataclass
class PageRenderInput:
    """One page's render-time inputs.

    ``image`` is always available (every page gets rasterized upstream
    for OCR/structure purposes regardless of kind — see
    ``app/pipeline/redact.py``). ``fitz_page`` is only set for a real
    PDF page PyMuPDF itself rendered; combined with ``page_kind ==
    "digital"`` it lets the renderer redact and keep that page as real
    vector content instead of flattening it to an image like a scanned
    page. ``dpi`` is the render DPI ``image``/``padded_bbox`` pixel
    coordinates are in — needed to convert back to the source page's
    point space when redacting ``fitz_page`` directly.
    """

    image: Image.Image
    page_kind: str = "scanned"
    fitz_page: object | None = None
    dpi: int = 200


def _normalize_page_input(item: "Image.Image | PageRenderInput") -> PageRenderInput:
    """Accept a bare ``PIL.Image`` (existing callers/tests — always
    treated as a plain raster page) or a full ``PageRenderInput``.
    """
    if isinstance(item, Image.Image):
        return PageRenderInput(image=item)
    return item


def _append_raster_page(
    doc: fitz.Document,
    image: Image.Image,
    regions: Sequence[PaintedRegion],
    fmt: str,
    jpeg_quality: int,
) -> None:
    redacted = _draw_on_image(image, regions)
    buf = io.BytesIO()
    if fmt == "jpeg":
        redacted.save(buf, format="JPEG", quality=jpeg_quality)
    else:
        redacted.save(buf, format="PNG")
    image_bytes = buf.getvalue()
    rect = fitz.Rect(0, 0, image.width, image.height)
    page = doc.new_page(width=image.width, height=image.height)
    page.insert_image(rect, stream=image_bytes)


def _append_vector_page(
    doc: fitz.Document,
    fitz_page: object,
    regions: Sequence[PaintedRegion],
    dpi: int,
) -> None:
    """Clone the single source page into a throwaway document, redact
    the clone — text first, brand/image zones second, see
    ``pdf_native_redactor`` — and merge the redacted clone into ``doc``.

    Cloning first, rather than mutating ``fitz_page`` in place, keeps
    the long-lived source ``fitz.Document`` (held alive by
    ``PageProcessState._fitz_doc`` for the whole session) pristine, so a
    session's ``regenerate()`` can re-render from scratch as many times
    as needed — the same guarantee the raster path gets for free from
    ``image.copy()`` inside ``_draw_on_image`` on every call.
    """
    clone_doc = fitz.open()
    try:
        clone_doc.insert_pdf(fitz_page.parent, from_page=fitz_page.number, to_page=fitz_page.number)
        clone_page = clone_doc[0]
        redact_text_regions(clone_page, regions, dpi)
        redact_image_regions(clone_page, regions, dpi)
        doc.insert_pdf(clone_doc, from_page=0, to_page=0)
    finally:
        clone_doc.close()


def render_redacted_pdf(
    pages: Sequence["Image.Image | PageRenderInput"],
    redactions: Sequence[PaintedRegion],
    filename: str,
    image_format: str = "jpeg",
    jpeg_quality: int = 85,
) -> bytes:
    """Assemble the output PDF page by page.

    A page is redacted as real vector content (see
    ``_append_vector_page``) when it's a ``PageRenderInput`` with
    ``page_kind == "digital"`` and a ``fitz_page``; every other page —
    a bare ``Image``, a non-digital ``PageRenderInput``, or a
    ``PageRenderInput`` whose ``fitz_page`` is unavailable (non-PDF
    input, or the pdf2image fallback path) — gets the white rect + black
    mock_value paint-then-flatten treatment. Brand zones are drawn last
    within each page's own region list. Mixed documents (e.g. a digital
    page followed by a scanned one) come out with the matching page type
    at each position, in original order.

    Args:
        pages: Per-page render inputs (or bare ``PIL.Image``\\s — see
            ``_normalize_page_input``).
        redactions: Duck-typed regions with page + padded_bbox.
        filename: Original upload name; used only as sanitized PDF title.
        image_format: "jpeg" (default) or "png" for a raster page's
            embedded image. A scanned/photographed page's noise
            compresses far worse under lossless PNG (often 5-10x larger)
            than JPEG, for no visible quality difference once redaction
            boxes are already painted on. Never used for a vector page.
        jpeg_quality: 1-95 Pillow JPEG quality, only used when
            ``image_format == "jpeg"``.

    Returns:
        Assembled PDF bytes.
    """
    fmt = image_format.lower()
    if fmt not in _VALID_IMAGE_FORMATS:
        logger.warning("unknown_redact_image_format format=%s falling_back=jpeg", image_format)
        fmt = "jpeg"

    normalized = [_normalize_page_input(p) for p in pages]

    ordered = _regions_in_draw_order(redactions)
    by_page: dict[int, list[PaintedRegion]] = {}
    for region in ordered:
        by_page.setdefault(region.page, []).append(region)

    doc = fitz.open()
    vector_page_count = 0
    try:
        for page_idx, page_input in enumerate(normalized):
            page_regions = by_page.get(page_idx, [])
            used_vector = False
            if page_input.page_kind == "digital" and page_input.fitz_page is not None:
                try:
                    _append_vector_page(doc, page_input.fitz_page, page_regions, page_input.dpi)
                    used_vector = True
                    vector_page_count += 1
                except Exception:
                    # Fail safe to the raster path for this one page rather
                    # than aborting the whole document — mirrors native_text
                    # .py's own "any extraction failure -> treat as scanned"
                    # rule. Genuine fitz.Page objects always support this;
                    # this only ever triggers for a page whose fitz_page is
                    # somehow not real vector content (should not happen in
                    # production, since classify_and_extract already required
                    # a working fitz.Page to reach "digital" in the first
                    # place — this is the same defense-in-depth as that rule).
                    logger.warning(
                        "vector_redaction_failed page_index=%s falling_back=raster",
                        page_idx,
                        exc_info=True,
                    )
            if not used_vector:
                _append_raster_page(doc, page_input.image, page_regions, fmt, jpeg_quality)

        doc.set_metadata({"title": _sanitize_filename(filename)})
        pdf_bytes = doc.tobytes()
    finally:
        doc.close()

    logger.info(
        "redacted_pdf_rendered page_count=%s vector_page_count=%s region_count=%s "
        "image_format=%s filename_hash=%s",
        len(normalized),
        vector_page_count,
        len(redactions),
        fmt,
        hashlib.sha256(filename.encode("utf-8", errors="replace")).hexdigest(),
    )
    return pdf_bytes
