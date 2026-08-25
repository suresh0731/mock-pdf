"""Paint white redaction boxes and black mock values onto original pages."""

import hashlib
import io
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import fitz
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

_MIN_FONT_SIZE = 6
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
    if size < _MIN_FONT_SIZE:
        return None
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


def render_redacted_pdf(
    original_pages: list[Image.Image],
    redactions: Sequence[PaintedRegion],
    filename: str,
    image_format: str = "jpeg",
    jpeg_quality: int = 85,
) -> bytes:
    """White rect + black mock_value. Brand zones drawn last. PyMuPDF assemble.

    Args:
        original_pages: Unmodified page images to paint on.
        redactions: Duck-typed regions with page + padded_bbox.
        filename: Original upload name; used only as sanitized PDF title.
        image_format: "jpeg" (default) or "png" for the embedded page image.
            A scanned/photographed page's noise compresses far worse under
            lossless PNG (often 5-10x larger) than JPEG, for no visible
            quality difference once redaction boxes are already painted on.
        jpeg_quality: 1-95 Pillow JPEG quality, only used when
            ``image_format == "jpeg"``.

    Returns:
        Assembled PDF bytes.
    """
    fmt = image_format.lower()
    if fmt not in _VALID_IMAGE_FORMATS:
        logger.warning("unknown_redact_image_format format=%s falling_back=jpeg", image_format)
        fmt = "jpeg"

    ordered = _regions_in_draw_order(redactions)
    by_page: dict[int, list[PaintedRegion]] = {}
    for region in ordered:
        by_page.setdefault(region.page, []).append(region)

    doc = fitz.open()
    try:
        for page_idx, image in enumerate(original_pages):
            redacted = _draw_on_image(image, by_page.get(page_idx, []))
            buf = io.BytesIO()
            if fmt == "jpeg":
                redacted.save(buf, format="JPEG", quality=jpeg_quality)
            else:
                redacted.save(buf, format="PNG")
            image_bytes = buf.getvalue()
            rect = fitz.Rect(0, 0, image.width, image.height)
            page = doc.new_page(width=image.width, height=image.height)
            page.insert_image(rect, stream=image_bytes)

        doc.set_metadata({"title": _sanitize_filename(filename)})
        pdf_bytes = doc.tobytes()
    finally:
        doc.close()

    logger.info(
        "redacted_pdf_rendered page_count=%s region_count=%s image_format=%s filename_hash=%s",
        len(original_pages),
        len(redactions),
        fmt,
        hashlib.sha256(filename.encode("utf-8", errors="replace")).hexdigest(),
    )
    return pdf_bytes
