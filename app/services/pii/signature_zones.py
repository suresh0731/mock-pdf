"""Signature-ink zone detection: the blank vertical gap immediately above
or below an already-detected signatory name/org row, plus native PDF
signature graphics on digital pages.

Docling's picture-block classifier (``brand_zones.detect_picture_zones``)
is trained on printed graphics — logos, photos, charts — and only
inconsistently recognizes handwritten signature ink as a ``picture``
block: verified directly against ``repii/`` sample scans, where two pen
signatures side by side on the same page had one boxed by Docling and the
other missed entirely (the second stroke wasn't dense/graphic-shaped
enough to trip the layout model). Adding a second ML model just for
signatures would trade one unreliable classifier for another dependency
to vendor/pin (see README's "Model weight versions" table) without fixing
the underlying issue: cursive ink on blank paper doesn't look like the
logos/photos either model was trained on.

Anchoring instead on the signatory name/org row ``field_extractor``
already finds in the bottom-of-page signature block
(``find_signature_anchor_bboxes``) needs no CV thresholds, no new
dependency, and no model: the ink zone is simply the blank space between
that anchor and whatever real OCR-recognized content (if any) sits
immediately above or below it in the same horizontal band, capped so it
can never balloon into unrelated page whitespace on a mostly-empty page.

On a digital (copyable-text) page that path is not enough: the signature
is often a real PDF object — an embedded image, a Stamp/Ink annotation,
a signature widget appearance, or vector ink paths — and native text
extracted from a stamp (e.g. ``BNI`` inside a seal) sits in the same
band and shrinks the ink-gap to nothing. When a ``fitz.Page`` is passed
in, this module also boxes those native graphics in the bottom-of-page
signature band so the vector redactor can delete them.

Never logs redacted text — only counts and integer bbox coords (SEC-001).
"""

from __future__ import annotations

import logging
from typing import Any

from app.models.pii_chunk import BBox
from app.services.ocr.ensemble_types import EnsembleWord
from app.services.pii.brand_zones import BrandZone
from app.services.pii.field_extractor import build_rows, find_signature_anchor_bboxes

logger = logging.getLogger(__name__)

# How far past the anchor row's own x-range (fraction of the anchor's own
# width, floored at a fixed pixel minimum) the ink-column search extends —
# a handwritten signature commonly runs a bit wider than the printed name
# below/above it.
_ZONE_X_MARGIN_FRACTION = 0.6
_ZONE_X_MARGIN_MIN_PX = 20
# Longest ink gap ever painted, as a multiple of the anchor row's own
# height — bounds the zone to roughly a plausible signature height so it
# can never grow into unrelated whitespace toward the top/bottom of a
# mostly-empty page (e.g. a signature anchor with nothing else below it
# all the way to the page edge).
_ZONE_MAX_GAP_HEIGHT_FACTOR = 6.0
# Anchor rows sitting flush against neighboring content (no room to sign)
# are skipped rather than painting a sliver.
_ZONE_MIN_GAP_PX = 6.0
# An ink zone already substantially covered by an existing footer/picture
# zone (e.g. Docling did catch this particular signature as a picture) is
# skipped so it isn't painted a second time.
_COVERAGE_SKIP_THRESHOLD = 0.6
# Same bottom-of-page band field_extractor uses for signatory rows. Native
# PDF images/annots/drawings only become signature zones if they reach this
# strip — a letterhead logo at the top is ``patch_images``, not a signature.
_NATIVE_BOTTOM_FRACTION = 0.3
# Skip a native image that is basically the whole page (a scanned-page
# XObject, not a signature stamp sitting on a digital letter).
_NATIVE_IMAGE_MAX_PAGE_FRACTION = 0.4
# Skip hairline table rules / tiny icons. Pixel floor is applied after
# point→pixel scaling so it tracks ``RedactOptions.dpi``.
_NATIVE_GRAPHIC_MIN_WIDTH_PX = 24
_NATIVE_GRAPHIC_MIN_HEIGHT_PX = 12
_NATIVE_GRAPHIC_MIN_PAGE_FRACTION = 0.0008
# Vector ink (Bezier strokes) vs table gridlines: a signature drawing
# has several curve items; a cell border is one thin rectangle.
_NATIVE_INK_MIN_CURVES = 3
_NATIVE_INK_ANNOT_TYPES = frozenset({"Stamp", "Ink"})
_PDF_WIDGET_TYPE_SIGNATURE = 7


def _overlap_area(a: BBox, b: BBox) -> int:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


def _covered_by_existing(bbox: BBox, existing_zones: list[BrandZone]) -> bool:
    area = bbox.w * bbox.h
    if area <= 0:
        return False
    return any(
        _overlap_area(bbox, zone.bbox) / area >= _COVERAGE_SKIP_THRESHOLD
        for zone in existing_zones
    )


def _clamp_bbox(bbox: BBox, page_w: int, page_h: int) -> BBox | None:
    x1 = max(0, bbox.x)
    y1 = max(0, bbox.y)
    x2 = min(page_w, bbox.x + bbox.w)
    y2 = min(page_h, bbox.y + bbox.h)
    if x2 <= x1 or y2 <= y1:
        return None
    return BBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)


def _in_band(word: EnsembleWord, left: float, right: float) -> bool:
    """True for a real (non-punctuation-noise) word overlapping [left, right).

    OCR occasionally hallucinates a tiny punctuation-only detection on a
    cursive stroke (a stray ``"-"``/``"'"``); requiring at least one
    alphanumeric character keeps such noise from being trusted as real
    bounding content and prematurely shrinking the ink gap.
    """
    if not any(ch.isalnum() for ch in word.text):
        return False
    return word.bbox.x + word.bbox.w > left and word.bbox.x < right


def _ink_gap_zones(anchor: BBox, words: list[EnsembleWord]) -> list[BBox]:
    """Blank-gap boxes directly above and/or below ``anchor``.

    Each is capped at ``_ZONE_MAX_GAP_HEIGHT_FACTOR`` times the anchor's
    own height and bounded by the nearest neighboring word's edge in the
    same horizontal band — so the zone only ever covers blank paper,
    never another word's own text. The anchor's own words never qualify
    as their own "neighboring content": every word in the anchor's row
    shares its y-range, so none can satisfy the strict above/below
    comparisons below.
    """
    x_margin = max(_ZONE_X_MARGIN_MIN_PX, anchor.w * _ZONE_X_MARGIN_FRACTION)
    left = anchor.x - x_margin
    right = anchor.x + anchor.w + x_margin
    max_gap = max(anchor.h, 1) * _ZONE_MAX_GAP_HEIGHT_FACTOR

    above_bottoms = [
        w.bbox.y + w.bbox.h
        for w in words
        if _in_band(w, left, right) and w.bbox.y + w.bbox.h <= anchor.y
    ]
    below_tops = [
        w.bbox.y
        for w in words
        if _in_band(w, left, right) and w.bbox.y >= anchor.y + anchor.h
    ]

    zones: list[BBox] = []
    gap_above = min(anchor.y - max(above_bottoms), max_gap) if above_bottoms else max_gap
    if gap_above >= _ZONE_MIN_GAP_PX:
        zones.append(
            BBox(x=int(left), y=int(anchor.y - gap_above), w=int(right - left), h=int(gap_above))
        )

    gap_below = (
        min(min(below_tops) - (anchor.y + anchor.h), max_gap) if below_tops else max_gap
    )
    if gap_below >= _ZONE_MIN_GAP_PX:
        zones.append(
            BBox(x=int(left), y=int(anchor.y + anchor.h), w=int(right - left), h=int(gap_below))
        )
    return zones


def _points_to_pixels_scale(dpi: int) -> float:
    return dpi / 72.0 if dpi else 1.0


def _rect_to_bbox(rect: Any, dpi: int) -> BBox | None:
    """PyMuPDF rect (PDF points, top-left origin) → pixel ``BBox``."""
    try:
        x0, y0, x1, y1 = float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)
    except (AttributeError, TypeError, ValueError):
        return None
    scale = _points_to_pixels_scale(dpi)
    x = int(round(x0 * scale))
    y = int(round(y0 * scale))
    w = int(round((x1 - x0) * scale))
    h = int(round((y1 - y0) * scale))
    if w <= 0 or h <= 0:
        return None
    return BBox(x=x, y=y, w=w, h=h)


def _in_bottom_band(bbox: BBox, page_h: int) -> bool:
    if page_h <= 0:
        return False
    threshold_y = page_h * (1 - _NATIVE_BOTTOM_FRACTION)
    return bbox.y + bbox.h > threshold_y


def _plausible_native_graphic(bbox: BBox, page_w: int, page_h: int) -> bool:
    area = bbox.w * bbox.h
    page_area = page_w * page_h
    if page_area <= 0 or area <= 0:
        return False
    if area / page_area >= _NATIVE_IMAGE_MAX_PAGE_FRACTION:
        return False
    if bbox.w < _NATIVE_GRAPHIC_MIN_WIDTH_PX or bbox.h < _NATIVE_GRAPHIC_MIN_HEIGHT_PX:
        return False
    if area / page_area < _NATIVE_GRAPHIC_MIN_PAGE_FRACTION:
        return False
    return True


def _append_native_zone(
    bbox: BBox,
    page: int,
    page_w: int,
    page_h: int,
    existing: list[BrandZone],
    result: list[BrandZone],
    *,
    require_bottom_band: bool,
) -> None:
    clamped = _clamp_bbox(bbox, page_w, page_h)
    if clamped is None or not _plausible_native_graphic(clamped, page_w, page_h):
        return
    if require_bottom_band and not _in_bottom_band(clamped, page_h):
        return
    skip_against = [*existing, *result]
    if _covered_by_existing(clamped, skip_against):
        return
    result.append(BrandZone(zone="signature", page=page, bbox=clamped, label="SIGNATURE"))


def _native_image_boxes(fitz_page: Any, dpi: int) -> list[BBox]:
    try:
        infos = fitz_page.get_image_info()
    except Exception:
        logger.warning("signature_zones: get_image_info failed", exc_info=True)
        return []
    boxes: list[BBox] = []
    for info in infos or []:
        bbox = info.get("bbox") if isinstance(info, dict) else None
        if bbox is None:
            continue
        converted = _rect_to_bbox(_RectView(bbox), dpi)
        if converted is not None:
            boxes.append(converted)
    return boxes


class _RectView:
    """Minimal ``.x0/.y0/.x1/.y1`` wrapper over a 4-tuple or fitz.Rect."""

    __slots__ = ("x0", "y0", "x1", "y1")

    def __init__(self, rect: Any) -> None:
        if hasattr(rect, "x0"):
            self.x0, self.y0, self.x1, self.y1 = rect.x0, rect.y0, rect.x1, rect.y1
        else:
            self.x0, self.y0, self.x1, self.y1 = (float(v) for v in rect[:4])


def _native_annot_boxes(fitz_page: Any, dpi: int) -> list[BBox]:
    boxes: list[BBox] = []
    try:
        widgets = fitz_page.widgets()
    except Exception:
        widgets = None
    if widgets:
        for widget in widgets:
            field_type = getattr(widget, "field_type", None)
            if field_type != _PDF_WIDGET_TYPE_SIGNATURE:
                continue
            converted = _rect_to_bbox(getattr(widget, "rect", None), dpi)
            if converted is not None:
                boxes.append(converted)
    try:
        annots = fitz_page.annots()
    except Exception:
        annots = None
    if annots:
        for annot in annots:
            try:
                kind = annot.type[1]
            except (AttributeError, TypeError, IndexError):
                continue
            if kind not in _NATIVE_INK_ANNOT_TYPES:
                continue
            converted = _rect_to_bbox(getattr(annot, "rect", None), dpi)
            if converted is not None:
                boxes.append(converted)
    return boxes


def _native_ink_drawing_boxes(fitz_page: Any, dpi: int) -> list[BBox]:
    try:
        drawings = fitz_page.get_drawings()
    except Exception:
        logger.warning("signature_zones: get_drawings failed", exc_info=True)
        return []
    boxes: list[BBox] = []
    for drawing in drawings or []:
        items = drawing.get("items") or []
        curve_count = sum(1 for item in items if item and item[0] == "c")
        if curve_count < _NATIVE_INK_MIN_CURVES:
            continue
        converted = _rect_to_bbox(drawing.get("rect"), dpi)
        if converted is not None:
            boxes.append(converted)
    return boxes


def _native_graphic_zones(
    fitz_page: Any,
    dpi: int,
    page_w: int,
    page_h: int,
    page: int,
    existing: list[BrandZone],
) -> list[BrandZone]:
    """Signature-shaped native PDF objects on a digital page.

    Independent of ink-gap geometry: stamp text in the native text layer
    must not be able to shrink these away. Errors collecting any one
    source (images / annots / drawings) are swallowed so a single
    PyMuPDF failure cannot drop the other two.
    """
    result: list[BrandZone] = []
    for box in _native_image_boxes(fitz_page, dpi):
        _append_native_zone(box, page, page_w, page_h, existing, result, require_bottom_band=True)
    for box in _native_annot_boxes(fitz_page, dpi):
        # Widgets/Stamp/Ink *are* signatures by PDF type — not restricted
        # to the bottom strip the way a generic embedded image is.
        _append_native_zone(box, page, page_w, page_h, existing, result, require_bottom_band=False)
    for box in _native_ink_drawing_boxes(fitz_page, dpi):
        _append_native_zone(box, page, page_w, page_h, existing, result, require_bottom_band=True)
    return result


def detect_signature_zones(
    ensemble_words: list[EnsembleWord],
    page_w: int,
    page_h: int,
    page: int,
    *,
    existing_zones: list[BrandZone] | None = None,
    enabled: bool = True,
    fitz_page: Any | None = None,
    dpi: int = 200,
) -> list[BrandZone]:
    """Ink-gap zones beside every detected bottom-of-page signatory anchor.

    Pure OCR-word geometry — no Docling, no OpenCV, no ML model. Anchors
    on the same signatory name/org rows ``field_extractor`` already
    recognizes (``find_signature_anchor_bboxes``), then paints the blank
    space immediately above and/or below each one, capped to a plausible
    signature height and clipped at the nearest real content on either
    side, so a handwritten signature gets covered even when Docling's
    picture classifier misses it (see module docstring).

    When ``fitz_page`` is provided (digital/vector pages), also boxes
    native signature graphics in the bottom-of-page band: embedded
    images, Stamp/Ink annotations, signature widgets, and curve-heavy
    vector ink. Those survive even if stamp text would have collapsed the
    ink-gap.

    Args:
        ensemble_words: Aligned OCR words for one page.
        page_w: Original page width in pixels.
        page_h: Original page height in pixels.
        page: Zero-based page index (logged only).
        existing_zones: Zones already computed (footer/picture) — an ink
            zone already substantially covered by one of these (Docling
            did catch this particular signature) is skipped.
        enabled: When False, return ``[]`` without inspecting words.
        fitz_page: Source ``fitz.Page`` for native graphic detection.
            ``None`` (scanned/image input) skips that extra pass.
        dpi: Render DPI used to scale PDF-point rects into pixel space.

    Returns:
        One ``BrandZone`` (``zone="signature"``, ``label="SIGNATURE"``)
        per surviving ink gap / native graphic. Invalid page size or
        ``enabled=False`` returns ``[]``.
    """
    if not enabled or page_w <= 0 or page_h <= 0:
        return []
    if not ensemble_words and fitz_page is None:
        return []

    resolved_existing = existing_zones or []
    result: list[BrandZone] = []
    if ensemble_words:
        rows = build_rows(ensemble_words)
        anchors = find_signature_anchor_bboxes(rows, page_h)
        for anchor in anchors:
            for zone_bbox in _ink_gap_zones(anchor, ensemble_words):
                clamped = _clamp_bbox(zone_bbox, page_w, page_h)
                if clamped is None or _covered_by_existing(clamped, resolved_existing):
                    continue
                result.append(
                    BrandZone(zone="signature", page=page, bbox=clamped, label="SIGNATURE")
                )

    if fitz_page is not None:
        native = _native_graphic_zones(
            fitz_page, dpi, page_w, page_h, page, [*resolved_existing, *result]
        )
        result.extend(native)

    logger.info("signature_zones_detected page=%s count=%s", page, len(result))
    logger.debug(
        "signature_zones_bboxes page=%s boxes=%s",
        page,
        [(z.bbox.x, z.bbox.y, z.bbox.w, z.bbox.h) for z in result],
    )
    return result
