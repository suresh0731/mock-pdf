"""Canonical→original inverse map and blur-tier padding.

Padding uses local 4/8/12 defaults. Does not read ``get_settings``.
Logs integer geometry only (SEC-001).
"""

import logging

from app.models.pii_chunk import BBox
from app.models.redact import BlurTier, PageTransform

logger = logging.getLogger(__name__)

PADDING_PX_BY_TIER: dict[str, int] = {"good": 4, "mild": 8, "severe": 12}
# Inset between a cell-clamped painted box and the cell's own border
# line. Sized to survive a couple of degrees of residual page skew
# (deskew is rotation-only; a ~80px-wide cell at 2° of leftover tilt
# shifts ~3px at the far edge) so the white patch never paints over
# the gridline itself.
_CELL_CLAMP_MARGIN_PX = 3


def canonical_to_original(bbox: BBox, transform: PageTransform) -> BBox:
    """original = canonical + (dx, dy); size unchanged.

    Args:
        bbox: Bounding box in canonical (post-preprocess) space.
        transform: Page crop offset. Negative ``dx``/``dy`` are allowed.

    Returns:
        Translated box. Not clamped; padding/clamp is ``apply_padding``.
    """
    mapped = BBox(
        x=bbox.x + transform.dx,
        y=bbox.y + transform.dy,
        w=bbox.w,
        h=bbox.h,
    )
    logger.debug(
        "canonical bbox translated",
        extra={
            "dx": transform.dx,
            "dy": transform.dy,
            "x": mapped.x,
            "y": mapped.y,
            "w": mapped.w,
            "h": mapped.h,
        },
    )
    return mapped


def _cell_inner_edges(cell: BBox, margin: int) -> tuple[float, float, float, float] | None:
    """Interior of ``cell`` inset by ``margin``, or the cell itself when
    there isn't room for the margin. ``None`` if the cell has no area."""
    if cell.w < 1 or cell.h < 1:
        return None
    x1 = cell.x + margin
    y1 = cell.y + margin
    x2 = cell.x + cell.w - margin
    y2 = cell.y + cell.h - margin
    if x2 > x1 and y2 > y1:
        return x1, y1, x2, y2
    return float(cell.x), float(cell.y), float(cell.x + cell.w), float(cell.y + cell.h)


# Minimum fraction of the *detected word's own height* (``bbox``, before
# padding) that a cell-clipped box must still retain vertically to be
# trusted. Deliberately checked on the vertical axis only, not full area:
# clipping a bbox's *width* down to a cell's own column bounds is the
# common, wanted case (an OCR union bbox that spilled sideways into a
# neighboring column must still shrink to that column, regardless of how
# little of its width survives — see
# ``test_apply_padding_hard_clips_ocr_overflow_to_cell``). Clipping a
# bbox's *height* down is a different, much riskier failure mode: it
# means the chosen cell (see ``redact.py``'s ``_enclosing_cell_bbox``)
# spans noticeably less than the word's own detected vertical extent,
# which real cells for a word's own row never do (a correctly-sized cell
# always covers its own row's full height) — in practice this happens
# when img2table misreads a faint/broken gridline and emits an
# undersized cell, or a multi-row word union gets matched to a
# single-row cell by a majority vote. Either way, trusting that clip
# would hard-crop the painted white box to a sliver and leave most of
# the real, detected PII text exposed outside it. When the clip would
# retain less than this fraction of the word's own height, the cell is
# treated as untrusted for the vertical axis and padding falls back to
# the neighbor/page-edge path below, which never shrinks past ``bbox``'s
# own vertical extent.
_MIN_CELL_CLIP_HEIGHT_COVERAGE = 0.6


def _bbox_height_fraction_in_range(bbox: BBox, y1: float, y2: float) -> float:
    """Fraction of ``bbox``'s own height inside the ``[y1, y2]`` vertical band."""
    iy1, iy2 = max(bbox.y, y1), min(bbox.y + bbox.h, y2)
    if iy2 <= iy1 or bbox.h <= 0:
        return 0.0
    return (iy2 - iy1) / bbox.h


def apply_padding(
    bbox: BBox,
    tier: BlurTier,
    page_w: int,
    page_h: int,
    *,
    cell_bbox: BBox | None = None,
    left_neighbor_x: float | None = None,
    right_neighbor_x: float | None = None,
) -> BBox:
    """Expand by tier px on each edge; clamp to containment context; w/h at least 1.

    Padding is clamped, in priority order, against:

    1. ``cell_bbox`` — the enclosing table-cell bbox, when the redacted
       span's words are known to belong to one (minus a safety margin, so
       the box never paints over the cell's own border line). This is a
       *hard* clip: an OCR/union bbox that spilled into a neighboring
       column is shrunk back into this cell rather than covering the
       neighbor's value. Takes priority over ``left_neighbor_x``/
       ``right_neighbor_x``. A cell that does not overlap ``bbox`` at all
       is ignored (treated as no cell), so a wildly wrong cell estimate
       cannot erase the detected word.
    2. ``left_neighbor_x``/``right_neighbor_x`` — the pre-computed
       midpoint x-position toward the nearest non-redacted word on the
       same visual row (prose/label/signature text with no table-cell
       context), reusing the same midpoint-clamp technique
       ``field_extractor._split_multiword_tokens`` uses for its interior
       sub-token boundaries.
    3. The page edge — the final fallback when neither of the above is
       available, matching this function's original (pre-cell/neighbor-
       aware) behavior exactly.

    When no usable ``cell_bbox`` is present, the result never shrinks
    past ``bbox`` itself: a too-tight neighbor estimate can only reduce
    the padding down to zero, never eat into the actual detected word.

    Args:
        bbox: Box in original-scan coordinates.
        tier: Blur tier ``good`` / ``mild`` / ``severe``.
        page_w: Page width in pixels.
        page_h: Page height in pixels.
        cell_bbox: Optional enclosing table-cell bbox, in the same
            (original-scan) coordinate space as ``bbox``.
        left_neighbor_x: Optional x-position (original-scan space) the
            padded box's left edge must not cross, when no ``cell_bbox``
            is known.
        right_neighbor_x: Optional x-position (original-scan space) the
            padded box's right edge must not cross, when no ``cell_bbox``
            is known.

    Returns:
        Padded and clamped box. Invalid page size still clamps (may be 1×1).
    """
    pad = PADDING_PX_BY_TIER.get(tier, 4)
    x1 = bbox.x - pad
    y1 = bbox.y - pad
    x2 = bbox.x + bbox.w + pad
    y2 = bbox.y + bbox.h + pad

    used_cell_clip = False
    if cell_bbox is not None:
        inner = _cell_inner_edges(cell_bbox, _CELL_CLAMP_MARGIN_PX)
        if inner is not None:
            ix1, iy1, ix2, iy2 = inner
            # Ignore a cell that shares no area with the word box — a bad
            # cell assignment must not be allowed to clip the word away.
            word_hits_cell = (
                min(bbox.x + bbox.w, cell_bbox.x + cell_bbox.w) > max(bbox.x, cell_bbox.x)
                and min(bbox.y + bbox.h, cell_bbox.y + cell_bbox.h) > max(bbox.y, cell_bbox.y)
            )
            if word_hits_cell:
                clipped_x1 = max(x1, ix1)
                clipped_y1 = max(y1, iy1)
                clipped_x2 = min(x2, ix2)
                clipped_y2 = min(y2, iy2)
                if (
                    clipped_x2 > clipped_x1
                    and clipped_y2 > clipped_y1
                    and _bbox_height_fraction_in_range(bbox, clipped_y1, clipped_y2)
                    >= _MIN_CELL_CLIP_HEIGHT_COVERAGE
                ):
                    x1, y1, x2, y2 = clipped_x1, clipped_y1, clipped_x2, clipped_y2
                    used_cell_clip = True

    if not used_cell_clip:
        if left_neighbor_x is not None:
            x1 = max(x1, left_neighbor_x)
        if right_neighbor_x is not None:
            x2 = min(x2, right_neighbor_x)
        # Never shrink past the box's own extent when we don't have a
        # trusted cell to clip to.
        x1 = min(x1, bbox.x)
        y1 = min(y1, bbox.y)
        x2 = max(x2, bbox.x + bbox.w)
        y2 = max(y2, bbox.y + bbox.h)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(page_w, x2)
    y2 = min(page_h, y2)
    padded = BBox(x=int(x1), y=int(y1), w=max(1, int(x2 - x1)), h=max(1, int(y2 - y1)))
    logger.debug(
        "bbox padded and clamped",
        extra={
            "tier": tier,
            "pad": pad,
            "page_w": page_w,
            "page_h": page_h,
            "cell_clamped": used_cell_clip,
            "neighbor_clamped": (not used_cell_clip)
            and (left_neighbor_x is not None or right_neighbor_x is not None),
            "x": padded.x,
            "y": padded.y,
            "w": padded.w,
            "h": padded.h,
        },
    )
    return padded
