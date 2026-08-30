"""Canonical→original inverse map and blur-tier padding.

Padding uses local 4/8/12 defaults. Does not read ``get_settings``.
Logs integer geometry only (SEC-001).
"""

import logging
import math

from app.models.pii_chunk import BBox
from app.models.redact import BlurTier, PageTransform

logger = logging.getLogger(__name__)

PADDING_PX_BY_TIER: dict[str, int] = {"good": 4, "mild": 8, "severe": 12}
# Inset between a cell-clamped painted box and the cell's own border
# line. Sized to survive a couple of degrees of residual page skew
# (deskew is rotation-only; a ~80px-wide cell at 2° of leftover tilt
# shifts ~3px at the far edge) so the white patch never paints over
# the gridline itself. Used as the floor for the skew-scaled margin
# below (``_cell_skew_margin_px``), and directly wherever a skew angle
# isn't available (defaults to 0.0, see ``apply_padding``).
_CELL_CLAMP_MARGIN_PX = 3
# A cell-clamped box now fills the *entire* cell interior (see
# ``apply_padding``'s docstring), not just the padded word box — so its
# edges sit right next to the cell's own gridline on every side, not
# only where the word happened to reach. A larger cell therefore needs a
# larger margin to absorb the same residual tilt. Angle is capped here
# (a residual well past this most likely means a bad skew estimate, not
# a genuinely more tilted scan) and so is the resulting margin, so a
# large cell on a heavily-skewed page still can't eat an unreasonable
# chunk of its own interior.
_MAX_CELL_SKEW_MARGIN_ANGLE_DEG = 5.0
_MAX_CELL_SKEW_MARGIN_PX = 14


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


def _cell_skew_margin_px(cell: BBox, skew_angle_deg: float) -> float:
    """Border-line safety margin for a full-cell-fill white patch.

    A point near one edge of a ``w``x``h`` cell shifts roughly
    ``dimension * sin(angle)`` pixels relative to the opposite edge under
    ``angle`` degrees of residual rotation, so the margin scales with the
    cell's own larger dimension rather than staying a flat few pixels —
    see the module-level comment on ``_MAX_CELL_SKEW_MARGIN_ANGLE_DEG``
    for why the angle and the resulting margin are both capped.
    """
    angle = min(abs(skew_angle_deg), _MAX_CELL_SKEW_MARGIN_ANGLE_DEG)
    shift = max(cell.w, cell.h) * math.sin(math.radians(angle))
    return min(_MAX_CELL_SKEW_MARGIN_PX, max(_CELL_CLAMP_MARGIN_PX, shift))


def _cell_inner_edges(cell: BBox, margin: float) -> tuple[float, float, float, float] | None:
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


def apply_padding(
    bbox: BBox,
    tier: BlurTier,
    page_w: int,
    page_h: int,
    *,
    cell_bbox: BBox | None = None,
    left_neighbor_x: float | None = None,
    right_neighbor_x: float | None = None,
    multiline: bool = False,
    redaction_id: str | None = None,
) -> BBox:
    """Expand by tier px on each edge; clamp to containment context; w/h at least 1.

    Padding is clamped, in priority order, against:

    1. ``cell_bbox`` — the enclosing table-cell bbox, when the redacted
       span's words are known to belong to one (minus a safety margin, so
       the box never paints over the cell's own border line). Horizontally
       this is a *hard* clip: an OCR/union bbox that spilled into a
       neighboring column is shrunk back into this cell rather than
       covering the neighbor's value — but only when ``bbox`` genuinely
       overflows the cell to begin with; when it already fits inside the
       cell, the margin only ever trims the *added* padding, never the
       real text (otherwise cell text sitting flush against its own
       left/right edge would lose a visible sliver on that side — a real
       redaction-leak bug this guards against). Vertically the clip is
       never hard: a cell a few px shorter than the word union it was
       matched to (common for a tightly packed multi-line row) must only
       reduce the added padding, never cut into the detected text's own
       height. Takes priority over ``left_neighbor_x``/``right_neighbor_x``.
       A cell that does not overlap ``bbox`` at all is ignored (treated as
       no cell), so a wildly wrong cell estimate cannot erase the detected
       word.
    2. ``left_neighbor_x``/``right_neighbor_x`` — the pre-computed
       midpoint x-position toward the nearest non-redacted word on the
       same visual row (prose/label/signature text with no table-cell
       context), reusing the same midpoint-clamp technique
       ``field_extractor._split_multiword_tokens`` uses for its interior
       sub-token boundaries.
    3. The page edge — the final fallback when neither of the above is
       available, matching this function's original (pre-cell/neighbor-
       aware) behavior exactly.

    In every path, the result never shrinks past ``bbox`` itself on the
    vertical axis, and never on the horizontal axis either unless ``bbox``
    already overflows ``cell_bbox`` horizontally: a too-tight neighbor/cell
    estimate can only reduce the padding down to zero, never eat into the
    actual detected word — the painted box must always fully cover the
    text it was computed from.

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
        multiline: When True, the redacted text already spans more than
            one visual line (e.g. a wrapped table-cell value), so no
            padding is added on the top/bottom edges — only the union's
            own vertical extent is used before any cell/neighbor clamp.
            Extra vertical slack on an already-tall, multi-line box risks
            bleeding into a tightly-packed neighboring row; left/right
            padding is unaffected.
        redaction_id: Optional caller-assigned correlation ID (e.g.
            ``"{page}:{start}:{end}"`` or a ``RedactionRegion.region_id``),
            stamped onto this call's debug/warning lines only so they can
            be joined with ``ensemble_mapper``'s "span mapped to bbox" and
            ``redaction_scorer``'s "redaction scored" lines for the same
            redaction. Never affects the returned geometry.

    Returns:
        Padded and clamped box. Invalid page size still clamps (may be 1×1).
    """
    pad = PADDING_PX_BY_TIER.get(tier, 4)
    pad_y = 0 if multiline else pad
    x1 = bbox.x - pad
    y1 = bbox.y - pad_y
    x2 = bbox.x + bbox.w + pad
    y2 = bbox.y + bbox.h + pad_y

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

                # Vertical: never trust the clip to cut into bbox itself.
                clipped_y1 = min(clipped_y1, bbox.y)
                clipped_y2 = max(clipped_y2, bbox.y + bbox.h)

                # Horizontal: only hard-clip below bbox's own extent when
                # bbox genuinely overflows the cell; when it already fits
                # (within a small jitter tolerance — the same margin used
                # for the border-line inset above, since an OCR union box
                # and a detected cell edge routinely differ by a couple of
                # px with no real spillover involved), floor back to bbox
                # so the border-line margin can't eat into flush-edge text.
                fits_x = (
                    bbox.x >= cell_bbox.x - _CELL_CLAMP_MARGIN_PX
                    and bbox.x + bbox.w <= cell_bbox.x + cell_bbox.w + _CELL_CLAMP_MARGIN_PX
                )
                if fits_x:
                    clipped_x1 = min(clipped_x1, bbox.x)
                    clipped_x2 = max(clipped_x2, bbox.x + bbox.w)

                if clipped_x2 > clipped_x1 and clipped_y2 > clipped_y1:
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
            "redaction_id": redaction_id,
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
    if used_cell_clip:
        _warn_if_cell_clamp_looks_oversized(bbox, padded, page_w, page_h, redaction_id=redaction_id)
    return padded


# A legitimate table cell can genuinely dwarf the short PII value sitting
# inside it (e.g. a wide "Bank Name" column holding a 3-letter value) — see
# test_apply_padding_neighbor_clamp_ignored_when_cell_bbox_present — so
# these thresholds intentionally stay generous and only *warn*, never
# change behavior. They exist to catch the much rarer, much worse case: a
# structure-extraction false positive (Docling/img2table misreading noise
# on a photographed/scanned page as a table border) that hands
# apply_padding one giant "cell" spanning an entire header/paragraph
# block, silently painting over everything inside it — see this module's
# apply_padding docstring: a cell-clamped box fills the *entire* cell
# interior, so a wrong cell this large paints far more than intended.
_SUSPICIOUS_CELL_CLAMP_BBOX_RATIO = 25.0
_SUSPICIOUS_CELL_CLAMP_PAGE_AREA_PCT = 0.10


def _warn_if_cell_clamp_looks_oversized(
    bbox: BBox, padded: BBox, page_w: int, page_h: int, *, redaction_id: str | None = None
) -> None:
    bbox_area = bbox.w * bbox.h
    padded_area = padded.w * padded.h
    page_area = page_w * page_h
    bbox_ratio = (padded_area / bbox_area) if bbox_area > 0 else float("inf")
    page_pct = (padded_area / page_area) if page_area > 0 else 0.0
    if bbox_ratio >= _SUSPICIOUS_CELL_CLAMP_BBOX_RATIO or page_pct >= _SUSPICIOUS_CELL_CLAMP_PAGE_AREA_PCT:
        logger.warning(
            "cell-clamped padding covers a suspiciously large area — likely a "
            "structure-extraction false positive (Docling/img2table misread a "
            "header/paragraph block as one table cell), not a real table cell; "
            "this redaction may be painting over far more than the detected text",
            extra={
                "redaction_id": redaction_id,
                "bbox_w": bbox.w,
                "bbox_h": bbox.h,
                "padded_w": padded.w,
                "padded_h": padded.h,
                "bbox_to_padded_area_ratio": round(bbox_ratio, 1),
                "padded_pct_of_page_area": round(page_pct, 3),
            },
        )
