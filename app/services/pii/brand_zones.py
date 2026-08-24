"""Percent-of-page footer zone unioned with Docling chrome, plus
position-agnostic picture-block detection.

The logo zone used to be a second fixed percent-of-page rectangle here
(top-right corner), which only ever covered a logo actually placed there
by one specific template. It has been removed now that
``detect_picture_zones`` covers any Docling-detected picture block
anywhere on the page — including a logo wherever it actually sits.

Logs page index, zone names, counts, and integer bbox coords only (SEC-001).
Never logs ``DocBlock.text`` or other block content.
"""

import logging
from dataclasses import dataclass
from typing import Literal

from app.models.pii_chunk import BBox
from app.services.structure.docling_adapter import DocBlock

logger = logging.getLogger(__name__)

UNION_BLOCK_TYPES = frozenset({"footer", "picture"})


@dataclass(frozen=True)
class BrandZone:
    """A heuristic brand chrome region on an original page.

    Attributes:
        zone: Zone kind, ``footer`` or ``picture``.
        page: Zero-based page index.
        bbox: Box in original-page pixels.
        label: Painted cover label, ``FOOTER`` or ``IMAGE``.
    """

    zone: Literal["footer", "picture"]
    page: int
    bbox: BBox
    label: str


def _pct_px(size: int, pct: float) -> int:
    """Convert a page dimension percent to pixels.

    Args:
        size: Page width or height in pixels.
        pct: Fraction of ``size``. Values ``<= 0`` yield 0.

    Returns:
        ``max(0, int(round(size * pct)))``.
    """
    if pct <= 0:
        return 0
    return max(0, int(round(size * pct)))


def _clamp_bbox(bbox: BBox, page_w: int, page_h: int) -> BBox | None:
    """Clamp a box to ``[0, page_w] × [0, page_h]``.

    Args:
        bbox: Box in original-page pixels (may extend off-page).
        page_w: Page width in pixels.
        page_h: Page height in pixels.

    Returns:
        Clamped box, or ``None`` if empty after clamp.
    """
    x1 = max(0, bbox.x)
    y1 = max(0, bbox.y)
    x2 = min(page_w, bbox.x + bbox.w)
    y2 = min(page_h, bbox.y + bbox.h)
    if x2 <= x1 or y2 <= y1:
        return None
    return BBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)


def _footer_box(page_w: int, page_h: int, bottom_pct: float) -> BBox:
    """Bottom ``bottom_pct`` of height, full width.

    Args:
        page_w: Page width in pixels.
        page_h: Page height in pixels.
        bottom_pct: Footer height as a fraction of page height.

    Returns:
        Unclamped footer seed box.
    """
    height = _pct_px(page_h, bottom_pct)
    return BBox(x=0, y=page_h - height, w=page_w, h=height)


def _boxes_overlap(a: BBox, b: BBox) -> bool:
    """True iff intersection area is greater than 0.

    Args:
        a: First box.
        b: Second box.

    Returns:
        Whether the two axis-aligned boxes share a positive-area overlap.
    """
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    return min(ax2, bx2) > max(a.x, b.x) and min(ay2, by2) > max(a.y, b.y)


def _union_bbox(a: BBox, b: BBox) -> BBox:
    """Axis-aligned union of two boxes.

    Args:
        a: First box.
        b: Second box.

    Returns:
        Smallest axis-aligned box covering both inputs.
    """
    x1, y1 = min(a.x, b.x), min(a.y, b.y)
    x2 = max(a.x + a.w, b.x + b.w)
    y2 = max(a.y + a.h, b.y + b.h)
    return BBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)


def _expand_zone(seed: BBox, blocks: list[DocBlock]) -> BBox:
    """Union ``seed`` with every eligible overlapping block bbox.

    Eligible types are ``footer`` and ``picture`` (case-insensitive).
    Block text is never read.

    Args:
        seed: Heuristic zone rectangle.
        blocks: Docling structure blocks for the page.

    Returns:
        Expanded (possibly unclamped) zone box.
    """
    expanded = seed
    for block in blocks:
        if block.block_type.lower() not in UNION_BLOCK_TYPES:
            continue
        if _boxes_overlap(expanded, block.bbox):
            expanded = _union_bbox(expanded, block.bbox)
    return expanded


def detect_brand_zones(
    page_w: int,
    page_h: int,
    page: int,
    blocks: list[DocBlock] | None = None,
    patch_footer: bool = True,
    footer_bottom_pct: float = 0.12,
) -> list[BrandZone]:
    """Percent-of-page footer zone, unioned with overlapping footer/picture.

    Args:
        page_w: Original page width in pixels.
        page_h: Original page height in pixels.
        page: Zero-based page index (logged on skip/detect).
        blocks: Optional Docling blocks. ``None`` is treated as empty.
        patch_footer: When False, omit the footer zone.
        footer_bottom_pct: Footer height as a fraction of page height.

    Returns:
        The footer ``BrandZone`` if it remains non-empty after clamp,
        else ``[]``. Invalid page size (``<= 0``) returns ``[]``.
    """
    if page_w <= 0 or page_h <= 0:
        logger.warning(
            "brand_zones_skipped invalid_page_size page=%s", page
        )
        return []

    resolved_blocks = blocks or []
    result: list[BrandZone] = []

    if patch_footer:
        seed = _footer_box(page_w, page_h, footer_bottom_pct)
        expanded = _expand_zone(seed, resolved_blocks)
        clamped = _clamp_bbox(expanded, page_w, page_h)
        if clamped is not None:
            result.append(
                BrandZone(
                    zone="footer", page=page, bbox=clamped, label="FOOTER"
                )
            )

    logger.info(
        "brand_zones_detected page=%s count=%s zones=%s",
        page,
        len(result),
        [z.zone for z in result],
    )
    logger.debug(
        "brand_zones_bboxes page=%s boxes=%s",
        page,
        [(z.zone, z.bbox.x, z.bbox.y, z.bbox.w, z.bbox.h) for z in result],
    )
    return result


def _overlap_area(a: BBox, b: BBox) -> int:
    """Intersection area of two boxes, or 0 if they don't overlap."""
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


_PICTURE_COVERAGE_THRESHOLD = 0.6


def _covered_by_existing(bbox: BBox, existing_zones: list["BrandZone"]) -> bool:
    """True if ``bbox`` is already substantially covered by one of
    ``existing_zones`` (e.g. a picture already unioned into the footer
    zone), so it isn't painted a second time.
    """
    area = bbox.w * bbox.h
    if area <= 0:
        return False
    for zone in existing_zones:
        if _overlap_area(bbox, zone.bbox) / area >= _PICTURE_COVERAGE_THRESHOLD:
            return True
    return False


def detect_picture_zones(
    page_w: int,
    page_h: int,
    page: int,
    blocks: list[DocBlock] | None = None,
    existing_zones: list[BrandZone] | None = None,
    enabled: bool = True,
    min_area_pct: float = 0.0015,
) -> list[BrandZone]:
    """Every detected picture/figure block, anywhere on the page.

    Unlike ``detect_brand_zones``'s fixed percent-of-page seeds, this
    covers a picture block at whatever position Docling found it — so a
    logo, stamp, signature scan, or watermark placed anywhere on the
    page (not just the top-right/bottom strip a single template's logo
    and footer happen to sit in) gets its own redaction zone.

    Args:
        page_w: Original page width in pixels.
        page_h: Original page height in pixels.
        page: Zero-based page index (logged on skip/detect).
        blocks: Optional Docling blocks. ``None`` is treated as empty.
        existing_zones: Zones already computed (e.g. from
            ``detect_brand_zones``) — a picture block already
            substantially covered by one of these is skipped so it
            isn't painted twice.
        enabled: When False, return ``[]`` without inspecting blocks.
        min_area_pct: Minimum picture-block area, as a fraction of page
            area, to qualify. Filters bullet icons/checkbox glyphs that
            Docling may still classify as ``picture``.

    Returns:
        One ``BrandZone`` per surviving picture block. Invalid page
        size (``<= 0``) or ``enabled=False`` returns ``[]``.
    """
    if not enabled:
        return []
    if page_w <= 0 or page_h <= 0:
        logger.warning(
            "picture_zones_skipped invalid_page_size page=%s", page
        )
        return []

    resolved_blocks = blocks or []
    resolved_existing = existing_zones or []
    page_area = page_w * page_h
    min_area = max(0.0, min_area_pct) * page_area

    result: list[BrandZone] = []
    for block in resolved_blocks:
        if block.block_type.lower() != "picture":
            continue
        clamped = _clamp_bbox(block.bbox, page_w, page_h)
        if clamped is None:
            continue
        if clamped.w * clamped.h < min_area:
            continue
        if _covered_by_existing(clamped, resolved_existing):
            continue
        result.append(
            BrandZone(zone="picture", page=page, bbox=clamped, label="IMAGE")
        )

    logger.info(
        "picture_zones_detected page=%s count=%s",
        page,
        len(result),
    )
    logger.debug(
        "picture_zones_bboxes page=%s boxes=%s",
        page,
        [(z.bbox.x, z.bbox.y, z.bbox.w, z.bbox.h) for z in result],
    )
    return result
