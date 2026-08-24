"""Percent-of-page logo/footer zones unioned with Docling chrome.

Logs page index, zone names, counts, and integer bbox coords only (SEC-001).
Never logs ``DocBlock.text`` or other block content.
"""

import logging
from dataclasses import dataclass
from typing import Literal

from app.models.pii_chunk import BBox
from app.services.structure.docling_adapter import DocBlock

logger = logging.getLogger(__name__)

UNION_BLOCK_TYPES = frozenset({"header", "footer", "picture"})


@dataclass(frozen=True)
class BrandZone:
    """A heuristic brand chrome region on an original page.

    Attributes:
        zone: Zone kind, ``logo`` or ``footer``.
        page: Zero-based page index.
        bbox: Box in original-page pixels.
        label: Painted cover label, ``LOGO`` or ``FOOTER``.
    """

    zone: Literal["logo", "footer"]
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


def _logo_box(page_w: int, page_h: int, top_pct: float, right_pct: float) -> BBox:
    """Top ``top_pct`` of height, right ``right_pct`` of width.

    Args:
        page_w: Page width in pixels.
        page_h: Page height in pixels.
        top_pct: Logo height as a fraction of page height.
        right_pct: Logo width as a fraction of page width.

    Returns:
        Unclamped logo seed box.
    """
    height = _pct_px(page_h, top_pct)
    width = _pct_px(page_w, right_pct)
    return BBox(x=page_w - width, y=0, w=width, h=height)


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

    Eligible types are ``header``, ``footer``, and ``picture``
    (case-insensitive). Block text is never read.

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
    patch_logo: bool = True,
    patch_footer: bool = True,
    logo_top_pct: float = 0.12,
    logo_right_pct: float = 0.28,
    footer_bottom_pct: float = 0.12,
) -> list[BrandZone]:
    """Percent-of-page zones, unioned with overlapping header/footer/picture.

    Args:
        page_w: Original page width in pixels.
        page_h: Original page height in pixels.
        page: Zero-based page index (logged on skip/detect).
        blocks: Optional Docling blocks. ``None`` is treated as empty.
        patch_logo: When False, omit the logo zone.
        patch_footer: When False, omit the footer zone.
        logo_top_pct: Logo height as a fraction of page height.
        logo_right_pct: Logo width as a fraction of page width.
        footer_bottom_pct: Footer height as a fraction of page height.

    Returns:
        Logo then footer ``BrandZone`` items that remain non-empty after
        clamp. Invalid page size (``<= 0``) returns ``[]``.
    """
    if page_w <= 0 or page_h <= 0:
        logger.warning(
            "brand_zones_skipped invalid_page_size page=%s", page
        )
        return []

    resolved_blocks = blocks or []
    result: list[BrandZone] = []

    if patch_logo:
        seed = _logo_box(page_w, page_h, logo_top_pct, logo_right_pct)
        expanded = _expand_zone(seed, resolved_blocks)
        clamped = _clamp_bbox(expanded, page_w, page_h)
        if clamped is not None:
            result.append(
                BrandZone(zone="logo", page=page, bbox=clamped, label="LOGO")
            )

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
