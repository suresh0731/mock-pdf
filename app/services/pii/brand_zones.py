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
        zone: Zone kind, ``footer``, ``picture``, or ``signature`` (see
            ``app.services.pii.signature_zones.detect_signature_zones``).
        page: Zero-based page index.
        bbox: Box in original-page pixels.
        label: Painted cover label, ``FOOTER``, ``IMAGE``, or
            ``SIGNATURE``.
    """

    zone: Literal["footer", "picture", "signature"]
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
# A picture whose own area is mostly a word patch is a table-cell logo
# sitting on already-mocked text (the tiny "IMAGE" over "DSDC_Bank" in
# the Bank column). Skip it; the word patch already covers that PII.
_PICTURE_OWN_OVERLAP_SKIP = 0.5
# A word patch whose own area is mostly inside the picture is OCR of
# logo/stamp ink — keep the IMAGE box as-is so it still covers the
# graphic; draw-order paints IMAGE last over that mock.
_WORD_INSIDE_PICTURE_COVERAGE = 0.6
_PICTURE_INSET_FRACTION = 0.15
_PICTURE_INSET_MIN_PX = 8
# The "picture inside a cell" skip only makes sense for the small icon
# case it was built for (a bank-column logo occupying a minor fraction of
# its cell). Without this ratio check, a real full-size logo (e.g. a
# letterhead) whose center merely happens to fall inside an unrelated,
# overly broad table-geometry cell (img2table's own table detection can
# produce a cell spanning a whole header block) would be silently
# excluded from redaction even though it isn't a small in-cell icon at
# all — see the header-table/logo overlap case in a real bank statement.
_PICTURE_CELL_AREA_RATIO_MAX = 0.5


def _bbox_center_xy(bbox: BBox) -> tuple[float, float]:
    return bbox.x + bbox.w / 2, bbox.y + bbox.h / 2


def _point_in_bbox(bbox: BBox, x: float, y: float) -> bool:
    return bbox.x <= x <= bbox.x + bbox.w and bbox.y <= y <= bbox.y + bbox.h


def _picture_center_in_cell(bbox: BBox, blocks: list[DocBlock]) -> bool:
    """True when the picture sits inside a table cell as a minor
    fraction of it (bank-column logos) — not merely whenever its center
    happens to fall inside some cell, which a real, page-spanning logo
    can do purely by coincidence against an overly broad table-geometry
    cell (see ``_PICTURE_CELL_AREA_RATIO_MAX``).
    """
    picture_area = bbox.w * bbox.h
    if picture_area <= 0:
        return False
    cx, cy = _bbox_center_xy(bbox)
    for block in blocks:
        if block.block_type.lower() != "cell":
            continue
        if not _point_in_bbox(block.bbox, cx, cy):
            continue
        cell_area = block.bbox.w * block.bbox.h
        if cell_area <= 0:
            continue
        if picture_area / cell_area <= _PICTURE_CELL_AREA_RATIO_MAX:
            return True
    return False


def _shrink_bbox_to_avoid(box: BBox, blocker: BBox) -> BBox | None:
    """Largest remaining piece of ``box`` after cutting off ``blocker``.

    Axis-aligned: keep the slab above, below, left, or right of the
    blocker, whichever retains the most area. ``None`` if nothing usable
    remains (blocker covers the whole box).
    """
    if _overlap_area(box, blocker) <= 0:
        return box
    box_x2, box_y2 = box.x + box.w, box.y + box.h
    blk_x2, blk_y2 = blocker.x + blocker.w, blocker.y + blocker.h
    candidates: list[BBox] = []
    above_h = blocker.y - box.y
    if above_h >= 1:
        candidates.append(BBox(x=box.x, y=box.y, w=box.w, h=above_h))
    below_h = box_y2 - blk_y2
    if below_h >= 1:
        candidates.append(BBox(x=box.x, y=blk_y2, w=box.w, h=below_h))
    left_w = blocker.x - box.x
    if left_w >= 1:
        candidates.append(BBox(x=box.x, y=box.y, w=left_w, h=box.h))
    right_w = box_x2 - blk_x2
    if right_w >= 1:
        candidates.append(BBox(x=blk_x2, y=box.y, w=right_w, h=box.h))
    if not candidates:
        return None
    return max(candidates, key=lambda b: b.w * b.h)


def _word_deeply_inside_picture(picture: BBox, word: BBox) -> bool:
    """True when ``word``'s center sits inset from every edge of ``picture``.

    Distinguishes logo-OCR (a small word in the middle of a stamp — keep
    IMAGE covering it) from an oversized signature box that only just
    contains the mocked name along its rim (clip IMAGE off that name).
    """
    margin_x = max(_PICTURE_INSET_MIN_PX, int(_PICTURE_INSET_FRACTION * picture.w))
    margin_y = max(_PICTURE_INSET_MIN_PX, int(_PICTURE_INSET_FRACTION * picture.h))
    inner_x1 = picture.x + margin_x
    inner_y1 = picture.y + margin_y
    inner_x2 = picture.x + picture.w - margin_x
    inner_y2 = picture.y + picture.h - margin_y
    if inner_x2 <= inner_x1 or inner_y2 <= inner_y1:
        return False
    cx, cy = _bbox_center_xy(word)
    return inner_x1 <= cx <= inner_x2 and inner_y1 <= cy <= inner_y2


def reconcile_picture_zones_with_text(
    zones: list[BrandZone],
    text_boxes: list[BBox],
    *,
    min_area: float = 0.0,
) -> list[BrandZone]:
    """Fit IMAGE zones around already-painted word patches.

    Word patches are collected first; IMAGE is painted last and would
    otherwise cover mocked names (signature stamps overlapping
    ``DTMLI`` / signatory lines, tiny bank logos overlapping
    ``DSDC_Bank``). Footer zones pass through unchanged.

    Per picture zone, against each overlapping word box:

    - Word mostly inside the picture *and* inset from its edges: leave
      IMAGE (logo/stamp OCR sitting in the middle of the graphic).
    - Picture mostly overlapping the word: drop IMAGE (cell logo).
    - Partial/adjacent overlap, including a name along the picture's
      rim: shrink IMAGE off the word box.
    """
    if not text_boxes:
        return zones
    result: list[BrandZone] = []
    for zone in zones:
        if zone.zone != "picture":
            result.append(zone)
            continue
        box = zone.bbox
        skip = False
        for text_box in text_boxes:
            overlap = _overlap_area(box, text_box)
            if overlap <= 0:
                continue
            word_area = text_box.w * text_box.h
            pic_area = box.w * box.h
            if (
                word_area > 0
                and overlap / word_area >= _WORD_INSIDE_PICTURE_COVERAGE
                and _word_deeply_inside_picture(box, text_box)
            ):
                continue
            if pic_area > 0 and overlap / pic_area >= _PICTURE_OWN_OVERLAP_SKIP:
                skip = True
                break
            shrunk = _shrink_bbox_to_avoid(box, text_box)
            if shrunk is None:
                skip = True
                break
            box = shrunk
        if skip:
            continue
        if box.w * box.h < min_area:
            continue
        if box != zone.bbox:
            zone = BrandZone(
                zone=zone.zone, page=zone.page, bbox=box, label=zone.label
            )
        result.append(zone)
    return result


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
        if _picture_center_in_cell(clamped, resolved_blocks):
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
