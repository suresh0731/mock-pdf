import logging

from app.models.pii_chunk import BBox
from app.models.redact import PageTransform
from app.services.pii.coordinate_map import (
    _CELL_CLAMP_MARGIN_PX,
    apply_padding,
    canonical_to_original,
)


def test_canonical_to_original_adds_dx_dy():
    bbox = BBox(x=120, y=340, w=110, h=18)
    transform = PageTransform(dx=12, dy=8)
    mapped = canonical_to_original(bbox, transform)
    assert mapped == BBox(x=132, y=348, w=110, h=18)


def test_canonical_to_original_zero_offset_unchanged():
    bbox = BBox(x=10, y=20, w=50, h=12)
    transform = PageTransform(dx=0, dy=0)
    assert canonical_to_original(bbox, transform) == bbox


def test_canonical_to_original_negative_offset():
    bbox = BBox(x=20, y=20, w=10, h=10)
    transform = PageTransform(dx=-5, dy=-3)
    assert canonical_to_original(bbox, transform) == BBox(x=15, y=17, w=10, h=10)


def test_apply_padding_severe_expands_and_clamps():
    bbox = BBox(x=20, y=20, w=30, h=10)
    padded = apply_padding(bbox, "severe", page_w=100, page_h=80)
    assert padded == BBox(x=8, y=8, w=54, h=34)


def test_apply_padding_severe_clamps_to_page_origin():
    bbox = BBox(x=0, y=0, w=10, h=10)
    padded = apply_padding(bbox, "severe", page_w=20, page_h=20)
    assert padded == BBox(x=0, y=0, w=20, h=20)


def test_apply_padding_good_uses_4px():
    bbox = BBox(x=20, y=20, w=10, h=10)
    padded = apply_padding(bbox, "good", page_w=100, page_h=100)
    assert padded == BBox(x=16, y=16, w=18, h=18)


def test_apply_padding_mild_uses_8px():
    bbox = BBox(x=20, y=20, w=10, h=10)
    padded = apply_padding(bbox, "mild", page_w=100, page_h=100)
    assert padded == BBox(x=12, y=12, w=26, h=26)


def test_apply_padding_logs_omit_text(caplog):
    with caplog.at_level(logging.DEBUG, logger="app.services.pii.coordinate_map"):
        apply_padding(BBox(x=20, y=20, w=10, h=10), "good", page_w=100, page_h=100)
    assert "ALPHA" not in caplog.text
    assert "BETA" not in caplog.text


# --- Cell-aware / neighbor-aware padding clamp --------------------------


def test_apply_padding_clamps_to_cell_bbox_instead_of_page():
    """Severe-tier padding (12px) would normally cross into the
    neighboring cell/page area; a cell_bbox clamp keeps it inside the
    cell's own bounds minus the border-line safety margin."""
    bbox = BBox(x=20, y=20, w=30, h=10)
    cell = BBox(x=15, y=15, w=40, h=20)
    padded = apply_padding(bbox, "severe", page_w=200, page_h=200, cell_bbox=cell)
    margin = _CELL_CLAMP_MARGIN_PX

    assert padded.x == cell.x + margin
    assert padded.y == cell.y + margin
    assert padded.x + padded.w == cell.x + cell.w - margin
    assert padded.y + padded.h == cell.y + cell.h - margin


def test_apply_padding_hard_clips_ocr_overflow_to_cell():
    """An OCR/union bbox that spilled into the neighboring column must
    shrink back into the assigned cell so the painted patch cannot cover
    the neighbor's value (e.g. USD amount next to a name cell) — but the
    vertical extent must still fully cover bbox; only the horizontal
    spillover is ever hard-clipped below bbox's own extent."""
    bbox = BBox(x=10, y=20, w=90, h=16)  # overflows well left of the cell
    cell = BBox(x=40, y=18, w=50, h=22)
    padded = apply_padding(bbox, "good", page_w=400, page_h=200, cell_bbox=cell)
    margin = _CELL_CLAMP_MARGIN_PX

    assert padded.x >= cell.x + margin
    assert padded.x + padded.w <= cell.x + cell.w - margin
    assert padded.y <= bbox.y
    assert padded.y + padded.h >= bbox.y + bbox.h


def test_apply_padding_never_clips_below_bbox_vertically_even_when_cell_is_short():
    """Regression: a cell a few px shorter than the word union it's
    matched to (e.g. img2table under-measuring a tightly-spaced,
    multi-line row) must never cause the painted box to cut into the
    real text's top/bottom edge — only the added padding may shrink."""
    bbox = BBox(x=811, y=412, w=81, h=82)
    cell = BBox(x=790, y=414, w=122, h=78)  # shorter than bbox itself
    padded = apply_padding(bbox, "good", page_w=2000, page_h=2000, cell_bbox=cell)

    assert padded.y <= bbox.y
    assert padded.y + padded.h >= bbox.y + bbox.h


def test_apply_padding_never_clips_below_bbox_horizontally_when_bbox_fits_cell():
    """Regression: when bbox already fits inside the cell horizontally,
    the border-line safety margin must only reduce the padding, never
    cut into text sitting flush against the cell's own edge."""
    bbox = BBox(x=562, y=426, w=163, h=54)
    cell = BBox(x=562, y=414, w=228, h=78)  # same left edge as bbox
    padded = apply_padding(bbox, "good", page_w=2000, page_h=2000, cell_bbox=cell)

    assert padded.x <= bbox.x
    assert padded.x + padded.w >= bbox.x + bbox.w


def test_apply_padding_multiline_skips_top_bottom_padding():
    bbox = BBox(x=20, y=20, w=100, h=50)
    padded = apply_padding(bbox, "severe", page_w=500, page_h=500, multiline=True)

    assert padded.y == bbox.y
    assert padded.y + padded.h == bbox.y + bbox.h
    # Left/right padding is unaffected.
    assert padded.x == bbox.x - 12
    assert padded.x + padded.w == bbox.x + bbox.w + 12


def test_apply_padding_multiline_still_covers_bbox_with_cell_clamp():
    bbox = BBox(x=562, y=426, w=163, h=54)
    cell = BBox(x=562, y=414, w=228, h=78)
    padded = apply_padding(
        bbox, "good", page_w=2000, page_h=2000, cell_bbox=cell, multiline=True
    )

    assert padded.y <= bbox.y
    assert padded.y + padded.h >= bbox.y + bbox.h
    assert padded.x <= bbox.x
    assert padded.x + padded.w >= bbox.x + bbox.w


def test_apply_padding_ignores_non_overlapping_cell():
    """A cell that doesn't overlap the word at all is ignored, so a bad
    cell assignment cannot clip the detected word away."""
    bbox = BBox(x=20, y=20, w=30, h=10)
    elsewhere = BBox(x=200, y=200, w=40, h=20)
    padded = apply_padding(bbox, "good", page_w=500, page_h=500, cell_bbox=elsewhere)
    assert padded == apply_padding(bbox, "good", page_w=500, page_h=500)


def test_apply_padding_clamps_to_neighbor_x_when_no_cell():
    bbox = BBox(x=100, y=20, w=30, h=10)
    padded = apply_padding(
        bbox, "severe", page_w=500, page_h=200, left_neighbor_x=95, right_neighbor_x=133
    )
    assert padded.x == 95
    assert padded.x + padded.w == 133


def test_apply_padding_neighbor_clamp_ignored_when_cell_bbox_present():
    """cell_bbox takes priority over neighbor clamps when both are given."""
    bbox = BBox(x=20, y=20, w=30, h=10)
    cell = BBox(x=0, y=0, w=200, h=200)
    with_neighbors = apply_padding(
        bbox, "severe", page_w=500, page_h=500, cell_bbox=cell, left_neighbor_x=45, right_neighbor_x=52
    )
    without_neighbors = apply_padding(bbox, "severe", page_w=500, page_h=500, cell_bbox=cell)
    # Neighbor x-values (45/52) would otherwise slice through bbox's own
    # padded extent; since cell_bbox is provided, they must be ignored
    # entirely and the result must match the cell-only clamp exactly.
    assert with_neighbors == without_neighbors


def test_apply_padding_no_containment_context_matches_page_edge_behavior():
    """With no cell/neighbor context, behavior is identical to the
    original page-edge-only clamp (regression guard)."""
    bbox = BBox(x=20, y=20, w=30, h=10)
    assert apply_padding(bbox, "severe", page_w=100, page_h=80) == BBox(x=8, y=8, w=54, h=34)
