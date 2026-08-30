"""Unit tests for the cell/neighbor containment lookups that feed
``apply_padding`` (see ``app.pipeline.redact._collect_redactions``).

These are pure geometry helpers, tested directly and independent of the
async pipeline / OCR / Docling wiring.
"""

from __future__ import annotations

from app.models.pii_chunk import BBox
from app.pipeline.redact import (
    _char_ranges_overlap,
    _enclosing_cell_bbox,
    _filter_words_to_cell,
    _line_wrap_clusters,
    _mask_claimed_ranges,
    _redaction_boxes_conflict,
    _row_clusters,
    _row_neighbor_clamp_x,
)
from app.services.ocr.ensemble_types import EnsembleWord
from app.services.structure.docling_adapter import DocBlock


def _word(text: str, x: int, y: int, w: int, h: int) -> EnsembleWord:
    return EnsembleWord(
        text=text,
        bbox=BBox(x=x, y=y, w=w, h=h),
        ocr_confidence=0.9,
        engine_agreement=1.0,
        engines=["tesseract"],
    )


def _cell(x: int, y: int, w: int, h: int, block_id: str = "cell-0") -> DocBlock:
    return DocBlock(block_id=block_id, block_type="cell", bbox=BBox(x=x, y=y, w=w, h=h), text="")


# --- _enclosing_cell_bbox ------------------------------------------------


def test_enclosing_cell_bbox_finds_substantially_containing_cell():
    bbox = BBox(x=20, y=20, w=30, h=10)
    cell = _cell(10, 10, 60, 30)
    assert _enclosing_cell_bbox(bbox, [cell]) == cell.bbox


def test_enclosing_cell_bbox_none_when_overlap_below_threshold():
    bbox = BBox(x=20, y=20, w=30, h=10)
    # Cell only covers the left third of the candidate's width.
    cell = _cell(10, 10, 20, 30)
    assert _enclosing_cell_bbox(bbox, [cell]) is None


def test_enclosing_cell_bbox_ignores_non_cell_blocks():
    bbox = BBox(x=20, y=20, w=30, h=10)
    paragraph = DocBlock(
        block_id="p-0", block_type="paragraph", bbox=BBox(x=0, y=0, w=200, h=200), text=""
    )
    assert _enclosing_cell_bbox(bbox, [paragraph]) is None


def test_enclosing_cell_bbox_empty_blocks_returns_none():
    bbox = BBox(x=20, y=20, w=30, h=10)
    assert _enclosing_cell_bbox(bbox, []) is None


def test_enclosing_cell_bbox_picks_highest_overlap_when_multiple_qualify():
    bbox = BBox(x=20, y=20, w=30, h=10)
    exact = _cell(20, 20, 30, 10, block_id="exact")  # 100% overlap
    partial = _cell(20, 20, 20, 10, block_id="partial")  # ~67% overlap, still clears threshold
    result = _enclosing_cell_bbox(bbox, [partial, exact])
    assert result == exact.bbox


def test_enclosing_cell_bbox_center_fallback_when_union_spans_two_cells():
    """An oversized OCR union covering a name cell and the amount cell
    to its left won't clear the 60% overlap threshold on either cell —
    fall back to the cell that contains the union's center."""
    # Union spans x=0..100; name cell is x=50..100 (center at 50).
    union = BBox(x=0, y=20, w=100, h=16)
    amount = _cell(0, 18, 48, 20, block_id="usd")
    name = _cell(50, 18, 50, 20, block_id="name")
    result = _enclosing_cell_bbox(union, [amount, name])
    assert result == name.bbox


def test_enclosing_cell_bbox_word_vote_picks_majority_cell():
    """Three name-word centers in the name cell beat one spilled amount
    word whose center landed in the USD cell."""
    union = BBox(x=0, y=20, w=100, h=16)
    amount = _cell(0, 18, 40, 20, block_id="usd")
    name = _cell(40, 18, 80, 20, block_id="name")
    word_bboxes = [
        BBox(x=10, y=22, w=20, h=12),  # USD
        BBox(x=45, y=22, w=20, h=12),
        BBox(x=70, y=22, w=20, h=12),
        BBox(x=95, y=22, w=20, h=12),
    ]
    result = _enclosing_cell_bbox(union, [amount, name], word_bboxes=word_bboxes)
    assert result == name.bbox


# --- _filter_words_to_cell -------------------------------------------------


def test_filter_words_to_cell_drops_word_outside_cell():
    """A merged_text char-range match that swept in one word from a
    paragraph line above the table (see the reading-order corruption
    this guards against) loses that word once a cell is known."""
    cell = _cell(260, 677, 200, 81)
    in_cell = _word("CUSTODIAN_A", 265, 690, 90, 40)
    stray_above = _word("kepada", 200, 650, 50, 20)  # center y=660, outside cell
    result = _filter_words_to_cell([stray_above, in_cell], cell.bbox)
    assert result == [in_cell]


def test_filter_words_to_cell_returns_none_when_all_words_coherent():
    """A genuine multi-line wrap (every word's center inside the same
    cell) is left untouched — no filtering needed."""
    cell = _cell(0, 0, 200, 200)
    line1 = _word("Maksima", 10, 10, 60, 20)
    line2 = _word("Plus", 10, 100, 60, 20)
    assert _filter_words_to_cell([line1, line2], cell.bbox) is None


def test_filter_words_to_cell_returns_none_when_no_word_matches():
    """All-or-nothing miss: if every word's center falls outside the
    cell, the cell itself is unreliable for this match — leave the
    original union untouched rather than discarding it entirely."""
    cell = _cell(1000, 1000, 50, 50)
    w1 = _word("foo", 0, 0, 10, 10)
    w2 = _word("bar", 20, 0, 10, 10)
    assert _filter_words_to_cell([w1, w2], cell.bbox) is None


def test_filter_words_to_cell_empty_input():
    cell = _cell(0, 0, 50, 50)
    assert _filter_words_to_cell([], cell.bbox) is None


# --- _row_neighbor_clamp_x ------------------------------------------------


def test_row_neighbor_clamp_x_returns_midpoints_on_both_sides():
    bbox = BBox(x=100, y=20, w=30, h=10)  # spans x in [100, 130], y in [20, 30]
    left_word = _word("Left", 50, 20, 30, 10)  # right edge at 80
    right_word = _word("Right", 150, 20, 30, 10)  # left edge at 150
    left_x, right_x = _row_neighbor_clamp_x(bbox, [left_word, right_word], excluded_ids=set())
    assert left_x == (80 + 100) / 2
    assert right_x == (130 + 150) / 2


def test_row_neighbor_clamp_x_ignores_words_on_a_different_row():
    bbox = BBox(x=100, y=20, w=30, h=10)
    far_below = _word("Below", 50, 500, 30, 10)  # same x-side, unrelated row
    left_x, right_x = _row_neighbor_clamp_x(bbox, [far_below], excluded_ids=set())
    assert left_x is None
    assert right_x is None


def test_row_neighbor_clamp_x_excludes_the_candidates_own_words():
    bbox = BBox(x=100, y=20, w=30, h=10)
    own_word = _word("Self", 50, 20, 30, 10)
    left_x, _ = _row_neighbor_clamp_x(bbox, [own_word], excluded_ids={id(own_word)})
    assert left_x is None


def test_row_neighbor_clamp_x_picks_nearest_neighbor_on_each_side():
    bbox = BBox(x=100, y=20, w=30, h=10)
    nearer = _word("Near", 80, 20, 10, 10)  # right edge at 90
    farther = _word("Far", 40, 20, 10, 10)  # right edge at 50
    left_x, _ = _row_neighbor_clamp_x(bbox, [farther, nearer], excluded_ids=set())
    assert left_x == (90 + 100) / 2


def test_row_neighbor_clamp_x_no_words_returns_none_none():
    bbox = BBox(x=100, y=20, w=30, h=10)
    assert _row_neighbor_clamp_x(bbox, [], excluded_ids=set()) == (None, None)


# --- overlap de-dupe helpers ---------------------------------------------


def test_char_ranges_overlap_nested_short_inside_long():
    assert _char_ranges_overlap(0, 18, [(0, 23)]) is True
    assert _char_ranges_overlap(0, 3, [(10, 20)]) is False


def test_mask_claimed_ranges_blanks_only_claimed_spans():
    text = "ABCDEFGHIJ"
    masked = _mask_claimed_ranges(text, [(2, 5), (8, 10)])
    assert masked == "AB\x00\x00\x00FGH\x00\x00"
    assert len(masked) == len(text)


def test_mask_claimed_ranges_no_claims_returns_original():
    text = "unchanged text"
    assert _mask_claimed_ranges(text, []) is text


def test_mask_claimed_ranges_clamps_out_of_bounds():
    text = "short"
    masked = _mask_claimed_ranges(text, [(-5, 3), (4, 100)])
    assert masked == "\x00\x00\x00r\x00"


def test_redaction_boxes_conflict_near_duplicate():
    cell = BBox(x=10, y=20, w=80, h=16)
    slightly_shifted = BBox(x=12, y=22, w=78, h=18)
    assert _redaction_boxes_conflict(cell, slightly_shifted) is True


def test_redaction_boxes_conflict_container_and_inner():
    inner = BBox(x=20, y=20, w=40, h=12)
    oversized = BBox(x=10, y=10, w=200, h=40)
    assert _redaction_boxes_conflict(inner, oversized) is True


def test_redaction_boxes_conflict_separate_cells():
    left = BBox(x=10, y=20, w=40, h=16)
    right = BBox(x=80, y=20, w=40, h=16)
    assert _redaction_boxes_conflict(left, right) is False


# --- _row_clusters / _line_wrap_clusters -----------------------------------


def test_row_clusters_groups_same_line_words_together():
    a = _word("Foo", 10, 20, 20, 10)
    b = _word("Bar", 40, 22, 20, 10)
    clusters = _row_clusters([a, b])
    assert clusters == [[a, b]]


def test_row_clusters_splits_words_on_different_lines():
    top = _word("Foo", 10, 20, 20, 10)
    bottom = _word("Bar", 10, 40, 20, 10)
    clusters = _row_clusters([top, bottom])
    assert clusters == [[top], [bottom]]


def test_row_clusters_splits_two_real_lines_despite_one_inflated_word_box():
    """A blurred/warped scan can make one OCR word's box come back tall
    enough to span two physical text lines (bbox.h far larger than its
    row's real line height) even though the words themselves sit on two
    clearly separate lines. Using that one inflated height as the overlap
    denominator would make the lines look >50% overlapping and collapse
    them into a single row — which then wrongly reports a two-line value
    as single-line (``multiline=False``) and lets ``apply_padding`` add
    the normal top/bottom pad on top of an already two-line-tall box."""
    line1_a = _word("TM", 10, 20, 20, 12)
    line1_b = _word("DPLK", 35, 20, 30, 12)
    # "Fixed" has a normal line-1 y-position but a bloated height that
    # bridges down into line 2's territory (y in [20, 55), vs. the other
    # line-1 words' y in [20, 32)).
    line1_bloated = _word("Fixed", 70, 20, 30, 35)
    line2 = _word("FundII", 10, 45, 30, 12)
    clusters = _row_clusters([line1_a, line1_b, line1_bloated, line2])
    assert len(clusters) == 2
    assert line2 not in clusters[0]
    assert clusters[1] == [line2]


def test_line_wrap_clusters_none_for_single_line_match():
    a = _word("John", 10, 20, 30, 10)
    b = _word("Smith", 45, 20, 30, 10)
    assert _line_wrap_clusters([a, b]) is None


def test_line_wrap_clusters_detects_extreme_ends_split():
    """First word(s) at the tail end of line 1 (large x), remaining
    word(s) at the head of line 2 (small x) — the line-wrap shape a
    name/value split by a real PDF line break actually takes. The union
    of both lines would span nearly the full line width; the per-line
    clusters should be returned instead."""
    line1_word = _word("Maria", 500, 20, 60, 12)  # tail of line 1
    line2_word = _word("Santos", 10, 40, 70, 12)  # head of line 2
    clusters = _line_wrap_clusters([line1_word, line2_word])
    assert clusters == [[line1_word], [line2_word]]


def test_line_wrap_clusters_none_for_column_aligned_wrapped_cell():
    """A wrapped table cell's lines stay column-aligned (same x) rather
    than spreading to opposite ends — must NOT be split."""
    line1_word = _word("Standard", 10, 20, 60, 12)
    line2_word = _word("Chartered", 10, 40, 65, 12)
    assert _line_wrap_clusters([line1_word, line2_word]) is None
