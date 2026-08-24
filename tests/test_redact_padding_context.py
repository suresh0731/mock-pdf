"""Unit tests for the cell/neighbor containment lookups that feed
``apply_padding`` (see ``app.pipeline.redact._collect_redactions``).

These are pure geometry helpers, tested directly and independent of the
async pipeline / OCR / Docling wiring.
"""

from __future__ import annotations

from app.models.pii_chunk import BBox
from app.pipeline.redact import _enclosing_cell_bbox, _row_neighbor_clamp_x
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
