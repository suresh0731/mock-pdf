"""Unit tests for the spillover fail-safe (see
``app.pipeline.redact._apply_spillover_safety_net`` and its helpers), which
absorbs an orphaned name-shaped OCR word into an adjacent already-redacted
span's bounding box instead of leaving it exposed as bare text.
"""

from __future__ import annotations

from PIL import Image

from app.models.pii_chunk import BBox
from app.models.redact import ConfidenceBreakdown, PageTransform, RedactionRegion
from app.pipeline.redact import (
    _apply_spillover_safety_net,
    _is_name_shaped,
    _nearest_adjacent_redaction,
    _union_bboxes,
)
from app.services.ocr.ensemble_types import EnsembleWord
from app.services.preprocess.canonical import CanonicalPage
from app.services.structure.docling_adapter import DocBlock


def _word(text: str, x: int, y: int, w: int = 60, h: int = 18) -> EnsembleWord:
    return EnsembleWord(
        text=text,
        bbox=BBox(x=x, y=y, w=w, h=h),
        ocr_confidence=0.9,
        engine_agreement=1.0,
        engines=["tesseract"],
    )


def _region(x: int, y: int, w: int, h: int, region_id: str = "r-0001") -> RedactionRegion:
    box = BBox(x=x, y=y, w=w, h=h)
    return RedactionRegion(
        region_id=region_id,
        page=0,
        entity_type="ORGANIZATION",
        canonical_bbox=box,
        original_bbox=box,
        padded_bbox=box,
        redaction_confidence=0.9,
        confidence_breakdown=ConfidenceBreakdown(
            presidio=0.9, ocr=0.9, engine_agreement=1.0, structural_context=0.0
        ),
        blur_tier="good",
        engines_seen=["tesseract"],
        mock_value="ORG_1",
        mapping_id="map_1",
        assignment_source="auto",
    )


def _canonical() -> CanonicalPage:
    image = Image.new("RGB", (720, 1100), "white")
    return CanonicalPage(
        page_index=0,
        original_image=image,
        canonical_image=image,
        transform=PageTransform(dx=0, dy=0, blur_tier="good"),
    )


# --- _is_name_shaped -------------------------------------------------------


def test_is_name_shaped_accepts_capitalized_alpha_word():
    assert _is_name_shaped(_word("Plus", 0, 0)) is True


def test_is_name_shaped_rejects_lowercase_word():
    assert _is_name_shaped(_word("plus", 0, 0)) is False


def test_is_name_shaped_rejects_digits():
    assert _is_name_shaped(_word("12345", 0, 0)) is False


def test_is_name_shaped_rejects_punctuation_only():
    assert _is_name_shaped(_word(":", 0, 0)) is False


def test_is_name_shaped_accepts_hyphenated_name():
    assert _is_name_shaped(_word("Al-Rashid", 0, 0)) is True


# --- _union_bboxes ----------------------------------------------------------


def test_union_bboxes_covers_both_inputs():
    a = BBox(x=10, y=10, w=20, h=10)
    b = BBox(x=40, y=10, w=20, h=10)
    result = _union_bboxes(a, b)
    assert result == BBox(x=10, y=10, w=50, h=10)


# --- _nearest_adjacent_redaction --------------------------------------------


def test_nearest_adjacent_redaction_finds_same_row_neighbor():
    word_bbox = BBox(x=100, y=20, w=40, h=18)
    region = _region(0, 20, 90, 18)  # right edge at 90, gap of 10 to word
    assert _nearest_adjacent_redaction(word_bbox, [region]) is region


def test_nearest_adjacent_redaction_none_when_gap_too_large():
    word_bbox = BBox(x=500, y=20, w=40, h=18)
    region = _region(0, 20, 90, 18)
    assert _nearest_adjacent_redaction(word_bbox, [region]) is None


def test_nearest_adjacent_redaction_none_for_different_row():
    word_bbox = BBox(x=100, y=500, w=40, h=18)
    region = _region(0, 20, 90, 18)
    assert _nearest_adjacent_redaction(word_bbox, [region]) is None


def test_nearest_adjacent_redaction_picks_nearest_of_several():
    word_bbox = BBox(x=200, y=20, w=40, h=18)
    near = _region(150, 20, 40, 18, region_id="near")  # right edge 190, gap 10
    far = _region(0, 20, 40, 18, region_id="far")  # right edge 40, gap 160
    assert _nearest_adjacent_redaction(word_bbox, [far, near]) is near


def test_nearest_adjacent_redaction_empty_list_returns_none():
    assert _nearest_adjacent_redaction(BBox(x=0, y=0, w=10, h=10), []) is None


# --- _apply_spillover_safety_net -------------------------------------------


def test_apply_spillover_absorbs_orphaned_adjacent_word():
    region = _region(0, 20, 90, 18)
    orphan = _word("Plus", 100, 20)
    words = [orphan]
    _apply_spillover_safety_net([region], words, [], _canonical())
    assert region.canonical_bbox.w >= 140
    assert "tesseract" in region.engines_seen


def test_apply_spillover_ignores_word_already_covered_by_a_redaction():
    region = _region(0, 20, 90, 18)
    covered_word = _word("Maksima", 0, 20, w=90)
    original_bbox = region.canonical_bbox
    _apply_spillover_safety_net([region], [covered_word], [], _canonical())
    assert region.canonical_bbox == original_bbox


def test_apply_spillover_ignores_non_name_shaped_word():
    region = _region(0, 20, 90, 18)
    number_word = _word("12345", 100, 20)
    original_bbox = region.canonical_bbox
    _apply_spillover_safety_net([region], [number_word], [], _canonical())
    assert region.canonical_bbox == original_bbox


def test_apply_spillover_no_op_when_no_redactions():
    orphan = _word("Plus", 100, 20)
    _apply_spillover_safety_net([], [orphan], [], _canonical())  # must not raise


def test_apply_spillover_does_not_absorb_word_in_neighboring_cell():
    """A name-shaped word whose center sits in a different table cell
    (the USD/amount column next door) must not be unioned into this
    redaction — that is the oversized ORG_09 patch that covered the
    neighboring cell's value."""
    region = _region(50, 20, 80, 18)
    orphan = _word("Amountish", 5, 20, w=30)
    name_cell = DocBlock(
        block_id="name",
        block_type="cell",
        bbox=BBox(x=45, y=15, w=90, h=30),
        text="",
    )
    usd_cell = DocBlock(
        block_id="usd",
        block_type="cell",
        bbox=BBox(x=0, y=15, w=40, h=30),
        text="",
    )
    original_bbox = region.canonical_bbox
    _apply_spillover_safety_net([region], [orphan], [name_cell, usd_cell], _canonical())
    assert region.canonical_bbox == original_bbox
