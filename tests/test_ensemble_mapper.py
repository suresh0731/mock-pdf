import logging

from app.models.pii_chunk import BBox
from app.services.ocr.ensemble_types import EnsembleWord
from app.services.pii.ensemble_mapper import map_span_to_ensemble_bbox, words_for_span


def _word(
    text: str,
    bbox: BBox,
    char_start: int,
    char_end: int,
    ocr: float = 1.0,
    agreement: float = 1.0,
) -> EnsembleWord:
    return EnsembleWord(
        text=text,
        bbox=bbox,
        ocr_confidence=ocr,
        engine_agreement=agreement,
        char_start=char_start,
        char_end=char_end,
    )


def test_map_span_union_bbox_for_multi_word():
    words = [
        _word("ALPHA", BBox(x=10, y=20, w=40, h=12), 0, 5),
        _word("BETA", BBox(x=55, y=22, w=50, h=10), 6, 10),
    ]
    box = map_span_to_ensemble_bbox(0, 10, words, "ALPHA BETA")
    assert box == BBox(x=10, y=20, w=95, h=12)


def test_map_span_returns_none_when_no_overlap():
    words = [_word("ALPHA", BBox(x=10, y=20, w=40, h=12), 0, 5)]
    assert map_span_to_ensemble_bbox(20, 24, words, "ALPHA") is None


def test_map_span_returns_none_when_start_gte_end():
    words = [_word("ALPHA", BBox(x=10, y=20, w=40, h=12), 0, 5)]
    assert map_span_to_ensemble_bbox(5, 5, words, "ALPHA") is None
    assert map_span_to_ensemble_bbox(8, 3, words, "ALPHA") is None


def test_map_span_returns_none_when_words_empty():
    assert map_span_to_ensemble_bbox(0, 4, [], "ALPHA") is None


def test_words_for_span_includes_partial_and_contained():
    words = [
        _word("ALPHA", BBox(x=0, y=0, w=10, h=10), 0, 5),
        _word("BETA", BBox(x=12, y=0, w=10, h=10), 6, 10),
        _word("BETA", BBox(x=24, y=0, w=10, h=10), 11, 16),
    ]
    matched = words_for_span(3, 12, words)
    assert matched == words


def test_words_for_span_invalid_span_empty():
    words = [_word("ALPHA", BBox(x=0, y=0, w=10, h=10), 0, 5)]
    assert words_for_span(4, 4, words) == []


def test_map_span_logs_omit_span_text(caplog):
    words = [
        _word("ALPHA", BBox(x=10, y=20, w=40, h=12), 0, 5),
        _word("BETA", BBox(x=55, y=22, w=50, h=10), 6, 10),
    ]
    with caplog.at_level(logging.DEBUG, logger="app.services.pii.ensemble_mapper"):
        map_span_to_ensemble_bbox(0, 10, words, "ALPHA BETA")
    assert "ALPHA" not in caplog.text
    assert "BETA" not in caplog.text
    assert "ALPHA BETA" not in caplog.text
