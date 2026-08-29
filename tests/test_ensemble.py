import logging
import sys

import pytest

from app.models.pii_chunk import BBox
from app.services.ocr.ensemble import (
    _format_engine_failure,
    align_word_boxes,
    merged_text_from_words,
)
from app.services.ocr.ensemble_types import EnsembleWord

SECRET_OCR_TOKEN = "SECRET_OCR_TOKEN"


def _word(
    text: str,
    x: int,
    y: int,
    w: int = 20,
    h: int = 10,
    confidence: float = 0.9,
) -> dict:
    """Build a canned engine word dict (synthetic tokens only)."""
    return {"text": text, "x": x, "y": y, "w": w, "h": h, "confidence": confidence}


# Fixture F3 — three-engine overlap, IoU ≥ 0.5. Union: (100, 50, 84, 16).
_F3_TESSERACT = [{"text": "INV-001", "x": 100, "y": 50, "w": 80, "h": 14, "confidence": 0.90}]
_F3_EASYOCR = [{"text": "INV-001", "x": 102, "y": 51, "w": 78, "h": 15, "confidence": 0.85}]
_F3_RAPIDOCR = [{"text": "INV-001", "x": 101, "y": 50, "w": 83, "h": 16, "confidence": 0.88}]
_F3_INPUT = [
    ("tesseract", _F3_TESSERACT),
    ("easyocr", _F3_EASYOCR),
    ("rapidocr", _F3_RAPIDOCR),
]


def test_ensemble_word_defaults_and_fields():
    word = EnsembleWord(
        text="A",
        bbox=BBox(x=0, y=0, w=10, h=8),
        ocr_confidence=0.9,
        engine_agreement=1.0,
    )
    assert word.engines == []
    assert word.page == 0
    assert word.char_start == 0
    assert word.char_end == 0
    assert word.text == "A"
    assert word.bbox == BBox(x=0, y=0, w=10, h=8)
    assert word.ocr_confidence == 0.9
    assert word.engine_agreement == 1.0


def test_three_engine_overlap_clusters_with_full_agreement():
    aligned = align_word_boxes(_F3_INPUT, iou_threshold=0.5, page=0)
    assert len(aligned) == 1
    assert aligned[0].engine_agreement == 1.0
    assert set(aligned[0].engines) == {"tesseract", "easyocr", "rapidocr"}


def test_three_engine_cluster_uses_union_bbox():
    aligned = align_word_boxes(_F3_INPUT, iou_threshold=0.5, page=0)
    assert len(aligned) == 1
    assert aligned[0].bbox == BBox(x=100, y=50, w=84, h=16)


def test_single_engine_agreement_is_one():
    aligned = align_word_boxes(
        [
            (
                "tesseract",
                [
                    _word("LEFT", x=10, y=10, w=20, h=10),
                    _word("RIGHT", x=200, y=80, w=20, h=10),
                ],
            )
        ],
        iou_threshold=0.5,
    )
    assert len(aligned) == 2
    for word in aligned:
        assert word.engine_agreement == 1.0
        assert word.engines == ["tesseract"]


def test_max_width_cluster_is_split():
    aligned = align_word_boxes(
        [
            ("tesseract", [_word("A", x=0, y=0, w=40, h=10)]),
            ("easyocr", [_word("B", x=10, y=0, w=40, h=10)]),
        ],
        iou_threshold=0.5,
        max_cluster_width_px=30,
    )
    assert len(aligned) == 2


def test_empty_input_returns_empty_list():
    assert align_word_boxes([]) == []


def test_empty_engine_word_lists_returns_empty():
    assert align_word_boxes([("tesseract", []), ("easyocr", [])]) == []


def test_invalid_words_skipped_no_exception():
    words = [
        {"text": "OK", "x": 0, "y": 0, "w": 12, "h": 8, "confidence": 0.8},
        {"text": ""},
        {"text": "X"},
        {"text": "X", "x": 0, "y": 0, "w": 0, "h": 8},
        "not-a-dict",
        {"text": "X", "x": "bad", "y": 0, "w": 8, "h": 8},
    ]
    aligned = align_word_boxes([("tesseract", words)])
    assert len(aligned) == 1
    assert aligned[0].text == "OK"


def test_majority_vote_picks_common_text():
    aligned = align_word_boxes(
        [
            ("tesseract", [_word("Invoice", x=10, y=10)]),
            ("easyocr", [_word("Invoice", x=10, y=10)]),
            ("rapidocr", [_word("lnvoice", x=10, y=10, confidence=0.99)]),
        ],
        iou_threshold=0.5,
    )
    assert len(aligned) == 1
    assert aligned[0].text == "Invoice"


def test_tie_breaks_to_highest_confidence():
    aligned = align_word_boxes(
        [
            ("tesseract", [_word("Alpha", x=10, y=10, confidence=0.6)]),
            ("easyocr", [_word("Beta", x=10, y=10, confidence=0.95)]),
        ],
        iou_threshold=0.5,
    )
    assert len(aligned) == 1
    assert aligned[0].text == "Beta"


def test_two_of_three_engines_agreement():
    aligned = align_word_boxes(
        [
            ("tesseract", [_word("Hello", x=10, y=10)]),
            ("easyocr", [_word("Hello", x=12, y=10)]),
            ("rapidocr", [_word("Other", x=200, y=80)]),
        ],
        iou_threshold=0.5,
    )
    overlap = next(word for word in aligned if word.text == "Hello")
    assert overlap.engine_agreement == pytest.approx(2 / 3)


def test_align_word_boxes_does_not_call_get_settings(monkeypatch):
    def _boom() -> None:
        raise AssertionError("get_settings must not be called")

    monkeypatch.setattr("app.config.get_settings", _boom)
    aligned = align_word_boxes(
        [("tesseract", [_word("OK", x=0, y=0)])],
        iou_threshold=0.5,
    )
    assert len(aligned) == 1


def test_reading_order_top_to_bottom_left_to_right():
    aligned = align_word_boxes(
        [
            (
                "tesseract",
                [
                    _word("B", x=10, y=80),
                    _word("A", x=50, y=10),
                    _word("C", x=10, y=10),
                ],
            )
        ],
        iou_threshold=0.5,
    )
    assert [word.text for word in aligned] == ["C", "A", "B"]


def test_reading_order_tolerates_same_row_y_jitter():
    """A flat (y, x) sort would let a slightly-higher word from a column
    further right jump ahead of a same-row word further left — corrupting
    merged_text's left-to-right order and silently breaking any exact
    substring match (dictionary scan/custom terms) spanning that row. Row-
    banding by vertical-center adjacency must keep them in visual reading
    order regardless of a few px of y noise between columns.
    """
    aligned = align_word_boxes(
        [
            (
                "tesseract",
                [
                    _word("Left", x=10, y=12, h=10),
                    _word("Right", x=200, y=8, h=10),  # same row, box starts 4px higher
                    _word("NextRow", x=10, y=80, h=10),
                ],
            )
        ],
        iou_threshold=0.5,
    )
    assert [word.text for word in aligned] == ["Left", "Right", "NextRow"]


def test_char_spans_cover_merged_text():
    aligned = align_word_boxes(
        [
            (
                "tesseract",
                [
                    _word("B", x=10, y=80),
                    _word("A", x=50, y=10),
                    _word("C", x=10, y=10),
                ],
            )
        ],
        iou_threshold=0.5,
    )
    merged = merged_text_from_words(aligned)
    assert merged == "C A B"
    spans: list[tuple[int, int]] = []
    for word in aligned:
        assert merged[word.char_start : word.char_end] == word.text
        spans.append((word.char_start, word.char_end))
    for index, (start_a, end_a) in enumerate(spans):
        for start_b, end_b in spans[index + 1 :]:
            assert end_a <= start_b or end_b <= start_a


def test_align_logs_contain_no_ocr_text(caplog):
    with caplog.at_level(logging.DEBUG, logger="app.services.ocr.ensemble"):
        align_word_boxes(
            [("tesseract", [_word(SECRET_OCR_TOKEN, x=0, y=0)])],
            iou_threshold=0.5,
        )
    assert SECRET_OCR_TOKEN not in caplog.text


def test_format_engine_failure_omits_exception_message():
    message = _format_engine_failure("tesseract", ValueError(SECRET_OCR_TOKEN))
    assert "tesseract" in message
    assert "ValueError" in message
    assert SECRET_OCR_TOKEN not in message


def test_ensemble_tests_do_not_load_gpu_engines():
    assert "easyocr" not in sys.modules
    assert "rapidocr" not in sys.modules


def test_table_bias_breaks_tie_toward_preferred_engine():
    aligned = align_word_boxes(
        [
            ("tesseract", [_word("Alpha", x=10, y=10, confidence=0.6)]),
            ("rapidocr", [_word("Beta", x=10, y=10, confidence=0.3)]),
        ],
        iou_threshold=0.5,
        table_regions=[BBox(x=0, y=0, w=200, h=200)],
        prefer_engine_in_tables="rapidocr",
    )
    assert len(aligned) == 1
    assert aligned[0].text == "Beta"


def test_table_bias_has_no_effect_outside_table_regions():
    aligned = align_word_boxes(
        [
            ("tesseract", [_word("Alpha", x=10, y=10, confidence=0.6)]),
            ("rapidocr", [_word("Beta", x=10, y=10, confidence=0.3)]),
        ],
        iou_threshold=0.5,
        table_regions=[BBox(x=500, y=500, w=50, h=50)],
        prefer_engine_in_tables="rapidocr",
    )
    assert len(aligned) == 1
    assert aligned[0].text == "Alpha"


def test_table_bias_does_not_override_agreement():
    """Bias only breaks ties; a real majority vote still wins."""
    aligned = align_word_boxes(
        [
            ("tesseract", [_word("Invoice", x=10, y=10)]),
            ("easyocr", [_word("Invoice", x=10, y=10)]),
            ("rapidocr", [_word("lnvoice", x=10, y=10, confidence=0.99)]),
        ],
        iou_threshold=0.5,
        table_regions=[BBox(x=0, y=0, w=200, h=200)],
        prefer_engine_in_tables="rapidocr",
    )
    assert len(aligned) == 1
    assert aligned[0].text == "Invoice"


def test_table_bias_none_when_engine_not_in_tie():
    """Preferred engine absent from the tie falls back to confidence."""
    aligned = align_word_boxes(
        [
            ("tesseract", [_word("Alpha", x=10, y=10, confidence=0.6)]),
            ("easyocr", [_word("Beta", x=10, y=10, confidence=0.95)]),
        ],
        iou_threshold=0.5,
        table_regions=[BBox(x=0, y=0, w=200, h=200)],
        prefer_engine_in_tables="rapidocr",
    )
    assert len(aligned) == 1
    assert aligned[0].text == "Beta"


def test_engine_filter_skips_non_whitelisted_engines():
    import asyncio
    from unittest.mock import patch

    from app.services.ocr.ensemble import ensemble_ocr_page

    with (
        patch("app.services.ocr.ensemble.tesseract_available", return_value=True),
        patch("app.services.ocr.ensemble.easyocr_available", return_value=True),
        patch("app.services.ocr.ensemble.rapidocr_available", return_value=True),
        patch(
            "app.services.ocr.ensemble.ocr_image_rapidocr",
            return_value=("Total", 0.9, [_word("Total", x=0, y=0)]),
        ) as mock_rapidocr,
        patch("app.services.ocr.ensemble.ocr_image_tesseract") as mock_tess,
        patch("app.services.ocr.ensemble.ocr_image_easyocr") as mock_easy,
    ):
        asyncio.run(ensemble_ocr_page(None, 0, "eng", ["en"], engine_filter=["rapidocr"]))
        mock_rapidocr.assert_called_once()
        mock_tess.assert_not_called()
        mock_easy.assert_not_called()


def test_default_policy_uses_only_the_primary_engine():
    """No engine_filter, default settings: exactly one engine runs, no vote."""
    import asyncio
    from unittest.mock import patch

    from app.services.ocr.ensemble import ensemble_ocr_page

    with (
        patch("app.services.ocr.ensemble.tesseract_available", return_value=True),
        patch("app.services.ocr.ensemble.easyocr_available", return_value=True),
        patch("app.services.ocr.ensemble.rapidocr_available", return_value=True),
        patch(
            "app.services.ocr.ensemble.ocr_image_rapidocr",
            return_value=("Total", 0.9, [_word("Total", x=0, y=0)]),
        ) as mock_rapidocr,
        patch("app.services.ocr.ensemble.ocr_image_tesseract") as mock_tess,
        patch("app.services.ocr.ensemble.ocr_image_easyocr") as mock_easy,
    ):
        merged, aligned, results = asyncio.run(ensemble_ocr_page(None, 0, "eng", ["en"]))
        mock_rapidocr.assert_called_once()
        mock_tess.assert_not_called()
        mock_easy.assert_not_called()
        assert len(results) == 1
        assert results[0].engine == "rapidocr"
        assert merged == "Total"
        assert aligned[0].engines == ["rapidocr"]


def test_default_policy_falls_back_when_primary_unavailable():
    """Primary engine missing: falls through to the next configured engine."""
    import asyncio
    from unittest.mock import patch

    from app.services.ocr.ensemble import ensemble_ocr_page

    with (
        patch("app.services.ocr.ensemble.rapidocr_available", return_value=False),
        patch("app.services.ocr.ensemble.tesseract_available", return_value=True),
        patch("app.services.ocr.ensemble.easyocr_available", return_value=True),
        patch(
            "app.services.ocr.ensemble.ocr_image_tesseract",
            return_value=("Fallback", 0.8, [_word("Fallback", x=0, y=0)]),
        ) as mock_tess,
        patch("app.services.ocr.ensemble.ocr_image_easyocr") as mock_easy,
    ):
        merged, aligned, results = asyncio.run(ensemble_ocr_page(None, 0, "eng", ["en"]))
        mock_tess.assert_called_once()
        mock_easy.assert_not_called()
        assert len(results) == 1
        assert results[0].engine == "tesseract"
        assert merged == "Fallback"


def test_default_policy_falls_back_when_primary_raises():
    """Primary engine raises: the failure is swallowed and logged, not fatal."""
    import asyncio
    from unittest.mock import patch

    from app.services.ocr.ensemble import ensemble_ocr_page

    with (
        patch("app.services.ocr.ensemble.rapidocr_available", return_value=True),
        patch("app.services.ocr.ensemble.ocr_image_rapidocr", side_effect=RuntimeError("boom")),
        patch("app.services.ocr.ensemble.tesseract_available", return_value=True),
        patch(
            "app.services.ocr.ensemble.ocr_image_tesseract",
            return_value=("Fallback", 0.8, [_word("Fallback", x=0, y=0)]),
        ),
        patch("app.services.ocr.ensemble.easyocr_available", return_value=False),
    ):
        merged, aligned, results = asyncio.run(ensemble_ocr_page(None, 0, "eng", ["en"]))
        assert results[0].engine == "tesseract"
        assert merged == "Fallback"


def test_default_policy_raises_when_all_configured_engines_fail():
    import asyncio
    from unittest.mock import patch

    from app.services.ocr.ensemble import ensemble_ocr_page

    with (
        patch("app.services.ocr.ensemble.rapidocr_available", return_value=False),
        patch("app.services.ocr.ensemble.tesseract_available", return_value=False),
        patch("app.services.ocr.ensemble.easyocr_available", return_value=False),
    ):
        with pytest.raises(RuntimeError):
            asyncio.run(ensemble_ocr_page(None, 0, "eng", ["en"]))


def test_multi_mode_setting_restores_legacy_parallel_vote():
    import asyncio
    from unittest.mock import patch

    from app.config import get_settings
    from app.services.ocr.ensemble import ensemble_ocr_page

    get_settings.cache_clear()
    try:
        with (
            patch.dict("os.environ", {"OCR_ENSEMBLE_MODE": "multi"}),
            patch("app.services.ocr.ensemble.tesseract_available", return_value=True),
            patch("app.services.ocr.ensemble.easyocr_available", return_value=True),
            patch("app.services.ocr.ensemble.rapidocr_available", return_value=True),
            patch(
                "app.services.ocr.ensemble.ocr_image_tesseract",
                return_value=("Invoice", 0.9, [_word("Invoice", x=10, y=10)]),
            ),
            patch(
                "app.services.ocr.ensemble.ocr_image_easyocr",
                return_value=("Invoice", 0.9, [_word("Invoice", x=10, y=10)]),
            ),
            patch(
                "app.services.ocr.ensemble.ocr_image_rapidocr",
                return_value=("Invoice", 0.9, [_word("Invoice", x=10, y=10)]),
            ),
        ):
            get_settings.cache_clear()
            merged, aligned, results = asyncio.run(ensemble_ocr_page(None, 0, "eng", ["en"]))
            assert len(results) == 3
            assert aligned[0].engine_agreement == 1.0
    finally:
        get_settings.cache_clear()
