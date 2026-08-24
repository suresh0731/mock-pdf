import logging

from app.models.pii_chunk import BBox
from app.models.redact import StructuralContext
from app.services.ocr.ensemble_types import EnsembleWord
from app.services.pii.redaction_scorer import score_redaction


def _word(
    text: str,
    ocr: float = 1.0,
    agreement: float = 1.0,
) -> EnsembleWord:
    return EnsembleWord(
        text=text,
        bbox=BBox(x=0, y=0, w=10, h=10),
        ocr_confidence=ocr,
        engine_agreement=agreement,
        char_start=0,
        char_end=len(text),
    )


def test_score_redaction_all_ones_is_one():
    ctx = StructuralContext(block_type="table", table_column="COL", join_iou=0.8)
    composite, breakdown = score_redaction(1.0, [_word("ALPHA")], ctx)
    assert composite == 1.0
    assert breakdown.presidio == 1.0
    assert breakdown.ocr == 1.0
    assert breakdown.engine_agreement == 1.0
    assert breakdown.structural_context == 1.0


def test_score_redaction_presidio_only_is_0_35():
    composite, breakdown = score_redaction(1.0, [], None)
    assert composite == 0.35
    assert breakdown.ocr == 0.0
    assert breakdown.engine_agreement == 0.0
    assert breakdown.structural_context == 0.0


def test_score_redaction_paragraph_structural_is_0_5():
    ctx = StructuralContext(block_type="paragraph")
    composite, breakdown = score_redaction(0.0, [], ctx)
    assert breakdown.structural_context == 0.5
    assert composite == 0.075


def test_score_redaction_orphan_structural_is_zero():
    composite, breakdown = score_redaction(0.0, [], None)
    assert composite == 0.0
    assert breakdown.structural_context == 0.0


def test_score_redaction_mean_of_two_words():
    words = [_word("ALPHA", ocr=1.0, agreement=1.0), _word("BETA", ocr=0.0, agreement=0.0)]
    composite, breakdown = score_redaction(0.0, words, None)
    assert breakdown.ocr == 0.5
    assert breakdown.engine_agreement == 0.5
    assert composite == 0.25


def test_score_redaction_logs_omit_word_text(caplog):
    with caplog.at_level(logging.DEBUG, logger="app.services.pii.redaction_scorer"):
        score_redaction(1.0, [_word("ALPHA")], None)
    assert "ALPHA" not in caplog.text
