"""Composite redaction confidence (1A Stage 3b weights).

Logs scores and word counts only. Never logs EnsembleWord.text (SEC-001).
"""

import logging

from app.models.redact import ConfidenceBreakdown, StructuralContext
from app.services.ocr.ensemble_types import EnsembleWord

logger = logging.getLogger(__name__)

WEIGHT_PRESIDIO = 0.35
WEIGHT_OCR = 0.25
WEIGHT_AGREEMENT = 0.25
WEIGHT_STRUCTURAL = 0.15

_PARAGRAPH_BLOCK_TYPES = frozenset({"paragraph", "text", "section_header"})


def _structural_component(ctx: StructuralContext | None) -> float:
    """1.0 labeled table/field; 0.5 paragraph; 0.0 orphan.

    Args:
        ctx: Spatial-join context, or ``None`` for an orphan span.

    Returns:
        Structural weight component in ``{0.0, 0.5, 1.0}``.
    """
    if ctx is None:
        return 0.0
    if ctx.table_column or ctx.block_label:
        return 1.0
    if ctx.block_type in _PARAGRAPH_BLOCK_TYPES:
        return 0.5
    return 0.0


def score_redaction(
    presidio_score: float,
    matched_words: list[EnsembleWord],
    structural_context: StructuralContext | None,
) -> tuple[float, ConfidenceBreakdown]:
    """Composite confidence per 1A Stage 3b.

    Args:
        presidio_score: Detector score for the span (0–1).
        matched_words: Ensemble words overlapping the span.
        structural_context: Spatial-join context, or ``None`` if orphan.

    Returns:
        Rounded composite score and a ``ConfidenceBreakdown`` with each
        field rounded to 4 decimals.
    """
    if matched_words:
        ocr = sum(w.ocr_confidence for w in matched_words) / len(matched_words)
        agreement = sum(w.engine_agreement for w in matched_words) / len(
            matched_words
        )
    else:
        ocr = 0.0
        agreement = 0.0
    structural = _structural_component(structural_context)
    total = (
        presidio_score * WEIGHT_PRESIDIO
        + ocr * WEIGHT_OCR
        + agreement * WEIGHT_AGREEMENT
        + structural * WEIGHT_STRUCTURAL
    )
    logger.debug(
        "redaction scored",
        extra={
            "presidio": round(presidio_score, 4),
            "ocr": round(ocr, 4),
            "agreement": round(agreement, 4),
            "structural": round(structural, 4),
            "total": round(total, 4),
            "word_count": len(matched_words),
        },
    )
    breakdown = ConfidenceBreakdown(
        presidio=round(presidio_score, 4),
        ocr=round(ocr, 4),
        engine_agreement=round(agreement, 4),
        structural_context=round(structural, 4),
    )
    return round(total, 4), breakdown
