"""Map a detector's character spans to ensemble-word bounding boxes.

Span matching is char-interval overlap only. ``merged_text`` is accepted for
pipeline positional compatibility and is never sliced or logged (SEC-001).
"""

import logging

from app.models.pii_chunk import BBox
from app.services.ocr.ensemble_types import EnsembleWord

logger = logging.getLogger(__name__)


def words_for_span(
    start: int, end: int, words: list[EnsembleWord]
) -> list[EnsembleWord]:
    """Words whose [char_start, char_end) overlaps [start, end).

    Args:
        start: Inclusive character offset of the span.
        end: Exclusive character offset of the span.
        words: Ensemble words with char offsets in the same space.

    Returns:
        Matching words in input order, or an empty list if the span is
        invalid (``start >= end``) or ``words`` is empty.
    """
    if start >= end or not words:
        return []
    return [w for w in words if w.char_start < end and w.char_end > start]


def union_bbox(words: list[EnsembleWord]) -> BBox | None:
    """Smallest bbox covering every word's box, or ``None`` if ``words`` is empty."""
    if not words:
        return None
    boxes = [w.bbox for w in words]
    x1 = min(b.x for b in boxes)
    y1 = min(b.y for b in boxes)
    x2 = max(b.x + b.w for b in boxes)
    y2 = max(b.y + b.h for b in boxes)
    return BBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)


def map_span_to_ensemble_bbox(
    start: int,
    end: int,
    words: list[EnsembleWord],
    merged_text: str,
    *,
    redaction_id: str | None = None,
) -> BBox | None:
    """Union bbox of words overlapping the span, or None if none / invalid.

    Args:
        start: Inclusive character offset of the span.
        end: Exclusive character offset of the span.
        words: Ensemble words with char offsets in the same space.
        merged_text: Unused positional arg kept for pipeline compatibility.
            Must not be sliced or logged.
        redaction_id: Optional caller-assigned correlation ID (e.g.
            ``"{page}:{start}:{end}"``), stamped onto this debug line only
            so it can be joined with ``coordinate_map``'s "bbox padded and
            clamped" and ``redaction_scorer``'s "redaction scored" lines
            for the same redaction — otherwise these three lines are only
            distinguishable by re-deriving from geometry/ordering. Never
            used for matching logic.

    Returns:
        Union bbox of overlapping words, or ``None`` when there is no match
        or the span is invalid.
    """
    del merged_text
    matched = words_for_span(start, end, words)
    box = union_bbox(matched)
    if box is None:
        logger.debug(
            "span mapping produced no bbox",
            extra={"redaction_id": redaction_id, "start": start, "end": end, "word_count": 0},
        )
        return None
    logger.debug(
        "span mapped to bbox",
        extra={
            "redaction_id": redaction_id,
            "start": start,
            "end": end,
            "word_count": len(matched),
            "x": box.x,
            "y": box.y,
            "w": box.w,
            "h": box.h,
        },
    )
    return box
