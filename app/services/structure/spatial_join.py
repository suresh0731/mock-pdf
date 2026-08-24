"""Spatial join of ensemble words onto Docling structure blocks."""

import logging

from app.models.pii_chunk import BBox
from app.models.redact import StructuralContext
from app.services.ocr.ensemble_types import EnsembleWord
from app.services.structure.docling_adapter import DocBlock

logger = logging.getLogger(__name__)

_LABELED_SCORE = 1.0
_PARAGRAPH_SCORE = 0.5
_ORPHAN_SCORE = 0.0
_HALF_SCORE_TYPES = frozenset({"paragraph", "text", "header", "section_header"})


def _iou(a: BBox, b: BBox) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def _nonempty(value: str | None) -> bool:
    return bool(value)


def structural_context_score(ctx: StructuralContext | None) -> float:
    """Score a joined structural context for redaction confidence.

    1.0 labeled table/field; 0.5 paragraph (and header); 0.0 orphan / other
    unlabeled.

    Args:
        ctx: Joined context, or None for an orphan word.

    Returns:
        Score in {0.0, 0.5, 1.0}.
    """
    if ctx is None:
        return _ORPHAN_SCORE
    if _nonempty(ctx.table_column) or _nonempty(ctx.block_label):
        return _LABELED_SCORE
    if ctx.block_type in _HALF_SCORE_TYPES:
        return _PARAGRAPH_SCORE
    return _ORPHAN_SCORE


def join_words_to_blocks(
    words: list[EnsembleWord],
    blocks: list[DocBlock],
) -> dict[int, StructuralContext]:
    """Map each word index to the block with strictly greatest IoU > 0.

    Orphan (no overlap): omit the index (caller treats missing as None).
    Tie: keep the first block that reached that IoU (`>` not `>=`).
    Empty words or blocks: `{}` — no exception.

    Args:
        words: Ensemble words for a page.
        blocks: Docling (or fixture) structure blocks.

    Returns:
        Word index → StructuralContext for overlapping words only.
    """
    result: dict[int, StructuralContext] = {}
    for idx, word in enumerate(words):
        best_block: DocBlock | None = None
        best_iou = 0.0
        for block in blocks:
            score = _iou(word.bbox, block.bbox)
            if score > best_iou:
                best_iou = score
                best_block = block
        if best_block is None or best_iou <= 0:
            continue
        result[idx] = StructuralContext(
            block_id=best_block.block_id,
            block_type=best_block.block_type,
            block_label=best_block.parent_label,
            table_column=best_block.table_column,
            table_row=best_block.table_row,
            join_iou=round(best_iou, 4),
        )
    logger.info(
        "words joined to blocks",
        extra={
            "joined_count": len(result),
            "orphan_count": max(0, len(words) - len(result)),
        },
    )
    return result
