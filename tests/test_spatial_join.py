import logging

from app.models.pii_chunk import BBox
from app.models.redact import StructuralContext
from app.services.ocr.ensemble_types import EnsembleWord
from app.services.structure.docling_adapter import DocBlock
from app.services.structure.spatial_join import (
    join_words_to_blocks,
    structural_context_score,
)


def _word(text: str, x: int, y: int, w: int, h: int) -> EnsembleWord:
    return EnsembleWord(
        text=text,
        bbox=BBox(x=x, y=y, w=w, h=h),
        ocr_confidence=0.9,
        engine_agreement=1.0,
        engines=["tesseract"],
        page=0,
        char_start=0,
        char_end=len(text),
    )


def _block(
    *,
    block_id: str,
    block_type: str,
    x: int,
    y: int,
    w: int,
    h: int,
    text: str = "",
    table_column: str | None = None,
    table_row: int | None = None,
    parent_label: str | None = None,
) -> DocBlock:
    return DocBlock(
        block_id=block_id,
        block_type=block_type,
        bbox=BBox(x=x, y=y, w=w, h=h),
        text=text,
        table_column=table_column,
        table_row=table_row,
        parent_label=parent_label,
    )


def test_join_word_to_table_cell_sets_column_row():
    word = _word("nric", 12, 12, 20, 10)
    cell = _block(
        block_id="cell-0",
        block_type="cell",
        x=10,
        y=10,
        w=40,
        h=20,
        table_column="NRIC",
        table_row=2,
    )
    paragraph = _block(
        block_id="p-0",
        block_type="paragraph",
        x=0,
        y=0,
        w=200,
        h=200,
    )
    result = join_words_to_blocks([word], [paragraph, cell])
    assert result[0].block_type == "cell"
    assert result[0].table_column == "NRIC"
    assert result[0].table_row == 2
    assert result[0].join_iou > 0


def test_join_word_with_no_overlap_is_orphan():
    word = _word("orphan", 0, 0, 10, 10)
    block = _block(block_id="b-0", block_type="paragraph", x=50, y=50, w=10, h=10)
    result = join_words_to_blocks([word], [block])
    assert 0 not in result
    assert structural_context_score(None) == 0.0


def test_score_labeled_cell_paragraph_orphan():
    labeled = StructuralContext(
        block_type="cell",
        table_column="NRIC",
    )
    paragraph = StructuralContext(block_type="paragraph")
    assert structural_context_score(labeled) == 1.0
    assert structural_context_score(paragraph) == 0.5
    assert structural_context_score(None) == 0.0


def test_score_header_is_half_unlabeled_footer_zero():
    header = StructuralContext(block_type="header")
    footer = StructuralContext(block_type="footer")
    assert structural_context_score(header) == 0.5
    assert structural_context_score(footer) == 0.0


def test_join_empty_inputs_return_empty_dict():
    block = _block(block_id="b-0", block_type="paragraph", x=0, y=0, w=10, h=10)
    word = _word("x", 0, 0, 10, 10)
    assert join_words_to_blocks([], [block]) == {}
    assert join_words_to_blocks([word], []) == {}


def test_join_passes_through_picture_block_type():
    word = _word("logo", 10, 10, 20, 20)
    picture = _block(block_id="pic-0", block_type="picture", x=0, y=0, w=50, h=50)
    result = join_words_to_blocks([word], [picture])
    assert result[0].block_type == "picture"


def test_join_logs_omit_word_and_block_text(caplog):
    word = _word("Standard Chartered", 0, 0, 20, 10)
    block = _block(
        block_id="b-0",
        block_type="paragraph",
        x=0,
        y=0,
        w=40,
        h=20,
        text="S1234567A",
    )
    with caplog.at_level(logging.INFO):
        join_words_to_blocks([word], [block])
    assert "Standard Chartered" not in caplog.text
    assert "S1234567A" not in caplog.text
