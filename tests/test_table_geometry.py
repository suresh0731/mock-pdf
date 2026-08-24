from collections import OrderedDict
from types import SimpleNamespace

from app.models.pii_chunk import BBox
from app.services.structure.docling_adapter import DocBlock
from app.services.structure.table_geometry import (
    extract_table_geometry,
    img2table_available,
    merge_table_geometry,
)


def _block(
    *,
    block_id: str,
    block_type: str,
    x: int,
    y: int,
    w: int,
    h: int,
    table_column: str | None = None,
    table_row: int | None = None,
) -> DocBlock:
    return DocBlock(
        block_id=block_id,
        block_type=block_type,
        bbox=BBox(x=x, y=y, w=w, h=h),
        text="",
        table_column=table_column,
        table_row=table_row,
    )


def _i2t_bbox(x1, y1, x2, y2):
    return SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2)


def _i2t_cell(x1, y1, x2, y2, value=None):
    return SimpleNamespace(bbox=_i2t_bbox(x1, y1, x2, y2), value=value)


def _i2t_table(x1, y1, x2, y2, rows: list[list[SimpleNamespace]]):
    content = OrderedDict(enumerate(rows))
    return SimpleNamespace(bbox=_i2t_bbox(x1, y1, x2, y2), title=None, content=content)


def test_img2table_available_reflects_real_import():
    # img2table is a pinned direct dependency (requirements.txt) so this
    # should be importable in the test environment.
    assert img2table_available() is True


def test_extract_table_geometry_returns_empty_when_unavailable(monkeypatch):
    monkeypatch.setattr(
        "app.services.structure.table_geometry.img2table_available", lambda: False
    )
    from PIL import Image

    assert extract_table_geometry(Image.new("RGB", (10, 10), "white")) == []


def test_extract_table_geometry_returns_empty_on_extraction_error(monkeypatch):
    from PIL import Image

    monkeypatch.setattr(
        "app.services.structure.table_geometry.img2table_available", lambda: True
    )

    def _boom(image):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.services.structure.table_geometry._run_img2table", _boom)
    assert extract_table_geometry(Image.new("RGB", (10, 10), "white")) == []


def test_extract_table_geometry_maps_table_and_cells(monkeypatch):
    from PIL import Image

    monkeypatch.setattr(
        "app.services.structure.table_geometry.img2table_available", lambda: True
    )
    table = _i2t_table(
        10,
        10,
        110,
        50,
        rows=[
            [_i2t_cell(10, 10, 60, 30), _i2t_cell(60, 10, 110, 30)],
            [_i2t_cell(10, 30, 60, 50), _i2t_cell(60, 30, 110, 50)],
        ],
    )
    monkeypatch.setattr(
        "app.services.structure.table_geometry._run_img2table", lambda image: [table]
    )

    blocks = extract_table_geometry(Image.new("RGB", (200, 200), "white"))
    tables = [b for b in blocks if b.block_type == "table"]
    cells = [b for b in blocks if b.block_type == "cell"]
    assert len(tables) == 1
    assert tables[0].bbox == BBox(x=10, y=10, w=100, h=40)
    assert len(cells) == 4
    assert {c.table_column for c in cells} == {"col_0", "col_1"}
    assert {c.table_row for c in cells} == {0, 1}
    # Never carries OCR/text content — geometry only (SEC-001).
    assert all(c.text == "" for c in cells)


def test_extract_table_geometry_skips_degenerate_cells(monkeypatch):
    from PIL import Image

    monkeypatch.setattr(
        "app.services.structure.table_geometry.img2table_available", lambda: True
    )
    table = _i2t_table(0, 0, 50, 50, rows=[[_i2t_cell(0, 0, 0, 20)]])
    monkeypatch.setattr(
        "app.services.structure.table_geometry._run_img2table", lambda image: [table]
    )
    blocks = extract_table_geometry(Image.new("RGB", (100, 100), "white"))
    assert all(b.block_type != "cell" for b in blocks)


def test_merge_returns_docling_blocks_unchanged_when_no_img2table():
    docling_blocks = [_block(block_id="dl-0", block_type="paragraph", x=0, y=0, w=10, h=10)]
    assert merge_table_geometry(docling_blocks, []) is docling_blocks


def test_merge_replaces_cells_when_img2table_table_overlaps():
    docling_table = _block(block_id="dl-0", block_type="table", x=0, y=0, w=100, h=100)
    docling_cell = _block(
        block_id="dl-1", block_type="cell", x=0, y=0, w=50, h=50, table_column="A"
    )
    img2table_table = _block(block_id="i2t-0", block_type="table", x=0, y=0, w=100, h=100)
    img2table_cell = _block(
        block_id="i2t-1", block_type="cell", x=0, y=0, w=50, h=50, table_column="col_0"
    )

    merged = merge_table_geometry(
        [docling_table, docling_cell], [img2table_table, img2table_cell]
    )

    assert docling_table in merged
    assert docling_cell not in merged
    assert img2table_cell in merged


def test_merge_keeps_docling_cells_when_no_overlapping_img2table_table():
    docling_table = _block(block_id="dl-0", block_type="table", x=0, y=0, w=100, h=100)
    docling_cell = _block(block_id="dl-1", block_type="cell", x=0, y=0, w=50, h=50)
    # Far away, no overlap with the Docling table.
    img2table_table = _block(block_id="i2t-0", block_type="table", x=500, y=500, w=50, h=50)
    img2table_cell = _block(block_id="i2t-1", block_type="cell", x=500, y=500, w=25, h=25)

    merged = merge_table_geometry(
        [docling_table, docling_cell], [img2table_table, img2table_cell]
    )

    assert docling_cell in merged
    assert img2table_table in merged
    assert img2table_cell in merged


def test_merge_preserves_non_table_blocks_and_orphan_cells():
    paragraph = _block(block_id="dl-p", block_type="paragraph", x=0, y=0, w=10, h=10)
    docling_table = _block(block_id="dl-0", block_type="table", x=0, y=0, w=100, h=100)
    # Orphan cell not spatially inside any Docling table — must not be dropped.
    orphan_cell = _block(block_id="dl-orphan", block_type="cell", x=900, y=900, w=10, h=10)
    img2table_table = _block(block_id="i2t-0", block_type="table", x=0, y=0, w=100, h=100)
    img2table_cell = _block(block_id="i2t-1", block_type="cell", x=0, y=0, w=50, h=50)

    merged = merge_table_geometry(
        [paragraph, docling_table, orphan_cell], [img2table_table, img2table_cell]
    )

    assert paragraph in merged
    assert orphan_cell in merged
    assert img2table_cell in merged


def test_merge_respects_min_table_iou_threshold():
    """Below-threshold overlap keeps Docling's own cells (no replacement) and
    treats the img2table table as a distinct, additional table rather than a
    corroborating match — both tables' cells end up present side by side."""
    docling_table = _block(block_id="dl-0", block_type="table", x=0, y=0, w=100, h=100)
    docling_cell = _block(block_id="dl-1", block_type="cell", x=0, y=0, w=50, h=50)
    # Small sliver overlap — IoU well under the 0.3 threshold.
    img2table_table = _block(block_id="i2t-0", block_type="table", x=95, y=95, w=100, h=100)
    img2table_cell = _block(block_id="i2t-1", block_type="cell", x=96, y=96, w=2, h=2)

    merged = merge_table_geometry(
        [docling_table, docling_cell],
        [img2table_table, img2table_cell],
        min_table_iou=0.3,
    )

    assert docling_cell in merged
    assert img2table_cell in merged
    assert img2table_table in merged
