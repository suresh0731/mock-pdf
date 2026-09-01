"""img2table-based deterministic table/cell geometry.

img2table (OpenCV border/line detection, no ML model weights) reads actual
table borders directly from the image, so the same page always yields the
same cell grid — unlike Docling's TableFormer, a learned model that
docling-project/docling#2081 confirms can "drop or merge cell texts during
prediction... especially when using OCR engines such as EasyOCR or
Tesseract" on OCR-driven scanned tables (exactly the class of document this
pipeline processes). See ``merge_table_geometry`` for how the two are
combined: img2table's cells are preferred for any table region it
corroborates; Docling's own cells are the fallback everywhere else, so table
coverage never regresses relative to Docling alone.

Like Docling, img2table is used for geometry (bboxes) only — text identity
for redaction always comes from the OCR ensemble. Called without an OCR
backend (``ocr=None``), so every ``TableCell.value`` is ``None`` and no
document text ever passes through this module (SEC-001).

img2table needs visible gridlines to detect borders — the opposite of what
OCR wants from ``line_removal.strip_table_lines`` — so callers must feed
this a non-line-stripped image variant (``CanonicalPage.original_image``),
not the line-stripped OCR canonical image. Both are pixel-registered 1:1
(see ``preprocess/canonical.py``), so bboxes from either are directly
comparable/usable against ``EnsembleWord`` geometry.
"""

import io
import logging

from PIL import Image

from app.models.pii_chunk import BBox
from app.services.structure.docling_adapter import DocBlock

logger = logging.getLogger(__name__)

_DEFAULT_MIN_TABLE_IOU = 0.3


def img2table_available() -> bool:
    """True if `import img2table` succeeds."""
    try:
        import img2table  # noqa: F401

        return True
    except ImportError:
        return False


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


def _contains_center(container: BBox, box: BBox) -> bool:
    cx = box.x + box.w / 2
    cy = box.y + box.h / 2
    return (
        container.x <= cx <= container.x + container.w
        and container.y <= cy <= container.y + container.h
    )


def _table_bbox(table: object) -> BBox | None:
    box = getattr(table, "bbox", None)
    if box is None:
        return None
    w = int(round(box.x2 - box.x1))
    h = int(round(box.y2 - box.y1))
    if w < 1 or h < 1:
        return None
    return BBox(x=int(round(box.x1)), y=int(round(box.y1)), w=w, h=h)


def _cell_bbox(cell: object) -> BBox | None:
    box = getattr(cell, "bbox", None)
    if box is None:
        return None
    w = int(round(box.x2 - box.x1))
    h = int(round(box.y2 - box.y1))
    if w < 1 or h < 1:
        return None
    return BBox(x=int(round(box.x1)), y=int(round(box.y1)), w=w, h=h)


_COLLAPSED_ROW_HEIGHT_FRACTION = 0.5


def _max_row_height_fraction(table: object) -> float:
    """Tallest row's height as a fraction of the whole table's own height.

    A genuinely bordered table's rows (even a multi-line-wrapped one) each
    take up a modest, fairly even slice of the table — see
    ``_prefer_bordered_tables``'s docstring for the two real measurements
    (~0.2-0.25) this is calibrated against. A borderless table the strict
    (``implicit_rows=False``) pass couldn't split at all instead collapses
    almost everything below the header into one giant row spanning nearly
    the table's full height (observed directly: ~0.97) — this ratio is
    what tells the two apart. Returns 1.0 (maximally "collapsed") for a
    degenerate zero-height table rather than raising.
    """
    content = getattr(table, "content", None) or {}
    table_box = _table_bbox(table)
    table_h = table_box.h if table_box is not None else 0
    if table_h <= 0:
        return 1.0
    max_row_h = 0.0
    for row in content.values():
        cell_boxes = [b for c in row if (b := _cell_bbox(c)) is not None]
        if not cell_boxes:
            continue
        row_top = min(b.y for b in cell_boxes)
        row_bottom = max(b.y + b.h for b in cell_boxes)
        max_row_h = max(max_row_h, row_bottom - row_top)
    return max_row_h / table_h if max_row_h else 1.0


def _prefer_bordered_tables(strict_tables: list, implicit_tables: list) -> list:
    """Reconcile img2table's two extraction passes per detected table region.

    ``implicit_rows=True`` (the ``implicit`` pass) infers a row break from
    text-line spacing whenever there's no real horizontal border at that
    y-position — necessary for a genuinely borderless table (e.g. a bank
    statement's transaction list with only vertical column rules), but it
    applies that same inference *inside* an ordinary bordered table too: a
    multi-line-wrapped cell's blank inter-line gap looks just like an
    inter-row gap to that heuristic, so a real, single, bordered row
    wrapping a fund name onto 2-3 lines gets sliced into one phantom
    sub-row per physical text line (observed directly: a 7-row bordered
    table became 14 "rows"). Those phantom sub-rows are never merged back
    by ``field_extractor._merge_oversegmented_cells`` — it deliberately
    trusts every img2table cell as already reading a real detected
    gridline — so the wrapped cell's own continuation lines end up
    assigned to a different "cell" than its first line, breaking
    multi-line PII redaction for that row.

    The ``strict`` pass (``implicit_rows=False``) never has this problem:
    a row boundary only ever comes from a real detected border, so a
    genuinely bordered table's row count from this pass is trustworthy
    ground truth. It just can't find *any* row structure in a borderless
    table (observed directly: the same real bordered-columns-only bank
    statement table collapsed from 31 rows to 2), so it can't simply
    replace the implicit pass outright either.

    Reconciling the two: for each strict-pass table whose tallest row
    doesn't dominate the table (below ``_COLLAPSED_ROW_HEIGHT_FRACTION`` —
    i.e. real borders actually did split it into multiple, reasonably
    even rows: the bordered-table case, wrapped cells included), keep the
    strict pass's cells for that region. Otherwise (the strict pass
    couldn't resolve real row structure there at all and collapsed most
    of the table into one row — the borderless case), fall back to the
    implicit pass's cells for that region, exactly matching this module's
    pre-existing behavior for borderless tables. A table only one pass
    detected at all passes through from whichever pass found it.
    """
    used_implicit_idx: set[int] = set()

    def _find_corroborating_implicit(
        box: BBox | None,
    ) -> tuple[int, object] | None:
        if box is None:
            return None
        for i, i_table in enumerate(implicit_tables):
            if i in used_implicit_idx:
                continue
            i_box = _table_bbox(i_table)
            if i_box is not None and _iou(box, i_box) >= _DEFAULT_MIN_TABLE_IOU:
                return i, i_table
        return None

    chosen: list = []
    for s_table in strict_tables:
        s_box = _table_bbox(s_table)
        match = _find_corroborating_implicit(s_box)
        collapsed = _max_row_height_fraction(s_table) >= _COLLAPSED_ROW_HEIGHT_FRACTION
        if collapsed and match is not None:
            chosen.append(match[1])
        else:
            chosen.append(s_table)
        if match is not None:
            used_implicit_idx.add(match[0])

    chosen.extend(
        i_table for i, i_table in enumerate(implicit_tables) if i not in used_implicit_idx
    )
    return chosen


def _run_img2table(image: Image.Image) -> list:
    """Run img2table's border-based extraction on `image`. Text-free (ocr=None).

    Runs twice — once with ``implicit_rows=False`` (real borders only),
    once with the original ``implicit_rows=True`` (borderless-table
    inference) — and reconciles the two per detected table region; see
    ``_prefer_bordered_tables`` for why neither pass alone is trustworthy
    for every table shape.
    """
    from img2table.document import Image as Img2TableImage

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    common_kwargs = dict(
        ocr=None,
        implicit_columns=True,
        borderless_tables=True,
        min_confidence=50,
    )
    strict_tables = Img2TableImage(src=buf.getvalue()).extract_tables(
        implicit_rows=False, **common_kwargs
    )
    implicit_tables = Img2TableImage(src=buf.getvalue()).extract_tables(
        implicit_rows=True, **common_kwargs
    )
    return _prefer_bordered_tables(strict_tables, implicit_tables)


def _table_to_blocks(table: object, start_idx: int) -> list[DocBlock]:
    table_box = _table_bbox(table)
    if table_box is None:
        return []
    blocks = [DocBlock(block_id=f"i2t-{start_idx}", block_type="table", bbox=table_box, text="")]
    idx = start_idx + 1
    content = getattr(table, "content", None) or {}
    for row_idx, row in enumerate(content.values()):
        for col_idx, cell in enumerate(row):
            cell_box = _cell_bbox(cell)
            if cell_box is None:
                continue
            blocks.append(
                DocBlock(
                    block_id=f"i2t-{idx}",
                    block_type="cell",
                    bbox=cell_box,
                    text="",
                    table_column=f"col_{col_idx}",
                    table_row=row_idx,
                )
            )
            idx += 1
    return blocks


def extract_table_geometry(image: Image.Image) -> list[DocBlock]:
    """Run img2table on `image`, returning table/cell DocBlocks (text-free).

    Degrades to `[]` (never raises) when img2table is unavailable, the
    extraction call fails, or the page genuinely has no detectable
    table borders — the last case is a legitimate result, not an error,
    since not every scanned page is a table.

    Args:
        image: A non-line-stripped page image (see module docstring).

    Returns:
        DocBlocks (`table` and `cell` types), or `[]` in degraded mode.
    """
    if not img2table_available():
        # Mirrors docling_adapter.extract_structure's "structure degraded"
        # warning: silently returning [] here (as before) is indistinguishable
        # from "this page has no detectable table borders", so a missing
        # package regresses every table's cell geometry back to Docling's
        # own TableFormer output — exactly the "drop or merge cell text"
        # failure mode this module exists to correct — with nothing in the
        # logs to say so.
        logger.warning("img2table degraded", extra={"reason": "unavailable"})
        return []
    try:
        tables = _run_img2table(image)
    except Exception:
        logger.warning("img2table extraction failed", extra={"reason": "convert_error"})
        return []
    blocks: list[DocBlock] = []
    for table in tables:
        blocks.extend(_table_to_blocks(table, len(blocks)))
    logger.info(
        "img2table geometry extracted",
        extra={
            "table_count": len(tables),
            "cell_count": sum(1 for b in blocks if b.block_type == "cell"),
        },
    )
    return blocks


def merge_table_geometry(
    docling_blocks: list[DocBlock],
    img2table_blocks: list[DocBlock],
    *,
    min_table_iou: float = _DEFAULT_MIN_TABLE_IOU,
) -> list[DocBlock]:
    """Merge img2table's table/cell geometry into Docling's block list.

    For each Docling `table` block, if an unclaimed img2table `table` block
    overlaps it with IoU >= `min_table_iou`, that region's Docling `cell`
    children are dropped and replaced with the img2table table's own cells
    (deterministic OpenCV border detection reading the same actual
    gridlines every run). A Docling table with no corroborating img2table
    table (e.g. faint/broken borders img2table couldn't resolve) keeps its
    own cells unchanged — table coverage never regresses relative to
    Docling alone. img2table tables with no overlapping Docling table (a
    table Docling missed entirely) are added as new blocks. Non-table/cell
    Docling blocks (paragraph, header, footer, label, picture) always pass
    through unchanged.

    Args:
        docling_blocks: Blocks from `docling_adapter.extract_structure`.
        img2table_blocks: Blocks from `extract_table_geometry`.
        min_table_iou: Overlap threshold gating cell replacement.

    Returns:
        Merged block list. Returns `docling_blocks` unchanged (no copy) when
        `img2table_blocks` is empty.
    """
    if not img2table_blocks:
        return docling_blocks

    docling_tables = [b for b in docling_blocks if b.block_type == "table"]
    docling_cells = [b for b in docling_blocks if b.block_type == "cell"]
    passthrough = [b for b in docling_blocks if b.block_type not in ("table", "cell")]

    i2t_tables = [b for b in img2table_blocks if b.block_type == "table"]
    i2t_cells = [b for b in img2table_blocks if b.block_type == "cell"]

    matched_i2t_ids: set[str] = set()
    claimed_docling_cell_ids: set[int] = set()
    merged_tables: list[DocBlock] = []
    merged_cells: list[DocBlock] = []
    replaced_count = 0

    for d_table in docling_tables:
        best_i2t: DocBlock | None = None
        best_iou = 0.0
        for i_table in i2t_tables:
            if i_table.block_id in matched_i2t_ids:
                continue
            score = _iou(d_table.bbox, i_table.bbox)
            if score > best_iou:
                best_iou, best_i2t = score, i_table

        own_cells = [c for c in docling_cells if _contains_center(d_table.bbox, c.bbox)]
        claimed_docling_cell_ids.update(id(c) for c in own_cells)
        merged_tables.append(d_table)

        if best_i2t is not None and best_iou >= min_table_iou:
            matched_i2t_ids.add(best_i2t.block_id)
            merged_cells.extend(c for c in i2t_cells if _contains_center(best_i2t.bbox, c.bbox))
            replaced_count += 1
        else:
            merged_cells.extend(own_cells)

    for i_table in i2t_tables:
        if i_table.block_id in matched_i2t_ids:
            continue
        merged_tables.append(i_table)
        merged_cells.extend(c for c in i2t_cells if _contains_center(i_table.bbox, c.bbox))

    orphan_docling_cells = [c for c in docling_cells if id(c) not in claimed_docling_cell_ids]

    logger.info(
        "table geometry merged",
        extra={
            "docling_tables": len(docling_tables),
            "img2table_tables": len(i2t_tables),
            "tables_replaced_with_img2table_cells": replaced_count,
        },
    )
    return passthrough + merged_tables + merged_cells + orphan_docling_cells
