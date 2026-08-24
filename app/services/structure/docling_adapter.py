"""Docling structure adapter mapping layout items to DocBlocks."""

import logging
import tempfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from app.models.pii_chunk import BBox

logger = logging.getLogger(__name__)

CANONICAL_BLOCK_TYPES = frozenset(
    {
        "paragraph",
        "table",
        "header",
        "label",
        "cell",
        "footer",
        "picture",
    }
)
_KEEP_EMPTY_TEXT_TYPES = frozenset({"header", "footer", "picture"})
_HEADER_KEYS = ("page_header", "section_header", "header", "title")
_FOOTER_KEYS = ("page_footer", "footer")
_PICTURE_KEYS = ("picture", "figure", "image", "chart")
_CELL_KEYS = ("table_cell", "cell")
_LABEL_KEYS = ("label", "caption", "key_value")


@dataclass
class DocBlock:
    """A mapped document structure block (paragraph, table, chrome, or cell)."""

    block_id: str
    block_type: str  # one of CANONICAL_BLOCK_TYPES
    bbox: BBox
    text: str
    table_column: str | None = None
    table_row: int | None = None
    parent_label: str | None = None


class StructureBackend(Protocol):
    """Replaceable Docling (or test) extractor. Returns already-mapped DocBlocks."""

    def extract(self, image: Image.Image, ocr_text: str | None = None) -> list[DocBlock]:
        """Extract structure blocks from a page image."""
        ...


def docling_available() -> bool:
    """True if `import docling` succeeds. Keep existing try/except ImportError."""
    try:
        import docling  # noqa: F401

        return True
    except ImportError:
        return False


def _normalize_label(raw_label: object) -> str:
    if raw_label is None:
        return ""
    value = getattr(raw_label, "value", raw_label)
    return str(value).strip().lower()


def canonical_block_type(raw_label: str | None) -> str:
    """Map Docling/enum labels to CANONICAL_BLOCK_TYPES. Unknown → 'paragraph'.

    Args:
        raw_label: Docling label, enum, or None.

    Returns:
        One of CANONICAL_BLOCK_TYPES.
    """
    text = _normalize_label(raw_label)
    if not text:
        mapped = "paragraph"
    elif any(key in text for key in _HEADER_KEYS):
        mapped = "header"
    elif any(key in text for key in _FOOTER_KEYS):
        mapped = "footer"
    elif any(key in text for key in _PICTURE_KEYS):
        mapped = "picture"
    elif any(key in text for key in _CELL_KEYS):
        mapped = "cell"
    elif "table" in text:
        mapped = "table"
    elif any(key in text for key in _LABEL_KEYS):
        mapped = "label"
    else:
        mapped = "paragraph"
    return mapped if mapped in CANONICAL_BLOCK_TYPES else "paragraph"


def bbox_from_prov(bbox: object, page_h: int) -> BBox | None:
    """Convert a Docling l/t/r/b provenance box to a pixel BBox.

    BOTTOMLEFT origin: y = page_h - t; h = t - b. w,h ≥ 1.

    Args:
        bbox: Object with l/t/r/b and optional coord_origin.
        page_h: Page height in pixels.

    Returns:
        Pixel BBox, or None if the provenance box is unusable.
    """
    try:
        left = getattr(bbox, "l", None)
        top = getattr(bbox, "t", None)
        right = getattr(bbox, "r", None)
        bottom = getattr(bbox, "b", None)
        if None in (left, top, right, bottom):
            return None
        left_f = float(left)
        top_f = float(top)
        right_f = float(right)
        bottom_f = float(bottom)
        origin = getattr(bbox, "coord_origin", None)
        origin_str = str(getattr(origin, "value", origin) or "").upper()
        if "BOTTOMLEFT" in origin_str or (
            "TOPLEFT" not in origin_str and top_f > bottom_f
        ):
            y = int(round(page_h - top_f))
            height = int(round(top_f - bottom_f))
        else:
            y = int(round(top_f))
            height = int(round(bottom_f - top_f))
        width = int(round(right_f - left_f))
        if width < 1 or height < 1:
            width = max(1, width)
            height = max(1, height)
        return BBox(x=int(round(left_f)), y=y, w=width, h=height)
    except (TypeError, ValueError, AttributeError):
        return None


def _unwrap_item(raw: object) -> object:
    if isinstance(raw, tuple) and raw:
        return raw[0]
    return raw


def _item_bbox(item: object, page_h: int) -> BBox | None:
    prov = getattr(item, "prov", None)
    if prov:
        try:
            page_prov = prov[0]
        except (IndexError, TypeError, KeyError):
            page_prov = None
        if page_prov is not None:
            raw = getattr(page_prov, "bbox", page_prov)
            box = bbox_from_prov(raw, page_h)
            if box is not None:
                return box
    raw = getattr(item, "bbox", None)
    if raw is None:
        return None
    if isinstance(raw, BBox):
        return raw
    return bbox_from_prov(raw, page_h)


def _item_text(item: object) -> str:
    text = getattr(item, "text", "") or ""
    return str(text).strip()


def _as_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_attr(obj: object, *names: str) -> object:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return None


def _as_nonempty_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _iter_table_cells(data: object) -> Iterable[object]:
    cells = getattr(data, "table_cells", None)
    if cells is None:
        cells = getattr(data, "grid", None)
    if cells is None:
        return
    try:
        for row in cells:
            if isinstance(row, (list, tuple)):
                yield from row
            else:
                yield row
    except TypeError:
        return


def _first_row_headers(cells: Iterable[object]) -> dict[int, str]:
    headers: dict[int, str] = {}
    for cell in cells:
        row_idx = _as_int(_first_attr(cell, "start_row_offset_idx", "row"))
        col_idx = _as_int(_first_attr(cell, "start_col_offset_idx", "col"))
        if row_idx != 0 or col_idx is None:
            continue
        text = _item_text(cell)
        if text:
            headers[col_idx] = text
    return headers


def _cell_column(cell: object, headers: dict[int, str]) -> str | None:
    explicit = _as_nonempty_str(getattr(cell, "table_column", None))
    if explicit:
        return explicit
    col_idx = _as_int(_first_attr(cell, "start_col_offset_idx", "col"))
    if col_idx is not None:
        return headers.get(col_idx)
    return None


def _cell_row(cell: object) -> int | None:
    return _as_int(_first_attr(cell, "table_row", "start_row_offset_idx", "row"))


def _type_counts(blocks: list[DocBlock]) -> dict[str, int]:
    return dict(Counter(block.block_type for block in blocks))


def map_docling_items(items: Iterable[object], page_h: int) -> list[DocBlock]:
    """Map Docling (or fixture) items to DocBlocks.

    Keeps empty-text header/footer/picture blocks. Drops other empty-text
    items and anything without a usable bbox.

    Args:
        items: Iterable of Docling items or (item, level) tuples.
        page_h: Page height in pixels for origin conversion.

    Returns:
        Mapped DocBlock list in encounter order.
    """
    blocks: list[DocBlock] = []
    idx = 0
    for raw in items:
        item = _unwrap_item(raw)
        box = _item_bbox(item, page_h)
        if box is None:
            continue
        raw_label = getattr(item, "label", None)
        label = (
            raw_label
            if raw_label is None or isinstance(raw_label, str)
            else _normalize_label(raw_label)
        )
        block_type = canonical_block_type(label)
        text = _item_text(item)
        data = getattr(item, "data", None)
        table_cells = list(_iter_table_cells(data)) if data is not None else []
        if table_cells:
            blocks.append(
                DocBlock(
                    block_id=f"dl-{idx}",
                    block_type="table",
                    bbox=box,
                    text=text,
                )
            )
            idx += 1
            try:
                headers = _first_row_headers(table_cells)
                for cell in table_cells:
                    cell_box = _item_bbox(cell, page_h)
                    if cell_box is None:
                        continue
                    blocks.append(
                        DocBlock(
                            block_id=f"dl-{idx}",
                            block_type="cell",
                            bbox=cell_box,
                            text=_item_text(cell),
                            table_column=_cell_column(cell, headers),
                            table_row=_cell_row(cell),
                        )
                    )
                    idx += 1
            except Exception:
                continue
            continue
        if not text and block_type not in _KEEP_EMPTY_TEXT_TYPES:
            continue
        blocks.append(
            DocBlock(
                block_id=f"dl-{idx}",
                block_type=block_type,
                bbox=box,
                text=text,
            )
        )
        idx += 1
    return blocks


def _blocks_from_docling(image: Image.Image) -> list[DocBlock]:
    from docling.document_converter import DocumentConverter

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "page.png"
        image.save(path, format="PNG")
        converter = DocumentConverter()
        result = converter.convert(str(path))
        doc = result.document
        return map_docling_items(doc.iterate_items(), image.height)


def extract_structure(
    image: Image.Image,
    ocr_text: str | None = None,
    *,
    enabled: bool = True,
    backend: StructureBackend | None = None,
) -> list[DocBlock]:
    """Extract document structure from a page image.

    `ocr_text` is accepted (never logged) and unused by live convert in v1.
    Degraded (return []): enabled=False; no backend and not
    docling_available(); backend/convert raises. Never crash.

    Args:
        image: Canonical page image.
        ocr_text: Optional OCR text. Non-str values are ignored.
        enabled: When False, skip extraction and return [].
        backend: Injected extractor; used instead of live Docling.

    Returns:
        Mapped DocBlocks, or [] in degraded mode.
    """
    if ocr_text is not None and not isinstance(ocr_text, str):
        ocr_text = None
    if not enabled:
        logger.warning("structure degraded", extra={"reason": "disabled"})
        return []
    try:
        if backend is not None:
            blocks = backend.extract(image, ocr_text)
        elif not docling_available():
            logger.warning("structure degraded", extra={"reason": "unavailable"})
            return []
        else:
            blocks = _blocks_from_docling(image)
    except Exception:
        logger.warning("structure degraded", extra={"reason": "convert_error"})
        return []
    logger.info(
        "structure extracted",
        extra={"block_count": len(blocks), "type_counts": _type_counts(blocks)},
    )
    return blocks
