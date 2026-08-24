import logging
from types import SimpleNamespace

import pytest
from PIL import Image

from app.models.pii_chunk import BBox
from app.services.ocr.ensemble_types import EnsembleWord
from app.services.structure.docling_adapter import (
    DocBlock,
    bbox_from_prov,
    canonical_block_type,
    extract_structure,
    map_docling_items,
)


@pytest.fixture
def blank_image() -> Image.Image:
    return Image.new("RGB", (200, 200), "white")


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


def _prov_item(
    label: str,
    text: str,
    l: float,
    t: float,
    r: float,
    b: float,
    origin: str = "TOPLEFT",
) -> SimpleNamespace:
    bbox = SimpleNamespace(l=l, t=t, r=r, b=b, coord_origin=origin)
    return SimpleNamespace(label=label, text=text, prov=[SimpleNamespace(bbox=bbox)])


def test_extract_structure_uses_injected_backend_without_docling(blank_image):
    expected = [_block(block_id="dl-0", block_type="paragraph", x=0, y=0, w=10, h=10)]

    class _Backend:
        def extract(self, image, ocr_text=None):
            return expected

    out = extract_structure(blank_image, ocr_text="S1234567A", backend=_Backend())
    assert out[0].block_id == "dl-0"


def test_extract_structure_returns_empty_when_unavailable(blank_image, monkeypatch):
    monkeypatch.setattr(
        "app.services.structure.docling_adapter.docling_available",
        lambda: False,
    )
    assert extract_structure(blank_image) == []


def test_extract_structure_returns_empty_when_disabled(blank_image):
    class _BoomBackend:
        def __init__(self) -> None:
            self.called = False

        def extract(self, image, ocr_text=None):
            self.called = True
            raise RuntimeError("backend should not be called when disabled")

    backend = _BoomBackend()
    assert extract_structure(blank_image, enabled=False, backend=backend) == []
    assert backend.called is False


def test_extract_structure_returns_empty_when_backend_raises(blank_image):
    class _FailingBackend:
        def extract(self, image, ocr_text=None):
            raise RuntimeError("boom")

    assert extract_structure(blank_image, backend=_FailingBackend()) == []


def test_extract_structure_ignores_non_str_ocr_text(blank_image):
    expected = [_block(block_id="dl-1", block_type="paragraph", x=1, y=1, w=2, h=2)]

    class _RecordingBackend:
        def __init__(self) -> None:
            self.ocr_text = "unset"

        def extract(self, image, ocr_text=None):
            self.ocr_text = ocr_text
            return expected

    backend = _RecordingBackend()
    out = extract_structure(blank_image, [_word("x", 0, 0, 1, 1)], backend=backend)
    assert backend.ocr_text is None
    assert out[0].block_id == "dl-1"


def test_canonical_block_type_maps_header_footer_picture():
    assert canonical_block_type("PAGE_HEADER") == "header"
    assert canonical_block_type("page_footer") == "footer"
    assert canonical_block_type("PICTURE") == "picture"
    assert canonical_block_type("Table-Cell") == "cell"
    assert canonical_block_type("TEXT") == "paragraph"


def test_map_keeps_empty_text_header_footer_picture():
    items = [
        _prov_item("picture", "", 500, 0, 700, 80),
        _prov_item("header", "", 0, 0, 200, 30),
        _prov_item("footer", "", 0, 180, 200, 200),
        _prov_item("paragraph", "", 10, 40, 50, 60),
    ]
    blocks = map_docling_items(items, page_h=200)
    types = [block.block_type for block in blocks]
    assert types == ["picture", "header", "footer"]
    assert all(block.text == "" for block in blocks)


def test_bbox_from_prov_topleft_and_bottomleft():
    topleft = SimpleNamespace(l=10, t=20, r=40, b=50, coord_origin="TOPLEFT")
    bottomleft = SimpleNamespace(l=10, t=80, r=40, b=50, coord_origin="BOTTOMLEFT")
    assert bbox_from_prov(topleft, 100) == BBox(x=10, y=20, w=30, h=30)
    assert bbox_from_prov(bottomleft, 100) == BBox(x=10, y=20, w=30, h=30)


def test_extract_structure_logs_omit_block_text(blank_image, caplog):
    secret_block = _block(
        block_id="dl-pii",
        block_type="paragraph",
        x=0,
        y=0,
        w=10,
        h=10,
        text="S1234567A",
    )

    class _Backend:
        def extract(self, image, ocr_text=None):
            return [secret_block]

    with caplog.at_level(logging.INFO):
        extract_structure(
            blank_image,
            ocr_text="Dian Wicaksono",
            backend=_Backend(),
        )
    assert "S1234567A" not in caplog.text
    assert "Dian Wicaksono" not in caplog.text
