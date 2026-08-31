"""Tests for the per-document OCR output diagnostic dump."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.models.mock import MockValidationError
from app.models.pii_chunk import BBox
from app.models.redact import PageTransform
from app.pipeline.page_state import PageProcessState
from app.services.ocr.ensemble_types import EnsembleWord
from app.services.preprocess.canonical import CanonicalPage
from app.services.redact.ocr_output_store import OcrOutputStore

SPECIAL_TEXT = "Jos\u00e9 O\u2019Br\u00e9\u00efn \ufffd"  # accents, smart quote, replacement char


def _page_state(page_index: int, page_kind: str, text: str) -> PageProcessState:
    canonical = CanonicalPage(
        page_index=page_index,
        original_image=None,
        canonical_image=None,
        transform=PageTransform(),
    )
    word = EnsembleWord(
        text=text,
        bbox=BBox(x=10, y=20, w=30, h=15),
        ocr_confidence=0.88,
        engine_agreement=1.0,
        engines=["rapidocr"],
        page=page_index,
        char_start=0,
        char_end=len(text),
    )
    return PageProcessState(
        canonical=canonical,
        merged_text=text,
        ensemble_words=[word],
        page_kind=page_kind,
    )


@pytest.fixture
def store(tmp_path: Path) -> OcrOutputStore:
    return OcrOutputStore(base_dir=tmp_path / "ocr-output")


def test_save_creates_one_file_per_document(store: OcrOutputStore, tmp_path: Path) -> None:
    states = [_page_state(0, "scanned", SPECIAL_TEXT), _page_state(1, "digital", "Plain text")]
    path = store.save("req-1", "statement.pdf", states)

    assert path == tmp_path / "ocr-output" / "req-1.json"
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["request_id"] == "req-1"
    assert payload["filename"] == "statement.pdf"
    assert payload["page_count"] == 2
    assert [p["page_index"] for p in payload["pages"]] == [0, 1]
    assert [p["page_kind"] for p in payload["pages"]] == ["scanned", "digital"]


def test_save_preserves_special_characters_verbatim(
    store: OcrOutputStore, tmp_path: Path
) -> None:
    """Non-ASCII/garbled characters must round-trip exactly — this file
    exists specifically to diagnose whether OCR itself produced a special
    character that later breaks fuzzy matching.
    """
    states = [_page_state(0, "scanned", SPECIAL_TEXT)]
    store.save("req-2", "doc.pdf", states)

    raw = (tmp_path / "ocr-output" / "req-2.json").read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert payload["pages"][0]["merged_text"] == SPECIAL_TEXT
    assert payload["pages"][0]["words"][0]["text"] == SPECIAL_TEXT


def test_save_includes_per_word_geometry_and_confidence(store: OcrOutputStore) -> None:
    states = [_page_state(0, "scanned", "Hello")]
    path = store.save("req-3", "doc.pdf", states)
    payload = json.loads(path.read_text(encoding="utf-8"))
    word = payload["pages"][0]["words"][0]
    assert word["x"] == 10 and word["y"] == 20 and word["w"] == 30 and word["h"] == 15
    assert word["confidence"] == 0.88
    assert word["engines"] == ["rapidocr"]
    assert word["engine_agreement"] == 1.0
    assert word["char_start"] == 0
    assert word["char_end"] == len("Hello")


def test_save_rejects_path_traversal_request_id(store: OcrOutputStore, tmp_path: Path) -> None:
    with pytest.raises(MockValidationError) as exc_info:
        store.save("../etc", "doc.pdf", [])
    assert exc_info.value.field == "request_id"
    outside = tmp_path / "etc"
    assert not outside.exists()
