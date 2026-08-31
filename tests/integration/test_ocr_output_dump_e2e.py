"""Regression test: RedactPipeline.run() writes the per-document OCR
output diagnostic dump (Settings.ocr_output_dump_enabled) — and skips it
cleanly when the flag is disabled.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

from PIL import Image

from app.config import Settings
from app.models.pii_chunk import BBox
from app.models.redact import RedactOptions
from app.pipeline.redact import RedactPipeline
from app.services.ocr.ensemble_types import EnsembleWord
from app.services.ocr.page_renderer import RenderedPage
from app.services.pii.mock_dictionary import MockDictionaryStore
from app.services.redact.audit_store import AuditStore
from app.services.redact.ledger_store import LedgerStore
from app.services.redact.ocr_output_store import OcrOutputStore

PERSON = "Jane Doe"
MERGED_TEXT = "Contacted by Jane Doe"
PERSON_START = MERGED_TEXT.find(PERSON)
PERSON_END = PERSON_START + len(PERSON)


def _blank_image() -> Image.Image:
    return Image.new("RGB", (200, 100), "white")


async def _fake_ocr(*args, **kwargs):
    word = EnsembleWord(
        text=PERSON,
        bbox=BBox(x=20, y=20, w=90, h=18),
        ocr_confidence=0.9,
        engine_agreement=1.0,
        engines=["tesseract"],
        page=0,
        char_start=PERSON_START,
        char_end=PERSON_END,
    )
    return MERGED_TEXT, [word], []


def _pipeline(tmp_path: Path, monkeypatch, **settings_overrides) -> tuple[RedactPipeline, MockDictionaryStore]:
    monkeypatch.setenv("SHARD_BASE_PATH", str(tmp_path))
    monkeypatch.setenv("MOCK_DICTIONARY_PATH", str(tmp_path / "mappings.json"))
    settings = Settings(
        shard_base_path=tmp_path,
        mock_dictionary_path=tmp_path / "mappings.json",
        field_detection_enabled=False,
        restrict_to_known_mappings=False,
        _env_file=None,
        **settings_overrides,
    )
    mock_store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    mock_store.resolve(PERSON)
    pipeline = RedactPipeline(
        settings=settings,
        mock_store=mock_store,
        ledger_store=LedgerStore(base_dir=tmp_path / "shards"),
        audit_store=AuditStore(),
        ocr_output_store=OcrOutputStore(base_dir=tmp_path / "ocr-output"),
    )
    return pipeline, mock_store


def _patch_ocr(monkeypatch) -> None:
    pages = [RenderedPage(image=_blank_image(), fitz_page=None)]
    monkeypatch.setattr("app.pipeline.redact.load_pages", lambda *a, **k: pages)
    monkeypatch.setattr(
        "app.pipeline.redact.ensemble_ocr_page", AsyncMock(side_effect=_fake_ocr)
    )
    monkeypatch.setattr("app.pipeline.redact.extract_structure", lambda *a, **k: [])


def test_run_writes_ocr_output_dump_by_default(tmp_path, monkeypatch):
    _patch_ocr(monkeypatch)
    pipeline, _store = _pipeline(tmp_path, monkeypatch)

    _, audit, _ = asyncio.run(pipeline.run(b"%PDF-1.4", "doc.pdf", RedactOptions()))

    dump_path = tmp_path / "ocr-output" / f"{audit.request_id}.json"
    assert dump_path.exists()
    payload = json.loads(dump_path.read_text(encoding="utf-8"))
    assert payload["request_id"] == audit.request_id
    assert payload["filename"] == "doc.pdf"
    assert payload["page_count"] == 1
    page = payload["pages"][0]
    assert page["page_kind"] == "scanned"
    assert page["merged_text"] == MERGED_TEXT
    assert page["words"][0]["text"] == PERSON


def test_ocr_output_dump_disabled_writes_nothing(tmp_path, monkeypatch):
    _patch_ocr(monkeypatch)
    pipeline, _store = _pipeline(tmp_path, monkeypatch, ocr_output_dump_enabled=False)

    _, audit, _ = asyncio.run(pipeline.run(b"%PDF-1.4", "doc.pdf", RedactOptions()))

    dump_dir = tmp_path / "ocr-output"
    assert not (dump_dir / f"{audit.request_id}.json").exists()


def test_regenerate_does_not_write_a_new_dump(tmp_path, monkeypatch):
    """regenerate() reuses cached page_states with nothing new to OCR —
    only run() should ever write a dump file."""
    _patch_ocr(monkeypatch)
    pipeline, _store = _pipeline(tmp_path, monkeypatch)

    _, audit, session = asyncio.run(pipeline.run(b"%PDF-1.4", "doc.pdf", RedactOptions()))
    dump_dir = tmp_path / "ocr-output"
    assert len(list(dump_dir.glob("*.json"))) == 1

    _, audit2, _ = asyncio.run(pipeline.regenerate(session.session_id))
    assert audit2.request_id != audit.request_id
    assert len(list(dump_dir.glob("*.json"))) == 1
