"""Regression test for the native-text bypass on a mixed document: one
page with a real embedded/copyable text layer (page 0) followed by a
rasterized-only scanned page (page 1). Confirms both branches run inside
a single ``RedactPipeline.run()`` call, `page_kind` is reported correctly
per page on the audit, OCR is only invoked for the scanned page, and
redactions still land correctly on both.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
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

ORG = "Acme Holdings"
PERSON = "Jane Doe"
SCANNED_MERGED = "Contacted by Jane Doe"
PERSON_START = SCANNED_MERGED.find(PERSON)
PERSON_END = PERSON_START + len(PERSON)


class FakeFitzPage:
    """Minimal fitz.Page double carrying a real embedded "Acme Holdings"."""

    rect = SimpleNamespace(width=200.0, height=100.0)

    def get_text(self, mode: str):
        if mode == "words":
            return [
                (10.0, 10.0, 50.0, 25.0, "Acme", 0, 0, 0),
                (55.0, 10.0, 120.0, 25.0, "Holdings", 0, 0, 1),
            ]
        if mode == "blocks":
            return [(10.0, 10.0, 120.0, 25.0, "Acme Holdings", 0, 0)]
        raise ValueError(mode)


def _blank_image() -> Image.Image:
    return Image.new("RGB", (200, 100), "white")


def _scanned_words() -> list[EnsembleWord]:
    return [
        EnsembleWord(
            text=PERSON,
            bbox=BBox(x=20, y=20, w=90, h=18),
            ocr_confidence=0.9,
            engine_agreement=1.0,
            engines=["tesseract"],
            page=1,
            char_start=PERSON_START,
            char_end=PERSON_END,
        )
    ]


async def _fake_ocr(*args, **kwargs):
    return SCANNED_MERGED, _scanned_words(), []


def _pipeline(tmp_path: Path, monkeypatch) -> tuple[RedactPipeline, MockDictionaryStore]:
    monkeypatch.setenv("SHARD_BASE_PATH", str(tmp_path))
    monkeypatch.setenv("MOCK_DICTIONARY_PATH", str(tmp_path / "mappings.json"))
    settings = Settings(
        shard_base_path=tmp_path,
        mock_dictionary_path=tmp_path / "mappings.json",
        field_detection_enabled=False,
        restrict_to_known_mappings=False,
        # Keep the synthetic 2-word digital page above threshold without
        # needing to fabricate a realistic 20+ word page (that heuristic
        # boundary is already covered by tests/test_native_text.py).
        native_text_min_words=2,
        native_text_min_coverage_pct=0.0,
        _env_file=None,
    )
    mock_store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    mock_store.resolve(ORG)
    mock_store.resolve(PERSON)
    pipeline = RedactPipeline(
        settings=settings,
        mock_store=mock_store,
        ledger_store=LedgerStore(base_dir=tmp_path / "shards"),
        audit_store=AuditStore(),
    )
    return pipeline, mock_store


def test_mixed_document_bypasses_ocr_on_digital_page_only(tmp_path, monkeypatch):
    pages = [
        RenderedPage(image=_blank_image(), fitz_page=FakeFitzPage()),
        RenderedPage(image=_blank_image(), fitz_page=None),
    ]
    monkeypatch.setattr("app.pipeline.redact.load_pages", lambda *a, **k: pages)
    ocr_mock = AsyncMock(side_effect=_fake_ocr)
    monkeypatch.setattr("app.pipeline.redact.ensemble_ocr_page", ocr_mock)
    monkeypatch.setattr("app.pipeline.redact.extract_structure", lambda *a, **k: [])

    pipeline, store = _pipeline(tmp_path, monkeypatch)

    _, audit, _ = asyncio.run(pipeline.run(b"%PDF-1.4", "doc.pdf", RedactOptions()))

    assert [p.page_kind for p in audit.pages] == ["digital", "scanned"]
    # OCR only ran once, for the scanned page — the digital page's native
    # text layer was used instead.
    ocr_mock.assert_called_once()

    org_region = next(r for r in audit.redactions if r.page == 0)
    person_region = next(r for r in audit.redactions if r.page == 1)
    assert org_region.mock_value == store.resolve(ORG).mock_value
    assert person_region.mock_value == store.resolve(PERSON).mock_value
