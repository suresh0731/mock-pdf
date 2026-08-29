"""Regression test: when every OCR engine fails/returns nothing for a page
(app.services.ocr.ensemble.ensemble_ocr_page raises), the pipeline must not
abort the whole document if that page still carries *some* native PDF text
layer -- even one too thin to have cleared the "digital" classification
threshold on its own (see app/services/ocr/native_text.py's "hybrid page"
case). A page with any real copyable text is strictly better redacted from
that text than dropped with a hard pipeline failure.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from PIL import Image

from app.config import Settings
from app.models.redact import RedactOptions
from app.pipeline.redact import RedactPipeline
from app.services.ocr.page_renderer import RenderedPage
from app.services.pii.mock_dictionary import MockDictionaryStore
from app.services.redact.audit_store import AuditStore
from app.services.redact.ledger_store import LedgerStore

PERSON = "Jane Doe"


class FakeFitzPage:
    """A "hybrid" page: a thin embedded text layer (2 words) that fails the
    digital-page word-count threshold on its own, but is still real,
    extractable text -- e.g. a baked-in OCR layer from a scan-to-PDF tool.
    """

    rect = SimpleNamespace(width=200.0, height=100.0)

    def get_text(self, mode: str):
        if mode == "words":
            return [
                (20.0, 20.0, 60.0, 35.0, "Jane", 0, 0, 0),
                (65.0, 20.0, 100.0, 35.0, "Doe", 0, 0, 1),
            ]
        if mode == "blocks":
            return [(20.0, 20.0, 100.0, 35.0, "Jane Doe", 0, 0)]
        raise ValueError(mode)


def _blank_image() -> Image.Image:
    return Image.new("RGB", (200, 100), "white")


async def _failing_ocr(*args, **kwargs):
    raise RuntimeError("All configured OCR engines failed or unavailable")


def _pipeline(tmp_path: Path, monkeypatch, **settings_overrides) -> tuple[RedactPipeline, MockDictionaryStore]:
    monkeypatch.setenv("SHARD_BASE_PATH", str(tmp_path))
    monkeypatch.setenv("MOCK_DICTIONARY_PATH", str(tmp_path / "mappings.json"))
    settings = Settings(
        shard_base_path=tmp_path,
        mock_dictionary_path=tmp_path / "mappings.json",
        field_detection_enabled=False,
        restrict_to_known_mappings=False,
        # Default thresholds (20 words / 2% coverage): this page's 2 words
        # must fail classification and be routed to OCR first.
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
    )
    return pipeline, mock_store


def test_ocr_failure_falls_back_to_native_text_when_available(tmp_path, monkeypatch):
    pages = [RenderedPage(image=_blank_image(), fitz_page=FakeFitzPage())]
    monkeypatch.setattr("app.pipeline.redact.load_pages", lambda *a, **k: pages)
    ocr_mock = AsyncMock(side_effect=_failing_ocr)
    monkeypatch.setattr("app.pipeline.redact.ensemble_ocr_page", ocr_mock)
    monkeypatch.setattr("app.pipeline.redact.extract_structure", lambda *a, **k: [])

    pipeline, store = _pipeline(tmp_path, monkeypatch)

    _, audit, _ = asyncio.run(pipeline.run(b"%PDF-1.4", "doc.pdf", RedactOptions()))

    ocr_mock.assert_called_once()
    # Classified "scanned" up front (2 words < default min_words=20), but
    # the page-level PipelineStageError never fires because native text
    # extraction rescues it -- reported "digital" since that's what was
    # actually used to redact it.
    assert [p.page_kind for p in audit.pages] == ["digital"]
    person_region = next(r for r in audit.redactions if r.page == 0)
    assert person_region.mock_value == store.resolve(PERSON).mock_value


def test_ocr_failure_skips_blank_page_by_default(tmp_path, monkeypatch):
    """A page with no OCR text AND no native text layer at all (no
    fitz_page) -- either genuinely blank or an unreadably degraded scan --
    must not abort the whole document by default
    (Settings.ocr_blank_page_skip_enabled=True): it's treated as blank
    (0 redactions possible on it) and the run still completes."""
    pages = [RenderedPage(image=_blank_image(), fitz_page=None)]
    monkeypatch.setattr("app.pipeline.redact.load_pages", lambda *a, **k: pages)
    ocr_mock = AsyncMock(side_effect=_failing_ocr)
    monkeypatch.setattr("app.pipeline.redact.ensemble_ocr_page", ocr_mock)
    monkeypatch.setattr("app.pipeline.redact.extract_structure", lambda *a, **k: [])

    pipeline, _store = _pipeline(tmp_path, monkeypatch)

    _, audit, _ = asyncio.run(pipeline.run(b"%PDF-1.4", "doc.pdf", RedactOptions()))

    assert [p.page_kind for p in audit.pages] == ["blank"]
    # A footer/brand-zone patch (position-based, no text needed) may still
    # land on a blank page -- only actual PII redactions are impossible.
    assert not any(r.page == 0 for r in audit.redactions if r.assignment_source != "brand")


def test_ocr_failure_still_raises_when_blank_page_skip_disabled(tmp_path, monkeypatch):
    """With Settings.ocr_blank_page_skip_enabled=False, a page with no OCR
    text and no native text layer must still surface the OCR failure --
    the stricter choice for a document type expected to always carry PII
    on every page."""
    pages = [RenderedPage(image=_blank_image(), fitz_page=None)]
    monkeypatch.setattr("app.pipeline.redact.load_pages", lambda *a, **k: pages)
    ocr_mock = AsyncMock(side_effect=_failing_ocr)
    monkeypatch.setattr("app.pipeline.redact.ensemble_ocr_page", ocr_mock)
    monkeypatch.setattr("app.pipeline.redact.extract_structure", lambda *a, **k: [])

    pipeline, _store = _pipeline(tmp_path, monkeypatch, ocr_blank_page_skip_enabled=False)

    from app.pipeline.errors import PipelineStageError

    try:
        asyncio.run(pipeline.run(b"%PDF-1.4", "doc.pdf", RedactOptions()))
    except PipelineStageError as exc:
        assert exc.stage == "ensemble_ocr"
    else:
        raise AssertionError("expected PipelineStageError")
