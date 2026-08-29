"""End-to-end regression: a mixed *real* PDF document (page 0 digital,
page 1 scanned) must come out of a full ``RedactPipeline.run()`` call as
a PDF whose page 0 is genuine vector content (redacted text truly
removed, non-redacted text still selectable, no flattened raster image)
and whose page 1 is the existing painted-then-flattened raster page —
exercising the full pipeline -> ``render_redacted_pdf`` assembly, not
just ``pdf_renderer``/``pdf_native_redactor`` in isolation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import fitz

from app.config import Settings
from app.models.pii_chunk import BBox
from app.models.redact import RedactOptions
from app.pipeline.redact import RedactPipeline
from app.services.ocr.ensemble_types import EnsembleWord
from app.services.pii.mock_dictionary import MockDictionaryStore
from app.services.redact.audit_store import AuditStore
from app.services.redact.ledger_store import LedgerStore

ORG = "Acme Holdings"
PERSON = "Jane Roe"
SCANNED_MERGED = "Contacted by Jane Roe"
PERSON_START = SCANNED_MERGED.find(PERSON)
PERSON_END = PERSON_START + len(PERSON)


def _build_source_pdf() -> bytes:
    """A real 2-page PDF: page 0 carries a genuine text layer (digital),
    page 1 has no text layer at all (scanned)."""
    doc = fitz.open()
    digital_page = doc.new_page(width=300, height=150)
    digital_page.insert_text((10, 30), "Client: Acme Holdings account statement")
    doc.new_page(width=300, height=150)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


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
        # Keep the synthetic ~5-word digital page above threshold without
        # needing a realistic 20+ word page.
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


def test_mixed_real_pdf_output_has_vector_digital_page_and_raster_scanned_page(tmp_path, monkeypatch):
    pdf_bytes = _build_source_pdf()
    ocr_mock = AsyncMock(side_effect=_fake_ocr)
    monkeypatch.setattr("app.pipeline.redact.ensemble_ocr_page", ocr_mock)
    monkeypatch.setattr("app.pipeline.redact.extract_structure", lambda *a, **k: [])

    pipeline, store = _pipeline(tmp_path, monkeypatch)

    pdf_out, audit, _ = asyncio.run(pipeline.run(pdf_bytes, "doc.pdf", RedactOptions()))

    assert [p.page_kind for p in audit.pages] == ["digital", "scanned"]

    out_doc = fitz.open(stream=pdf_out, filetype="pdf")
    try:
        assert len(out_doc) == 2

        # Page 0: real vector content. The redacted org name is gone,
        # the surrounding non-redacted text survives verbatim, the mock
        # value was drawn back in as real text, and no raster image was
        # embedded (which a flattened page always would have).
        page0_text = out_doc[0].get_text()
        assert "Client:" in page0_text
        assert "account statement" in page0_text
        assert ORG not in page0_text
        assert store.resolve(ORG).mock_value in page0_text
        assert len(out_doc[0].get_images()) == 0

        # Page 1: the existing painted-then-flattened raster path —
        # no extractable text, exactly one embedded page image.
        assert out_doc[1].get_text().strip() == ""
        assert len(out_doc[1].get_images()) == 1
    finally:
        out_doc.close()
