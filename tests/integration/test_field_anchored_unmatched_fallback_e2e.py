"""End-to-end test for the field-anchored "unmatched" fallback wired into
``RedactPipeline._collect_redactions`` (see
``MockDictionaryStore.best_unambiguous_match``): a field-anchored table-cell
value (e.g. a "Nama Rekening" column entry) so badly OCR-garbled it misses
``lookup()``'s trusted-fuzzy bar under ``restrict_to_known_mappings`` must
still get redacted with the closest known entry's mock value — rather than
being silently dropped and left fully exposed in the output — as long as
that match is decisively unambiguous. Modeled on the real production bug:
"PT 8NILIFE INSURANCE" (digit-substituted, merged "PT BNI LIFE INSURANCE")
in a BNI Life redemption-instruction letter.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

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

_MERGED = "A/C Name : PT 8NILIFE INSURANCE"


def _words() -> list[EnsembleWord]:
    pieces = ["A/C", "Name", ":", "PT", "8NILIFE", "INSURANCE"]
    words = []
    x = 0
    for piece in pieces:
        start = _MERGED.index(piece)
        words.append(
            EnsembleWord(
                text=piece,
                bbox=BBox(x=x, y=200, w=max(30, len(piece) * 12), h=20),
                ocr_confidence=0.9,
                engine_agreement=1.0,
                engines=["tesseract"],
                char_start=start,
                char_end=start + len(piece),
            )
        )
        x += max(30, len(piece) * 12) + 10
    return words


async def _fake_ocr(*args, **kwargs):
    return _MERGED, _words(), []


def _pipeline(tmp_path: Path, monkeypatch) -> tuple[RedactPipeline, MockDictionaryStore]:
    monkeypatch.setattr("app.pipeline.redact.ensemble_ocr_page", _fake_ocr)
    monkeypatch.setattr(
        "app.pipeline.redact.load_pages",
        lambda *a, **k: [RenderedPage(image=Image.new("RGB", (720, 1100), "white"))],
    )
    monkeypatch.setattr("app.pipeline.redact.extract_structure", lambda *a, **k: [])

    settings = Settings(
        shard_base_path=tmp_path,
        mock_dictionary_path=tmp_path / "mappings.json",
        field_detection_enabled=True,
        dictionary_scan_enabled=False,
        restrict_to_known_mappings=True,
        _env_file=None,
    )
    mock_store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    pipeline = RedactPipeline(
        settings=settings,
        mock_store=mock_store,
        ledger_store=LedgerStore(base_dir=tmp_path / "shards"),
        audit_store=AuditStore(),
    )
    return pipeline, mock_store


def test_garbled_field_anchored_value_falls_back_to_closest_known_entry(tmp_path, monkeypatch):
    pipeline, store = _pipeline(tmp_path, monkeypatch)
    known = store.upsert("PT BNI LIFE INSURANCE", "Client-30608778491")
    # An unrelated entry that must not be confused with the real one.
    store.upsert("Sompo Insurance Indonesia", "ORG_UNRELATED")

    _, audit, _ = asyncio.run(pipeline.run(b"%PDF-1.4", "doc.pdf", RedactOptions()))

    org_region = next(
        (r for r in audit.redactions if r.entity_type == "ORGANIZATION" and r.mock_value),
        None,
    )
    assert org_region is not None, "garbled field-anchored value must still be redacted"
    assert org_region.mapping_id == known.mapping_id
    assert org_region.mock_value == known.mock_value
    # No spurious new dictionary entry was minted for the garbled text.
    assert len(store.list()) == 2


def test_ambiguous_garbled_value_is_still_dropped_not_guessed(tmp_path, monkeypatch):
    """Same OCR garbling, but now the dictionary contains two entries close
    enough to the garbled text that picking either would be a guess — the
    fallback must stay silent (no redaction, no wrong-client mock) rather
    than risk painting the wrong client's mock value.
    """
    pipeline, store = _pipeline(tmp_path, monkeypatch)
    # Neither entry is close enough to "PT 8NILIFE INSURANCE" to win
    # decisively (best_unambiguous_match's margin check must reject both).
    store.upsert("PT ABC Life Insurance", "ORG_ABC")
    store.upsert("PT XYZ Life Insurance", "ORG_XYZ")

    _, audit, _ = asyncio.run(pipeline.run(b"%PDF-1.4", "doc.pdf", RedactOptions()))

    org_region = next(
        (r for r in audit.redactions if r.entity_type == "ORGANIZATION" and r.mock_value),
        None,
    )
    assert org_region is None
    # Still no new dictionary entry created for the unmatched text.
    assert len(store.list()) == 2
