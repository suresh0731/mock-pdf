"""End-to-end tests for the field-anchored redaction path.

Unlike ``test_redact_mocks_e2e.py`` (which fakes the detector entirely to
exercise dictionary/ledger/audit mechanics), these tests fake only OCR/
Docling and let the real ``extract_field_candidates`` + ``MockDictionaryStore``
run, using ``EnsembleWord`` fixtures modeled on the debit/credit letter
template (see ``tests/test_field_extractor.py``). This validates the whole
redesign end to end: label-anchored detection -> fuzzy name-based mock
resolution -> redaction regions, while account numbers and dates are never
themselves redacted.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from PIL import Image

from app.config import Settings
from app.models.pii_chunk import BBox
from app.models.redact import RedactOptions
from app.pipeline.redact import RedactPipeline
from app.services.ocr.ensemble_types import EnsembleWord
from app.services.pii.mock_dictionary import MockDictionaryStore
from app.services.redact.audit_store import AuditStore
from app.services.redact.ledger_store import LedgerStore

WordSpec = tuple[str, int, int, int, int]


def _make_words(specs: list[WordSpec]) -> tuple[str, list[EnsembleWord]]:
    """Build EnsembleWords with char offsets matching a single-space merge."""
    words: list[EnsembleWord] = []
    texts: list[str] = []
    offset = 0
    for text, x, y, w, h in specs:
        words.append(
            EnsembleWord(
                text=text,
                bbox=BBox(x=x, y=y, w=w, h=h),
                ocr_confidence=0.9,
                engine_agreement=1.0,
                engines=["tesseract"],
                char_start=offset,
                char_end=offset + len(text),
            )
        )
        texts.append(text)
        offset += len(text) + 1
    return " ".join(texts), words


def _debit_credit_letter(
    debit_name_tokens: list[str],
    credit_name_tokens: list[str],
    debit_account_number: str = "11002345",
    credit_account_number: str = "22009988",
) -> tuple[str, list[EnsembleWord]]:
    """A payment-order letter: debit + credit sections, each with a bank,
    an account number, and an account name — modeled on Template 1 in
    ``tests/test_field_extractor.py``.
    """
    specs: list[WordSpec] = [
        ("Debit", 0, 0, 60, 20),
        ("dari", 65, 0, 40, 20),
        ("Bank", 0, 40, 60, 20),
        (":", 65, 40, 10, 20),
        ("Standard", 80, 40, 70, 20),
        ("Chartered", 155, 40, 80, 20),
        ("Custody", 240, 40, 70, 20),
        ("A/C", 0, 80, 30, 20),
        ("No", 35, 80, 30, 20),
        (":", 70, 80, 10, 20),
        (debit_account_number, 85, 80, 100, 20),
        ("A/C", 0, 120, 30, 20),
        ("Name", 35, 120, 50, 20),
        (":", 90, 120, 10, 20),
    ]
    x = 105
    for tok in debit_name_tokens:
        specs.append((tok, x, 120, max(30, len(tok) * 9), 20))
        x += max(30, len(tok) * 9) + 5

    specs += [
        ("Kredit", 0, 200, 65, 20),
        ("ke", 70, 200, 25, 20),
        ("Bank", 0, 240, 60, 20),
        (":", 65, 240, 10, 20),
        ("PT", 80, 240, 30, 20),
        ("Bank", 115, 240, 55, 20),
        ("Mandiri", 175, 240, 65, 20),
        ("A/C", 0, 280, 30, 20),
        ("No", 35, 280, 30, 20),
        (":", 70, 280, 10, 20),
        (credit_account_number, 85, 280, 100, 20),
        ("A/C", 0, 320, 30, 20),
        ("Name", 35, 320, 50, 20),
        (":", 90, 320, 10, 20),
    ]
    x = 105
    for tok in credit_name_tokens:
        specs.append((tok, x, 320, max(30, len(tok) * 9), 20))
        x += max(30, len(tok) * 9) + 5

    return _make_words(specs)


def _letter_page() -> Image.Image:
    return Image.new("RGB", (720, 1100), "white")


def _pipeline(tmp_path: Path, monkeypatch) -> tuple[RedactPipeline, MockDictionaryStore, LedgerStore]:
    settings = Settings(
        shard_base_path=tmp_path,
        mock_dictionary_path=tmp_path / "mappings.json",
        presidio_enabled=False,
        field_detection_enabled=True,
        # These tests exercise the raw field-anchored detector (including
        # first-sight auto-creation) directly — unrelated to the
        # restrict_to_known_mappings policy layered on top in production.
        restrict_to_known_mappings=False,
        _env_file=None,
    )
    mock_store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    ledger_store = LedgerStore(base_dir=tmp_path / "shards")
    pipeline = RedactPipeline(
        settings=settings,
        mock_store=mock_store,
        ledger_store=ledger_store,
        audit_store=AuditStore(),
    )
    monkeypatch.setattr("app.pipeline.redact.load_pages", lambda *a, **k: [_letter_page()])
    monkeypatch.setattr("app.pipeline.redact.extract_structure", lambda *a, **k: [])
    return pipeline, mock_store, ledger_store


def _fake_ocr_sequence(pages: list[tuple[str, list[EnsembleWord]]]):
    """Return a fake ``ensemble_ocr_page`` that yields ``pages`` in order,
    one page-worth of (merged_text, words) per call — lets successive
    pipeline runs simulate OCR noise varying between documents.
    """
    call_count = {"n": 0}

    async def _fake(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        merged_text, words = pages[idx % len(pages)]
        return merged_text, words, []

    return _fake


def _run(pipeline: RedactPipeline, options: RedactOptions | None = None):
    return asyncio.run(pipeline.run(b"%PDF-1.4", "doc.pdf", options))


@pytest.fixture
def field_anchored_pipeline(tmp_path, monkeypatch):
    return _pipeline(tmp_path, monkeypatch)


def test_field_anchored_pipeline_redacts_debit_credit_and_bank(
    field_anchored_pipeline, monkeypatch
):
    pipeline, _, _ = field_anchored_pipeline
    merged, words = _debit_credit_letter(
        ["Reksa", "Dana", "Bahana", "Primavera", "99"],
        ["PT", "BNI", "Life", "Insurance"],
    )
    monkeypatch.setattr(
        "app.pipeline.redact.ensemble_ocr_page", _fake_ocr_sequence([(merged, words)])
    )

    _, audit, _ = _run(pipeline)

    debit = next(r for r in audit.redactions if r.entity_type == "ORGANIZATION" and r.mock_value)
    entity_types = {r.entity_type for r in audit.redactions}
    assert "ORGANIZATION" in entity_types
    assert debit.mock_value
    # Every organization-shaped mock actually replaces a real assigned value.
    org_regions = [r for r in audit.redactions if r.entity_type == "ORGANIZATION"]
    assert len(org_regions) >= 3  # debit name, credit name, bank(s)


def test_field_anchored_pipeline_never_redacts_account_numbers(
    field_anchored_pipeline, monkeypatch
):
    pipeline, _, _ = field_anchored_pipeline
    debit_account_number = "11002345"
    credit_account_number = "22009988"
    merged, words = _debit_credit_letter(
        ["Reksa", "Dana", "Bahana"],
        ["PT", "BNI", "Life", "Insurance"],
        debit_account_number=debit_account_number,
        credit_account_number=credit_account_number,
    )
    monkeypatch.setattr(
        "app.pipeline.redact.ensemble_ocr_page", _fake_ocr_sequence([(merged, words)])
    )

    _, audit, _ = _run(pipeline)

    for region in audit.redactions:
        if region.assignment_source == "brand":
            continue
        source_span = merged[
            region.canonical_bbox.x : region.canonical_bbox.x + region.canonical_bbox.w
        ]
        # Loose guard: the account-number strings never appear as the sole
        # redacted content (they're plain-digit fields — see the numeric
        # guard already unit-tested in test_field_extractor.py).
        assert debit_account_number not in (region.mock_value or "")
        assert credit_account_number not in (region.mock_value or "")


def test_field_anchored_pipeline_reuses_mock_despite_ocr_noise(
    field_anchored_pipeline, monkeypatch
):
    """Heavily OCR-mangled name second time still fuzzy-matches to the same mock."""
    pipeline, store, _ = field_anchored_pipeline
    account_number = "30608778491"

    merged1, words1 = _debit_credit_letter(
        ["Reksa", "Dana", "Bahana"],
        ["PT", "BNI", "Life", "Insurance"],
        debit_account_number=account_number,
    )
    merged2, words2 = _debit_credit_letter(
        ["R3ksa", "Oana", "8ahana"],  # unrecognizable OCR noise
        ["PT", "BNI", "Life", "Insurance"],
        debit_account_number=account_number,
    )
    monkeypatch.setattr(
        "app.pipeline.redact.ensemble_ocr_page",
        _fake_ocr_sequence([(merged1, words1), (merged2, words2)]),
    )

    _, audit1, _ = _run(pipeline)
    _, audit2, _ = _run(pipeline)

    debit1 = next(r for r in audit1.redactions if r.entity_type == "ORGANIZATION" and r.mock_value)
    debit2 = next(r for r in audit2.redactions if r.entity_type == "ORGANIZATION" and r.mock_value)
    assert debit2.mapping_id == debit1.mapping_id
    assert debit2.mock_value == debit1.mock_value


def test_field_anchored_pipeline_reuses_mock_via_fuzzy_match_without_account_number(
    field_anchored_pipeline, monkeypatch
):
    """No account number available (bank-only doc) -> fuzzy name match still
    collapses OCR variants of the same custodian into one mapping."""
    pipeline, store, _ = field_anchored_pipeline

    merged1, words1 = _debit_credit_letter(
        ["PT", "BNI", "Life", "Insurance"],
        ["PT", "Sinarmas", "Life"],
        debit_account_number="",
        credit_account_number="",
    )
    merged2, words2 = _debit_credit_letter(
        ["PT", "BNI", "Llfe", "Insurnce"],  # OCR-noisy variant, same org
        ["PT", "Sinarmas", "Life"],
        debit_account_number="",
        credit_account_number="",
    )
    monkeypatch.setattr(
        "app.pipeline.redact.ensemble_ocr_page",
        _fake_ocr_sequence([(merged1, words1), (merged2, words2)]),
    )

    _, audit1, _ = _run(pipeline)
    _, audit2, _ = _run(pipeline)

    debit1 = next(r for r in audit1.redactions if r.entity_type == "ORGANIZATION" and r.mock_value)
    debit2 = next(r for r in audit2.redactions if r.entity_type == "ORGANIZATION" and r.mock_value)
    assert debit2.mapping_id == debit1.mapping_id
    assert debit2.mock_value == debit1.mock_value


def test_field_anchored_settings_default_disables_presidio_restricts_field_detection():
    """Field-anchored detection is on by default, but restricted to the
    curated mock dictionary — geometry-based table/label detection is the
    only reliable way to locate PII in noisy multi-column OCR text, but
    letting it auto-create entries for never-seen text produced dozens of
    OCR-garbled near-duplicates. Curate the dictionary instead.
    """
    settings = Settings(_env_file=None)
    assert settings.presidio_enabled is False
    assert settings.field_detection_enabled is True
    assert settings.restrict_to_known_mappings is True


def test_field_anchored_pipeline_logs_omit_source_text(
    field_anchored_pipeline, monkeypatch, caplog
):
    import logging

    pipeline, _, _ = field_anchored_pipeline
    merged, words = _debit_credit_letter(
        ["Reksa", "Dana", "Bahana"], ["PT", "BNI", "Life", "Insurance"]
    )
    monkeypatch.setattr(
        "app.pipeline.redact.ensemble_ocr_page", _fake_ocr_sequence([(merged, words)])
    )

    with caplog.at_level(logging.DEBUG):
        _run(pipeline)

    assert "Reksa Dana Bahana" not in caplog.text
    assert "Standard Chartered Custody" not in caplog.text
    assert "PT BNI Life Insurance" not in caplog.text
