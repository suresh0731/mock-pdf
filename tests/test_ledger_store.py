"""Tests for per-job substitution ledger persistence (PII store)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.models.mock import LedgerEntry, MockValidationError, SubstitutionLedger
from app.services.redact.ledger_store import LedgerStore

_SOURCE = "Standard Chartered Custody"


def assert_no_pii_in_logs(caplog_text: str, forbidden: str) -> None:
    """Assert logs omit a PII needle and its case-folded form."""
    assert forbidden not in caplog_text
    assert forbidden.casefold() not in caplog_text.casefold()


@pytest.fixture
def ledger_store(tmp_path: Path) -> LedgerStore:
    """Ledger store rooted under a tmp shards directory."""
    return LedgerStore(base_dir=tmp_path / "shards")


def _sample_ledger(
    request_id: str = "req-1",
    brand_zones: list[dict] | None = None,
) -> SubstitutionLedger:
    return SubstitutionLedger(
        request_id=request_id,
        created_at=datetime.now(timezone.utc),
        entries=[
            LedgerEntry(
                mapping_id="map_abcd1234",
                source_text=_SOURCE,
                mock_value="XXX",
                entity_type="ORGANIZATION",
                assignment_source="user",
                hit_count=2,
                pages=[0, 1],
            )
        ],
        brand_zones=brand_zones or [],
    )


def test_ledger_save_get_roundtrip_includes_source_text(ledger_store: LedgerStore) -> None:
    saved = _sample_ledger()
    ledger_store.save(saved)
    loaded = ledger_store.get("req-1")
    assert loaded is not None
    assert loaded.request_id == "req-1"
    assert len(loaded.entries) == 1
    entry = loaded.entries[0]
    assert entry.source_text == _SOURCE
    assert entry.mock_value == "XXX"
    assert entry.mapping_id == "map_abcd1234"
    assert entry.pages == [0, 1]


def test_ledger_get_unknown_returns_none(ledger_store: LedgerStore) -> None:
    assert ledger_store.get("no-such") is None


def test_ledger_save_creates_request_subdir(
    ledger_store: LedgerStore, tmp_path: Path
) -> None:
    path = ledger_store.save(_sample_ledger())
    expected = tmp_path / "shards" / "req-1" / "ledger.json"
    assert path == expected
    assert path.exists()


def test_ledger_rejects_path_traversal_request_id(
    ledger_store: LedgerStore, tmp_path: Path
) -> None:
    ledger = _sample_ledger(request_id="../etc")
    with pytest.raises(MockValidationError) as exc_info:
        ledger_store.save(ledger)
    assert exc_info.value.field == "request_id"
    assert exc_info.value.reason == "invalid"
    outside = tmp_path / "etc"
    assert not outside.exists()
    shards = tmp_path / "shards"
    if shards.exists():
        for child in shards.rglob("*"):
            assert ".." not in child.name


def test_ledger_logs_omit_source_text(
    ledger_store: LedgerStore, caplog: pytest.LogCaptureFixture
) -> None:
    ledger = _sample_ledger()
    with caplog.at_level(logging.DEBUG):
        ledger_store.save(ledger)
        ledger_store.get("req-1")
    assert_no_pii_in_logs(caplog.text, _SOURCE)


def test_ledger_roundtrip_brand_zones(ledger_store: LedgerStore) -> None:
    zones = [{"page": 0, "zone": "logo"}]
    ledger_store.save(_sample_ledger(brand_zones=zones))
    loaded = ledger_store.get("req-1")
    assert loaded is not None
    assert loaded.brand_zones == zones
