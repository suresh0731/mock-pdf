"""Tests for optional bulk preload of known name/account -> mock mappings."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from app.services.pii.mock_dictionary import MockDictionaryStore
from app.services.pii.seed_loader import load_seed_entries


def test_load_seed_entries_returns_zero_when_path_is_none(tmp_path: Path) -> None:
    store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    assert load_seed_entries(store, None) == 0
    assert store.list() == []


def test_load_seed_entries_returns_zero_when_file_missing(tmp_path: Path) -> None:
    store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    missing = tmp_path / "does_not_exist.json"
    assert load_seed_entries(store, missing) == 0
    assert store.list() == []


def test_load_seed_entries_inserts_wrapped_entries(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.json"
    seed_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "source_text": "Standard Chartered Custody",
                        "mock_value": "CUSTODIAN_A",
                        "entity_type": "ORGANIZATION",
                        "field_role": "bank_name",
                    },
                    {
                        "source_text": "PT BNI Life Insurance",
                        "mock_value": "CUSTODIAN_B",
                        "account_number": "30608778491",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    inserted = load_seed_entries(store, seed_path)
    assert inserted == 2

    resolved = store.resolve("Standard Chartered Custody", "ORGANIZATION")
    assert resolved.mock_value == "CUSTODIAN_A"
    assert resolved.field_role == "bank_name"

    by_account = store.resolve("anything at all", "ORGANIZATION", account_number="30608778491")
    assert by_account.mock_value == "CUSTODIAN_B"


def test_load_seed_entries_accepts_bare_list_shape(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.json"
    seed_path.write_text(
        json.dumps([{"source_text": "Acme Corp", "mock_value": "ORG_SEED"}]),
        encoding="utf-8",
    )
    store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    assert load_seed_entries(store, seed_path) == 1
    assert store.resolve("Acme Corp", "ORGANIZATION").mock_value == "ORG_SEED"


def test_load_seed_entries_never_overwrites_existing_mapping(tmp_path: Path) -> None:
    store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    manual = store.upsert("Standard Chartered Custody", "USER_OVERRIDE")

    seed_path = tmp_path / "seed.json"
    seed_path.write_text(
        json.dumps(
            {"entries": [{"source_text": "Standard Chartered Custody", "mock_value": "SEED_VALUE"}]}
        ),
        encoding="utf-8",
    )
    inserted = load_seed_entries(store, seed_path)
    assert inserted == 0

    resolved = store.resolve("Standard Chartered Custody", "ORGANIZATION")
    assert resolved.mapping_id == manual.mapping_id
    assert resolved.mock_value == "USER_OVERRIDE"


def test_load_seed_entries_skips_malformed_rows(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.json"
    seed_path.write_text(
        json.dumps(
            {
                "entries": [
                    {"source_text": "Good Org", "mock_value": "ORG_OK"},
                    {"source_text": "Missing Mock Value"},
                    {"mock_value": "No Source"},
                    "not a dict",
                    {"source_text": 123, "mock_value": "Bad Types"},
                ]
            }
        ),
        encoding="utf-8",
    )
    store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    inserted = load_seed_entries(store, seed_path)
    assert inserted == 1
    assert store.resolve("Good Org", "ORGANIZATION").mock_value == "ORG_OK"


def test_load_seed_entries_returns_zero_for_corrupt_json(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.json"
    seed_path.write_text("{not valid json", encoding="utf-8")
    store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    assert load_seed_entries(store, seed_path) == 0


def test_load_seed_entries_returns_zero_for_invalid_shape(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.json"
    seed_path.write_text(json.dumps({"entries": "not a list"}), encoding="utf-8")
    store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    assert load_seed_entries(store, seed_path) == 0


def test_load_seed_entries_logs_omit_source_text(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    seed_path = tmp_path / "seed.json"
    secret_name = "Standard Chartered Custody"
    seed_path.write_text(
        json.dumps({"entries": [{"source_text": secret_name, "mock_value": "CUSTODIAN_A"}]}),
        encoding="utf-8",
    )
    store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    with caplog.at_level(logging.INFO):
        load_seed_entries(store, seed_path)
    assert secret_name not in caplog.text
    assert secret_name.casefold() not in caplog.text.casefold()
