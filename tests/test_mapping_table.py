"""Unit tests for mapping table helpers (no browser, no NiceGUI page)."""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest

from app.ui.mapping_table import (
    ROW_KEYS,
    build_mapping_panel,
    build_mapping_toolbar,
    parse_create,
    parse_override,
    rows_from_entries,
)

LEDGER_SOURCE = "Standard Chartered Custody"

LEDGER_ENTRY = {
    "source_text": LEDGER_SOURCE,
    "mock_value": "XXX",
    "entity_type": "ORGANIZATION",
    "assignment_source": "user",
    "hit_count": 11,
    "mapping_id": "map_00a",
    "pages": [0, 1],
}


def test_rows_from_entries_maps_ledger_row() -> None:
    rows = rows_from_entries([LEDGER_ENTRY])
    assert len(rows) == 1
    row = rows[0]
    assert row["source_text"] == LEDGER_SOURCE
    assert row["mock_value"] == "XXX"
    assert row["entity_type"] == "ORGANIZATION"
    assert row["assignment_source"] == "user"
    assert row["hit_count"] == 11
    assert row["mapping_id"] == "map_00a"
    assert "pages" not in row


def test_rows_from_entries_includes_field_role_and_account_number() -> None:
    entry = {
        **LEDGER_ENTRY,
        "field_role": "debit_account_name",
        "account_number": "30608778491",
    }
    rows = rows_from_entries([entry])
    assert rows[0]["field_role"] == "debit_account_name"
    assert rows[0]["account_number"] == "30608778491"


def test_rows_from_entries_empty_list() -> None:
    assert rows_from_entries([]) == []


def test_rows_from_entries_none() -> None:
    assert rows_from_entries(None) == []


def test_rows_from_entries_drops_extra_keys() -> None:
    entry = {
        "source_text": LEDGER_SOURCE,
        "mock_value": "XXX",
        "entity_type": "ORGANIZATION",
        "assignment_source": "user",
        "hit_count": 11,
        "mapping_id": "map_00a",
        "normalized": "standard chartered custody",
        "created_at": "2026-08-19T01:00:00Z",
    }
    rows = rows_from_entries([entry])
    assert set(rows[0].keys()) == set(ROW_KEYS)


def test_rows_from_entries_defaults_missing_fields() -> None:
    rows = rows_from_entries([{"mapping_id": "map_x"}])
    assert rows == [
        {
            "source_text": "",
            "mock_value": "",
            "entity_type": "",
            "field_role": "",
            "account_number": "",
            "assignment_source": "",
            "hit_count": 0,
            "mapping_id": "map_x",
        }
    ]


def test_rows_from_entries_skips_non_dicts() -> None:
    mixed: list = [{"source_text": "A", "mock_value": "X"}, "bad", 1]
    rows = rows_from_entries(mixed)
    assert len(rows) == 1
    assert rows[0]["source_text"] == "A"


def test_parse_override_valid_bank_a() -> None:
    assert parse_override("map_00a", "BANK_A") == {
        "mapping_id": "map_00a",
        "mock_value": "BANK_A",
    }


def test_parse_override_strips_whitespace() -> None:
    assert parse_override("  map_00a  ", "  BANK_A  ") == {
        "mapping_id": "map_00a",
        "mock_value": "BANK_A",
    }


def test_parse_override_raises_on_blank_mock() -> None:
    with pytest.raises(ValueError):
        parse_override("map_00a", "")


def test_parse_override_raises_on_whitespace_mock() -> None:
    with pytest.raises(ValueError):
        parse_override("map_00a", "   ")


def test_parse_override_blank_does_not_invoke_callback() -> None:
    called: list[object] = []

    def on_override(payload: object) -> None:
        called.append(payload)

    with pytest.raises(ValueError):
        payload = parse_override("map_00a", "")
        on_override(payload)

    assert called == []


def test_parse_override_raises_on_blank_mapping_id() -> None:
    with pytest.raises(ValueError):
        parse_override("", "BANK_A")


def test_parse_override_omits_field_role_and_account_when_not_given() -> None:
    payload = parse_override("map_00a", "BANK_A")
    assert "field_role" not in payload
    assert "account_number" not in payload


def test_parse_override_includes_field_role_and_account_when_given() -> None:
    payload = parse_override("map_00a", "BANK_A", "debit_account_name", "111")
    assert payload["field_role"] == "debit_account_name"
    assert payload["account_number"] == "111"


def test_parse_override_empty_field_role_clears_it() -> None:
    payload = parse_override("map_00a", "BANK_A", "", "")
    assert payload["field_role"] == ""
    assert payload["account_number"] == ""


def test_parse_override_passes_through_legacy_field_role_unvalidated() -> None:
    # Rows created before the current FIELD_ROLE_OPTIONS list (or hand-
    # edited) may carry a role outside it; re-saving must not corrupt it.
    payload = parse_override("map_00a", "BANK_A", "addressee")
    assert payload["field_role"] == "addressee"


def test_parse_create_valid() -> None:
    payload = parse_create("Acme Corp", "ORG_09", "ORGANIZATION", "bank_name", "999")
    assert payload == {
        "source_text": "Acme Corp",
        "mock_value": "ORG_09",
        "entity_type": "ORGANIZATION",
        "field_role": "bank_name",
        "account_number": "999",
    }


def test_parse_create_defaults_entity_type_to_custom() -> None:
    payload = parse_create("Acme Corp", "ORG_09", "")
    assert payload["entity_type"] == "CUSTOM"
    assert "field_role" not in payload
    assert "account_number" not in payload


def test_parse_create_raises_on_blank_source() -> None:
    with pytest.raises(ValueError):
        parse_create("", "ORG_09", "ORGANIZATION")


def test_parse_create_raises_on_blank_mock() -> None:
    with pytest.raises(ValueError):
        parse_create("Acme Corp", "  ", "ORGANIZATION")


def test_parse_create_accepts_any_field_role_string() -> None:
    payload = parse_create("Acme Corp", "ORG_09", "ORGANIZATION", "custom_role")
    assert payload["field_role"] == "custom_role"


def test_build_mapping_panel_is_callable() -> None:
    assert callable(build_mapping_panel)


def test_build_mapping_toolbar_is_callable() -> None:
    assert callable(build_mapping_toolbar)


def test_mapping_table_module_has_no_pipeline_import() -> None:
    source = Path("app/ui/mapping_table.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("app.pipeline")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith("app.pipeline")
        if isinstance(node, ast.Name):
            assert node.id != "RedactPipeline"
        if isinstance(node, ast.Attribute):
            assert node.attr != "RedactPipeline"


def test_helpers_do_not_log_source_text(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    rows_from_entries([LEDGER_ENTRY])
    parse_override("map_00a", "BANK_A")
    for record in caplog.records:
        message = record.getMessage()
        assert LEDGER_SOURCE not in message
        assert "source_text" not in message
