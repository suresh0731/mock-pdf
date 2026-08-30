"""Tests for CSV import/export of the mock dictionary (UI upload/download)."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from app.services.pii.mapping_csv import (
    EXPORT_COLUMNS,
    TEMPLATE_COLUMNS,
    export_mappings_csv,
    import_mappings_csv,
    save_skipped_rows_report,
    template_csv,
)
from app.services.pii.mock_dictionary import MockDictionaryStore


def _rows(csv_text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(csv_text)))


def test_export_mappings_csv_header_matches_export_columns() -> None:
    text = export_mappings_csv([])
    header = text.splitlines()[0].split(",")
    assert header == list(EXPORT_COLUMNS)


def test_export_mappings_csv_writes_one_row_per_entry() -> None:
    entries = [
        {
            "source_text": "Standard Chartered Custody",
            "mock_value": "CUSTODIAN_A",
            "mapping_id": "map_abc123",
            "assignment_source": "auto",
            "hit_count": 3,
        }
    ]
    text = export_mappings_csv(entries)
    rows = _rows(text)
    assert len(rows) == 1
    assert rows[0]["source_text"] == "Standard Chartered Custody"
    assert rows[0]["mock_value"] == "CUSTODIAN_A"


def test_template_csv_header_matches_template_columns() -> None:
    text = template_csv()
    header = text.splitlines()[0].split(",")
    assert header == list(TEMPLATE_COLUMNS)


def test_template_csv_has_worked_examples() -> None:
    rows = _rows(template_csv())
    assert len(rows) >= 2
    assert all(r["source_text"] and r["mock_value"] for r in rows)


def test_import_mappings_csv_inserts_new_rows(tmp_path: Path) -> None:
    store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    csv_text = (
        "source_text,mock_value\n"
        "Standard Chartered Custody,CUSTODIAN_A\n"
        "Reksa Dana Bahana,FUND_A\n"
    )
    result = import_mappings_csv(store, csv_text)
    assert result.inserted == 2
    assert result.skipped_existing == 0
    assert result.skipped_invalid == 0

    resolved = store.resolve("Standard Chartered Custody")
    assert resolved.mock_value == "CUSTODIAN_A"


def test_import_mappings_csv_round_trips_an_exported_file(tmp_path: Path) -> None:
    """The exact file a user downloads must be re-uploadable unchanged."""
    store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    store.upsert("Acme Corp", "ORG_SEED")
    exported = export_mappings_csv([e.model_dump(mode="json") for e in store.list()])

    other_store = MockDictionaryStore(snapshot_path=tmp_path / "other.json")
    result = import_mappings_csv(other_store, exported)
    assert result.inserted == 1
    assert other_store.resolve("Acme Corp").mock_value == "ORG_SEED"


def test_import_mappings_csv_never_overwrites_existing_mapping(tmp_path: Path) -> None:
    store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    manual = store.upsert("Standard Chartered Custody", "USER_OVERRIDE")
    csv_text = "source_text,mock_value\nStandard Chartered Custody,SEED_VALUE\n"
    result = import_mappings_csv(store, csv_text)
    assert result.inserted == 0
    assert result.skipped_existing == 1

    resolved = store.resolve("Standard Chartered Custody")
    assert resolved.mapping_id == manual.mapping_id
    assert resolved.mock_value == "USER_OVERRIDE"


def test_import_mappings_csv_skips_rows_missing_required_fields(tmp_path: Path) -> None:
    store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    csv_text = (
        "source_text,mock_value\n"
        "Good Org,ORG_OK\n"
        ",Missing Source\n"
        "Missing Mock,\n"
    )
    result = import_mappings_csv(store, csv_text)
    assert result.inserted == 1
    assert result.skipped_invalid == 2
    assert store.resolve("Good Org").mock_value == "ORG_OK"


def test_import_mappings_csv_empty_file_returns_zero_counts(tmp_path: Path) -> None:
    store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    result = import_mappings_csv(store, "")
    assert result.inserted == 0
    assert result.skipped_existing == 0
    assert result.skipped_invalid == 0


def test_import_mappings_csv_accepts_template_with_no_data_rows(tmp_path: Path) -> None:
    store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    header_only = "source_text,mock_value\n"
    result = import_mappings_csv(store, header_only)
    assert result.inserted == 0


def test_import_mappings_csv_ignores_extra_legacy_columns(tmp_path: Path) -> None:
    """A CSV exported by an older version (with entity_type/field_role/
    account_number columns) must still import cleanly — extra columns are
    simply ignored."""
    store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    csv_text = (
        "source_text,mock_value,entity_type,field_role,account_number\n"
        "Acme Corp,ORG_A,ORGANIZATION,bank_name,12345\n"
    )
    result = import_mappings_csv(store, csv_text)
    assert result.inserted == 1
    entry = next(e for e in store.list() if e.source_text == "Acme Corp")
    assert entry.mock_value == "ORG_A"


def test_import_mappings_csv_records_reason_for_each_skipped_row(tmp_path: Path) -> None:
    store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    store.upsert("Standard Chartered Custody", "USER_OVERRIDE")
    csv_text = (
        "source_text,mock_value\n"
        "Standard Chartered Custody,SEED_VALUE\n"
        ",Missing Source\n"
        "Missing Mock,\n"
    )
    result = import_mappings_csv(store, csv_text)
    reasons = {(row.row_number, row.reason) for row in result.skipped_rows}
    assert reasons == {
        (1, "existing_mapping"),
        (2, "missing_source_text"),
        (3, "missing_mock_value"),
    }


def test_save_skipped_rows_report_returns_none_when_nothing_skipped(tmp_path: Path) -> None:
    store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    result = import_mappings_csv(store, "source_text,mock_value\nAcme Corp,ORG_A\n")
    report_dir = tmp_path / "import-reports"
    assert save_skipped_rows_report(result, report_dir) is None
    assert not report_dir.exists()


def test_save_skipped_rows_report_writes_a_timestamped_file_with_reasons(
    tmp_path: Path,
) -> None:
    store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    store.upsert("Standard Chartered Custody", "USER_OVERRIDE")
    csv_text = "source_text,mock_value\nStandard Chartered Custody,SEED_VALUE\n,Bad Row\n"
    result = import_mappings_csv(store, csv_text)

    report_dir = tmp_path / "import-reports"
    report_path = save_skipped_rows_report(result, report_dir)

    assert report_path is not None
    assert report_path.parent == report_dir
    assert report_path.name.startswith("mapping-import-rejected-")
    rows = _rows(report_path.read_text(encoding="utf-8"))
    assert len(rows) == 2
    reasons = {row["reason"] for row in rows}
    assert reasons == {"existing_mapping", "missing_source_text"}
    assert any(row["source_text"] == "Standard Chartered Custody" for row in rows)


def test_save_skipped_rows_report_writes_one_file_per_upload(tmp_path: Path) -> None:
    """Repeated uploads each leave a separate trace file for debugging."""
    store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    store.upsert("Acme Corp", "USER_OVERRIDE")
    csv_text = "source_text,mock_value\nAcme Corp,SEED_VALUE\n"
    report_dir = tmp_path / "import-reports"

    save_skipped_rows_report(import_mappings_csv(store, csv_text), report_dir)
    save_skipped_rows_report(import_mappings_csv(store, csv_text), report_dir)

    written = list(report_dir.glob("mapping-import-rejected-*.csv"))
    assert len(written) >= 1
    for path in written:
        assert _rows(path.read_text(encoding="utf-8"))
