"""CSV import/export for the mock dictionary.

Lets QA download the current mappings, edit them in Excel, and re-upload —
and lets anyone pre-seed known custodian/bank/account names via the exact
same file shape, so a "template" is just this file with some rows filled in
and ``account_number`` left blank where an entity has no associated account
(e.g. a bank or counterparty org named only by a label, not a numbered
account).

Upload never clobbers an existing mapping (see ``seed_loader.insert_new_rows``)
— a row whose ``source_text`` already has a mapping is skipped, not
overwritten, so a re-upload can't silently undo a QA correction.
"""

from __future__ import annotations

import csv
import io
from typing import Any

from app.models.mock import MockDictionaryStoreProtocol
from app.services.pii.seed_loader import InsertRowsResult, insert_new_rows

EXPORT_COLUMNS: tuple[str, ...] = (
    "source_text",
    "mock_value",
    "entity_type",
    "field_role",
    "account_number",
    "mapping_id",
    "assignment_source",
    "hit_count",
)

# Only these columns are read back on upload — the rest of EXPORT_COLUMNS
# are informational (mapping_id/assignment_source/hit_count are ignored if
# present, and not required if absent), so a downloaded mappings file and a
# from-scratch template are interchangeable on re-upload.
TEMPLATE_COLUMNS: tuple[str, ...] = (
    "source_text",
    "mock_value",
    "entity_type",
    "field_role",
    "account_number",
)

_TEMPLATE_EXAMPLE_ROWS: tuple[tuple[str, ...], ...] = (
    ("Standard Chartered Custody", "CUSTODIAN_A", "ORGANIZATION", "bank_name", ""),
    (
        "Reksa Dana Bahana Primavera 99",
        "FUND_A",
        "ORGANIZATION",
        "debit_account_name",
        "11002345",
    ),
)


def export_mappings_csv(entries: list[dict[str, Any]]) -> str:
    """Serialize current mock-dictionary entries to CSV text.

    Args:
        entries: Mock entry dicts (e.g. from ``MockEntry.model_dump()``).

    Returns:
        CSV text with a header row, using ``EXPORT_COLUMNS`` order. Missing
        fields become empty cells.
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=EXPORT_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for entry in entries:
        writer.writerow({col: entry.get(col) if entry.get(col) is not None else "" for col in EXPORT_COLUMNS})
    return buffer.getvalue()


def template_csv() -> str:
    """A starter CSV with headers and a couple of worked examples.

    Returns:
        CSV text using ``TEMPLATE_COLUMNS`` — the same shape ``import_
        mappings_csv`` reads, so it can be filled in and uploaded directly.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(TEMPLATE_COLUMNS)
    for row in _TEMPLATE_EXAMPLE_ROWS:
        writer.writerow(row)
    return buffer.getvalue()


def import_mappings_csv(
    store: MockDictionaryStoreProtocol, csv_text: str
) -> InsertRowsResult:
    """Parse an uploaded CSV (export or template shape) and insert new rows.

    Args:
        store: Target mock dictionary store.
        csv_text: Raw CSV file contents. Must have a header row containing
            at least ``source_text`` and ``mock_value``; ``entity_type``,
            ``field_role``, and ``account_number`` are optional per row.

    Returns:
        Counts of inserted vs. skipped (existing mapping vs. malformed row).
        A CSV with no parseable header/rows returns all-zero counts.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = [dict(row) for row in reader]
    return insert_new_rows(store, rows)
