"""Optional bulk preload of known name/account-number -> mock mappings.

Answers "do I need to provide the list or will it auto-detect": both.
Known recurring custodians/funds/banks can be seeded here for a
guaranteed-correct mock from document 1; anything unseeded is still
auto-learned by ``MockDictionaryStore.resolve`` and remembered via its
write-through snapshot.

Loaded once at process startup (``mock_seed_path`` config, unset = skip).
Never logs seed source_text/mock_value (SEC-001) — only counts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.models.mock import MockDictionaryStoreProtocol
from app.services.pii.mock_dictionary import normalize_source

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InsertRowsResult:
    """Outcome of a bulk non-clobbering insert (seed file or CSV upload)."""

    inserted: int
    skipped_existing: int
    skipped_invalid: int


def insert_new_rows(
    store: MockDictionaryStoreProtocol, rows: list[dict[str, Any]]
) -> InsertRowsResult:
    """Insert rows shaped like ``{source_text, mock_value}`` without
    clobbering existing mappings.

    A row is skipped whenever its normalized ``source_text`` already has a
    mapping — this is what lets a later reseed/reupload never silently
    overwrite a user's manual correction back to an older value. Shared by
    the JSON seed-file loader and the CSV mapping-upload endpoint so both
    have identical, once-reviewed semantics.

    Args:
        store: Target mock dictionary store.
        rows: Parsed row dicts (already JSON- or CSV-decoded).

    Returns:
        Counts of inserted vs. skipped (existing mapping vs. malformed row).
    """
    existing = {entry.normalized for entry in store.list()}
    inserted = 0
    skipped_existing = 0
    skipped_invalid = 0
    for row in rows:
        if not isinstance(row, dict):
            skipped_invalid += 1
            continue
        source_text = row.get("source_text")
        mock_value = row.get("mock_value")
        if (
            not isinstance(source_text, str)
            or not source_text.strip()
            or not isinstance(mock_value, str)
            or not mock_value.strip()
        ):
            skipped_invalid += 1
            continue
        normalized = normalize_source(source_text)
        if not normalized:
            skipped_invalid += 1
            continue
        if normalized in existing:
            skipped_existing += 1
            continue

        try:
            store.upsert(source_text, mock_value)
        except Exception as exc:  # noqa: BLE001 - bulk rows must not crash the caller
            logger.warning("mapping row skipped: %s", type(exc).__name__)
            skipped_invalid += 1
            continue
        existing.add(normalized)
        inserted += 1

    return InsertRowsResult(
        inserted=inserted, skipped_existing=skipped_existing, skipped_invalid=skipped_invalid
    )


def load_seed_entries(
    store: MockDictionaryStoreProtocol, seed_path: Path | str | None
) -> int:
    """Preload known source-text -> mock rows without clobbering existing ones.

    Args:
        store: Target mock dictionary store.
        seed_path: JSON file path (``{"entries": [...]}`` or a bare list of
            ``{source_text, mock_value}`` objects), or ``None``/missing to
            skip entirely.

    Returns:
        Number of new rows inserted. ``0`` if the path is unset, missing,
        empty, or the file is not valid JSON.
    """
    if seed_path is None:
        return 0
    path = Path(seed_path)
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("mock seed load failed: %s", type(exc).__name__)
        return 0

    rows: Any = payload.get("entries") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        logger.warning("mock seed load failed: invalid shape")
        return 0

    result = insert_new_rows(store, rows)
    logger.info("mock seed loaded inserted=%s", result.inserted)
    return result.inserted
