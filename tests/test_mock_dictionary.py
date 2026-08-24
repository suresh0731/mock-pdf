"""Tests for mock dictionary models, store, and PII-safe logging."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.models.mock import (
    LedgerEntry,
    MockEntry,
    MockMappingNotFound,
    MockValidationError,
)
from app.services.pii.mock_dictionary import (
    MockDictionaryStore,
    MockMockDictionary,
    deterministic_auto_mock_value,
    normalize_source,
)

_SOURCE = "Standard Chartered Custody"
_SOURCE_NORMALIZED = "standard chartered custody"


def assert_no_pii_in_logs(caplog_text: str, forbidden: str) -> None:
    """Assert logs omit a PII needle and its case-folded form."""
    assert forbidden not in caplog_text
    assert forbidden.casefold() not in caplog_text.casefold()


@pytest.fixture
def store(tmp_path: Path) -> MockDictionaryStore:
    """Real store writing a tmp snapshot (no audit path, no settings)."""
    return MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")


def test_mock_entry_roundtrip_json() -> None:
    now = datetime.now(timezone.utc)
    entry = MockEntry(
        mapping_id="map_abcd1234",
        source_text=_SOURCE,
        normalized=_SOURCE_NORMALIZED,
        mock_value="MOCK_01",
        assignment_source="auto",
        hit_count=1,
        created_at=now,
        updated_at=now,
    )
    restored = MockEntry.model_validate_json(entry.model_dump_json())
    assert restored.source_text == _SOURCE
    assert restored.normalized == _SOURCE_NORMALIZED
    assert restored.mock_value == "MOCK_01"
    assert restored.mapping_id == "map_abcd1234"
    assert restored.assignment_source == "auto"
    assert restored.hit_count == 1


def test_ledger_entry_includes_source_text() -> None:
    entry = LedgerEntry(
        mapping_id="map_abcd1234",
        source_text=_SOURCE,
        mock_value="XXX",
        entity_type="ORGANIZATION",
        assignment_source="user",
        hit_count=1,
        pages=[0],
    )
    dumped = json.loads(entry.model_dump_json())
    assert "source_text" in dumped
    assert dumped["source_text"] == _SOURCE


def test_mapping_not_found_exposes_code() -> None:
    exc = MockMappingNotFound("map_missing")
    assert exc.code == "MAPPING_NOT_FOUND"
    assert exc.mapping_id == "map_missing"
    message = str(exc)
    assert "map_missing" in message
    assert _SOURCE not in message
    assert _SOURCE_NORMALIZED not in message


def test_resolve_assigns_deterministic_mock_for_unseen_name(
    store: MockDictionaryStore,
) -> None:
    entry = store.resolve(_SOURCE)
    assert entry.mock_value == deterministic_auto_mock_value(_SOURCE_NORMALIZED)
    assert entry.mock_value.startswith("MOCK_")
    assert entry.assignment_source == "auto"
    assert entry.hit_count == 1
    assert entry.normalized == _SOURCE_NORMALIZED
    assert entry.source_text == _SOURCE
    assert entry.mapping_id.startswith("map_")
    assert len(entry.mapping_id) == 12


def test_resolve_reuses_mapping_when_casing_and_spaces_differ(
    store: MockDictionaryStore,
) -> None:
    first = store.resolve(_SOURCE)
    second = store.resolve("  STANDARD   chartered  custody ")
    assert second.mapping_id == first.mapping_id
    assert second.mock_value == first.mock_value
    assert second.hit_count == 2


def test_resolve_applies_irrespective_of_pii_category(
    store: MockDictionaryStore,
) -> None:
    """The store has no notion of category — resolve() doesn't even take
    one, so the same patch applies no matter what a caller would have
    called the span."""
    first = store.resolve(_SOURCE)
    second = store.resolve(_SOURCE)
    assert second.mapping_id == first.mapping_id
    assert second.mock_value == first.mock_value


def test_resolve_user_mock_sets_assignment_source_user(store: MockDictionaryStore) -> None:
    store.resolve(_SOURCE)
    updated = store.resolve(_SOURCE, user_mock="XXX")
    assert updated.mock_value == "XXX"
    assert updated.assignment_source == "user"


def test_override_updates_mock_value(store: MockDictionaryStore) -> None:
    created = store.resolve(_SOURCE)
    updated = store.override(created.mapping_id, "BANK_A")
    assert updated.mapping_id == created.mapping_id
    assert updated.mock_value == "BANK_A"
    assert updated.assignment_source == "user"


def test_new_store_reuses_snapshot_mock_not_new_counter(tmp_path: Path) -> None:
    snapshot = tmp_path / "mappings.json"
    store_a = MockDictionaryStore(snapshot_path=snapshot)
    first = store_a.resolve(_SOURCE)
    store_b = MockDictionaryStore(snapshot_path=snapshot)
    reused = store_b.resolve(_SOURCE)
    assert reused.mapping_id == first.mapping_id
    assert reused.mock_value == deterministic_auto_mock_value(_SOURCE_NORMALIZED)


def test_deterministic_mock_value_same_across_independent_stores(tmp_path: Path) -> None:
    """The core determinism property: no shared snapshot/history required —
    the same real value gets the same mock label purely from its content.
    """
    store_a = MockDictionaryStore(snapshot_path=tmp_path / "a.json")
    store_b = MockDictionaryStore(snapshot_path=tmp_path / "b.json")
    # store_a resolves an unrelated value first, simulating a different
    # "run history" than store_b — must not affect store_b's result.
    store_a.resolve("Acme Holdings")
    entry_a = store_a.resolve(_SOURCE)
    entry_b = store_b.resolve(_SOURCE)
    assert entry_a.mock_value == entry_b.mock_value
    assert entry_a.mock_value == deterministic_auto_mock_value(_SOURCE_NORMALIZED)


def test_delete_then_resolve_recovers_same_deterministic_mock(store: MockDictionaryStore) -> None:
    first = store.resolve(_SOURCE)
    store.delete(first.mapping_id)
    second = store.resolve(_SOURCE)
    assert second.mapping_id != first.mapping_id
    assert second.mock_value == first.mock_value
    ids = {row.mapping_id for row in store.list()}
    assert first.mapping_id not in ids
    assert second.mapping_id in ids


def test_override_unknown_id_raises_mapping_not_found(
    store: MockDictionaryStore, tmp_path: Path
) -> None:
    snapshot = tmp_path / "mappings.json"
    assert not snapshot.exists()
    with pytest.raises(MockMappingNotFound) as exc_info:
        store.override("map_missing", "XXX")
    assert exc_info.value.code == "MAPPING_NOT_FOUND"
    assert not snapshot.exists()


def test_delete_unknown_id_raises_mapping_not_found(store: MockDictionaryStore) -> None:
    with pytest.raises(MockMappingNotFound):
        store.delete("map_missing")


def test_resolve_empty_source_raises_validation_error(
    store: MockDictionaryStore, tmp_path: Path
) -> None:
    snapshot = tmp_path / "mappings.json"
    for empty in ("", "   "):
        with pytest.raises(MockValidationError) as exc_info:
            store.resolve(empty)
        assert exc_info.value.field == "source_text"
        assert exc_info.value.reason == "empty"
        assert empty.strip() not in str(exc_info.value) or empty.strip() == ""
    assert store.list() == []
    if snapshot.exists():
        payload = json.loads(snapshot.read_text(encoding="utf-8"))
        assert payload.get("entries") == []


def test_upsert_empty_source_raises_validation_error(store: MockDictionaryStore) -> None:
    with pytest.raises(MockValidationError) as exc_info:
        store.upsert("", "XXX")
    assert exc_info.value.field == "source_text"
    assert store.list() == []


def test_upsert_blank_mock_raises_validation_error(store: MockDictionaryStore) -> None:
    with pytest.raises(MockValidationError) as exc_info:
        store.upsert("Acme", "  ")
    assert exc_info.value.field == "mock_value"
    assert store.list() == []


def test_upsert_creates_user_assignment(store: MockDictionaryStore) -> None:
    entry = store.upsert("Acme Corp", "XXX")
    assert entry.assignment_source == "user"
    assert entry.mock_value == "XXX"


def test_auto_mock_uses_generic_prefix_regardless_of_name(store: MockDictionaryStore) -> None:
    entry = store.resolve("Dian Wicaksono")
    assert entry.mock_value == deterministic_auto_mock_value("dian wicaksono")
    assert entry.mock_value.startswith("MOCK_")


def test_normalize_source_casefold_and_whitespace() -> None:
    assert normalize_source("  Foo\tBAR  ") == "foo bar"


def test_list_returns_copies(store: MockDictionaryStore) -> None:
    created = store.resolve(_SOURCE)
    rows = store.list()
    assert len(rows) == 1
    rows[0].mock_value = "MUTATED"
    again = store.list()
    assert again[0].mock_value == created.mock_value
    assert again[0].mock_value == deterministic_auto_mock_value(_SOURCE_NORMALIZED)


def test_dictionary_logs_omit_source_text(
    store: MockDictionaryStore, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG):
        created = store.resolve(_SOURCE)
        store.resolve(_SOURCE, user_mock="XXX")
        store.override(created.mapping_id, "BANK_A")
        store.upsert("Acme Corp", "YYY")
        store.delete(created.mapping_id)
    assert_no_pii_in_logs(caplog.text, _SOURCE)
    assert_no_pii_in_logs(caplog.text, _SOURCE_NORMALIZED)
    assert_no_pii_in_logs(caplog.text, "Acme Corp")


def test_gitignore_excludes_mock_dictionary_path() -> None:
    gitignore = Path(__file__).resolve().parents[1] / ".gitignore"
    text = gitignore.read_text(encoding="utf-8")
    assert "data/mock-dictionary/" in text


def test_mock_mock_dictionary_has_no_snapshot_io(tmp_path: Path) -> None:
    before = {p.name for p in tmp_path.iterdir()}
    double = MockMockDictionary()
    entry = double.resolve("A")
    assert entry.mock_value == deterministic_auto_mock_value("a")
    after = {p.name for p in tmp_path.iterdir()}
    assert after == before


# --- Fuzzy matching applies irrespective of PII category ------------------


def test_resolve_fuzzy_matches_ocr_variant_within_threshold(
    store: MockDictionaryStore,
) -> None:
    first = store.resolve(_SOURCE)
    # ratio to _SOURCE ~0.85, clears the default 0.85 threshold.
    second = store.resolve("Stan Chartered Custdn")
    assert second.mapping_id == first.mapping_id
    assert second.hit_count == 2


def test_resolve_fuzzy_match_not_scoped_by_any_category(
    store: MockDictionaryStore,
) -> None:
    """There is no entity_type/field_role concept at all any more — a
    near-identical OCR variant always collapses onto the existing
    mapping, whatever category a caller might have detected either
    span under."""
    first_entry = store.resolve(_SOURCE)
    # A near-identical OCR variant (not an exact-normalized match).
    second_entry = store.resolve("Standard Chartered Custdy")
    assert second_entry.mapping_id == first_entry.mapping_id


def test_resolve_unrelated_text_below_threshold_gets_new_entry(
    store: MockDictionaryStore,
) -> None:
    first = store.resolve(_SOURCE)
    second = store.resolve("PT BNI Life Insurance")
    assert second.mapping_id != first.mapping_id
    assert second.mock_value != first.mock_value
    assert second.mock_value == deterministic_auto_mock_value("pt bni life insurance")


def test_resolve_fuzzy_threshold_is_configurable(tmp_path: Path) -> None:
    variant = "Chartered Custody"  # ratio to _SOURCE ~0.79: below 0.85, above 0.7

    strict_store = MockDictionaryStore(
        snapshot_path=tmp_path / "strict.json", fuzzy_threshold=0.85
    )
    strict_first = strict_store.resolve(_SOURCE)
    strict_second = strict_store.resolve(variant)
    assert strict_second.mapping_id != strict_first.mapping_id

    lenient_store = MockDictionaryStore(
        snapshot_path=tmp_path / "lenient.json", fuzzy_threshold=0.7
    )
    lenient_first = lenient_store.resolve(_SOURCE)
    lenient_second = lenient_store.resolve(variant)
    assert lenient_second.mapping_id == lenient_first.mapping_id


# --- Prefix-collision ambiguity detection --------------------------------


def test_find_prefix_collisions_flags_truncated_subset(store: MockDictionaryStore) -> None:
    """The real 'Maksima Plus' case: a shorter entry's tokens are a strict
    subset of a longer entry's — token_sort_ratio alone (0.945) wouldn't
    force a re-check, but token_set_ratio flags it immediately.

    Both entries are seeded via ``upsert`` (curated, user-assigned) rather
    than ``resolve`` — a plain ``resolve`` of the second string would
    itself fuzzy-collapse into the first (ratio ~0.945 clears the default
    0.85 threshold), which is exactly the ambiguity this function exists
    to flag rather than silently resolve.
    """
    short_entry = store.upsert("Blife Link Saham Maksima", "ORG_SHORT")
    long_entry = store.upsert("Blife Link Saham Maksima Plus", "ORG_LONG")

    collisions = store.find_prefix_collisions("blife link saham maksima")
    assert {c.mapping_id for c in collisions} == {long_entry.mapping_id}

    collisions_reverse = store.find_prefix_collisions("blife link saham maksima plus")
    assert {c.mapping_id for c in collisions_reverse} == {short_entry.mapping_id}


def test_find_prefix_collisions_empty_when_no_containment_relationship(
    store: MockDictionaryStore,
) -> None:
    store.upsert(_SOURCE, "BANK_A")
    store.upsert("PT BNI Life Insurance", "ORG_B")
    assert store.find_prefix_collisions("standard chartered custody") == []


def test_find_prefix_collisions_excludes_given_mapping_id(store: MockDictionaryStore) -> None:
    entry = store.upsert("Blife Link Saham Maksima Plus", "ORG_LONG")
    collisions = store.find_prefix_collisions(
        "blife link saham maksima", exclude_mapping_id=entry.mapping_id
    )
    assert collisions == []


def test_find_prefix_collisions_blank_normalized_returns_empty(
    store: MockDictionaryStore,
) -> None:
    store.upsert(_SOURCE, "BANK_A")
    assert store.find_prefix_collisions("") == []


def test_resolve_logs_prefix_collision_warning_without_leaking_text(
    store: MockDictionaryStore, caplog: pytest.LogCaptureFixture
) -> None:
    """Both entries pre-exist as separate curated (seed) rows — e.g. two
    genuinely distinct real fund names — so a later OCR read that exactly
    matches the shorter one (an exact-normalized-text hit, bypassing fuzzy
    matching entirely) still gets flagged as ambiguous against the longer
    one still on file.
    """
    short_entry = store.upsert("Blife Link Saham Maksima", "ORG_SHORT")
    long_entry = store.upsert("Blife Link Saham Maksima Plus", "ORG_LONG")

    with caplog.at_level(logging.WARNING):
        resolved = store.resolve("Blife Link Saham Maksima")
    assert resolved.mapping_id == short_entry.mapping_id

    assert "mock_prefix_collision_detected" in caplog.text
    assert short_entry.mapping_id in caplog.text
    assert long_entry.mapping_id in caplog.text
    assert_no_pii_in_logs(caplog.text, "Blife Link Saham Maksima")


def test_resolve_no_collision_warning_for_unrelated_entries(
    store: MockDictionaryStore, caplog: pytest.LogCaptureFixture
) -> None:
    store.resolve(_SOURCE)
    with caplog.at_level(logging.WARNING):
        store.resolve("PT BNI Life Insurance")
    assert "mock_prefix_collision_detected" not in caplog.text
