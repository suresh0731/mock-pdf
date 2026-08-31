"""In-memory mock dictionary with write-through JSON snapshot.

This module is a PII store: ``source_text`` is persisted in the snapshot
only. Logs may include mapping_id and assignment_source — never
source_text, normalized text, or a mock paired with its source.

Matching (exact and fuzzy) is by name text only. There is no notion of
PII category, structural role, or account number scoping a match — the
same source text always resolves to the same stored mapping regardless of
what category the caller detected it under.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from app.models.mock import MockEntry, MockMappingNotFound, MockValidationError
from app.services.pii.name_matcher import (
    GENERIC_ORG_TOKENS,
    MIN_STRIPPED_LENGTH,
    is_deliberate_family_pair,
    is_token_subset_collision,
    strip_generic_org_tokens,
    token_sort_ratio,
)
from app.utils.atomic_write import atomic_write_text

logger = logging.getLogger(__name__)

DEFAULT_FUZZY_THRESHOLD = 0.85

# Length (hex chars) of the default content-derived auto-mock suffix. 6 hex
# chars = 24 bits (~16.7M combinations) — comfortably collision-free for a
# per-organization dictionary of tens to low hundreds of entries; the rare
# collision is still handled deterministically by extending the hash length
# (see ``_InMemoryDictionary._auto_mock_value``), never by an order-
# dependent counter.
_AUTO_SUFFIX_LENGTH = 6
_AUTO_SUFFIX_FALLBACK_LENGTHS = (8, 12, 16, 32, 64)

# Manually curated (assignment_source == "user") entries are verified ground
# truth, so matching against them uses a lower threshold than auto-vs-auto
# matching. Originally 0.65, calibrated against real OCR ensemble output on
# scanned bank letters (see tests/test_mock_dictionary.py): garbled
# table-cell reads of a known organization ("Cnartercd Standerd Custpay",
# "Terop Blife Stabill Link Pendapatin") score 0.70-0.96 against their clean
# source text, while unrelated organizations never exceed ~0.58 in that
# small hand-curated set.
#
# That separation doesn't hold once the dictionary is bulk-loaded from a
# large real-world mapping table (hundreds of rows): sibling entities that
# share a generic corporate shell — "PT <NAME> LIFE ASSURANCE" — routinely
# score in the high 70s/80s against each other on nothing but the shared
# "PT"/"LIFE"/"ASSURANCE" tokens (measured directly against
# complete_client_mappings.csv: "PT Prudential Life Assurance" vs. the
# unrelated "PT MNC Life Assurance" scores 0.7755 token_sort_ratio), which
# used to clear 0.65 and get the wrong client's mock painted over a
# correctly-OCR'd name. Raised to 0.90 — above even the auto-vs-auto
# default — because a bulk-loaded dictionary this size needs a *stricter*
# bar than a handful of manually verified rows, not a looser one. This
# still doesn't fully separate the family-of-similarly-named-funds case
# (many genuinely distinct "PT PRUDENTIAL LIFE ASSURANCE - <FUND CODE>"
# rows score >=0.90 against each other) — that needs spatial/contextual
# disambiguation, not a name-similarity threshold, and remains a known gap.
_TRUSTED_FUZZY_THRESHOLD = 0.90

# Fallback acceptance path for a trusted entry that misses
# _TRUSTED_FUZZY_THRESHOLD only because the two spellings disagree on
# generic corporate/banking boilerplate (see
# name_matcher.strip_generic_org_tokens) — e.g. the curated table's own
# "Standard Chartered Bank - Custody" vs. an OCR read of "Standard
# Chartered Custody" scores 0.88 raw (below 0.90) but 1.0 once both sides
# drop "Bank"/"Custody". Deliberately still *below* 1.0 itself (not an
# exact-string requirement) to tolerate a little OCR noise on the
# distinguishing words that are left after stripping (e.g. "Standerd" for
# "Standard"). Calibrated against this codebase's full ~600-row real
# client mapping table: at 0.92, every pair of *different* real
# organizations/funds in that table that newly clears this path also
# turned out to be the same institution spelled two different ways
# (already an existing mock-mapping ambiguity the table itself contains,
# surfaced via find_prefix_collisions) rather than two distinct entities
# — see tests/test_mock_dictionary.py for the specific cases this was
# checked against (Prudential vs. MNC-shaped family confusion in
# particular must stay rejected).
_GENERIC_STRIP_RATIO_FLOOR = 0.92

# Every auto-assigned mock value shares one generic prefix — matching (and
# the mock value itself) is by name text only, irrespective of whatever PII
# category the caller detected the span as.
_AUTO_MOCK_PREFIX = "MOCK"


class _Snapshot(BaseModel):
    """On-disk dictionary snapshot (entries include source_text).

    ``counters`` is accepted-and-ignored for backward compatibility with
    snapshots written before auto mock values became content-deterministic
    (see ``deterministic_auto_mock_value``) — it's no longer read or
    written, but an old file containing it must still load cleanly.
    """

    version: int = 1
    counters: dict[str, int] = {}
    entries: list[MockEntry]


def normalize_source(source_text: str) -> str:
    """Case-fold, collapse whitespace, and strip.

    Args:
        source_text: Raw span or custom term text.

    Returns:
        Normalized lookup key. May be empty if ``source_text`` is blank.
    """
    return " ".join(source_text.casefold().split())


def _require_normalized(source_text: str) -> str:
    """Return the lookup key or raise without embedding the rejected text."""
    if not source_text.strip():
        raise MockValidationError("source_text", "empty")
    normalized = normalize_source(source_text)
    if not normalized:
        raise MockValidationError("source_text", "empty")
    return normalized


def _nonblank(value: str | None) -> str | None:
    """Return stripped text, or None if missing/whitespace-only."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _auto_suffix_hash(prefix: str, normalized_source_text: str, length: int) -> str:
    digest = hashlib.sha256(f"{prefix}\x1f{normalized_source_text}".encode("utf-8")).hexdigest()
    return digest[:length].upper()


def deterministic_auto_mock_value(normalized_source_text: str) -> str:
    """Content-derived auto mock value: ``MOCK_{HASH}``.

    Same ``normalized_source_text`` always yields the same mock value on
    every machine, independent of run history, order-of-first-sight, or
    the PII category the caller detected the span as — replaces the old
    sequential per-type counter (``ORG_01``, ``ORG_02``, ...), where the
    label a never-seen value got depended on how many other new values
    had already been auto-assigned on that particular machine.

    This is the *collision-naive* default-length form used directly by
    callers that just need a deterministic label (e.g. tests predicting an
    expected value). ``_InMemoryDictionary._auto_mock_value`` wraps this
    with a deterministic collision-extension fallback for the rare case
    where two different real values happen to hash to the same suffix
    within one dictionary.

    Args:
        normalized_source_text: Already-normalized lookup key (see
            ``normalize_source``) — never the raw/unnormalized text.

    Returns:
        A ``MOCK_{HASH}`` mock value, e.g. ``"MOCK_1F3A9C"``.
    """
    return (
        f"{_AUTO_MOCK_PREFIX}_"
        f"{_auto_suffix_hash(_AUTO_MOCK_PREFIX, normalized_source_text, _AUTO_SUFFIX_LENGTH)}"
    )


class _InMemoryDictionary:
    """Shared resolve/list/upsert/override/delete indexes. No disk I/O."""

    def __init__(self, fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD) -> None:
        self._by_normalized: dict[str, MockEntry] = {}
        self._by_id: dict[str, MockEntry] = {}
        self._fuzzy_threshold = fuzzy_threshold
        self._lock = threading.RLock()

    def _after_mutate(self) -> None:
        """Persist hook. Overridden by ``MockDictionaryStore``."""

    def _log(self, action: str, entry: MockEntry) -> None:
        logger.info(
            "mock_%s mapping_id=%s assignment_source=%s",
            action,
            entry.mapping_id,
            entry.assignment_source,
        )

    def _new_mapping_id(self) -> str:
        while True:
            mapping_id = f"map_{secrets.token_hex(4)}"
            if mapping_id not in self._by_id:
                return mapping_id

    def _auto_mock_value(self, normalized: str) -> str:
        """Deterministic auto mock value, extending the hash on collision.

        A collision (a *different* normalized value already holding the
        same mock_value string) is vanishingly unlikely at the default
        suffix length but handled deterministically rather than ignored:
        the suffix length grows until the candidate is unique, so the
        result still depends only on ``normalized`` — never on insertion
        order.
        """
        for length in (_AUTO_SUFFIX_LENGTH, *_AUTO_SUFFIX_FALLBACK_LENGTHS):
            candidate = f"{_AUTO_MOCK_PREFIX}_{_auto_suffix_hash(_AUTO_MOCK_PREFIX, normalized, length)}"
            collision = any(
                entry.mock_value == candidate and entry.normalized != normalized
                for entry in self._by_id.values()
            )
            if not collision:
                return candidate
        return candidate

    def _best_match(self, normalized: str) -> tuple[str | None, float]:
        """Best-scoring entry and its ratio, regardless of threshold (or
        ``(None, 0.0)`` if the dictionary is empty).

        Shared scoring core for ``_find_fuzzy_match`` (applies the
        threshold) and ``best_match_score`` (a pure, threshold-free ratio
        used by the maximal-munch window-extension probe — see
        ``app.pipeline.redact``). Matching is by name text only,
        irrespective of any PII category — see ``_find_fuzzy_match``.
        """
        best_id: str | None = None
        best_ratio = 0.0
        for entry in self._by_id.values():
            ratio = token_sort_ratio(normalized, entry.normalized)
            if ratio > best_ratio:
                best_ratio = ratio
                best_id = entry.mapping_id
        return best_id, best_ratio

    def _find_fuzzy_match(self, normalized: str) -> str | None:
        """Best entry above its applicable fuzzy threshold, or None.

        Matching applies irrespective of any PII category the caller
        detected the span as — a name resolves to the same mapping no
        matter which category it was flagged under. Manually curated
        (``assignment_source == "user"``) entries are verified ground
        truth, so they're compared at a lower threshold than auto-vs-auto
        matching: the same real value can legitimately be found by a
        different extraction mode (e.g. a table column vs. an address
        block) and should still collapse onto the one curated row rather
        than spawning a fresh auto-assigned duplicate.

        This threshold check can't fully share ``_best_match``'s single
        best-ratio-wins result: the *applicable* threshold itself depends
        on whether that particular entry is trusted, not on the winning
        entry only, so a stricter re-scan against each candidate's own
        threshold is kept here rather than trying to fold both thresholds
        into one shared comparison.

        A trusted entry that misses its own threshold gets one more
        chance via ``strip_generic_org_tokens`` (see
        ``_GENERIC_STRIP_RATIO_FLOOR``) — restricted to trusted entries
        only, since an auto-assigned entry was never human-verified and
        so doesn't earn the extra leniency curated rows get. The
        candidate side also gets a fuzzy pass against *this entry's own*
        generic tokens (e.g. "Custody" misread as "Cursdy") rather than
        the full generic word list — see ``strip_generic_org_tokens``'s
        ``fuzzy_against`` docstring for why that scoping matters. Both
        acceptance paths still rank purely by the *raw* ratio, so the
        single highest-raw-similarity entry always wins regardless of
        which path let it in.
        """
        best_id: str | None = None
        best_ratio = -1.0
        for entry in self._by_id.values():
            trusted = entry.assignment_source == "user"
            threshold = _TRUSTED_FUZZY_THRESHOLD if trusted else self._fuzzy_threshold
            ratio = token_sort_ratio(normalized, entry.normalized)
            accepted = ratio >= threshold
            if not accepted and trusted:
                entry_stripped = strip_generic_org_tokens(entry.normalized)
                entry_generic_present = frozenset(
                    token
                    for raw_token in entry.normalized.split()
                    if (token := raw_token.strip(".,")) in GENERIC_ORG_TOKENS
                )
                candidate_stripped = strip_generic_org_tokens(
                    normalized, fuzzy_against=entry_generic_present
                )
                accepted = (
                    len(candidate_stripped) >= MIN_STRIPPED_LENGTH
                    and len(entry_stripped) >= MIN_STRIPPED_LENGTH
                    and not is_deliberate_family_pair(candidate_stripped, entry_stripped)
                    and token_sort_ratio(candidate_stripped, entry_stripped)
                    >= _GENERIC_STRIP_RATIO_FLOOR
                )
            if accepted and ratio > best_ratio:
                best_ratio = ratio
                best_id = entry.mapping_id
        return best_id

    def best_match_score(self, normalized: str) -> float:
        """Best fuzzy-match ratio against known entries, or ``0.0``.

        Pure read: unlike ``resolve``/``lookup``, never mutates
        ``hit_count``/``updated_at`` on a match — safe to call speculatively
        to score several candidate text windows (see the maximal-munch
        window-extension probe in ``app.pipeline.redact``) without
        recording a hit for windows that don't end up being used.

        Args:
            normalized: Already-normalized candidate text (see
                ``normalize_source``).

        Returns:
            Ratio in ``[0.0, 1.0]``. ``0.0`` if ``normalized`` is blank or
            the dictionary is empty.
        """
        if not normalized.strip():
            return 0.0
        with self._lock:
            _, ratio = self._best_match(normalized)
        return ratio

    def find_prefix_collisions(
        self, normalized: str, *, exclude_mapping_id: str | None = None
    ) -> list[MockEntry]:
        """Other entries whose tokens strictly contain, or are strictly
        contained by, ``normalized``'s tokens.

        Uses ``rapidfuzz``'s ``token_set_ratio`` (via
        ``is_token_subset_collision``, 100 whenever one string's token
        set is a subset of the other's) instead of a hand-rolled prefix
        check — this is the built-in ambiguity signal for pairs like
        ``"blife link saham maksima"`` / ``"blife link saham maksima
        plus"``, which ``token_sort_ratio`` alone can't reliably
        distinguish from ordinary OCR character drift (0.945 vs 1.0 is
        too close to force a re-check on ratio alone).

        This is a detection/observability signal only — see
        ``_resolve_or_lookup``, which logs (never redirects) on a hit.
        Neither string shape here can safely decide *which* entry is
        correct on its own: a genuinely short real value and an
        unrelated, longer one can legitimately coexist in the same
        dictionary. Actually disambiguating needs spatial evidence (the
        lookahead/lookbehind window-extension probe over adjacent OCR
        words), which lives where that geometry is available
        (``field_extractor``/``redact`` pipeline), not here.

        A pair matching ``is_deliberate_family_pair`` (the longer entry
        is the shorter one plus a hyphen-delimited suffix, e.g. "dplk
        axa mandiri" / "dplk axa mandiri - ppip pu") is excluded
        entirely rather than flagged: a mapping table that names dozens
        of genuinely distinct sub-accounts this way would otherwise have
        roughly half its rows collide with each other by construction
        (every "PARENT - CHILD" row collides with its "PARENT" row and
        every sibling "PARENT - OTHER_CHILD" row), burying the rare,
        actually-ambiguous case (e.g. "Maksima" vs. "Maksima Plus",
        plain concatenation with no delimiter) in noise.

        Args:
            normalized: Already-normalized candidate text (see
                ``normalize_source``).
            exclude_mapping_id: Optional mapping id to omit from results
                (typically the entry ``normalized`` already resolved to).

        Returns:
            Colliding entries (deep copies), possibly empty. Order
            unspecified.
        """
        if not normalized.strip():
            return []
        with self._lock:
            return [
                entry.model_copy(deep=True)
                for entry in self._by_id.values()
                if entry.mapping_id != exclude_mapping_id
                and entry.normalized != normalized
                and is_token_subset_collision(normalized, entry.normalized)
                and not is_deliberate_family_pair(normalized, entry.normalized)
            ]

    def _log_prefix_collision_if_any(
        self, normalized: str, resolved_mapping_id: str | None
    ) -> None:
        """Warn (mapping ids/counts only, never text — SEC-001) when a
        resolution is ambiguous under ``find_prefix_collisions``.

        Alias rows that share the same ``mock_value`` (e.g. ``"SCB"`` and
        ``"Standard Chartered Bank"`` both painting ``DSDC_Bank``) are not
        a real ambiguity — either mapping produces the same redaction —
        so they are omitted. Only a colliding entry with a *different*
        mock is logged, which is the Maksima vs. Maksima Plus case the
        maximal-munch probe then has to disambiguate. A deliberate
        base+suffix family member (see ``find_prefix_collisions``'s
        ``_is_deliberate_family_pair`` filter) never reaches here at all
        — it's excluded from the collision list itself, not filtered out
        by this method.
        """
        collisions = self.find_prefix_collisions(
            normalized, exclude_mapping_id=resolved_mapping_id
        )
        if resolved_mapping_id is not None:
            resolved = self._by_id.get(resolved_mapping_id)
            if resolved is not None:
                collisions = [
                    entry
                    for entry in collisions
                    if entry.mock_value != resolved.mock_value
                ]
        if collisions:
            logger.warning(
                "mock_prefix_collision_detected mapping_id=%s colliding_mapping_ids=%s",
                resolved_mapping_id,
                sorted(entry.mapping_id for entry in collisions),
            )

    def _resolve_or_lookup(
        self,
        source_text: str,
        user_mock: str | None,
        *,
        create_if_missing: bool,
    ) -> MockEntry | None:
        """Shared implementation for ``resolve`` (may create) and ``lookup``
        (read-only — returns ``None`` instead of auto-assigning).

        Lookup order: exact normalized text match, then a fuzzy name match
        (see ``_find_fuzzy_match``), then — only when ``create_if_missing``
        — a fresh auto-assigned entry. Matching never depends on any PII
        category the caller detected the span as: the same name text
        always resolves to the same mapping.
        """
        normalized = _require_normalized(source_text)
        override = _nonblank(user_mock)
        now = datetime.now(timezone.utc)
        with self._lock:
            existing = self._by_normalized.get(normalized)

            if existing is None:
                fuzzy_id = self._find_fuzzy_match(normalized)
                if fuzzy_id is not None:
                    existing = self._by_id.get(fuzzy_id)

            if existing is not None:
                self._log_prefix_collision_if_any(normalized, existing.mapping_id)
                existing.hit_count += 1
                existing.updated_at = now
                if override is not None:
                    existing.mock_value = override
                    existing.assignment_source = "user"
                self._by_normalized.setdefault(normalized, existing)
                self._after_mutate()
                self._log("resolve", existing)
                return existing.model_copy(deep=True)

            if not create_if_missing:
                return None

            self._log_prefix_collision_if_any(normalized, None)

            assignment_source: Literal["auto", "user"]
            if override is not None:
                mock_value = override
                assignment_source = "user"
            else:
                mock_value = self._auto_mock_value(normalized)
                assignment_source = "auto"

            entry = MockEntry(
                mapping_id=self._new_mapping_id(),
                source_text=source_text,
                normalized=normalized,
                mock_value=mock_value,
                assignment_source=assignment_source,
                hit_count=1,
                created_at=now,
                updated_at=now,
            )
            self._by_id[entry.mapping_id] = entry
            self._by_normalized[normalized] = entry
            self._after_mutate()
            self._log("resolve", entry)
            return entry.model_copy(deep=True)

    def resolve(
        self,
        source_text: str,
        user_mock: str | None = None,
    ) -> MockEntry:
        """Lookup or auto-assign a mock. Non-blank user_mock becomes user.

        Matching is by name text only, irrespective of the PII category
        the caller detected the span as — the same patch applies no
        matter what category a name is found under.

        Args:
            source_text: Original span text (PII).
            user_mock: Optional override for this resolve.

        Returns:
            The stored mapping (copy). Always creates one if nothing matches.

        Raises:
            MockValidationError: If ``source_text`` is empty/whitespace.
        """
        entry = self._resolve_or_lookup(source_text, user_mock, create_if_missing=True)
        assert entry is not None  # create_if_missing=True never returns None
        return entry

    def lookup(self, source_text: str) -> MockEntry | None:
        """Read-only match against known entries — never auto-assigns.

        Same lookup order as ``resolve`` (exact normalized, then fuzzy
        match), but returns ``None`` instead of creating a new
        auto-assigned mapping when nothing matches. Used when detection is
        restricted to the curated dictionary only.

        Args:
            source_text: Original span text (PII).

        Returns:
            The stored mapping (copy), or ``None`` if nothing matches.

        Raises:
            MockValidationError: If ``source_text`` is empty/whitespace.
        """
        return self._resolve_or_lookup(source_text, None, create_if_missing=False)

    def list(self) -> list[MockEntry]:
        """Return deep copies of all entries.

        Returns:
            Snapshot of current mappings (order unspecified).
        """
        with self._lock:
            return [row.model_copy(deep=True) for row in self._by_id.values()]

    def upsert(self, source_text: str, mock_value: str) -> MockEntry:
        """Create or replace a user-assigned mapping.

        Args:
            source_text: Original text to map (PII).
            mock_value: Non-blank replacement painted on the PDF.

        Returns:
            The stored mapping (copy).

        Raises:
            MockValidationError: Empty source_text or blank mock_value.
        """
        normalized = _require_normalized(source_text)
        cleaned = _nonblank(mock_value)
        if cleaned is None:
            raise MockValidationError("mock_value", "empty")
        now = datetime.now(timezone.utc)
        with self._lock:
            existing = self._by_normalized.get(normalized)
            if existing is not None:
                existing.mock_value = cleaned
                existing.assignment_source = "user"
                existing.updated_at = now
                self._after_mutate()
                self._log("upsert", existing)
                return existing.model_copy(deep=True)

            entry = MockEntry(
                mapping_id=self._new_mapping_id(),
                source_text=source_text,
                normalized=normalized,
                mock_value=cleaned,
                assignment_source="user",
                hit_count=0,
                created_at=now,
                updated_at=now,
            )
            self._by_id[entry.mapping_id] = entry
            self._by_normalized[normalized] = entry
            self._after_mutate()
            self._log("upsert", entry)
            return entry.model_copy(deep=True)

    def override(self, mapping_id: str, mock_value: str) -> MockEntry:
        """Replace the stored mock_value for an existing mapping.

        Args:
            mapping_id: Stable id from a prior resolve/upsert.
            mock_value: Non-blank user replacement.

        Returns:
            The updated mapping (copy).

        Raises:
            MockMappingNotFound: Unknown ``mapping_id``.
            MockValidationError: Blank ``mock_value``.
        """
        cleaned = _nonblank(mock_value)
        with self._lock:
            entry = self._by_id.get(mapping_id)
            if entry is None:
                raise MockMappingNotFound(mapping_id)
            if cleaned is None:
                raise MockValidationError("mock_value", "empty")
            entry.mock_value = cleaned
            entry.assignment_source = "user"
            entry.updated_at = datetime.now(timezone.utc)
            self._after_mutate()
            self._log("override", entry)
            return entry.model_copy(deep=True)

    def delete(self, mapping_id: str) -> None:
        """Forget a mapping. Re-resolving the same text later gets back the
        same deterministic auto mock value (a fresh, unrelated mapping_id).

        Args:
            mapping_id: Stable id to remove.

        Raises:
            MockMappingNotFound: Unknown ``mapping_id``.
        """
        with self._lock:
            entry = self._by_id.get(mapping_id)
            if entry is None:
                raise MockMappingNotFound(mapping_id)
            del self._by_id[mapping_id]
            self._by_normalized.pop(entry.normalized, None)
            self._log("delete", entry)
            self._after_mutate()

    def clear_all(self) -> int:
        """Remove every mapping from memory and the persisted snapshot.

        Returns:
            Number of mappings removed.
        """
        with self._lock:
            cleared_count = len(self._by_id)
            self._by_id.clear()
            self._by_normalized.clear()
            self._after_mutate()
            logger.info("mock_clear_all count=%s", cleared_count)
            return cleared_count


class MockDictionaryStore(_InMemoryDictionary):
    """Thread-safe mock dictionary with write-through snapshot.

    Args:
        snapshot_path: JSON file under the injected PII volume.
    """

    def __init__(
        self, snapshot_path: Path, fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD
    ) -> None:
        super().__init__(fuzzy_threshold=fuzzy_threshold)
        self._snapshot_path = snapshot_path
        self._load()

    def _after_mutate(self) -> None:
        self._persist()

    def _load(self) -> None:
        path = self._snapshot_path
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("corrupt mock dictionary snapshot") from exc
        snapshot = _Snapshot.model_validate(payload)
        skipped_dupes = 0
        for entry in snapshot.entries:
            if entry.mapping_id in self._by_id:
                skipped_dupes += 1
                continue
            self._by_id[entry.mapping_id] = entry
            self._by_normalized[entry.normalized] = entry
        if skipped_dupes:
            logger.warning(
                "mock_dictionary_duplicate_mapping_ids skipped=%s", skipped_dupes
            )

    def _persist(self) -> None:
        snapshot = _Snapshot(
            version=1,
            entries=list(self._by_id.values()),
        )
        atomic_write_text(self._snapshot_path, snapshot.model_dump_json(indent=2))


class MockMockDictionary(_InMemoryDictionary):
    """In-memory Protocol double. No snapshot I/O. For Stories 8–9."""
