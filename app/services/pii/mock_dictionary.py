"""In-memory mock dictionary with write-through JSON snapshot.

This module is a PII store: ``source_text`` is persisted in the snapshot
only. Logs may include mapping_id, entity_type, and assignment_source —
never source_text, normalized text, or a mock paired with its source.
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
from app.services.pii.name_matcher import is_token_subset_collision, token_sort_ratio

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
# truth, so matching against them uses a much lower threshold than auto-vs-
# auto matching. Calibrated against real OCR ensemble output on scanned bank
# letters (see tests/test_mock_dictionary.py): garbled table-cell reads of a
# known organization ("Cnartercd Standerd Custpay", "Terop Blife Stabill
# Link Pendapatin") score 0.70-0.96 against their clean source text, while
# unrelated organizations never exceed ~0.58 — comfortable separation.
_TRUSTED_FUZZY_THRESHOLD = 0.65

_PREFIX_ALIASES: dict[str, str] = {
    "ORGANIZATION": "ORG",
    "ORG": "ORG",
    "PHONE_NUMBER": "PHONE",
    "PHONE": "PHONE",
    "EMAIL_ADDRESS": "EMAIL",
    "EMAIL": "EMAIL",
    "ADDRESS": "ADDR",
    "ADDR": "ADDR",
    "LOCATION": "ADDR",
    "NRIC": "ID",
    "US_SSN": "ID",
    "ID": "ID",
    "ACCOUNT_NUMBER": "ACCT",
    "ACCOUNT": "ACCT",
    "ACCT": "ACCT",
    "DATE_TIME": "DATE",
    "DATE": "DATE",
}


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


def prefix_for_entity_type(entity_type: str) -> str:
    """Map a Presidio/custom entity type to an auto-mock prefix.

    Args:
        entity_type: Detector or user label (e.g. ``ORGANIZATION``).

    Returns:
        Prefix such as ``ORG``, ``PERSON``, or the first 8 alphanumeric
        characters of the uppercased type (``CUSTOM`` if empty).
    """
    key = entity_type.upper().replace(" ", "_")
    if key in _PREFIX_ALIASES:
        return _PREFIX_ALIASES[key]
    alnum = "".join(ch for ch in key if ch.isalnum())
    return alnum[:8] if alnum else "CUSTOM"


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


def deterministic_auto_mock_value(entity_type: str, normalized_source_text: str) -> str:
    """Content-derived auto mock value: ``{PREFIX}_{HASH}``.

    Same ``(entity_type, normalized_source_text)`` always yields the same
    mock value on every machine, independent of run history or
    order-of-first-sight — replaces the old sequential per-type counter
    (``ORG_01``, ``ORG_02``, ...), where the label a never-seen value got
    depended on how many other new values had already been auto-assigned
    on that particular machine.

    This is the *collision-naive* default-length form used directly by
    callers that just need a deterministic label (e.g. tests predicting an
    expected value). ``_InMemoryDictionary._auto_mock_value`` wraps this
    with a deterministic collision-extension fallback for the rare case
    where two different real values happen to hash to the same suffix
    within one dictionary.

    Args:
        entity_type: Detector or user label (e.g. ``ORGANIZATION``).
        normalized_source_text: Already-normalized lookup key (see
            ``normalize_source``) — never the raw/unnormalized text.

    Returns:
        A ``{PREFIX}_{HASH}`` mock value, e.g. ``"ORG_1F3A9C"``.
    """
    prefix = prefix_for_entity_type(entity_type)
    return f"{prefix}_{_auto_suffix_hash(prefix, normalized_source_text, _AUTO_SUFFIX_LENGTH)}"


class _InMemoryDictionary:
    """Shared resolve/list/upsert/override/delete indexes. No disk I/O."""

    def __init__(self, fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD) -> None:
        self._by_normalized: dict[str, MockEntry] = {}
        self._by_id: dict[str, MockEntry] = {}
        self._by_account_number: dict[str, str] = {}
        self._fuzzy_threshold = fuzzy_threshold
        self._lock = threading.RLock()

    def _after_mutate(self) -> None:
        """Persist hook. Overridden by ``MockDictionaryStore``."""

    def _log(self, action: str, entry: MockEntry) -> None:
        logger.info(
            "mock_%s mapping_id=%s entity_type=%s assignment_source=%s",
            action,
            entry.mapping_id,
            entry.entity_type,
            entry.assignment_source,
        )

    def _new_mapping_id(self) -> str:
        while True:
            mapping_id = f"map_{secrets.token_hex(4)}"
            if mapping_id not in self._by_id:
                return mapping_id

    def _auto_mock_value(self, entity_type: str, normalized: str) -> str:
        """Deterministic auto mock value, extending the hash on collision.

        A collision (a *different* normalized value already holding the
        same mock_value string) is vanishingly unlikely at the default
        suffix length but handled deterministically rather than ignored:
        the suffix length grows until the candidate is unique, so the
        result still depends only on ``(entity_type, normalized)`` — never
        on insertion order.
        """
        prefix = prefix_for_entity_type(entity_type)
        for length in (_AUTO_SUFFIX_LENGTH, *_AUTO_SUFFIX_FALLBACK_LENGTHS):
            candidate = f"{prefix}_{_auto_suffix_hash(prefix, normalized, length)}"
            collision = any(
                entry.mock_value == candidate and entry.normalized != normalized
                for entry in self._by_id.values()
            )
            if not collision:
                return candidate
        return candidate

    def _best_match(
        self, normalized: str, entity_type: str, field_role: str | None
    ) -> tuple[str | None, float]:
        """Best same-type entry and its ratio, regardless of threshold (or
        ``(None, 0.0)`` if there are no same-type/role-eligible entries).

        Shared scoring core for ``_find_fuzzy_match`` (applies the
        threshold) and ``best_match_score`` (a pure, threshold-free ratio
        used by the maximal-munch window-extension probe — see
        ``app.pipeline.redact``). See ``_find_fuzzy_match`` for the
        trusted-vs-auto scoping rationale.
        """
        best_id: str | None = None
        best_ratio = 0.0
        for entry in self._by_id.values():
            if entry.entity_type != entity_type:
                continue
            trusted = entry.assignment_source == "user"
            if not trusted and field_role and entry.field_role and entry.field_role != field_role:
                continue
            ratio = token_sort_ratio(normalized, entry.normalized)
            if ratio > best_ratio:
                best_ratio = ratio
                best_id = entry.mapping_id
        return best_id, best_ratio

    def _find_fuzzy_match(
        self, normalized: str, entity_type: str, field_role: str | None
    ) -> str | None:
        """Best same-type entry above its applicable fuzzy threshold, or None.

        Scoped to the same ``entity_type``. Auto-assigned entries are also
        scoped to the same ``field_role`` (when both the candidate and the
        entry state one) and compared at the standard, stricter threshold —
        this is how OCR variants of an *unverified* name collapse into a
        single mapping instead of accumulating duplicates, without risking
        two genuinely different auto-detected organizations merging.

        Manually curated (``assignment_source == "user"``) entries are
        verified ground truth, so they're compared at a lower threshold and
        without the ``field_role`` restriction: the same real value can
        legitimately be found by a different extraction mode (e.g. a table
        column vs. an address block, or as the debit side in one letter and
        the credit side in another) and should still collapse onto the one
        curated row rather than spawning a fresh auto-assigned duplicate.

        This threshold check can't fully share ``_best_match``'s single
        best-ratio-wins result: the *applicable* threshold itself depends
        on whether that particular entry is trusted, not on the winning
        entry only, so a stricter re-scan against each candidate's own
        threshold is kept here rather than trying to fold both thresholds
        into one shared comparison.
        """
        best_id: str | None = None
        best_ratio = -1.0
        for entry in self._by_id.values():
            if entry.entity_type != entity_type:
                continue
            trusted = entry.assignment_source == "user"
            if not trusted and field_role and entry.field_role and entry.field_role != field_role:
                continue
            threshold = _TRUSTED_FUZZY_THRESHOLD if trusted else self._fuzzy_threshold
            ratio = token_sort_ratio(normalized, entry.normalized)
            if ratio >= threshold and ratio > best_ratio:
                best_ratio = ratio
                best_id = entry.mapping_id
        return best_id

    def best_match_score(
        self, normalized: str, entity_type: str, field_role: str | None = None
    ) -> float:
        """Best fuzzy-match ratio against known same-type entries, or ``0.0``.

        Pure read: unlike ``resolve``/``lookup``, never mutates
        ``hit_count``/``updated_at`` on a match — safe to call speculatively
        to score several candidate text windows (see the maximal-munch
        window-extension probe in ``app.pipeline.redact``) without
        recording a hit for windows that don't end up being used.

        Args:
            normalized: Already-normalized candidate text (see
                ``normalize_source``).
            entity_type: Scopes the search to same-type entries only.
            field_role: Optional structural role, used the same way
                ``_find_fuzzy_match`` uses it (auto-assigned entries only).

        Returns:
            Ratio in ``[0.0, 1.0]``. ``0.0`` if ``normalized`` is blank or
            no same-type/role-eligible entry exists.
        """
        if not normalized.strip():
            return 0.0
        with self._lock:
            _, ratio = self._best_match(normalized, entity_type, field_role)
        return ratio

    def find_prefix_collisions(
        self, normalized: str, entity_type: str, *, exclude_mapping_id: str | None = None
    ) -> list[MockEntry]:
        """Other same-type entries whose tokens strictly contain, or are
        strictly contained by, ``normalized``'s tokens.

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

        Args:
            normalized: Already-normalized candidate text (see
                ``normalize_source``).
            entity_type: Scopes the search to same-type entries only.
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
                if entry.entity_type == entity_type
                and entry.mapping_id != exclude_mapping_id
                and entry.normalized != normalized
                and is_token_subset_collision(normalized, entry.normalized)
            ]

    def _log_prefix_collision_if_any(
        self, normalized: str, entity_type: str, resolved_mapping_id: str | None
    ) -> None:
        """Warn (mapping ids/counts only, never text — SEC-001) when a
        resolution is ambiguous under ``find_prefix_collisions``."""
        collisions = self.find_prefix_collisions(
            normalized, entity_type, exclude_mapping_id=resolved_mapping_id
        )
        if collisions:
            logger.warning(
                "mock_prefix_collision_detected mapping_id=%s entity_type=%s "
                "colliding_mapping_ids=%s",
                resolved_mapping_id,
                entity_type,
                sorted(entry.mapping_id for entry in collisions),
            )

    def _resolve_or_lookup(
        self,
        source_text: str,
        entity_type: str,
        user_mock: str | None,
        account_number: str | None,
        field_role: str | None,
        *,
        create_if_missing: bool,
    ) -> MockEntry | None:
        """Shared implementation for ``resolve`` (may create) and ``lookup``
        (read-only — returns ``None`` instead of auto-assigning).

        Lookup order: exact normalized text match, then (if ``account_number``
        is given and already known) that account's mapping regardless of
        name spelling, then a fuzzy name match (see ``_find_fuzzy_match``),
        then — only when ``create_if_missing`` — a fresh auto-assigned entry.
        """
        normalized = _require_normalized(source_text)
        override = _nonblank(user_mock)
        acct = _nonblank(account_number)
        now = datetime.now(timezone.utc)
        with self._lock:
            existing = self._by_normalized.get(normalized)

            if existing is None and acct is not None:
                mapped_id = self._by_account_number.get(acct)
                if mapped_id is not None:
                    existing = self._by_id.get(mapped_id)

            if existing is None:
                fuzzy_id = self._find_fuzzy_match(normalized, entity_type, field_role)
                if fuzzy_id is not None:
                    existing = self._by_id.get(fuzzy_id)

            if existing is not None:
                self._log_prefix_collision_if_any(normalized, entity_type, existing.mapping_id)
                existing.hit_count += 1
                existing.updated_at = now
                if override is not None:
                    existing.mock_value = override
                    existing.assignment_source = "user"
                if acct is not None and not existing.account_number:
                    # Only fill in a missing account number, never overwrite
                    # one already on file: a fuzzy name match can pair with
                    # an OCR-mangled digit string from *this* particular
                    # read, and blindly overwriting would corrupt a stable
                    # (often curated, ground-truth) account number with
                    # that noise.
                    existing.account_number = acct
                    self._by_account_number[acct] = existing.mapping_id
                if field_role is not None and not existing.field_role:
                    existing.field_role = field_role
                self._by_normalized.setdefault(normalized, existing)
                self._after_mutate()
                self._log("resolve", existing)
                return existing.model_copy(deep=True)

            if not create_if_missing:
                return None

            self._log_prefix_collision_if_any(normalized, entity_type, None)

            assignment_source: Literal["auto", "user"]
            if override is not None:
                mock_value = override
                assignment_source = "user"
            else:
                mock_value = self._auto_mock_value(entity_type, normalized)
                assignment_source = "auto"

            entry = MockEntry(
                mapping_id=self._new_mapping_id(),
                source_text=source_text,
                normalized=normalized,
                mock_value=mock_value,
                entity_type=entity_type,
                assignment_source=assignment_source,
                hit_count=1,
                created_at=now,
                updated_at=now,
                account_number=acct,
                field_role=field_role,
            )
            self._by_id[entry.mapping_id] = entry
            self._by_normalized[normalized] = entry
            if acct is not None:
                self._by_account_number[acct] = entry.mapping_id
            self._after_mutate()
            self._log("resolve", entry)
            return entry.model_copy(deep=True)

    def resolve(
        self,
        source_text: str,
        entity_type: str,
        user_mock: str | None = None,
        account_number: str | None = None,
        field_role: str | None = None,
    ) -> MockEntry:
        """Lookup or auto-assign a mock. Non-blank user_mock becomes user.

        Args:
            source_text: Original span text (PII).
            entity_type: Detector type used for the auto prefix.
            user_mock: Optional override for this resolve.
            account_number: Optional stable key; numbers don't drift with
                OCR the way names do, so this takes priority over fuzzy
                name matching when present.
            field_role: Optional structural role (e.g.
                ``debit_account_name``) used to scope fuzzy matching.

        Returns:
            The stored mapping (copy). Always creates one if nothing matches.

        Raises:
            MockValidationError: If ``source_text`` is empty/whitespace.
        """
        entry = self._resolve_or_lookup(
            source_text, entity_type, user_mock, account_number, field_role, create_if_missing=True
        )
        assert entry is not None  # create_if_missing=True never returns None
        return entry

    def lookup(
        self,
        source_text: str,
        entity_type: str,
        account_number: str | None = None,
        field_role: str | None = None,
    ) -> MockEntry | None:
        """Read-only match against known entries — never auto-assigns.

        Same lookup order as ``resolve`` (exact normalized, then account
        number, then fuzzy match), but returns ``None`` instead of creating
        a new auto-assigned mapping when nothing matches. Used when
        detection is restricted to the curated dictionary only.

        Args:
            source_text: Original span text (PII).
            entity_type: Detector type, used to scope matching.
            account_number: Optional stable key checked before fuzzy match.
            field_role: Optional structural role used to scope fuzzy match.

        Returns:
            The stored mapping (copy), or ``None`` if nothing matches.

        Raises:
            MockValidationError: If ``source_text`` is empty/whitespace.
        """
        return self._resolve_or_lookup(
            source_text, entity_type, None, account_number, field_role, create_if_missing=False
        )

    def list(self) -> list[MockEntry]:
        """Return deep copies of all entries.

        Returns:
            Snapshot of current mappings (order unspecified).
        """
        with self._lock:
            return [row.model_copy(deep=True) for row in self._by_id.values()]

    def upsert(
        self,
        source_text: str,
        mock_value: str,
        entity_type: str = "CUSTOM",
        account_number: str | None = None,
        field_role: str | None = None,
    ) -> MockEntry:
        """Create or replace a user-assigned mapping.

        Args:
            source_text: Original text to map (PII).
            mock_value: Non-blank replacement painted on the PDF.
            entity_type: Type label for a newly created row.
            account_number: Optional stable key for later account-number
                priority lookups via ``resolve``.
            field_role: Optional structural role for a newly created row.

        Returns:
            The stored mapping (copy).

        Raises:
            MockValidationError: Empty source_text or blank mock_value.
        """
        normalized = _require_normalized(source_text)
        cleaned = _nonblank(mock_value)
        if cleaned is None:
            raise MockValidationError("mock_value", "empty")
        acct = _nonblank(account_number)
        now = datetime.now(timezone.utc)
        with self._lock:
            existing = self._by_normalized.get(normalized)
            if existing is not None:
                existing.mock_value = cleaned
                existing.assignment_source = "user"
                existing.updated_at = now
                if acct is not None:
                    existing.account_number = acct
                    self._by_account_number[acct] = existing.mapping_id
                if field_role is not None:
                    existing.field_role = field_role
                self._after_mutate()
                self._log("upsert", existing)
                return existing.model_copy(deep=True)

            entry = MockEntry(
                mapping_id=self._new_mapping_id(),
                source_text=source_text,
                normalized=normalized,
                mock_value=cleaned,
                entity_type=entity_type,
                assignment_source="user",
                hit_count=0,
                created_at=now,
                updated_at=now,
                account_number=acct,
                field_role=field_role,
            )
            self._by_id[entry.mapping_id] = entry
            self._by_normalized[normalized] = entry
            if acct is not None:
                self._by_account_number[acct] = entry.mapping_id
            self._after_mutate()
            self._log("upsert", entry)
            return entry.model_copy(deep=True)

    def override(
        self,
        mapping_id: str,
        mock_value: str,
        field_role: str | None = None,
        account_number: str | None = None,
    ) -> MockEntry:
        """Replace mock_value (and optionally field_role/account_number).

        Unlike ``resolve()``'s conservative "fill in only if missing"
        behaviour — needed there because an auto-detected candidate's own
        account number/role can be OCR noise — this is an explicit,
        UI/API-driven correction, so ``field_role``/``account_number`` are
        allowed to overwrite an existing value, not just fill a blank one.

        Args:
            mapping_id: Stable id from a prior resolve/upsert.
            mock_value: Non-blank user replacement.
            field_role: When given, replaces the stored role. An empty
                string clears it; ``None`` (default) leaves it untouched.
            account_number: When given, replaces the stored account number
                (and its reverse-lookup index entry). An empty string
                clears it; ``None`` (default) leaves it untouched.

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
            if field_role is not None:
                entry.field_role = _nonblank(field_role)
            if account_number is not None:
                if entry.account_number is not None:
                    self._by_account_number.pop(entry.account_number, None)
                new_acct = _nonblank(account_number)
                entry.account_number = new_acct
                if new_acct is not None:
                    self._by_account_number[new_acct] = entry.mapping_id
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
        for entry in snapshot.entries:
            self._by_id[entry.mapping_id] = entry
            self._by_normalized[entry.normalized] = entry
            if entry.account_number:
                self._by_account_number[entry.account_number] = entry.mapping_id

    def _persist(self) -> None:
        snapshot = _Snapshot(
            version=1,
            entries=list(self._by_id.values()),
        )
        path = self._snapshot_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)


class MockMockDictionary(_InMemoryDictionary):
    """In-memory Protocol double. No snapshot I/O. For Stories 8–9."""
