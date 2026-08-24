"""Mock dictionary and substitution-ledger models, errors, and Protocols."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field


class MockMappingNotFound(Exception):
    """Unknown mapping_id. Story 8 maps to 404 MAPPING_NOT_FOUND."""

    def __init__(self, mapping_id: str) -> None:
        self.mapping_id = mapping_id
        self.code = "MAPPING_NOT_FOUND"
        super().__init__(f"mapping not found: {mapping_id}")


class MockValidationError(Exception):
    """Empty source_text / blank mock_value. Do not embed the rejected text."""

    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        self.code = "VALIDATION_ERROR"
        super().__init__(f"{field}: {reason}")


class MockEntry(BaseModel):
    """A source-to-mock mapping stored in the dictionary (PII store)."""

    mapping_id: str
    source_text: str
    normalized: str
    mock_value: str
    assignment_source: Literal["auto", "user"]
    hit_count: int
    created_at: datetime
    updated_at: datetime


class LedgerEntry(BaseModel):
    """One substitution row for a job (includes source_text)."""

    mapping_id: str
    source_text: str
    mock_value: str
    entity_type: str
    assignment_source: Literal["auto", "user"]
    hit_count: int
    pages: list[int] = Field(default_factory=list)


class SubstitutionLedger(BaseModel):
    """Per-job ledger of source-to-mock substitutions (PII store)."""

    request_id: str
    created_at: datetime
    entries: list[LedgerEntry] = Field(default_factory=list)
    brand_zones: list[dict[str, object]] = Field(default_factory=list)


class MockDictionaryStoreProtocol(Protocol):
    """Contract for the mock dictionary (Stories 8–9 inject doubles).

    Matching is by name text only — irrespective of any PII category the
    caller detected the span as, the same source text always resolves to
    the same stored mapping.
    """

    def resolve(
        self,
        source_text: str,
        user_mock: str | None = None,
    ) -> MockEntry: ...

    def lookup(
        self,
        source_text: str,
    ) -> MockEntry | None: ...

    def list(self) -> list[MockEntry]: ...

    def upsert(
        self,
        source_text: str,
        mock_value: str,
    ) -> MockEntry: ...

    def override(
        self,
        mapping_id: str,
        mock_value: str,
    ) -> MockEntry: ...

    def delete(self, mapping_id: str) -> None: ...

    def find_prefix_collisions(
        self, normalized: str, *, exclude_mapping_id: str | None = None
    ) -> list[MockEntry]: ...

    def best_match_score(self, normalized: str) -> float: ...


class LedgerStoreProtocol(Protocol):
    """Contract for per-job substitution ledger persistence."""

    def save(self, ledger: SubstitutionLedger) -> Path: ...

    def get(self, request_id: str) -> SubstitutionLedger | None: ...
