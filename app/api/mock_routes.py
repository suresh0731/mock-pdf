"""Mock dictionary and substitution-ledger HTTP routes.

Exposes `/v1/mocks*` CRUD and `GET /v1/redact/ledger/{request_id}`.
Success bodies may include `source_text`; error responses and logs must not.
The router is not mounted here — Story 10 / Integration includes it.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, NoReturn, Protocol, runtime_checkable

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ValidationError, field_validator

from app.config import get_settings

logger = logging.getLogger(__name__)

_NOT_FOUND_TYPE_NAMES = frozenset(
    {
        "MappingNotFoundError",
        "MockMappingNotFoundError",
        "MockMappingNotFound",
    }
)


class MappingNotFoundError(Exception):
    """Unknown mapping id. ``code`` is ``MAPPING_NOT_FOUND``."""

    code = "MAPPING_NOT_FOUND"


@dataclass
class AuthSettings:
    """Auth flags injected into route dependencies (tests override this)."""

    admin_auth_enabled: bool
    api_key: str


@runtime_checkable
class MockDictionaryStoreProtocol(Protocol):
    """Structural contract for the Story 5 mock dictionary store."""

    def list(self) -> list[Any]:
        """Return all persisted mock mappings."""
        ...

    def upsert(self, source_text: str, mock_value: str) -> Any:
        """Create or update a mapping by source text."""
        ...

    def override(self, mapping_id: str, mock_value: str) -> Any:
        """Replace the stored mock_value."""
        ...

    def delete(self, mapping_id: str) -> None:
        """Remove a mapping by id."""
        ...


@runtime_checkable
class LedgerStoreProtocol(Protocol):
    """Structural contract for the per-job substitution ledger store."""

    def get(self, request_id: str) -> Any | None:
        """Return the ledger for ``request_id``, or None if missing."""
        ...


def _non_blank(value: str) -> str:
    """Reject whitespace-only strings after strip."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("must be non-blank")
    return value.strip()


class CreateMockRequest(BaseModel):
    """Body for POST /v1/mocks (validated inside the handler, not by FastAPI)."""

    source_text: str
    mock_value: str

    @field_validator("source_text", "mock_value")
    @classmethod
    def validate_non_blank(cls, value: str) -> str:
        """Require a non-blank value after stripping whitespace."""
        return _non_blank(value)


class OverrideMockRequest(BaseModel):
    """Body for PUT /v1/mocks/{mapping_id}."""

    mock_value: str

    @field_validator("mock_value")
    @classmethod
    def validate_non_blank(cls, value: str) -> str:
        """Require a non-blank mock_value after stripping whitespace."""
        return _non_blank(value)


def get_auth_settings() -> AuthSettings:
    """Load auth flags from process settings (production only)."""
    settings = get_settings()
    return AuthSettings(
        admin_auth_enabled=settings.admin_auth_enabled,
        api_key=settings.api_key,
    )


@lru_cache
def get_mock_store() -> MockDictionaryStoreProtocol:
    """Process-wide mock dictionary constructed from settings path."""
    from app.services.pii.mock_dictionary import MockDictionaryStore
    from app.services.pii.seed_loader import load_seed_entries

    settings = get_settings()
    store = MockDictionaryStore(
        snapshot_path=settings.mock_dictionary_path,
        fuzzy_threshold=settings.fuzzy_match_threshold,
    )
    load_seed_entries(store, settings.mock_seed_path)
    return store


@lru_cache
def get_ledger_store() -> LedgerStoreProtocol:
    """Process-wide ledger store under ``{shard_base}/shards``."""
    from app.services.redact.ledger_store import LedgerStore

    return LedgerStore(base_dir=get_settings().shard_base_path / "shards")


@lru_cache
def get_ocr_output_store() -> "OcrOutputStore":
    """Process-wide OCR output dump store under ``{shard_base}/ocr-output``."""
    from app.services.redact.ocr_output_store import OcrOutputStore

    return OcrOutputStore(base_dir=get_settings().shard_base_path / "ocr-output")


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    auth: AuthSettings = Depends(get_auth_settings),
) -> None:
    """Reject the request when admin auth is on and the API key does not match."""
    if auth.admin_auth_enabled and auth.api_key:
        if x_api_key != auth.api_key:
            raise HTTPException(
                401,
                detail={
                    "error": {
                        "code": "UNAUTHORIZED",
                        "message": "Invalid API key",
                    }
                },
            )


def _error_response(
    status: int,
    code: str,
    message: str,
    request_id: str | None = None,
) -> HTTPException:
    """Build the standard ``{error: {code, message, request_id, timestamp}}`` envelope."""
    error: dict[str, Any] = {
        "code": code,
        "message": message,
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return HTTPException(status_code=status, detail={"error": error})


def _as_dict(entry: Any) -> dict[str, Any]:
    """Serialize a store entry for a success body (may include source_text)."""
    dump = getattr(entry, "model_dump", None)
    if callable(dump):
        dumped = dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    if isinstance(entry, dict):
        return dict(entry)
    data = getattr(entry, "__dict__", None)
    if isinstance(data, dict):
        return dict(data)
    return {
        "mapping_id": getattr(entry, "mapping_id", None),
        "source_text": getattr(entry, "source_text", None),
        "normalized": getattr(entry, "normalized", None),
        "mock_value": getattr(entry, "mock_value", None),
        "assignment_source": getattr(entry, "assignment_source", None),
        "hit_count": getattr(entry, "hit_count", None),
        "created_at": getattr(entry, "created_at", None),
        "updated_at": getattr(entry, "updated_at", None),
    }


def _is_mapping_not_found(exc: Exception) -> bool:
    """True for local MappingNotFoundError or Story 5 duck-typed equivalents."""
    if isinstance(exc, MappingNotFoundError):
        return True
    if type(exc).__name__ in _NOT_FOUND_TYPE_NAMES:
        return True
    return getattr(exc, "code", "") == "MAPPING_NOT_FOUND"


def _reraise_store_error(
    exc: Exception, mapping_id: str | None = None
) -> NoReturn:
    """Map store failures to HTTP errors without leaking PII from ``exc``."""
    if isinstance(exc, HTTPException):
        raise exc
    if _is_mapping_not_found(exc):
        raise _error_response(
            404, "MAPPING_NOT_FOUND", "Mapping not found"
        ) from None
    logger.error("Unexpected store error mapping_id=%s", mapping_id or "-")
    raise _error_response(
        500, "INTERNAL_ERROR", "Unexpected server error"
    ) from None


def _validate_body(model: type[BaseModel], payload: dict[str, Any]) -> Any:
    """Validate a raw dict; raise generic 422 so PII never appears in 4xx."""
    try:
        return model.model_validate(payload)
    except ValidationError:
        raise _error_response(
            422, "VALIDATION_ERROR", "Invalid request body"
        ) from None


router = APIRouter(
    prefix="/v1",
    tags=["mocks"],
    dependencies=[Depends(require_api_key)],
)


@router.get("/mocks")
async def list_mocks(
    store: MockDictionaryStoreProtocol = Depends(get_mock_store),
) -> dict[str, Any]:
    """Return all stored mappings.

    Args:
        store: Injected mock dictionary store.

    Returns:
        ``{count, entries}`` where entries may include ``source_text``.
    """
    try:
        entries = store.list()
    except Exception as exc:
        _reraise_store_error(exc)
    serialized = [_as_dict(e) for e in entries]
    logger.info("mock list count=%s", len(serialized))
    return {"count": len(serialized), "entries": serialized}


@router.post("/mocks")
async def upsert_mock(
    payload: dict[str, Any],
    store: MockDictionaryStoreProtocol = Depends(get_mock_store),
) -> JSONResponse:
    """Create or update a mapping. 201 if mapping_id was new, else 200.

    Args:
        payload: Raw JSON object (validated inside the handler).
        store: Injected mock dictionary store.

    Returns:
        JSONResponse with the MockEntry body and 201 or 200.
    """
    req = _validate_body(CreateMockRequest, payload)
    try:
        existing_ids = {_as_dict(e).get("mapping_id") for e in store.list()}
        entry = store.upsert(req.source_text, req.mock_value)
    except Exception as exc:
        _reraise_store_error(exc)
    data = _as_dict(entry)
    mapping_id = data.get("mapping_id")
    status_code = 200 if mapping_id in existing_ids else 201
    logger.info(
        "mock upsert mapping_id=%s assignment_source=%s",
        mapping_id,
        data.get("assignment_source"),
    )
    return JSONResponse(content=data, status_code=status_code)


@router.put("/mocks/{mapping_id}")
async def override_mock(
    mapping_id: str,
    payload: dict[str, Any],
    store: MockDictionaryStoreProtocol = Depends(get_mock_store),
) -> dict[str, Any]:
    """Override mock_value for an existing mapping.

    Args:
        mapping_id: Target mapping identifier.
        payload: Raw JSON object with ``mock_value``.
        store: Injected mock dictionary store.

    Returns:
        Updated MockEntry (may include ``source_text``).
    """
    req = _validate_body(OverrideMockRequest, payload)
    try:
        entry = store.override(mapping_id, req.mock_value)
    except Exception as exc:
        _reraise_store_error(exc, mapping_id=mapping_id)
    data = _as_dict(entry)
    logger.info(
        "mock override mapping_id=%s assignment_source=%s",
        mapping_id,
        data.get("assignment_source"),
    )
    return data


@router.delete("/mocks/{mapping_id}", status_code=204)
async def delete_mock(
    mapping_id: str,
    store: MockDictionaryStoreProtocol = Depends(get_mock_store),
) -> Response:
    """Delete a mapping by id.

    Args:
        mapping_id: Target mapping identifier.
        store: Injected mock dictionary store.

    Returns:
        Empty 204 response.
    """
    try:
        store.delete(mapping_id)
    except Exception as exc:
        _reraise_store_error(exc, mapping_id=mapping_id)
    logger.info("mock delete mapping_id=%s", mapping_id)
    return Response(status_code=204)


def _csv_attachment(csv_text: str, filename: str) -> Response:
    """A CSV response that browsers download instead of rendering inline."""
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/mocks/export")
async def export_mocks(
    store: MockDictionaryStoreProtocol = Depends(get_mock_store),
) -> Response:
    """Download all current mappings as CSV (editable in Excel, re-uploadable).

    Args:
        store: Injected mock dictionary store.

    Returns:
        CSV attachment; ``source_text`` and ``mock_value`` are included.
    """
    from app.services.pii.mapping_csv import export_mappings_csv

    try:
        entries = store.list()
    except Exception as exc:
        _reraise_store_error(exc)
    csv_text = export_mappings_csv([_as_dict(e) for e in entries])
    logger.info("mock export count=%s", len(entries))
    return _csv_attachment(csv_text, "mock-mappings.csv")


@router.get("/mocks/template")
async def download_mock_template() -> Response:
    """Download a starter CSV — same columns as ``/mocks/export``.

    Returns:
        CSV attachment with headers and two worked examples.
    """
    from app.services.pii.mapping_csv import template_csv

    return _csv_attachment(template_csv(), "mock-mappings-template.csv")


@router.post("/mocks/import")
async def import_mocks(
    file: UploadFile = File(...),
    store: MockDictionaryStoreProtocol = Depends(get_mock_store),
) -> dict[str, Any]:
    """Bulk-add mappings from an uploaded CSV (export or template shape).

    Existing mappings are never overwritten — a row is skipped whenever its
    ``source_text`` already has a mapping, so re-uploading a previously
    downloaded file (or one with overlapping rows) is always safe.

    Args:
        file: Uploaded CSV file (``.csv``).
        store: Injected mock dictionary store.

    Returns:
        ``{inserted, skipped_existing, skipped_invalid}`` row counts, plus
        ``rejected_report_path`` when any rows were skipped.
    """
    from app.services.pii.mapping_csv import import_mappings_csv, save_skipped_rows_report

    raw = await file.read()
    try:
        csv_text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise _error_response(
            422, "VALIDATION_ERROR", "File is not valid UTF-8 text"
        ) from None
    try:
        result = import_mappings_csv(store, csv_text)
    except Exception as exc:
        _reraise_store_error(exc)
    report_path = save_skipped_rows_report(result, get_settings().mock_import_report_dir)
    logger.info(
        "mock import inserted=%s skipped_existing=%s skipped_invalid=%s report_written=%s",
        result.inserted,
        result.skipped_existing,
        result.skipped_invalid,
        report_path is not None,
    )
    body: dict[str, Any] = {
        "inserted": result.inserted,
        "skipped_existing": result.skipped_existing,
        "skipped_invalid": result.skipped_invalid,
    }
    if report_path is not None:
        body["rejected_report_path"] = str(report_path)
    return body


@router.get("/redact/ledger/{request_id}")
async def get_ledger(
    request_id: str,
    ledger: LedgerStoreProtocol = Depends(get_ledger_store),
) -> dict[str, Any]:
    """Return the substitution ledger for a redact job.

    Args:
        request_id: Job identifier (UUID).
        ledger: Injected ledger store.

    Returns:
        SubstitutionLedger (may include ``source_text``).
    """
    try:
        found = ledger.get(request_id)
    except Exception as exc:
        _reraise_store_error(exc)
    if found is None:
        raise _error_response(
            404, "LEDGER_NOT_FOUND", "No ledger for request_id"
        )
    logger.info("ledger get request_id=%s", request_id)
    return _as_dict(found)
