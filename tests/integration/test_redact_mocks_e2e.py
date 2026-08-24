"""Integration tests for mock dictionary, ledger, brand zones, and HTTP wiring."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from starlette.middleware.cors import CORSMiddleware

from app.api.mock_routes import (
    AuthSettings,
    get_auth_settings,
    get_ledger_store,
    get_mock_store,
)
from app.config import Settings, get_settings
from app.main import create_app
from app.models.pii_chunk import BBox
from app.models.redact import (
    ConfidenceBreakdown,
    CustomRedactTerm,
    RedactionRegion,
    RedactOptions,
)
from app.pipeline.redact import RedactPipeline, _user_mock_for_term
from app.services.ocr.ensemble_types import EnsembleWord
from app.services.pii.mock_dictionary import MockDictionaryStore
from app.services.redact.audit_store import AuditStore
from app.services.redact.ledger_store import LedgerStore
from app.ui.redact_portal import persist_mapping_override
from app.utils.audit import audit_log

ORG = "Acme Holdings"
PERSON = "Jane Doe"
MERGED = "Acme Holdings contacted Jane Doe"
ORG_START = MERGED.find(ORG)
ORG_END = ORG_START + len(ORG)
PERSON_START = MERGED.find(PERSON)
PERSON_END = PERSON_START + len(PERSON)
PERSON_RE = re.compile(r"PERSON_[0-9A-F]{2,}")


def _contains_key(payload: object, key: str) -> bool:
    if isinstance(payload, dict):
        if key in payload:
            return True
        return any(_contains_key(v, key) for v in payload.values())
    if isinstance(payload, list):
        return any(_contains_key(item, key) for item in payload)
    return False


def _words(page: int = 0) -> list[EnsembleWord]:
    return [
        EnsembleWord(
            text=ORG,
            bbox=BBox(x=40, y=200, w=180, h=18),
            ocr_confidence=0.95,
            engine_agreement=1.0,
            engines=["tesseract"],
            page=page,
            char_start=ORG_START,
            char_end=ORG_END,
        ),
        EnsembleWord(
            text=PERSON,
            bbox=BBox(x=40, y=240, w=90, h=18),
            ocr_confidence=0.94,
            engine_agreement=1.0,
            engines=["tesseract"],
            page=page,
            char_start=PERSON_START,
            char_end=PERSON_END,
        ),
    ]


async def _fake_ocr(*args, **kwargs):
    page = args[1] if len(args) > 1 else 0
    return MERGED, _words(page), []


def _fake_pii(text, locale=None):
    return [
        {"start": ORG_START, "end": ORG_END, "entity_type": "ORGANIZATION", "score": 0.99},
        {"start": PERSON_START, "end": PERSON_END, "entity_type": "PERSON", "score": 0.98},
    ]


def _letter_page() -> Image.Image:
    return Image.new("RGB", (720, 1100), "white")


def _clear_caches() -> None:
    get_settings.cache_clear()
    get_mock_store.cache_clear()
    get_ledger_store.cache_clear()


@pytest.fixture
def patch_ocr_pii(monkeypatch):
    monkeypatch.setattr("app.pipeline.redact.ensemble_ocr_page", _fake_ocr)
    monkeypatch.setattr("app.pipeline.redact.detect_pii", _fake_pii)
    monkeypatch.setattr("app.pipeline.redact.load_pages", lambda *a, **k: [_letter_page()])
    monkeypatch.setattr("app.pipeline.redact.extract_structure", lambda *a, **k: [])


@pytest.fixture
def wired_app(tmp_path, monkeypatch, patch_ocr_pii):
    monkeypatch.setenv("MOCK_DICTIONARY_PATH", str(tmp_path / "mappings.json"))
    monkeypatch.setenv("SHARD_BASE_PATH", str(tmp_path))
    monkeypatch.setenv("ADMIN_AUTH_ENABLED", "false")
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("PRESIDIO_ENABLED", "true")
    monkeypatch.setenv("FIELD_DETECTION_ENABLED", "false")
    monkeypatch.setenv("RESTRICT_TO_KNOWN_MAPPINGS", "false")
    _clear_caches()
    application = create_app()
    with TestClient(application) as client:
        yield client, tmp_path
    _clear_caches()


def _pipeline(tmp_path: Path, monkeypatch) -> tuple[RedactPipeline, MockDictionaryStore, LedgerStore]:
    monkeypatch.setenv("SHARD_BASE_PATH", str(tmp_path))
    monkeypatch.setenv("MOCK_DICTIONARY_PATH", str(tmp_path / "mappings.json"))
    _clear_caches()
    # These tests exercise dictionary/ledger/audit mechanics via a fake
    # detector, independent of field-anchored detection specifics — opt
    # into the legacy Presidio path explicitly and disable the new
    # geometry-only detector so it can't add/remove regions here.
    settings = Settings(
        shard_base_path=tmp_path,
        mock_dictionary_path=tmp_path / "mappings.json",
        presidio_enabled=True,
        field_detection_enabled=False,
        restrict_to_known_mappings=False,
        _env_file=None,
    )
    mock_store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    ledger_store = LedgerStore(base_dir=tmp_path / "shards")
    return (
        RedactPipeline(
            settings=settings,
            mock_store=mock_store,
            ledger_store=ledger_store,
            audit_store=AuditStore(),
        ),
        mock_store,
        ledger_store,
    )


def _run(pipeline: RedactPipeline, options: RedactOptions | None = None):
    return asyncio.run(pipeline.run(b"%PDF-1.4", "doc.pdf", options))


def _post_redact(client: TestClient) -> object:
    return client.post(
        "/v1/redact",
        files={"file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"options": "{}"},
    )


def _org_region(audit) -> object:
    return next(r for r in audit.redactions if r.entity_type == "ORGANIZATION")


def _person_region(audit) -> object:
    return next(r for r in audit.redactions if r.entity_type == "PERSON")


def test_redact_options_patch_flags_default_true():
    opts = RedactOptions()
    assert opts.patch_logo is True
    assert opts.patch_footer is True


def test_redaction_region_accepts_mock_fields():
    box = BBox(x=1, y=2, w=3, h=4)
    region = RedactionRegion(
        region_id="r-0001",
        page=0,
        entity_type="ORGANIZATION",
        canonical_bbox=box,
        original_bbox=box,
        padded_bbox=box,
        redaction_confidence=0.9,
        confidence_breakdown=ConfidenceBreakdown(
            presidio=0.9, ocr=0.8, engine_agreement=1.0, structural_context=0.0
        ),
        blur_tier="good",
        mock_value="ORG_01",
        mapping_id="map_ab",
        assignment_source="auto",
    )
    dumped = region.model_dump()
    assert region.mock_value == "ORG_01"
    assert region.mapping_id == "map_ab"
    assert region.assignment_source == "auto"
    assert "source_text" not in dumped


def test_settings_expose_mock_path_and_zone_percents(monkeypatch):
    monkeypatch.delenv("MOCK_DICTIONARY_PATH", raising=False)
    monkeypatch.delenv("LOGO_ZONE_TOP_PCT", raising=False)
    monkeypatch.delenv("LOGO_ZONE_RIGHT_PCT", raising=False)
    monkeypatch.delenv("FOOTER_ZONE_BOTTOM_PCT", raising=False)
    settings = Settings(_env_file=None)
    assert str(settings.mock_dictionary_path).endswith("mappings.json")
    assert settings.logo_zone_top_pct == 0.12
    assert settings.logo_zone_right_pct == 0.28
    assert settings.footer_zone_bottom_pct == 0.12


def test_redact_pipeline_enabled_flag_unchanged(monkeypatch):
    monkeypatch.delenv("REDACT_PIPELINE_ENABLED", raising=False)
    assert Settings(_env_file=None).redact_pipeline_enabled is True


def test_pipeline_reuses_mock_across_two_jobs(tmp_path, monkeypatch, patch_ocr_pii):
    pipeline, _, _ = _pipeline(tmp_path, monkeypatch)
    _, audit1, _ = _run(pipeline)
    _, audit2, _ = _run(pipeline)
    org1 = _org_region(audit1)
    org2 = _org_region(audit2)
    assert org1.mock_value == org2.mock_value
    assert org1.mapping_id == org2.mapping_id


def test_pipeline_user_override_xxx_used_on_later_job(tmp_path, monkeypatch, patch_ocr_pii):
    pipeline, store, _ = _pipeline(tmp_path, monkeypatch)
    _, audit1, _ = _run(pipeline)
    org1 = _org_region(audit1)
    store.override(org1.mapping_id, "XXX")
    _, audit2, _ = _run(pipeline)
    org2 = _org_region(audit2)
    assert org2.mock_value == "XXX"
    assert org2.assignment_source == "user"


def test_pipeline_unseen_person_gets_person_nn(tmp_path, monkeypatch, patch_ocr_pii):
    pipeline, store, _ = _pipeline(tmp_path, monkeypatch)
    _, audit, _ = _run(pipeline)
    person = _person_region(audit)
    assert PERSON_RE.fullmatch(person.mock_value)
    assert any(e.mock_value == person.mock_value for e in store.list())


def test_structure_extraction_receives_original_not_line_stripped_image(tmp_path, monkeypatch):
    """Docling/img2table must see the pristine scan, not the line-stripped
    canonical image OCR uses — strip_table_lines erases essentially every
    rule on tightly-ruled tables, handing the structure model strictly less
    information than the original scan had (see app/pipeline/redact.py)."""
    pipeline, _, _ = _pipeline(tmp_path, monkeypatch)
    page = Image.new("RGB", (400, 200), "white")
    ImageDraw.Draw(page).line([(10, 100), (390, 100)], fill="black", width=2)

    captured: dict[str, Image.Image] = {}

    def _record_structure(image, *a, **k):
        captured["image"] = image
        return []

    monkeypatch.setattr("app.pipeline.redact.load_pages", lambda *a, **k: [page])
    monkeypatch.setattr("app.pipeline.redact.ensemble_ocr_page", _fake_ocr)
    monkeypatch.setattr("app.pipeline.redact.extract_structure", _record_structure)
    monkeypatch.setattr("app.pipeline.redact.extract_table_geometry", lambda *a, **k: [])

    asyncio.run(pipeline._build_page_states(b"%PDF-1.4", "doc.pdf", RedactOptions()))

    # original_image is an untouched copy of the source page — pixel-identical
    # comparison confirms structure extraction did not instead receive the
    # line-stripped (and possibly sauvola-binarized) OCR canonical image.
    assert captured["image"].convert("RGB").tobytes() == page.convert("RGB").tobytes()


def test_pipeline_default_options_emit_logo_and_footer(tmp_path, monkeypatch, patch_ocr_pii):
    pipeline, _, _ = _pipeline(tmp_path, monkeypatch)
    _, audit, _ = _run(pipeline)
    brand = [r for r in audit.redactions if r.assignment_source == "brand"]
    logos = [r for r in brand if r.mock_value == "LOGO"]
    footers = [r for r in brand if r.mock_value == "FOOTER"]
    assert logos and footers
    assert all(r.mapping_id is None for r in logos + footers)


def test_pipeline_custom_mock_label_upserts_user(tmp_path, monkeypatch, patch_ocr_pii):
    pipeline, store, _ = _pipeline(tmp_path, monkeypatch)
    opts = RedactOptions(custom_redactions=[CustomRedactTerm(search_value=ORG, mock_label="XXX")])
    _, audit, _ = _run(pipeline, opts)
    row = next(e for e in store.list() if e.source_text == ORG)
    assert row.assignment_source == "user"
    assert row.mock_value == "XXX"
    org = _org_region(audit)
    assert org.mock_value == "XXX"
    assert org.assignment_source == "user"


def test_pipeline_custom_default_label_is_not_user_override(tmp_path, monkeypatch, patch_ocr_pii):
    monkeypatch.setattr("app.pipeline.redact.detect_pii", lambda *a, **k: [])
    pipeline, store, _ = _pipeline(tmp_path, monkeypatch)
    opts = RedactOptions(custom_redactions=[CustomRedactTerm(search_value=ORG)])
    _, audit, _ = _run(pipeline, opts)
    custom = next(r for r in audit.redactions if r.entity_type == "CUSTOM")
    assert custom.assignment_source == "auto"
    assert custom.mock_value != "CUSTOM"
    assert custom.mock_value.startswith("CUSTOM_")
    row = next(e for e in store.list() if e.source_text == ORG)
    assert row.assignment_source == "auto"


def test_pipeline_saves_ledger_with_source_mock(tmp_path, monkeypatch, patch_ocr_pii):
    pipeline, _, ledger_store = _pipeline(tmp_path, monkeypatch)
    _, audit, _ = _run(pipeline)
    ledger = ledger_store.get(audit.request_id)
    assert ledger is not None
    match = next(e for e in ledger.entries if e.source_text == ORG)
    org = _org_region(audit)
    assert match.mock_value == org.mock_value


def test_pipeline_audit_regions_omit_source_text(tmp_path, monkeypatch, patch_ocr_pii):
    pipeline, _, _ = _pipeline(tmp_path, monkeypatch)
    _, audit, _ = _run(pipeline)
    assert _contains_key(audit.model_dump(), "source_text") is False


def test_pipeline_logs_omit_original_pii(tmp_path, monkeypatch, patch_ocr_pii, caplog):
    pipeline, _, _ = _pipeline(tmp_path, monkeypatch)
    with caplog.at_level(logging.DEBUG):
        _run(pipeline)
    text = caplog.text
    assert ORG not in text
    assert PERSON not in text


def test_user_mock_for_term_custom_is_none():
    assert _user_mock_for_term(CustomRedactTerm(search_value="X")) is None
    assert _user_mock_for_term(CustomRedactTerm(search_value="X", mock_label="CUSTOM")) is None
    assert _user_mock_for_term(CustomRedactTerm(search_value="X", mock_label="XXX")) == "XXX"


def test_audit_store_refuses_source_text(tmp_path, monkeypatch):
    monkeypatch.setenv("SHARD_BASE_PATH", str(tmp_path))
    _clear_caches()

    class DummyAudit:
        request_id = "req_bad"

        def model_dump(self, mode="json"):
            return {"request_id": "req_bad", "pages": [{"source_text": ORG}]}

    store = AuditStore()
    with pytest.raises(ValueError, match="source_text"):
        store.save(DummyAudit())
    assert not (store.base / "req_bad.json").exists()


def test_audit_store_allows_mock_value(tmp_path, monkeypatch, patch_ocr_pii):
    pipeline, _, _ = _pipeline(tmp_path, monkeypatch)
    _, audit, _ = _run(pipeline)
    path = tmp_path / "audit" / "requests" / f"{audit.request_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert _contains_key(data, "source_text") is False
    assert any(r.get("mock_value") for r in data["redactions"])


def test_audit_log_drops_source_text_kwarg(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("SHARD_BASE_PATH", str(tmp_path))
    _clear_caches()
    with caplog.at_level(logging.INFO):
        audit_log("x", source_text=ORG, request_id="r1")
    assert ORG not in caplog.text
    jsonl = (tmp_path / "audit" / "audit.jsonl").read_text(encoding="utf-8")
    assert ORG not in jsonl


def test_mock_router_mounted_at_v1_mocks(wired_app):
    client, _ = wired_app
    response = client.get("/v1/mocks")
    assert response.status_code == 200
    body = response.json()
    assert "count" in body
    assert "entries" in body


def test_mock_router_not_double_prefixed(wired_app):
    client, _ = wired_app
    assert client.get("/v1/v1/mocks").status_code == 404


def test_redact_response_includes_ledger_header(wired_app):
    client, _ = wired_app
    response = _post_redact(client)
    assert response.status_code == 200
    assert response.headers["X-Ledger-Id"] == response.headers["X-Request-Id"]


def test_cors_middleware_still_present():
    application = create_app()
    assert any(m.cls is CORSMiddleware for m in application.user_middleware)


def test_auth_enabled_post_mocks_401():
    application = create_app()
    application.dependency_overrides[get_auth_settings] = lambda: AuthSettings(True, "secret")
    store = MagicMock()
    application.dependency_overrides[get_mock_store] = lambda: store
    client = TestClient(application)
    response = client.post("/v1/mocks", json={"source_text": ORG, "mock_value": "XXX"})
    assert response.status_code == 401
    store.upsert.assert_not_called()


def test_portal_override_helper_calls_store(tmp_path):
    store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    entry = store.resolve(ORG, "ORGANIZATION")
    updated = persist_mapping_override(store, {"mapping_id": entry.mapping_id, "mock_value": "XXX"})
    assert updated.mock_value == "XXX"


def test_portal_module_imports_mapping_panel():
    from app.ui.redact_portal import _build_mappings_tab

    assert "build_mapping_panel" in inspect.getsource(_build_mappings_tab)


def test_portal_does_not_log_source_text(tmp_path, caplog):
    store = MockDictionaryStore(snapshot_path=tmp_path / "mappings.json")
    entry = store.resolve(ORG, "ORGANIZATION")
    with caplog.at_level(logging.DEBUG):
        persist_mapping_override(store, {"mapping_id": entry.mapping_id, "mock_value": "XXX"})
    assert ORG not in caplog.text


def test_e2e_two_jobs_reuse_org_mock(wired_app):
    client, _ = wired_app
    first = _post_redact(client)
    second = _post_redact(client)
    assert first.status_code == 200
    assert second.status_code == 200
    audit1 = client.get(f"/v1/redact/audit/{first.headers['X-Request-Id']}").json()
    audit2 = client.get(f"/v1/redact/audit/{second.headers['X-Request-Id']}").json()
    org1 = next(r for r in audit1["redactions"] if r["entity_type"] == "ORGANIZATION")
    org2 = next(r for r in audit2["redactions"] if r["entity_type"] == "ORGANIZATION")
    assert org1["mock_value"] == org2["mock_value"]
    assert org1["mapping_id"] == org2["mapping_id"]


def test_e2e_put_override_xxx_then_later_job(wired_app):
    client, _ = wired_app
    first = _post_redact(client)
    audit1 = client.get(f"/v1/redact/audit/{first.headers['X-Request-Id']}").json()
    org = next(r for r in audit1["redactions"] if r["entity_type"] == "ORGANIZATION")
    put = client.put(f"/v1/mocks/{org['mapping_id']}", json={"mock_value": "XXX"})
    assert put.status_code == 200
    later = _post_redact(client)
    audit = client.get(f"/v1/redact/audit/{later.headers['X-Request-Id']}").json()
    org2 = next(r for r in audit["redactions"] if r["entity_type"] == "ORGANIZATION")
    assert org2["mock_value"] == "XXX"


def test_e2e_person_auto_person_nn(wired_app):
    client, _ = wired_app
    response = _post_redact(client)
    audit = client.get(f"/v1/redact/audit/{response.headers['X-Request-Id']}").json()
    person = next(r for r in audit["redactions"] if r["entity_type"] == "PERSON")
    assert PERSON_RE.fullmatch(person["mock_value"])


def test_e2e_default_logo_footer_zones(wired_app):
    client, _ = wired_app
    response = _post_redact(client)
    audit = client.get(f"/v1/redact/audit/{response.headers['X-Request-Id']}").json()
    mocks = {r["mock_value"] for r in audit["redactions"] if r["assignment_source"] == "brand"}
    assert "LOGO" in mocks
    assert "FOOTER" in mocks


def test_e2e_get_audit_has_no_source_text(wired_app):
    client, _ = wired_app
    response = _post_redact(client)
    audit = client.get(f"/v1/redact/audit/{response.headers['X-Request-Id']}")
    assert audit.status_code == 200
    body = audit.json()
    assert _contains_key(body, "source_text") is False
    assert any(r.get("mock_value") for r in body["redactions"])


def test_e2e_get_ledger_and_mocks_have_mappings(wired_app):
    client, _ = wired_app
    response = _post_redact(client)
    request_id = response.headers["X-Request-Id"]
    ledger = client.get(f"/v1/redact/ledger/{request_id}")
    mocks = client.get("/v1/mocks")
    assert ledger.status_code == 200
    assert mocks.status_code == 200
    entry = next(e for e in ledger.json()["entries"] if e["source_text"] == ORG)
    assert entry["mock_value"]
    listed = next(e for e in mocks.json()["entries"] if e["source_text"] == ORG)
    assert listed["mock_value"] == entry["mock_value"]


def test_e2e_auth_on_post_mocks_401(tmp_path, monkeypatch):
    monkeypatch.setenv("SHARD_BASE_PATH", str(tmp_path))
    monkeypatch.setenv("MOCK_DICTIONARY_PATH", str(tmp_path / "mappings.json"))
    _clear_caches()
    application = create_app()
    application.dependency_overrides[get_auth_settings] = lambda: AuthSettings(True, "secret")
    store = MagicMock()
    application.dependency_overrides[get_mock_store] = lambda: store
    client = TestClient(application)
    response = client.post("/v1/mocks", json={"source_text": ORG, "mock_value": "XXX"})
    assert response.status_code == 401
    store.upsert.assert_not_called()


def test_e2e_logs_omit_original_pii(wired_app, caplog):
    client, _ = wired_app
    with caplog.at_level(logging.DEBUG):
        assert _post_redact(client).status_code == 200
    assert ORG not in caplog.text
    assert PERSON not in caplog.text


def test_e2e_ledger_and_audit_paths_isolated(wired_app):
    client, tmp_path = wired_app
    response = _post_redact(client)
    request_id = response.headers["X-Request-Id"]
    ledger_path = tmp_path / "shards" / request_id / "ledger.json"
    audit_path = tmp_path / "audit" / "requests" / f"{request_id}.json"
    assert ledger_path.is_file()
    assert audit_path.is_file()
    assert "shards" in str(ledger_path)
    assert ledger_path.name == "ledger.json"
    assert "audit" in str(audit_path)
    assert _contains_key(json.loads(audit_path.read_text(encoding="utf-8")), "source_text") is False


def test_e2e_export_mocks_returns_csv_attachment(wired_app):
    client, _ = wired_app
    _post_redact(client)
    response = client.get("/v1/mocks/export")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    assert "mock-mappings.csv" in response.headers["content-disposition"]
    lines = response.text.splitlines()
    assert lines[0].split(",")[:2] == ["source_text", "mock_value"]
    assert any(ORG in line for line in lines[1:])


def test_e2e_download_template_returns_csv_with_examples(wired_app):
    client, _ = wired_app
    response = client.get("/v1/mocks/template")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "mock-mappings-template.csv" in response.headers["content-disposition"]
    lines = response.text.splitlines()
    assert lines[0] == "source_text,mock_value,entity_type,field_role,account_number"
    assert len(lines) >= 3


def test_e2e_import_mocks_inserts_new_rows(wired_app):
    client, _ = wired_app
    csv_text = (
        "source_text,mock_value,entity_type,field_role,account_number\n"
        "Bank Mandiri,BANK_A,ORGANIZATION,bank_name,\n"
    )
    response = client.post(
        "/v1/mocks/import",
        files={"file": ("upload.csv", csv_text.encode("utf-8"), "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body == {"inserted": 1, "skipped_existing": 0, "skipped_invalid": 0}
    mocks = client.get("/v1/mocks").json()
    assert any(e["source_text"] == "Bank Mandiri" for e in mocks["entries"])


def test_e2e_import_mocks_never_overwrites_existing(wired_app):
    client, _ = wired_app
    _post_redact(client)
    mocks = client.get("/v1/mocks").json()
    existing = next(e for e in mocks["entries"] if e["source_text"] == ORG)
    csv_text = (
        "source_text,mock_value,entity_type,field_role,account_number\n"
        f"{ORG},SEED_OVERRIDE,ORGANIZATION,,\n"
    )
    response = client.post(
        "/v1/mocks/import",
        files={"file": ("upload.csv", csv_text.encode("utf-8"), "text/csv")},
    )
    assert response.status_code == 200
    assert response.json()["skipped_existing"] == 1
    mocks_after = client.get("/v1/mocks").json()
    unchanged = next(e for e in mocks_after["entries"] if e["source_text"] == ORG)
    assert unchanged["mock_value"] == existing["mock_value"]


def test_e2e_reimporting_own_export_inserts_nothing_new(wired_app):
    """A downloaded file re-uploaded to the same store is a safe no-op."""
    client, _ = wired_app
    _post_redact(client)
    exported = client.get("/v1/mocks/export").text
    response = client.post(
        "/v1/mocks/import",
        files={"file": ("mock-mappings.csv", exported.encode("utf-8"), "text/csv")},
    )
    assert response.status_code == 200
    assert response.json()["inserted"] == 0


def test_e2e_import_mocks_auth_gated_401():
    application = create_app()
    application.dependency_overrides[get_auth_settings] = lambda: AuthSettings(True, "secret")
    store = MagicMock()
    application.dependency_overrides[get_mock_store] = lambda: store
    client = TestClient(application)
    response = client.post(
        "/v1/mocks/import",
        files={"file": ("upload.csv", b"source_text,mock_value\nA,B\n", "text/csv")},
    )
    assert response.status_code == 401
    store.upsert.assert_not_called()


def test_e2e_redact_disabled_returns_503(tmp_path, monkeypatch, patch_ocr_pii):
    monkeypatch.setenv("SHARD_BASE_PATH", str(tmp_path))
    monkeypatch.setenv("MOCK_DICTIONARY_PATH", str(tmp_path / "mappings.json"))
    monkeypatch.setenv("REDACT_PIPELINE_ENABLED", "false")
    _clear_caches()
    with TestClient(create_app()) as client:
        response = _post_redact(client)
    _clear_caches()
    assert response.status_code == 503
    assert response.json()["detail"]["error"]["code"] == "PIPELINE_DISABLED"
