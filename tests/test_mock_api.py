"""HTTP and unit tests for the mock dictionary API (Story 8)."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.api.mock_routes import (
    AuthSettings,
    MappingNotFoundError,
    _as_dict,
    _is_mapping_not_found,
    get_auth_settings,
    get_ledger_store,
    get_mock_store,
    require_api_key,
    router,
)

PII = "Standard Chartered Custody"


def _entry(**overrides):
    base = {
        "mapping_id": "map_00a",
        "source_text": PII,
        "normalized": "standard chartered custody",
        "mock_value": "XXX",
        "entity_type": "ORGANIZATION",
        "assignment_source": "user",
        "hit_count": 1,
        "created_at": "2026-08-19T01:00:00+00:00",
        "updated_at": "2026-08-19T01:00:00+00:00",
    }
    base.update(overrides)
    return base


def make_client(auth_enabled=False, api_key="secret", store=None, ledger=None):
    if store is None:
        store = MagicMock()
        store.list.return_value = []
        store.upsert.return_value = _entry()
        store.override.return_value = _entry()
        store.delete.return_value = None
    if ledger is None:
        ledger = MagicMock()
        ledger.get.return_value = None
    application = FastAPI()
    application.include_router(router)
    application.dependency_overrides[get_auth_settings] = lambda: AuthSettings(
        admin_auth_enabled=auth_enabled, api_key=api_key
    )
    application.dependency_overrides[get_mock_store] = lambda: store
    application.dependency_overrides[get_ledger_store] = lambda: ledger
    return TestClient(application), store, ledger


def _error_code(response) -> str:
    return response.json()["detail"]["error"]["code"]


def _assert_logs_omit_pii(caplog) -> None:
    for record in caplog.records:
        assert PII not in record.getMessage()
        assert PII not in str(record.msg)
        assert PII not in str(record.args)


def _upsert_call(store) -> tuple[str | None, str | None, str | None]:
    args, kwargs = store.upsert.call_args
    source = kwargs.get("source_text", args[0] if len(args) > 0 else None)
    mock_value = kwargs.get("mock_value", args[1] if len(args) > 1 else None)
    entity_type = kwargs.get(
        "entity_type", args[2] if len(args) > 2 else None
    )
    return source, mock_value, entity_type


# --- Task 1 helper unit tests -------------------------------------------------


def test_as_dict_from_mapping():
    entry = SimpleNamespace(
        mapping_id="map_00a",
        source_text=PII,
        normalized="standard chartered custody",
        mock_value="XXX",
        entity_type="ORGANIZATION",
        assignment_source="user",
        hit_count=1,
        created_at="2026-08-19T01:00:00+00:00",
        updated_at="2026-08-19T01:00:00+00:00",
    )
    result = _as_dict(entry)
    assert result["mapping_id"] == "map_00a"
    assert result["source_text"] == PII
    assert result["normalized"] == "standard chartered custody"
    assert result["mock_value"] == "XXX"
    assert result["entity_type"] == "ORGANIZATION"
    assert result["assignment_source"] == "user"
    assert result["hit_count"] == 1


def test_is_mapping_not_found_by_class():
    assert _is_mapping_not_found(MappingNotFoundError()) is True


def test_is_mapping_not_found_by_code():
    exc = Exception("missing")
    exc.code = "MAPPING_NOT_FOUND"
    assert _is_mapping_not_found(exc) is True


def test_is_mapping_not_found_by_story5_name():
    class MockMappingNotFound(Exception):
        pass

    assert _is_mapping_not_found(MockMappingNotFound()) is True


def test_require_api_key_rejects_when_enabled():
    auth = AuthSettings(admin_auth_enabled=True, api_key="secret")
    for header in (None, "nope"):
        with pytest.raises(HTTPException) as exc_info:
            require_api_key(x_api_key=header, auth=auth)
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail["error"]["code"] == "UNAUTHORIZED"


def test_require_api_key_allows_when_disabled():
    auth = AuthSettings(admin_auth_enabled=False, api_key="secret")
    assert require_api_key(x_api_key=None, auth=auth) is None


# --- Task 2 / 3 / 4 HTTP tests -----------------------------------------------


def test_list_mocks_200_when_auth_disabled():
    store = MagicMock()
    store.list.return_value = [_entry()]
    client, _, _ = make_client(store=store)

    response = client.get("/v1/mocks")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["entries"][0]["source_text"] == PII
    assert body["entries"][0]["mock_value"] == "XXX"


def test_list_mocks_filters_entity_type():
    store = MagicMock()
    store.list.return_value = [
        _entry(mapping_id="map_org", entity_type="ORGANIZATION"),
        _entry(mapping_id="map_per", entity_type="PERSON", mock_value="NAME_A"),
    ]
    client, _, _ = make_client(store=store)

    response = client.get("/v1/mocks", params={"entity_type": "PERSON"})

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["entries"][0]["entity_type"] == "PERSON"
    assert body["entries"][0]["mapping_id"] == "map_per"


def test_post_mocks_201_new():
    client, store, _ = make_client()
    store.list.return_value = []
    store.upsert.return_value = _entry()

    response = client.post(
        "/v1/mocks",
        json={
            "source_text": PII,
            "mock_value": "XXX",
            "entity_type": "ORGANIZATION",
        },
    )

    assert response.status_code == 201
    assert response.json()["mapping_id"] == "map_00a"
    store.upsert.assert_called_once()
    source, mock_value, entity_type = _upsert_call(store)
    assert source == PII
    assert mock_value == "XXX"
    assert entity_type == "ORGANIZATION"


def test_post_mocks_200_existing():
    client, store, _ = make_client()
    store.list.return_value = [_entry()]
    store.upsert.return_value = _entry()

    response = client.post(
        "/v1/mocks",
        json={
            "source_text": PII,
            "mock_value": "XXX",
            "entity_type": "ORGANIZATION",
        },
    )

    assert response.status_code == 200
    store.upsert.assert_called_once()


def test_post_mocks_default_entity_type_custom():
    client, store, _ = make_client()
    store.list.return_value = []

    response = client.post(
        "/v1/mocks",
        json={"source_text": PII, "mock_value": "XXX"},
    )

    assert response.status_code == 201
    store.upsert.assert_called_once()
    _, _, entity_type = _upsert_call(store)
    assert entity_type == "CUSTOM"


def test_put_override_200():
    client, store, _ = make_client()
    store.override.return_value = _entry(mock_value="BANK_A")

    response = client.put("/v1/mocks/map_00a", json={"mock_value": "BANK_A"})

    assert response.status_code == 200
    assert response.json()["mock_value"] == "BANK_A"
    store.override.assert_called_once_with(
        "map_00a", "BANK_A", field_role=None, account_number=None
    )


def test_put_unknown_mapping_404():
    client, store, _ = make_client()
    store.override.side_effect = MappingNotFoundError()

    response = client.put("/v1/mocks/map_00a", json={"mock_value": "BANK_A"})

    assert response.status_code == 404
    assert _error_code(response) == "MAPPING_NOT_FOUND"
    assert PII not in response.text


def test_delete_204():
    client, store, _ = make_client()
    store.delete.return_value = None

    response = client.delete("/v1/mocks/map_00a")

    assert response.status_code == 204
    assert response.content == b""
    store.delete.assert_called_once_with("map_00a")


def test_delete_unknown_mapping_404():
    client, store, _ = make_client()
    store.delete.side_effect = MappingNotFoundError()

    response = client.delete("/v1/mocks/map_00a")

    assert response.status_code == 404
    assert _error_code(response) == "MAPPING_NOT_FOUND"


def test_get_ledger_200_includes_source_text():
    client, _, ledger = make_client()
    ledger.get.return_value = {
        "request_id": "req-1",
        "created_at": "2026-08-19T01:00:00+00:00",
        "entries": [{"source_text": PII, "mock_value": "XXX"}],
        "brand_zones": [],
    }

    response = client.get("/v1/redact/ledger/req-1")

    assert response.status_code == 200
    assert response.json()["entries"][0]["source_text"] == PII


def test_get_ledger_unknown_404():
    client, _, ledger = make_client()
    ledger.get.return_value = None

    response = client.get("/v1/redact/ledger/missing")

    assert response.status_code == 404
    assert _error_code(response) == "LEDGER_NOT_FOUND"


@pytest.mark.parametrize(
    "method,path,json_body",
    [
        ("GET", "/v1/mocks", None),
        ("POST", "/v1/mocks", {"source_text": PII, "mock_value": "XXX"}),
        ("PUT", "/v1/mocks/map_00a", {"mock_value": "BANK_A"}),
        ("DELETE", "/v1/mocks/map_00a", None),
        ("GET", "/v1/redact/ledger/req-1", None),
    ],
)
@pytest.mark.parametrize("header", [None, "wrong"])
def test_auth_401_no_mutation(method, path, json_body, header):
    client, store, ledger = make_client(auth_enabled=True, api_key="secret")
    headers = {} if header is None else {"X-API-Key": header}

    response = client.request(method, path, json=json_body, headers=headers)

    assert response.status_code == 401
    assert _error_code(response) == "UNAUTHORIZED"
    store.list.assert_not_called()
    store.upsert.assert_not_called()
    store.override.assert_not_called()
    store.delete.assert_not_called()
    ledger.get.assert_not_called()


def test_auth_200_with_valid_key():
    client, store, _ = make_client(auth_enabled=True, api_key="secret")
    store.list.return_value = []

    response = client.get("/v1/mocks", headers={"X-API-Key": "secret"})

    assert response.status_code == 200


@pytest.mark.parametrize("source_text", ["", "   "])
def test_post_empty_source_422_no_upsert(source_text):
    client, store, _ = make_client()

    response = client.post(
        "/v1/mocks",
        json={"source_text": source_text, "mock_value": "XXX"},
    )

    assert response.status_code == 422
    assert _error_code(response) == "VALIDATION_ERROR"
    store.upsert.assert_not_called()


def test_post_missing_mock_value_422_no_upsert():
    client, store, _ = make_client()

    response = client.post("/v1/mocks", json={"source_text": PII})

    assert response.status_code == 422
    assert _error_code(response) == "VALIDATION_ERROR"
    store.upsert.assert_not_called()


@pytest.mark.parametrize("body", [{}, {"mock_value": ""}])
def test_put_missing_mock_value_422_no_override(body):
    client, store, _ = make_client()

    response = client.put("/v1/mocks/map_00a", json=body)

    assert response.status_code == 422
    assert _error_code(response) == "VALIDATION_ERROR"
    store.override.assert_not_called()


def test_error_bodies_omit_source_text_pii():
    client, store, ledger = make_client()
    store.override.side_effect = MappingNotFoundError()
    ledger.get.return_value = None

    cases = [
        client.post("/v1/mocks", json={"source_text": PII}),
        client.put("/v1/mocks/map_missing", json={"mock_value": "BANK_A"}),
        client.get("/v1/redact/ledger/missing"),
    ]
    auth_client, _, _ = make_client(auth_enabled=True, api_key="secret")
    cases.append(
        auth_client.post(
            "/v1/mocks",
            json={"source_text": PII, "mock_value": "XXX"},
        )
    )

    for response in cases:
        assert response.status_code in {401, 404, 422}
        assert PII not in response.text
        error = response.json()["detail"]["error"]
        assert "source_text" not in error
        assert PII not in str(error)


def test_logs_omit_source_text_on_post(caplog):
    caplog.set_level(logging.INFO, logger="app.api.mock_routes")
    client, store, _ = make_client()
    store.list.return_value = []
    store.upsert.return_value = _entry()

    response = client.post(
        "/v1/mocks",
        json={"source_text": PII, "mock_value": "XXX"},
    )

    assert response.status_code == 201
    _assert_logs_omit_pii(caplog)


def test_logs_omit_source_text_on_error(caplog):
    caplog.set_level(logging.INFO, logger="app.api.mock_routes")
    client, store, _ = make_client()
    store.override.side_effect = MappingNotFoundError()

    post = client.post("/v1/mocks", json={"source_text": PII})
    put = client.put("/v1/mocks/map_00a", json={"mock_value": "BANK_A"})

    assert post.status_code == 422
    assert put.status_code == 404
    _assert_logs_omit_pii(caplog)


def test_unexpected_store_error_500_generic(caplog):
    caplog.set_level(logging.INFO, logger="app.api.mock_routes")
    client, store, _ = make_client()
    store.list.return_value = []
    store.upsert.side_effect = RuntimeError(PII)

    response = client.post(
        "/v1/mocks",
        json={"source_text": PII, "mock_value": "XXX"},
    )

    assert response.status_code == 500
    assert _error_code(response) == "INTERNAL_ERROR"
    assert response.json()["detail"]["error"]["message"] == "Unexpected server error"
    assert PII not in response.text
    _assert_logs_omit_pii(caplog)
