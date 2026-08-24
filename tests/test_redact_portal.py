"""Unit tests for redact portal preview helpers."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app
from app.models.redact import RedactOptions
from app.services.redact.session_store import RedactSession, session_store
from app.ui.redact_portal import _client_state, _mocked_filename, _pdf_viewer_html, setup_ui


def test_pdf_viewer_uses_served_url_not_data_uri() -> None:
    _client_state["preview-test"] = {
        "file_bytes": b"%PDF-1.4",
        "redacted_bytes": None,
        "filename": "doc.pdf",
        "session_id": None,
    }
    html = _pdf_viewer_html("original", "preview-test", "Original")
    assert "/portal/preview/original?client_id=preview-test" in html
    assert "data:application/pdf" not in html
    assert "height:100%" in html


def test_portal_preview_route_serves_original_pdf() -> None:
    app = create_app()
    setup_ui(app)
    _client_state["route-test"] = {
        "file_bytes": b"%PDF-1.4 sample",
        "redacted_bytes": None,
        "filename": "pii-test.pdf",
        "session_id": None,
    }
    client = TestClient(app)
    response = client.get("/portal/preview/original?client_id=route-test")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content == b"%PDF-1.4 sample"


def test_portal_preview_route_serves_redacted_pdf() -> None:
    app = create_app()
    setup_ui(app)
    _client_state["route-test-2"] = {
        "file_bytes": b"%PDF-1.4 orig",
        "redacted_bytes": b"%PDF-1.4 redacted",
        "filename": "pii-test.pdf",
        "session_id": "sess",
    }
    client = TestClient(app)
    response = client.get("/portal/preview/redacted?client_id=route-test-2")
    assert response.status_code == 200
    assert response.content == b"%PDF-1.4 redacted"


def test_portal_preview_route_serves_redacted_pdf_via_session_after_client_gone() -> None:
    """The long-running pipeline run can outlive the NiceGUI websocket (e.g. a
    reconnect assigns a brand new client_id mid-run) — the redacted preview
    must still resolve from session_store, not the now-gone client_state.
    """
    app = create_app()
    setup_ui(app)
    session_store.put(
        RedactSession(
            session_id="sess-durable",
            file_bytes=b"%PDF-1.4 orig",
            filename="pii-test.pdf",
            options=RedactOptions(),
            last_pdf=b"%PDF-1.4 redacted via session",
        )
    )
    _client_state.pop("route-test-gone", None)
    client = TestClient(app)
    response = client.get("/portal/preview/redacted?session_id=sess-durable")
    assert response.status_code == 200
    assert response.content == b"%PDF-1.4 redacted via session"


def test_portal_preview_route_404_for_unknown_session() -> None:
    app = create_app()
    setup_ui(app)
    client = TestClient(app)
    response = client.get("/portal/preview/redacted?session_id=does-not-exist")
    assert response.status_code == 404


def test_mocked_filename_uses_original_stem() -> None:
    assert _mocked_filename("pii-test.pdf") == "mocked_pii-test.pdf"
    assert _mocked_filename("folder/bill.png") == "mocked_bill.pdf"
    assert _mocked_filename(None) == "mocked_document.pdf"


def test_portal_download_mocked_as_attachment_via_session() -> None:
    app = create_app()
    setup_ui(app)
    session_store.put(
        RedactSession(
            session_id="sess-dl",
            file_bytes=b"%PDF-1.4 orig",
            filename="pii-test.pdf",
            options=RedactOptions(),
            last_pdf=b"%PDF-1.4 mocked",
        )
    )
    client = TestClient(app)
    response = client.get("/portal/download/mocked?session_id=sess-dl")
    assert response.status_code == 200
    assert response.content == b"%PDF-1.4 mocked"
    assert response.headers["content-type"] == "application/pdf"
    disposition = response.headers["content-disposition"]
    assert "attachment" in disposition
    assert "mocked_pii-test.pdf" in disposition


def test_portal_download_mocked_via_client_state() -> None:
    app = create_app()
    setup_ui(app)
    _client_state["dl-client"] = {
        "file_bytes": b"%PDF-1.4 orig",
        "redacted_bytes": b"%PDF-1.4 mocked-client",
        "filename": "scan.pdf",
        "session_id": None,
    }
    client = TestClient(app)
    response = client.get("/portal/download/mocked?client_id=dl-client")
    assert response.status_code == 200
    assert response.content == b"%PDF-1.4 mocked-client"
    assert "mocked_scan.pdf" in response.headers["content-disposition"]


def test_portal_download_mocked_404_when_missing() -> None:
    app = create_app()
    setup_ui(app)
    client = TestClient(app)
    response = client.get("/portal/download/mocked?session_id=nope")
    assert response.status_code == 404
