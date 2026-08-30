"""NiceGUI portal for PDF Text Mocker — pure Python UI, no npm build.

Two tabs keep each task focused: "Mock document" (upload, replace private
text, preview) and "Mock dictionary" (review/edit/download/upload the
source → mock list). No login, no user identity, no auth screen — this is
a single shared workspace for whoever has the portal open.
"""

from __future__ import annotations

import time
from typing import Any

from app.api.mock_routes import get_mock_store
from app.models.redact import RedactOptions
from app.pipeline.errors import PipelineStageError
from app.pipeline.redact import RedactPipeline
from app.services.ocr.engines import easyocr_available, rapidocr_available, tesseract_available
from app.services.pii.custom_redact import parse_custom_terms
from app.services.redact.session_store import session_store
from app.ui.mapping_table import (
    build_mapping_panel,
    build_mapping_toolbar,
    parse_create,
    parse_override,
)

# Shared UI state per browser tab (NiceGUI client id)
_client_state: dict[str, dict[str, Any]] = {}


def _state() -> dict[str, Any]:
    from nicegui import ui

    client_id = ui.context.client.id
    if client_id not in _client_state:
        _client_state[client_id] = {
            "file_bytes": None,
            "redacted_bytes": None,
            "filename": None,
            "session_id": None,
        }
    return _client_state[client_id]


def _preview_media_type(filename: str | None, kind: str) -> str:
    if kind == "redacted":
        return "application/pdf"
    name = (filename or "").lower()
    if name.endswith(".png"):
        return "image/png"
    if name.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if name.endswith((".tif", ".tiff")):
        return "image/tiff"
    return "application/pdf"


def _mocked_filename(original: str | None) -> str:
    """Safe download name for the mocked PDF, e.g. bill.pdf → mocked_bill.pdf."""
    raw = (original or "document.pdf").replace("\\", "/").rsplit("/", 1)[-1]
    stem = raw.rsplit(".", 1)[0] if raw else "document"
    stem = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in stem).strip() or "document"
    return f"mocked_{stem}.pdf"


def _pdf_viewer_html(
    kind: str, ident: str, title: str = "PDF", *, id_param: str = "client_id", cache_bust: bool = False
) -> str:
    """Embed a PDF via a served URL (data URLs break for multi-MB files)."""
    query = f"{id_param}={ident}"
    if cache_bust:
        query = f"{query}&v={int(time.time())}"
    src = f"/portal/preview/{kind}?{query}"
    return (
        f'<div style="width:100%;height:100%;min-height:100%;border-radius:8px;overflow:hidden;">'
        f'<iframe title="{title}" src="{src}" '
        'width="100%" height="100%" '
        'style="border:none;background:#07111f;">'
        "</iframe></div>"
    )


def persist_mapping_override(store: Any, payload: dict[str, Any]) -> Any:
    """Apply a mapping-table edit (mock value only)."""
    cleaned = parse_override(payload["mapping_id"], payload["mock_value"])
    return store.override(cleaned["mapping_id"], cleaned["mock_value"])


def persist_mapping_create(store: Any, payload: dict[str, Any]) -> Any:
    """Add a brand-new mapping from the mapping panel's "Add mapping" dialog."""
    cleaned = parse_create(payload["source_text"], payload["mock_value"])
    return store.upsert(cleaned["source_text"], cleaned["mock_value"])


def _ocr_engine_options() -> dict[str, str]:
    """Value → label options for the OCR engine picker, from live availability.

    "Auto" uses the deterministic default policy (one primary engine with a
    documented fallback order, see ``Settings.ocr_primary_engine``); picking
    a specific engine forces that one alone (e.g. RapidOCR for a table-heavy
    statement where it reads better than EasyOCR/Tesseract).
    """
    options = {"auto": "Auto (recommended)"}
    if tesseract_available():
        options["tesseract"] = "Tesseract"
    if easyocr_available():
        options["easyocr"] = "EasyOCR"
    if rapidocr_available():
        options["rapidocr"] = "RapidOCR"
    return options


def _empty_placeholder(label: str) -> str:
    return (
        f'<div style="display:flex;align-items:center;justify-content:center;text-align:center;'
        f'height:100%;min-height:180px;background:#07111f;border-radius:8px;color:#94a3b8;'
        f'padding:16px;font-family:system-ui,sans-serif;font-size:1rem;">{label}</div>'
    )


_PORTAL_CSS = """
<style>
  html, body, .q-layout, .q-page, .q-page-container, .nicegui-content {
    background: #07111f !important;
    color: #f8fafc;
    width: 100%;
  }
  .nicegui-content { padding: 0 !important; height: 100%; }
  .q-tab-panel { padding: 0 !important; height: 100%; width: 100% !important; }
  .q-tab-panels { height: 100%; width: 100% !important; }
  .q-tab-panel > * { width: 100% !important; }
  .pdf-preview-row {
    width: 100% !important;
    display: flex !important;
    flex-wrap: nowrap !important;
  }
  .pdf-preview-row > * {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    max-width: none !important;
  }
  .q-header { background: #0a1a2e !important; border-bottom: 1px solid #2a4366; }
  .q-tab { font-weight: 600; text-transform: none; letter-spacing: 0; }
  .action-bar {
    background: #0e1c30;
    border-bottom: 1px solid #2a4366;
  }
  .action-card {
    background: #15263f;
    border: 1px solid #2a4366;
    border-radius: 12px;
  }
  .upload-dropzone .q-uploader {
    width: 100%;
    background: rgba(45, 212, 191, 0.08);
    border: 2px dashed #2dd4bf !important;
    border-radius: 12px;
    min-height: 112px;
    box-shadow: none;
  }
  .upload-dropzone .q-uploader__header {
    background: transparent !important;
    min-height: 112px;
    padding: 14px 16px;
  }
  .upload-dropzone .q-uploader__title {
    color: #f8fafc !important;
    font-size: 1rem;
    font-weight: 700;
  }
  .upload-dropzone .q-uploader__subtitle { color: #94a3b8 !important; }
  .upload-dropzone .q-uploader__header .q-btn {
    background: #0f766e !important;
    color: #fff !important;
    width: auto !important;
    min-width: 120px;
    height: 44px !important;
    padding: 0 16px;
    border-radius: 8px !important;
    box-shadow: 0 4px 14px rgba(15, 118, 110, 0.5);
  }
  .upload-dropzone .q-uploader__header .q-btn .q-icon { display: none; }
  .upload-dropzone .q-uploader__header .q-btn::after {
    content: "Choose file";
    font-size: 13px;
    font-weight: 700;
    white-space: nowrap;
  }
  .upload-dropzone.compact .q-uploader,
  .upload-dropzone.compact .q-uploader__header { min-height: 64px; }
  .upload-dropzone.compact .q-uploader__header .q-btn::after { content: "Choose CSV"; }
  .upload-dropzone .q-uploader__list { display: none; }
  .primary-cta .q-btn {
    font-size: 1.05rem;
    font-weight: 700;
    min-height: 48px;
    padding: 8px 18px;
  }
</style>
"""


def _live_client(element: Any) -> Any | None:
    """Return the NiceGUI client for an element, or None if it was torn down."""
    try:
        client = element.client
    except RuntimeError:
        return None
    if client.is_deleted:
        return None
    return client


def setup_ui(fastapi_app) -> None:
    from fastapi import HTTPException
    from fastapi.responses import Response
    from nicegui import ui

    @fastapi_app.get("/portal/preview/{kind}")
    async def portal_preview(
        kind: str, client_id: str | None = None, session_id: str | None = None
    ) -> Response:
        """Serve the original upload or redacted PDF for the preview iframes.

        ``redacted`` is served from ``session_store`` keyed by ``session_id``
        rather than the per-websocket ``_client_state``: the pipeline run can
        take minutes (OCR ensemble), long enough for the browser's NiceGUI
        websocket to drop and silently reconnect under a brand new
        ``client_id`` — at which point the old ``client_id`` gets popped by
        ``on_disconnect`` and this route would 404 even though the redacted
        PDF was produced successfully. ``session_store`` entries are keyed by
        the pipeline's own ``session_id`` and don't depend on any websocket
        being alive, so they survive that reconnect. ``client_id`` stays the
        lookup for ``original`` (no session exists yet before the first run)
        and as a fallback if a caller has no ``session_id`` yet.
        """
        if kind not in ("original", "redacted"):
            raise HTTPException(status_code=404)
        if kind == "redacted" and session_id:
            session = session_store.get(session_id)
            if session is None or not session.last_pdf:
                raise HTTPException(status_code=404)
            data: bytes | None = session.last_pdf
            media_type = "application/pdf"
        else:
            st = _client_state.get(client_id) if client_id else None
            if not st:
                raise HTTPException(status_code=404)
            data = st["file_bytes"] if kind == "original" else st.get("redacted_bytes")
            if not data:
                raise HTTPException(status_code=404)
            media_type = _preview_media_type(st.get("filename"), kind)
        return Response(
            content=data,
            media_type=media_type,
            headers={
                "Content-Disposition": "inline",
                "Cache-Control": "no-store",
            },
        )

    @fastapi_app.get("/portal/download/mocked")
    async def portal_download_mocked(
        session_id: str | None = None, client_id: str | None = None
    ) -> Response:
        """Download the mocked PDF as an attachment (not inline preview)."""
        data: bytes | None = None
        filename = "mocked_document.pdf"
        if session_id:
            session = session_store.get(session_id)
            if session and session.last_pdf:
                data = session.last_pdf
                filename = _mocked_filename(session.filename)
        if data is None and client_id:
            st = _client_state.get(client_id)
            if st and st.get("redacted_bytes"):
                data = st["redacted_bytes"]
                filename = _mocked_filename(st.get("filename"))
        if not data:
            raise HTTPException(status_code=404)
        return Response(
            content=data,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Cache-Control": "no-store",
            },
        )

    @ui.page("/")
    def redact_portal() -> None:
        portal_client = ui.context.client
        portal_client.on_disconnect(lambda: _client_state.pop(portal_client.id, None))

        ui.add_head_html(_PORTAL_CSS)
        ui.colors(primary="#0f766e", secondary="#1e3a5f", accent="#f59e0b", positive="#10b981")

        with ui.header().classes("items-center px-6 py-3 text-white gap-6"):
            ui.icon("find_replace", size="md").classes("text-teal-300")
            ui.label("PDF Text Mocker").classes("text-xl font-bold tracking-tight")
            tabs = ui.tabs().props("indicator-color=teal-3 active-color=white").classes(
                "text-slate-300"
            )
            with tabs:
                redact_tab = ui.tab("redact", label="Mock document", icon="picture_as_pdf")
                mappings_tab = ui.tab("mappings", label="Mock dictionary", icon="table_view")

        with ui.tab_panels(tabs, value=redact_tab).classes("w-full").style(
            "height: calc(100vh - 64px)"
        ):
            with ui.tab_panel(redact_tab).classes("w-full h-full"):
                _build_redact_tab(ui)
            with ui.tab_panel(mappings_tab).classes("w-full h-full overflow-y-auto"):
                _build_mappings_tab(ui)

    ui.run_with(
        fastapi_app,
        mount_path="/",
        title="PDF Text Mocker",
        storage_secret="pii-redact-local-dev-secret-change-in-prod",
        reconnect_timeout=600,
    )


def _build_redact_tab(ui: Any) -> None:
    """Upload → extra names → replace private text → preview."""
    with ui.column().classes("w-full h-full no-wrap gap-0 overflow-hidden").style(
        "width:100%;height:100%"
    ):
        with ui.element("div").classes("action-bar w-full shrink-0 px-6 py-4"):
            with ui.row().classes("w-full items-stretch gap-4 flex-wrap"):
                with ui.column().classes("action-card w-[340px] shrink-0 gap-2 p-4"):
                    ui.label("1 · Upload").classes(
                        "text-xs font-semibold text-teal-300 uppercase tracking-wide"
                    )
                    upload_info = ui.label("No file selected").classes(
                        "text-sm text-slate-300 break-all"
                    )

                    async def on_upload(e) -> None:
                        st = _state()
                        uploaded = e.file
                        st["file_bytes"] = await uploaded.read()
                        st["redacted_bytes"] = None
                        st["filename"] = uploaded.name
                        st["session_id"] = None
                        client_id = ui.context.client.id
                        upload_info.set_text(uploaded.name)
                        original_view.set_content(_pdf_viewer_html("original", client_id, "Original"))
                        redacted_view.set_content(
                            _empty_placeholder("Click Mock this PDF to see the result")
                        )
                        status_label.set_text("File loaded — click Mock this PDF")
                        audit_card.set_visibility(False)
                        download_btn.set_enabled(False)

                    with ui.element("div").classes("upload-dropzone w-full"):
                        ui.upload(
                            label="Choose PDF or image",
                            auto_upload=True,
                            on_upload=on_upload,
                            max_file_size=25 * 1024 * 1024,
                        ).props(
                            'accept=".pdf,.png,.jpg,.jpeg,.tiff,.tif" '
                            "hide-upload-btn bordered color=primary"
                        ).classes("w-full")
                    ui.label("PDF, PNG, JPG or TIFF · up to 25 MB").classes("text-xs text-slate-500")

                with ui.column().classes("action-card flex-grow min-w-[280px] gap-2 p-4"):
                    ui.label("2 · Extra text to replace (optional)").classes(
                        "text-xs font-semibold text-teal-300 uppercase tracking-wide"
                    )
                    ui.label(
                        "Add names or numbers the automatic scan might miss — one per line. "
                        "Use value=MOCK_LABEL to pick the replacement."
                    ).classes("text-xs text-slate-400 leading-relaxed")
                    custom_terms = ui.textarea(
                        placeholder="e.g.\nAcme Corp\nJohn Smith=MOCK_NAME\n555-0100",
                    ).classes("w-full").props("outlined dark rows=3")
                    with ui.row().classes("w-full gap-4"):
                        patch_footer = ui.checkbox("Cover footer", value=True)
                        patch_images = ui.checkbox("Cover images/graphics", value=True)
                    ocr_engine_select = ui.select(
                        _ocr_engine_options(),
                        label="OCR engine",
                        value="auto",
                    ).classes("w-full").props("outlined dark dense options-dark")
                    ui.label(
                        "Auto uses the deterministic default engine order. Pick one to "
                        "force it — e.g. RapidOCR for table-heavy statements EasyOCR reads "
                        "poorly."
                    ).classes("text-xs text-slate-500 leading-relaxed")

                with ui.column().classes("action-card w-[300px] shrink-0 gap-3 p-4"):
                    ui.label("3 · Replace private text").classes(
                        "text-xs font-semibold text-teal-300 uppercase tracking-wide"
                    )
                    with ui.row().classes("items-center gap-2"):
                        status_label = ui.label("Upload a document to begin").classes(
                            "text-sm text-amber-300"
                        )
                        loading = ui.spinner(size="md", color="primary")
                        loading.set_visibility(False)
                    with ui.element("div").classes("primary-cta w-full"):
                        ui.button(
                            "Mock this PDF",
                            icon="find_replace",
                            on_click=lambda: run_redact(regenerate=False),
                        ).classes("w-full").props("color=primary unelevated")
                    ui.button(
                        "Run again (faster)",
                        icon="refresh",
                        on_click=lambda: run_redact(regenerate=True),
                    ).classes("w-full").props("color=accent outline")
                    ui.label("Run again reuses the last scan — only replacements change.").classes(
                        "text-xs text-slate-500 leading-relaxed"
                    )

            audit_card = ui.card().classes("w-full mt-3 action-card p-3 gap-1")
            audit_card.set_visibility(False)
            with audit_card:
                ui.label("Last run").classes(
                    "text-xs font-semibold text-teal-300 uppercase tracking-wide"
                )
                audit_stats = ui.label("").classes("text-sm text-slate-300")
                audit_regions = ui.label("").classes("text-xs text-slate-400")
                audit_engines = ui.label("").classes("text-xs text-slate-500")

            async def run_redact(*, regenerate: bool = False) -> None:
                client = ui.context.client
                st = _state()
                if regenerate and not st.get("session_id"):
                    ui.notify("Mock the document first", type="warning")
                    return
                if not regenerate and not st.get("file_bytes"):
                    ui.notify("Upload a file first", type="warning")
                    return

                loading.set_visibility(True)
                status_label.set_text("Working… reading the page and finding private text")
                try:
                    pipeline = RedactPipeline()
                    terms_text = custom_terms.value or ""
                    selected_engine = ocr_engine_select.value
                    opts = RedactOptions(
                        custom_redactions=parse_custom_terms(terms_text),
                        patch_footer=bool(patch_footer.value),
                        patch_images=bool(patch_images.value),
                        ocr_engines=(
                            [selected_engine] if selected_engine and selected_engine != "auto" else None
                        ),
                    )
                    if regenerate:
                        pdf_bytes, audit, session = await pipeline.regenerate(
                            st["session_id"],
                            custom_terms_text=terms_text,
                            options=opts,
                        )
                    else:
                        pdf_bytes, audit, session = await pipeline.run(
                            st["file_bytes"],
                            st["filename"] or "upload.pdf",
                            opts,
                        )
                    st["session_id"] = session.session_id
                    st["redacted_bytes"] = pdf_bytes
                    if client.is_deleted:
                        return
                    with client:
                        redacted_view.set_content(
                            _pdf_viewer_html(
                                "redacted",
                                session.session_id,
                                "Mocked copy",
                                id_param="session_id",
                                cache_bust=True,
                            )
                        )
                        count = audit.summary.get("redaction_count", 0)
                        ms = audit.processing_ms
                        status_label.set_text(f"Done — {count} replacements in {ms}ms")
                        audit_card.set_visibility(True)
                        audit_stats.set_text(
                            f"Pages: {audit.page_count} · Avg confidence: "
                            f"{audit.summary.get('avg_confidence', 0):.0%} · "
                            f"Blur: {audit.summary.get('blur_tiers', {})}"
                        )
                        types: dict[str, int] = {}
                        for r in audit.redactions:
                            types[r.entity_type] = types.get(r.entity_type, 0) + 1
                        type_str = ", ".join(f"{k}: {v}" for k, v in sorted(types.items()))
                        audit_regions.set_text(type_str or "Nothing replaced")
                        engines_used = sorted({e for r in audit.redactions for e in r.engines_seen})
                        audit_engines.set_text(
                            f"OCR engines used: {', '.join(engines_used)}"
                            if engines_used
                            else "OCR engines used: none reported"
                        )
                        download_btn.set_enabled(True)
                        ui.notify("Mocked PDF ready — click Download PDF to save it", type="positive")
                except PipelineStageError as exc:
                    if not client.is_deleted:
                        with client:
                            status_label.set_text(f"Error: {exc}")
                            ui.notify(str(exc), type="negative")
                except Exception:
                    if not client.is_deleted:
                        with client:
                            status_label.set_text("Error: processing failed")
                            ui.notify("Processing failed", type="negative")
                finally:
                    if not client.is_deleted:
                        with client:
                            loading.set_visibility(False)

        with ui.row().classes("pdf-preview-row w-full flex-grow gap-4 px-6 py-4 min-w-0 min-h-0").style(
            "background:#07111f;height:0;width:100%"
        ):
            with ui.card().classes("p-3 h-full").style(
                "min-height:0;flex:1 1 0;min-width:0;background:#0e1c30;border:1px solid #2a4366"
            ):
                with ui.row().classes("items-center gap-2 mb-2"):
                    ui.icon("description", size="sm").classes("text-slate-400")
                    ui.label("Original").classes("text-base font-semibold text-white")
                original_view = ui.html(
                    _empty_placeholder("Upload a PDF to preview"), sanitize=False
                ).classes("w-full").style("height: calc(100% - 28px);width:100%")

            with ui.card().classes("p-3 h-full").style(
                "min-height:0;flex:1 1 0;min-width:0;background:#0e1c30;border:1px solid #2a4366"
            ):
                with ui.row().classes("items-center gap-2 mb-2 w-full"):
                    ui.icon("auto_fix_high", size="sm").classes("text-teal-300")
                    ui.label("Mocked copy").classes("text-base font-semibold text-white")
                    ui.space()

                    def _download_mocked() -> None:
                        st = _state()
                        session_id = st.get("session_id")
                        client_id = ui.context.client.id
                        if not session_id and not st.get("redacted_bytes"):
                            ui.notify("Mock the PDF first", type="warning")
                            return
                        if session_id:
                            url = f"/portal/download/mocked?session_id={session_id}"
                        else:
                            url = f"/portal/download/mocked?client_id={client_id}"
                        filename = _mocked_filename(st.get("filename"))
                        ui.run_javascript(
                            "const a = document.createElement('a');"
                            f"a.href = {url!r};"
                            f"a.download = {filename!r};"
                            "document.body.appendChild(a); a.click(); a.remove();"
                        )

                    download_btn = ui.button(
                        "Download PDF",
                        icon="download",
                        on_click=_download_mocked,
                    ).props("color=primary unelevated dense")
                    download_btn.set_enabled(False)
                redacted_view = ui.html(
                    _empty_placeholder("Mocked copy appears here"), sanitize=False
                ).classes("w-full").style("height: calc(100% - 36px);width:100%")


def _build_mappings_tab(ui: Any) -> None:
    """Full-width dictionary review: table, inline override, CSV import/export."""
    with ui.column().classes("w-full gap-4 p-6"):
        with ui.row().classes("w-full items-start justify-between gap-4"):
            with ui.column().classes("gap-1"):
                ui.label("Mock dictionary").classes("text-2xl font-bold text-white")
                ui.label(
                    "Every original value maps to a stable fake stand-in here. "
                    "Download it for QA, or download the template to pre-fill known "
                    "custodians, banks, and account names before a run."
                ).classes("text-sm text-slate-400 max-w-2xl")
            stats_label = ui.label("").classes("text-sm text-slate-300 whitespace-nowrap")

        def _current_entries() -> list[dict[str, Any]]:
            return [
                e.model_dump() if hasattr(e, "model_dump") else dict(e)
                for e in get_mock_store().list()
            ]

        def _update_stats(entries: list[dict[str, Any]]) -> None:
            auto = sum(1 for e in entries if e.get("assignment_source") == "auto")
            user = sum(1 for e in entries if e.get("assignment_source") == "user")
            stats_label.set_text(f"{len(entries)} mappings · {auto} auto-learned · {user} user-set")

        with ui.card().classes("w-full p-4").style("background:#15263f;border:1px solid #2a4366"):
            def _download_mappings() -> None:
                from app.services.pii.mapping_csv import export_mappings_csv

                csv_text = export_mappings_csv(_current_entries())
                ui.download(csv_text.encode("utf-8"), "mock-mappings.csv", media_type="text/csv")

            def _download_template() -> None:
                from app.services.pii.mapping_csv import template_csv

                ui.download(template_csv().encode("utf-8"), "mock-mappings-template.csv", media_type="text/csv")

            async def _import_file(e: Any) -> None:
                from app.services.pii.mapping_csv import import_mappings_csv

                try:
                    text = await e.file.text("utf-8-sig")
                except Exception:
                    ui.notify("Could not read that file as text", type="negative")
                    return
                try:
                    result = import_mappings_csv(get_mock_store(), text)
                except Exception:
                    ui.notify("Could not parse that CSV", type="negative")
                    return
                ui.notify(
                    f"Added {result.inserted} · skipped {result.skipped_existing} existing, "
                    f"{result.skipped_invalid} invalid",
                    type="positive" if result.inserted else "warning",
                )
                _reload_mapping_panel()

            def _clear_all_mappings() -> None:
                try:
                    cleared_count = get_mock_store().clear_all()
                    ui.notify(
                        f"Cleared {cleared_count} mappings and the mapping cache",
                        type="positive",
                    )
                except Exception:
                    ui.notify("Could not clear mappings", type="negative")
                _reload_mapping_panel()

            build_mapping_toolbar(
                _download_mappings,
                _download_template,
                _import_file,
                _clear_all_mappings,
            )

        mapping_container = ui.column().classes("w-full")

        def _reload_mapping_panel() -> None:
            client = _live_client(mapping_container)
            if client is None:
                return
            with client:
                entries = _current_entries()
                _update_stats(entries)
                mapping_container.clear()
                with mapping_container:
                    build_mapping_panel(
                        entries,
                        on_override=_on_override,
                        on_refresh=_reload_mapping_panel,
                        on_create=_on_create,
                        on_delete=_on_delete,
                    )

        def _on_override(payload: dict[str, Any]) -> None:
            try:
                persist_mapping_override(get_mock_store(), payload)
                ui.notify("Mapping updated", type="positive")
            except Exception:
                ui.notify("Could not update mapping", type="negative")
            _reload_mapping_panel()

        def _on_create(payload: dict[str, Any]) -> None:
            try:
                persist_mapping_create(get_mock_store(), payload)
                ui.notify("Mapping added", type="positive")
            except Exception:
                ui.notify("Could not add mapping", type="negative")
            _reload_mapping_panel()

        def _on_delete(mapping_id: str) -> None:
            try:
                get_mock_store().delete(mapping_id)
                ui.notify("Mapping deleted", type="positive")
            except Exception:
                ui.notify("Could not delete mapping", type="negative")
            _reload_mapping_panel()

        _reload_mapping_panel()
