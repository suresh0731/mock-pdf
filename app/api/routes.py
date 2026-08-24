import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response

from app.config import get_settings
from app.models.redact import RedactOptions
from app.pipeline.errors import PipelineStageError
from app.pipeline.redact import RedactPipeline
from app.services.ocr.engines import easyocr_available, rapidocr_available, tesseract_available
from app.services.pii.detector import detect_pii
from app.services.redact.audit_store import AuditStore
from app.services.redact.session_store import session_store
from app.services.structure.docling_adapter import docling_available

router = APIRouter(prefix="/v1")


def _check_api_key(x_api_key: str | None) -> None:
    settings = get_settings()
    if settings.admin_auth_enabled and settings.api_key:
        if x_api_key != settings.api_key:
            raise HTTPException(401, detail={"error": {"code": "UNAUTHORIZED", "message": "Invalid API key"}})


def _error_response(
    status: int,
    code: str,
    message: str,
    request_id: str | None = None,
    stage: str | None = None,
) -> HTTPException:
    error: dict = {
        "code": code,
        "message": message,
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if stage:
        error["details"] = {"stage": stage}
    return HTTPException(status_code=status, detail={"error": error})


def _opencv_available() -> bool:
    try:
        import cv2  # noqa: F401

        return True
    except ImportError:
        return False


@router.get("/health")
async def health():
    settings = get_settings()
    return {
        "status": "ok",
        "tesseract": "ok" if tesseract_available() else "missing",
        "easyocr": "ok" if easyocr_available() else "missing",
        "rapidocr": "ok" if rapidocr_available() else "missing",
        "docling": "ok" if docling_available() else "missing",
        "opencv": "ok" if _opencv_available() else "missing",
        "pipeline": "redact-v1",
        "redact_pipeline_enabled": settings.redact_pipeline_enabled,
    }


@router.post("/redact")
async def redact(
    file: UploadFile = File(...),
    options: str = Form("{}"),
    response_mode: str = Form("pdf"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    settings = get_settings()
    _check_api_key(x_api_key)

    if not settings.redact_pipeline_enabled:
        raise _error_response(503, "PIPELINE_DISABLED", "Redact pipeline is disabled")

    content = await file.read()
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise _error_response(400, "FILE_TOO_LARGE", "File too large")

    ext = Path(file.filename or "upload").suffix.lower().lstrip(".")
    allowed = {x.strip() for x in settings.allowed_extensions.split(",")}
    if ext not in allowed:
        raise _error_response(400, "INVALID_FILE_TYPE", f"Unsupported file type: {ext}")

    try:
        opts = RedactOptions(**json.loads(options or "{}"))
    except Exception as exc:
        raise _error_response(422, "VALIDATION_ERROR", str(exc)) from exc

    pipeline = RedactPipeline()
    try:
        pdf_bytes, audit, session = await pipeline.run(content, file.filename or "upload", opts)
    except PipelineStageError as exc:
        raise _error_response(503, "PIPELINE_STAGE_FAILED", str(exc), stage=exc.stage) from exc

    base_name = Path(file.filename or "document").stem
    headers = {
        "X-Request-Id": audit.request_id,
        "X-Ledger-Id": audit.request_id,
        "X-Session-Id": session.session_id,
        "X-Redaction-Count": str(audit.summary.get("redaction_count", 0)),
        "X-Page-Count": str(audit.page_count),
        "X-Processing-Ms": str(audit.processing_ms),
        "Content-Disposition": f'attachment; filename="redacted_{base_name}.pdf"',
    }

    if response_mode == "multipart":
        boundary = "pii-redact-boundary"
        audit_json = audit.model_dump_json(indent=2)
        body = (
            f"--{boundary}\r\n"
            f"Content-Type: application/pdf\r\n"
            f'Content-Disposition: attachment; filename="redacted_{base_name}.pdf"\r\n\r\n'
        ).encode() + pdf_bytes + (
            f"\r\n--{boundary}\r\n"
            f"Content-Type: application/json\r\n"
            f'Content-Disposition: attachment; filename="audit.json"\r\n\r\n'
        ).encode() + audit_json.encode() + f"\r\n--{boundary}--\r\n".encode()
        headers["Content-Type"] = f"multipart/mixed; boundary={boundary}"
        return Response(content=body, headers=headers)

    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)


@router.get("/redact/audit/{request_id}")
async def redact_audit(request_id: str, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    _check_api_key(x_api_key)
    audit = AuditStore().get(request_id)
    if not audit:
        raise _error_response(404, "AUDIT_NOT_FOUND", f"No audit for request_id: {request_id}")
    return audit


@router.post("/redact/regenerate/{session_id}")
async def redact_regenerate(
    session_id: str,
    custom_terms: str = Form(""),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    _check_api_key(x_api_key)
    settings = get_settings()
    if not settings.redact_pipeline_enabled:
        raise _error_response(503, "PIPELINE_DISABLED", "Redact pipeline is disabled")

    pipeline = RedactPipeline()
    try:
        pdf_bytes, audit, session = await pipeline.regenerate(session_id, custom_terms_text=custom_terms)
    except PipelineStageError as exc:
        raise _error_response(503, "PIPELINE_STAGE_FAILED", str(exc), stage=exc.stage) from exc

    base_name = Path(session.filename or "document").stem
    headers = {
        "X-Request-Id": audit.request_id,
        "X-Ledger-Id": audit.request_id,
        "X-Session-Id": session.session_id,
        "X-Redaction-Count": str(audit.summary.get("redaction_count", 0)),
        "Content-Disposition": f'attachment; filename="redacted_{base_name}.pdf"',
    }
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)


@router.get("/redact/session/{session_id}/original")
async def session_original_pdf(session_id: str):
    session = session_store.get(session_id)
    if not session:
        raise _error_response(404, "SESSION_NOT_FOUND", "Session not found")
    return Response(content=session.file_bytes, media_type="application/pdf")


@router.get("/redact/session/{session_id}/redacted")
async def session_redacted_pdf(session_id: str):
    session = session_store.get(session_id)
    if not session or not session.last_pdf:
        raise _error_response(404, "SESSION_NOT_FOUND", "Redacted PDF not available")
    return Response(content=session.last_pdf, media_type="application/pdf")


@router.post("/ocr-only")
async def ocr_only(file: UploadFile = File(...), locale: str | None = Form(None)):
    from app.services.ocr.engine import ocr_page
    from app.services.ocr.ensemble import ensemble_ocr_page
    from app.services.ocr.page_renderer import load_pages
    from app.services.preprocess.blur import detect_blur_tier

    content = await file.read()
    pages = load_pages(content, file.filename or "upload")
    page_results = []
    for i, page in enumerate(pages):
        tier, variance = detect_blur_tier(page)
        merged, words, engines = await ensemble_ocr_page(page, i, "eng", ["en"])
        page_results.append(
            {
                "page": i,
                "text": merged,
                "blur_tier": tier,
                "blur_variance": variance,
                "ensemble_word_count": len(words),
                "engines": [e.engine for e in engines],
            }
        )
    full = "\n".join(p["text"] for p in page_results)
    detections = detect_pii(full, locale)
    return {"text": full, "pages": page_results, "detections": detections}


@router.get("/recognizers")
async def list_recognizers():
    settings = get_settings()
    files = list(settings.recognizers_dir.glob("*.yaml")) if settings.recognizers_dir.exists() else []
    return {"recognizers": [f.name for f in files]}


@router.post("/recognizers/reload")
async def reload_recognizers():
    from app.services.pii import detector

    detector._analyzer = None
    return {"status": "reloaded"}
