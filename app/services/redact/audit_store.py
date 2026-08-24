import json
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.models.redact import RedactAuditResponse


def _payload_has_source_text(payload: object) -> bool:
    """True if any dict key is source_text (recursive)."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "source_text":
                return True
            if _payload_has_source_text(value):
                return True
        return False
    if isinstance(payload, list):
        return any(_payload_has_source_text(item) for item in payload)
    return False


class AuditStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.base = self.settings.shard_base_path / "audit" / "requests"
        self.base.mkdir(parents=True, exist_ok=True)

    def save(self, audit: RedactAuditResponse) -> Path:
        data = audit.model_dump(mode="json")
        if _payload_has_source_text(data):
            raise ValueError("audit payload must not contain source_text")
        path = self.base / f"{audit.request_id}.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    def get(self, request_id: str) -> RedactAuditResponse | None:
        path = self.base / f"{request_id}.json"
        if not path.exists():
            return None
        return RedactAuditResponse.model_validate_json(path.read_text(encoding="utf-8"))

    def log_summary(self, audit: RedactAuditResponse) -> None:
        from app.utils.audit import audit_log

        audit_log(
            "redact_complete",
            request_id=audit.request_id,
            page_count=audit.page_count,
            redaction_count=audit.summary.get("redaction_count", 0),
            avg_confidence=audit.summary.get("avg_confidence", 0),
            blur_tiers=audit.summary.get("blur_tiers", {}),
            processing_ms=audit.processing_ms,
        )
