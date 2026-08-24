import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)


_DENIED_KEYS = frozenset(
    {"text", "value", "token", "corrected_value", "source_text", "normalized"}
)


def _keep_audit_kwarg(key: str) -> bool:
    lowered = key.lower()
    if lowered in _DENIED_KEYS:
        return False
    if "pii" in lowered or "value" in lowered:
        return False
    return True


def audit_log(action: str, **kwargs: Any) -> None:
    settings = get_settings()
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": action,
        **{k: v for k, v in kwargs.items() if _keep_audit_kwarg(k)},
    }
    logger.info("audit %s", json.dumps(record))
    audit_dir = settings.shard_base_path / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    with (audit_dir / "audit.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
