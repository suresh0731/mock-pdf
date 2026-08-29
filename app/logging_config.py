"""Central logging setup.

The rest of the codebase already logs plenty of structured detail via
``logger.info(msg, extra={...})``/``logger.debug(msg, extra={...})`` (blur
tier, structure block counts, img2table merges, padding clamp decisions,
OCR engine fallback, ...) but ``logging.basicConfig()``'s default formatter
only prints ``levelname``/``name``/``message`` — every ``extra`` field is
silently dropped from the actual log line, never a parsing/attribute error,
just invisible. That's the biggest reason two machines' logs "look the
same" even when the underlying decisions differ: the data was always being
computed, just never rendered.

``JsonLogFormatter`` renders every extra field alongside the standard
attributes as one JSON object per line, so two runs (e.g. laptop vs.
server) can be diffed directly (``jq`` per field, or a plain text diff)
instead of comparing screenshots of annotated images.
"""

from __future__ import annotations

import json
import logging
import time

# Attributes every stdlib LogRecord carries — anything else on the record
# came from an explicit ``extra={...}`` call and is worth surfacing.
_STANDARD_RECORD_ATTRS = frozenset(
    logging.LogRecord(
        "", 0, "", 0, "", (), None
    ).__dict__.keys()
) | {"message", "asctime"}


class JsonLogFormatter(logging.Formatter):
    """One JSON object per log line, including any ``extra=`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _STANDARD_RECORD_ATTRS or key in payload:
                continue
            try:
                json.dumps(value)
            except TypeError:
                value = repr(value)
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        try:
            return json.dumps(payload, default=str)
        except (TypeError, ValueError):
            payload["message"] = str(payload.get("message", ""))
            return json.dumps(payload, default=str)


class PlainLogFormatter(logging.Formatter):
    """Human-readable formatter that still appends any ``extra=`` fields.

    Same information as :class:`JsonLogFormatter`, laid out as
    ``LEVEL logger: message {extra=fields}`` for local terminal reading
    instead of log-aggregator ingestion.
    """

    def format(self, record: logging.LogRecord) -> str:
        base = f"{record.levelname} {record.name}: {record.getMessage()}"
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_RECORD_ATTRS
        }
        if extras:
            base += " " + " ".join(f"{k}={v!r}" for k, v in sorted(extras.items()))
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Replace the root logger's handlers with one that renders ``extra=`` fields.

    Idempotent — safe to call more than once (e.g. once from ``app.main``
    and again from a script), always resets to exactly one handler so
    output is never duplicated.

    Args:
        level: Standard logging level name (``DEBUG``/``INFO``/...).
            ``DEBUG`` additionally surfaces the per-redaction padding
            clamp decision logged in
            ``app.services.pii.coordinate_map.apply_padding`` — the most
            direct way to see *why* a padded box ended up the size it did
            on a given machine.
        fmt: ``"json"`` (default, greppable/diffable/log-aggregator
            friendly) or ``"plain"`` (human-readable terminal output).
    """
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    formatter: logging.Formatter = JsonLogFormatter() if fmt == "json" else PlainLogFormatter()
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(resolved_level)
    root.handlers.clear()
    root.addHandler(handler)
