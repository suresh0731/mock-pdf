"""Startup OCR-engine availability check.

Replaces the previous "call GET /v1/health yourself and notice something's
missing" pattern: compares the actually-available OCR engines against an
explicit required set from ``Settings`` and either fails fast or logs a loud,
unmissable warning — instead of the ensemble silently degrading to whichever
subset of engines happened to import successfully on a given machine.
"""

import logging
from dataclasses import dataclass, field

from app.config import get_settings
from app.services.ocr.engines import (
    easyocr_available,
    rapidocr_available,
    tesseract_available,
)

logger = logging.getLogger(__name__)

_AVAILABILITY_CHECKS = {
    "tesseract": tesseract_available,
    "easyocr": easyocr_available,
    "rapidocr": rapidocr_available,
}


@dataclass
class EngineCheckResult:
    required: list[str] = field(default_factory=list)
    available: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing and not self.unknown


def check_required_engines() -> EngineCheckResult:
    """Pure check: which required engines are available right now.

    Never raises; unknown engine names (typos in ``OCR_REQUIRED_ENGINES``)
    are reported separately from genuinely missing/uninstalled engines.
    """
    settings = get_settings()
    required = [name.strip() for name in settings.ocr_required_engines.split(",") if name.strip()]
    available: list[str] = []
    missing: list[str] = []
    unknown: list[str] = []
    for name in required:
        probe = _AVAILABILITY_CHECKS.get(name)
        if probe is None:
            unknown.append(name)
            continue
        if probe():
            available.append(name)
        else:
            missing.append(name)
    return EngineCheckResult(required=required, available=available, missing=missing, unknown=unknown)


def enforce_required_engines() -> EngineCheckResult:
    """Run :func:`check_required_engines` and act on the result.

    In strict mode (``Settings.ocr_strict_engine_check``), any missing or
    unknown required engine raises ``RuntimeError`` immediately at startup.
    Otherwise, logs a loud warning so the gap is visible without blocking
    local development on a partially-configured machine.
    """
    result = check_required_engines()
    settings = get_settings()
    if not result.ok:
        message = (
            "OCR engine availability check failed: required=%s available=%s "
            "missing=%s unknown=%s. Install the missing engine(s) or update "
            "OCR_REQUIRED_ENGINES in Settings/.env." % (
                result.required,
                result.available,
                result.missing,
                result.unknown,
            )
        )
        if settings.ocr_strict_engine_check:
            raise RuntimeError(message)
        logger.warning(message)
    else:
        logger.info("OCR engine availability check passed: %s", result.available)
    return result
