"""Startup OCR-engine availability check.

Replaces the previous "call GET /v1/health yourself and notice something's
missing" pattern: compares the actually-available OCR engines against an
explicit required set from ``Settings`` and either fails fast or logs a loud,
unmissable warning — instead of the ensemble silently degrading to whichever
subset of engines happened to import successfully on a given machine.
"""

import logging
import platform
import sys
from dataclasses import dataclass, field
from importlib import metadata as _importlib_metadata

from app.config import get_settings
from app.services.ocr.engines import (
    easyocr_available,
    rapidocr_available,
    tesseract_available,
)
from app.services.structure.docling_adapter import docling_available

logger = logging.getLogger(__name__)

_AVAILABILITY_CHECKS = {
    "tesseract": tesseract_available,
    "easyocr": easyocr_available,
    "rapidocr": rapidocr_available,
}

# Package name -> module + attribute probed for a version string. Anything
# missing/import-failing is reported as "not-installed" rather than raising —
# this fingerprint must never block startup, only describe it.
_VERSIONED_PACKAGES: tuple[tuple[str, str], ...] = (
    ("opencv-python", "cv2"),
    ("numpy", "numpy"),
    ("pillow", "PIL"),
    ("scikit-image", "skimage"),
    ("docling", "docling"),
    ("rapidocr", "rapidocr"),
    ("onnxruntime", "onnxruntime"),
    ("easyocr", "easyocr"),
    ("pytesseract", "pytesseract"),
    ("pymupdf", "fitz"),
)


def _package_version(dist_name: str, module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except ImportError:
        return None
    version = getattr(module, "__version__", None)
    if version:
        return str(version)
    # Some packages (e.g. rapidocr) don't set __version__ on the module
    # itself — fall back to the installed distribution's own metadata,
    # which pip/uv always populate regardless of what the package exposes.
    try:
        return _importlib_metadata.version(dist_name)
    except _importlib_metadata.PackageNotFoundError:
        return "unknown"


def _tesseract_binary_version() -> str | None:
    if not tesseract_available():
        return None
    try:
        import pytesseract

        return str(pytesseract.get_tesseract_version())
    except Exception:  # noqa: BLE001 - version probe must never fail startup
        return None


def log_environment_fingerprint() -> dict[str, object]:
    """Log (and return) one consolidated snapshot of this machine's setup.

    Intended to be called once at startup (see ``app.main``'s lifespan) so
    that "laptop produced result A, server produced result B for the same
    input" can be root-caused by diffing two log lines instead of two
    screenshots — every dependency that can legitimately vary
    machine-to-machine and change pipeline output (OCR engine
    availability, library versions, OS/Python version) in one place.
    """
    package_versions = {
        name: _package_version(name, module) for name, module in _VERSIONED_PACKAGES
    }
    fingerprint = {
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "engines_available": {
            "tesseract": tesseract_available(),
            "easyocr": easyocr_available(),
            "rapidocr": rapidocr_available(),
            "docling": docling_available(),
        },
        "tesseract_binary_version": _tesseract_binary_version(),
        "package_versions": package_versions,
    }
    logger.info("environment fingerprint", extra=fingerprint)
    return fingerprint


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
