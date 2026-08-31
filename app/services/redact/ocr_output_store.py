"""Per-document OCR output dump — a diagnostic artifact, not a product
feature.

Writes exactly one JSON file per processed document
(``{base_dir}/{request_id}.json``) containing every page's raw OCR
output — merged text plus per-word geometry/confidence/engine
attribution — straight from the ensemble/native-text extraction, before
any PII detection or mock-dictionary fuzzy matching runs. Because
detection/fuzzy-matching operate on exactly this text, this file is the
fastest way to tell whether a missed
``Settings.fuzzy_dictionary_scan_threshold`` (default 0.90) match is
caused by a genuinely garbled/misread character the OCR engine itself
produced (a smart quote, a diacritic dropped, a digit misread as a
letter, ...) rather than a bug in the matching logic — ``json.dumps``
with ``ensure_ascii=False`` preserves the exact characters verbatim
instead of masking them behind a lossy display font, while still
escaping any control/non-printable character via the standard
``\\uXXXX`` JSON string form, so even a genuinely corrupt/non-printable
OCR read is visible rather than silently invisible in a terminal.

Every field written here is real source document text — the same
``source_text`` sensitivity ``SubstitutionLedger`` already isolates —
so this deliberately never shares a store, directory, or response model
with ``AuditStore`` (which explicitly forbids ``source_text``) and is
never surfaced to the UI/API. See ``Settings.ocr_output_dump_enabled``
to disable writing this file entirely.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from app.models.mock import MockValidationError
from app.utils.atomic_write import atomic_write_text

if TYPE_CHECKING:
    from app.pipeline.page_state import PageProcessState
    from app.services.ocr.ensemble_types import EnsembleWord

logger = logging.getLogger(__name__)


def _validate_request_id(request_id: str) -> None:
    """Reject empty or path-escaping request ids (no write)."""
    if (
        not request_id
        or not request_id.strip()
        or "/" in request_id
        or "\\" in request_id
        or ".." in request_id
    ):
        raise MockValidationError("request_id", "invalid")


def _word_dict(word: "EnsembleWord") -> dict:
    return {
        "text": word.text,
        "x": word.bbox.x,
        "y": word.bbox.y,
        "w": word.bbox.w,
        "h": word.bbox.h,
        "confidence": word.ocr_confidence,
        "engine_agreement": word.engine_agreement,
        "engines": word.engines,
        "char_start": word.char_start,
        "char_end": word.char_end,
    }


def _page_dict(state: "PageProcessState") -> dict:
    return {
        "page_index": state.page_index,
        "page_kind": state.page_kind,
        "word_count": len(state.ensemble_words),
        "merged_text": state.merged_text,
        "words": [_word_dict(w) for w in state.ensemble_words],
    }


class OcrOutputStore:
    """One JSON file per document under an injected base directory.

    Args:
        base_dir: Root for ``{request_id}.json`` files.
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir

    def _path(self, request_id: str) -> Path:
        _validate_request_id(request_id)
        return self._base_dir / f"{request_id}.json"

    def save(
        self,
        request_id: str,
        filename: str,
        page_states: "list[PageProcessState]",
    ) -> Path:
        """Write the OCR output for every page of one document.

        Args:
            request_id: Same id as the accompanying ``RedactAuditResponse``
                (must not contain ``/``, ``\\``, or ``..``).
            filename: Original uploaded filename, for operator readability.
            page_states: This request's built ``PageProcessState`` list —
                already carries each page's final ``merged_text``/
                ``ensemble_words`` (native-text or OCR-ensemble output,
                whichever path that page took).

        Returns:
            Path the file was written to.
        """
        payload = {
            "request_id": request_id,
            "filename": filename,
            "page_count": len(page_states),
            "pages": [_page_dict(state) for state in page_states],
        }
        path = self._path(request_id)
        atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))
        logger.info(
            "ocr_output_save request_id=%s page_count=%s",
            request_id,
            len(page_states),
        )
        return path
