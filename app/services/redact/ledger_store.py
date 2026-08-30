"""Per-job substitution ledger persistence (isolated PII store).

``source_text`` is written to ``{base_dir}/{request_id}/ledger.json`` only.
Logs include ``request_id`` and ``entry_count`` — never entry bodies.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from app.models.mock import MockValidationError, SubstitutionLedger
from app.utils.atomic_write import atomic_write_text

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


class LedgerStore:
    """JSON ledger files under an injected base directory.

    Args:
        base_dir: Root for ``{request_id}/ledger.json`` shards.
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._lock = threading.RLock()

    def _ledger_path(self, request_id: str) -> Path:
        _validate_request_id(request_id)
        return self._base_dir / request_id / "ledger.json"

    def save(self, ledger: SubstitutionLedger) -> Path:
        """Atomically persist a ledger. Returns the final file path.

        Args:
            ledger: Per-job substitutions including ``source_text``.

        Returns:
            Path to ``ledger.json`` under ``base_dir / request_id``.

        Raises:
            MockValidationError: If ``request_id`` is empty or traverses.
        """
        path = self._ledger_path(ledger.request_id)
        with self._lock:
            atomic_write_text(path, ledger.model_dump_json(indent=2))
            logger.info(
                "ledger_save request_id=%s entry_count=%s",
                ledger.request_id,
                len(ledger.entries),
            )
            return path

    def get(self, request_id: str) -> SubstitutionLedger | None:
        """Load a ledger, or None if the file is missing.

        Args:
            request_id: Job id (must not contain ``/``, ``\\``, or ``..``).

        Returns:
            Restored ledger including ``source_text``, or None.

        Raises:
            MockValidationError: If ``request_id`` is empty or traverses.
        """
        path = self._ledger_path(request_id)
        with self._lock:
            if not path.exists():
                logger.info(
                    "ledger_get request_id=%s entry_count=%s",
                    request_id,
                    0,
                )
                return None
            ledger = SubstitutionLedger.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            logger.info(
                "ledger_get request_id=%s entry_count=%s",
                request_id,
                len(ledger.entries),
            )
            return ledger
