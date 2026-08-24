"""Folder-watch ingestion — an additive alternative to the UI/API upload.

Watches ``Settings.watch_input_dir`` for new files and redacts them one at
a time (never in parallel, per the plan's requirement), reusing the exact
same :class:`RedactPipeline` the NiceGUI portal and ``POST /v1/redact``
already call — no redaction logic lives here, only file lifecycle.

Lifecycle per file:

1. A file is only picked up once its size/mtime has stayed unchanged
   across two consecutive scans (cheap protection against reading a file
   that's still being copied/written into the folder).
2. On successful redaction, the redacted PDF is written to
   ``Settings.watch_output_dir`` and the original is removed from
   ``watch_input_dir`` so it's never reprocessed.
3. On any failure, the original file is left exactly where it is — the
   next scan retries it from scratch. Nothing is ever moved on error.

Disabled by default (``Settings.watch_enabled=False``); see
``app/main.py`` for how it's started/stopped alongside the FastAPI app.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable, Protocol

from app.config import Settings, get_settings
from app.models.redact import RedactAuditResponse, RedactOptions
from app.pipeline.errors import PipelineStageError

logger = logging.getLogger(__name__)


class RedactPipelineProtocol(Protocol):
    async def run(
        self, file_bytes: bytes, filename: str, options: RedactOptions | None = None
    ) -> tuple[bytes, RedactAuditResponse, object]: ...


def _default_pipeline() -> RedactPipelineProtocol:
    from app.pipeline.redact import RedactPipeline

    return RedactPipeline()


class FolderWatcher:
    """Polls one directory, redacts sequentially, files results into another."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        pipeline_factory: Callable[[], RedactPipelineProtocol] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._pipeline_factory = pipeline_factory or _default_pipeline
        # path -> (size, mtime) fingerprint recorded on the previous scan;
        # a file only becomes eligible once its fingerprint repeats.
        self._last_seen: dict[Path, tuple[int, float]] = {}

    def _eligible_extensions(self) -> set[str]:
        return {x.strip().lower() for x in self.settings.allowed_extensions.split(",") if x.strip()}

    def _candidate_files(self) -> list[Path]:
        input_dir = self.settings.watch_input_dir
        if not input_dir.exists():
            return []
        allowed = self._eligible_extensions()
        try:
            entries = sorted(input_dir.iterdir())
        except OSError:
            return []
        return [p for p in entries if p.is_file() and p.suffix.lower().lstrip(".") in allowed]

    def _stable_files(self) -> list[Path]:
        """Update the fingerprint snapshot; return files unchanged since the last scan."""
        stable: list[Path] = []
        seen_now: dict[Path, tuple[int, float]] = {}
        for path in self._candidate_files():
            try:
                stat = path.stat()
            except OSError:
                continue
            fingerprint = (stat.st_size, stat.st_mtime)
            seen_now[path] = fingerprint
            if self._last_seen.get(path) == fingerprint:
                stable.append(path)
        self._last_seen = seen_now
        return stable

    @staticmethod
    def _unique_path(path: Path) -> Path:
        """Avoid clobbering an existing output file with the same stem."""
        if not path.exists():
            return path
        stem, suffix = path.stem, path.suffix
        counter = 1
        while True:
            candidate = path.with_name(f"{stem}_{counter}{suffix}")
            if not candidate.exists():
                return candidate
            counter += 1

    async def process_once(self) -> int:
        """Redact every currently-stable file, strictly one at a time.

        Returns the number of files successfully redacted this pass.
        """
        processed = 0
        for path in self._stable_files():
            if await self._process_file(path):
                processed += 1
        return processed

    async def _process_file(self, path: Path) -> bool:
        output_dir = self.settings.watch_output_dir
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.exception("watch: could not create output dir %s — left %s in place", output_dir, path)
            return False

        try:
            file_bytes = path.read_bytes()
        except OSError as exc:
            logger.warning("watch: could not read %s (%s) — will retry next scan", path, exc)
            self._last_seen.pop(path, None)
            return False

        pipeline = self._pipeline_factory()
        try:
            pdf_bytes, audit, _session = await pipeline.run(file_bytes, path.name, RedactOptions())
        except PipelineStageError as exc:
            logger.error(
                "watch: redaction failed for %s at stage=%s: %s — left in place for retry",
                path, exc.stage, exc,
            )
            return False
        except Exception:
            logger.exception("watch: unexpected error redacting %s — left in place for retry", path)
            return False

        output_path = self._unique_path(output_dir / f"redacted_{path.stem}.pdf")
        try:
            output_path.write_bytes(pdf_bytes)
        except OSError:
            logger.exception("watch: could not write redacted output for %s — left in place for retry", path)
            return False

        try:
            path.unlink()
        except OSError:
            logger.exception(
                "watch: redacted %s -> %s but could not remove the original — may reprocess",
                path, output_path,
            )
            return False

        self._last_seen.pop(path, None)
        logger.info(
            "watch: redacted %s -> %s (%d replacements)",
            path.name, output_path.name, audit.summary.get("redaction_count", 0),
        )
        return True

    async def run_forever(self) -> None:
        """Poll indefinitely until the enclosing task is cancelled."""
        logger.info(
            "watch: watching %s -> %s every %ss",
            self.settings.watch_input_dir, self.settings.watch_output_dir, self.settings.watch_poll_seconds,
        )
        while True:
            try:
                await self.process_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("watch: scan cycle failed unexpectedly")
            await asyncio.sleep(self.settings.watch_poll_seconds)
