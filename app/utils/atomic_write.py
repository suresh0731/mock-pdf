"""Atomic same-directory temp-file writes, with Windows lock retry.

``Path.replace()``/``os.replace()`` is atomic and dependable on POSIX, but on
Windows it can raise ``PermissionError`` (WinError 5, "Access is denied")
when the destination is transiently held open by another process — a virus
scanner, a search indexer, a backup tool, or even a second reader in this
same process (e.g. the UI reading the snapshot while a request thread is
mid-write). That's not a real write conflict — the new content is already
fully and safely on disk in the temp file by the time replace() is called —
it just needs the other handle to close, which is normally momentary. This
wraps the rename in a bounded retry with backoff before falling back to a
slower but still-safe delete-then-rename, and, only as a last resort, a
non-atomic in-place write — so a snapshot/ledger update is never silently
lost to a machine-specific file-locking quirk.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_RETRIES = 6
_DEFAULT_BASE_DELAY_SECONDS = 0.05


def atomic_write_text(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    retries: int = _DEFAULT_RETRIES,
    base_delay_seconds: float = _DEFAULT_BASE_DELAY_SECONDS,
) -> None:
    """Write ``text`` to ``path`` via a same-directory temp file + rename.

    Args:
        path: Final destination file. Parent directory is created if missing.
        text: Full file contents (the whole file is rewritten each call).
        encoding: Text encoding for both the temp file and any fallback write.
        retries: Attempts at the atomic rename before falling back to
            delete-then-rename. Each attempt after the first waits
            ``base_delay_seconds * 2**attempt`` first (~1.55s total across
            the default 6 attempts) — long enough for a transient
            antivirus/indexer hold to clear, short enough to never look hung.
        base_delay_seconds: Backoff unit between retries.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding=encoding)

    last_error: OSError | None = None
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(base_delay_seconds * (2**attempt))

    logger.warning(
        "atomic_write_replace_denied path=%s attempts=%s error=%s"
        " - retrying with delete-then-rename",
        path,
        retries,
        type(last_error).__name__ if last_error else "unknown",
    )
    try:
        if path.exists():
            os.remove(path)
        os.replace(tmp, path)
        return
    except OSError as exc:
        last_error = exc

    logger.warning(
        "atomic_write_fallback_inplace path=%s error=%s - writing in place (non-atomic)",
        path,
        type(last_error).__name__ if last_error else "unknown",
    )
    try:
        path.write_text(text, encoding=encoding)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
