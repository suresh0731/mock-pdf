"""Tests for the atomic-write-with-Windows-retry helper."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.utils.atomic_write import atomic_write_text


def test_atomic_write_text_writes_file_and_cleans_up_tmp(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    atomic_write_text(path, "hello")
    assert path.read_text(encoding="utf-8") == "hello"
    assert not (tmp_path / "snapshot.json.tmp").exists()


def test_atomic_write_text_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "snapshot.json"
    atomic_write_text(path, "hello")
    assert path.read_text(encoding="utf-8") == "hello"


def test_atomic_write_text_overwrites_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    atomic_write_text(path, "first")
    atomic_write_text(path, "second")
    assert path.read_text(encoding="utf-8") == "second"


def test_atomic_write_text_retries_transient_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient WinError-5-style PermissionError on replace() must not
    lose the write — it should succeed once the lock clears."""
    path = tmp_path / "snapshot.json"
    real_replace = os.replace
    calls = {"count": 0}

    def flaky_replace(src, dst):
        calls["count"] += 1
        if calls["count"] < 3:
            raise PermissionError("simulated transient lock")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)
    atomic_write_text(path, "hello", base_delay_seconds=0.001)
    assert path.read_text(encoding="utf-8") == "hello"
    assert calls["count"] == 3


def test_atomic_write_text_falls_back_to_in_place_write_when_permanently_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the rename never succeeds, the content must still land on disk
    rather than being silently dropped."""
    path = tmp_path / "snapshot.json"
    path.write_text("original", encoding="utf-8")

    def always_denied(src, dst):
        raise PermissionError("simulated permanent lock")

    monkeypatch.setattr(os, "replace", always_denied)
    atomic_write_text(path, "hello", retries=2, base_delay_seconds=0.001)
    assert path.read_text(encoding="utf-8") == "hello"
    assert not (tmp_path / "snapshot.json.tmp").exists()


def test_atomic_write_text_logs_warning_on_retry_exhaustion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "snapshot.json"

    def always_denied(src, dst):
        raise PermissionError("simulated permanent lock")

    monkeypatch.setattr(os, "replace", always_denied)
    with caplog.at_level("WARNING"):
        atomic_write_text(path, "hello", retries=2, base_delay_seconds=0.001)
    assert "atomic_write_replace_denied" in caplog.text
    assert "atomic_write_fallback_inplace" in caplog.text
