"""Unit tests for the folder-watch ingestion extension (see
app/services/watch/folder_watcher.py) -- one-file-at-a-time redaction of a
watched input directory into an output directory, with failed files left
untouched for automatic retry.
"""

from __future__ import annotations

import asyncio

import pytest

from app.config import Settings
from app.pipeline.errors import PipelineStageError
from app.services.watch.folder_watcher import FolderWatcher


class _FakeAudit:
    def __init__(self, redaction_count: int = 1):
        self.summary = {"redaction_count": redaction_count}


class _FakePipeline:
    """Records every file it was asked to redact and how it should behave."""

    def __init__(self, *, fail_on: set[str] | None = None, raise_unexpected_on: set[str] | None = None):
        self.fail_on = fail_on or set()
        self.raise_unexpected_on = raise_unexpected_on or set()
        self.calls: list[str] = []
        self.concurrent = 0
        self.max_concurrent = 0

    async def run(self, file_bytes: bytes, filename: str, options=None):
        self.calls.append(filename)
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            await asyncio.sleep(0.01)
            if filename in self.raise_unexpected_on:
                raise RuntimeError("boom")
            if filename in self.fail_on:
                raise PipelineStageError("ensemble_ocr", f"OCR failed for {filename}")
            return b"%PDF-fake-" + file_bytes, _FakeAudit(), object()
        finally:
            self.concurrent -= 1


def _settings(tmp_path, **overrides) -> Settings:
    kwargs = {
        "watch_input_dir": tmp_path / "input",
        "watch_output_dir": tmp_path / "output",
        "watch_poll_seconds": 0.01,
        "allowed_extensions": "pdf,png",
    }
    kwargs.update(overrides)
    return Settings(**kwargs)


def _touch(path, content: bytes = b"hello") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


@pytest.mark.asyncio
async def test_new_file_is_not_processed_on_first_scan_only(tmp_path):
    """A brand-new file isn't stable yet on the scan where it first appears."""
    settings = _settings(tmp_path)
    _touch(settings.watch_input_dir / "a.pdf")
    pipeline = _FakePipeline()
    watcher = FolderWatcher(settings=settings, pipeline_factory=lambda: pipeline)

    processed_first = await watcher.process_once()
    assert processed_first == 0
    assert pipeline.calls == []

    processed_second = await watcher.process_once()
    assert processed_second == 1
    assert pipeline.calls == ["a.pdf"]


@pytest.mark.asyncio
async def test_successful_redaction_writes_output_and_removes_original(tmp_path):
    settings = _settings(tmp_path)
    input_path = settings.watch_input_dir / "statement.pdf"
    _touch(input_path)
    pipeline = _FakePipeline()
    watcher = FolderWatcher(settings=settings, pipeline_factory=lambda: pipeline)

    await watcher.process_once()  # records fingerprint
    await watcher.process_once()  # stable now, processed

    assert not input_path.exists()
    output_files = list(settings.watch_output_dir.glob("*.pdf"))
    assert len(output_files) == 1
    assert output_files[0].name == "redacted_statement.pdf"
    assert output_files[0].read_bytes().startswith(b"%PDF-fake-")


@pytest.mark.asyncio
async def test_pipeline_stage_error_leaves_original_file_in_place(tmp_path):
    settings = _settings(tmp_path)
    input_path = settings.watch_input_dir / "bad.pdf"
    _touch(input_path)
    pipeline = _FakePipeline(fail_on={"bad.pdf"})
    watcher = FolderWatcher(settings=settings, pipeline_factory=lambda: pipeline)

    await watcher.process_once()
    await watcher.process_once()

    assert input_path.exists()
    if settings.watch_output_dir.exists():
        assert list(settings.watch_output_dir.glob("*")) == []


@pytest.mark.asyncio
async def test_unexpected_exception_also_leaves_original_file_in_place(tmp_path):
    settings = _settings(tmp_path)
    input_path = settings.watch_input_dir / "crash.pdf"
    _touch(input_path)
    pipeline = _FakePipeline(raise_unexpected_on={"crash.pdf"})
    watcher = FolderWatcher(settings=settings, pipeline_factory=lambda: pipeline)

    await watcher.process_once()
    await watcher.process_once()

    assert input_path.exists()


@pytest.mark.asyncio
async def test_failed_file_is_retried_on_a_later_scan_once_it_succeeds(tmp_path):
    settings = _settings(tmp_path)
    input_path = settings.watch_input_dir / "flaky.pdf"
    _touch(input_path)
    pipeline = _FakePipeline(fail_on={"flaky.pdf"})
    watcher = FolderWatcher(settings=settings, pipeline_factory=lambda: pipeline)

    await watcher.process_once()
    await watcher.process_once()
    assert input_path.exists()
    assert pipeline.calls == ["flaky.pdf"]

    # Fix the "flakiness" and let the next scan retry successfully.
    pipeline.fail_on.clear()
    await watcher.process_once()
    assert not input_path.exists()
    assert pipeline.calls == ["flaky.pdf", "flaky.pdf"]


@pytest.mark.asyncio
async def test_multiple_files_are_processed_sequentially_never_in_parallel(tmp_path):
    settings = _settings(tmp_path)
    for name in ("one.pdf", "two.pdf", "three.pdf"):
        _touch(settings.watch_input_dir / name, content=name.encode())
    pipeline = _FakePipeline()
    watcher = FolderWatcher(settings=settings, pipeline_factory=lambda: pipeline)

    await watcher.process_once()  # fingerprint pass
    processed = await watcher.process_once()

    assert processed == 3
    assert pipeline.max_concurrent == 1
    assert sorted(pipeline.calls) == ["one.pdf", "three.pdf", "two.pdf"]


@pytest.mark.asyncio
async def test_output_name_collision_gets_a_unique_suffix(tmp_path):
    settings = _settings(tmp_path)
    settings.watch_output_dir.mkdir(parents=True, exist_ok=True)
    (settings.watch_output_dir / "redacted_bill.pdf").write_bytes(b"existing")
    input_path = settings.watch_input_dir / "bill.pdf"
    _touch(input_path)
    pipeline = _FakePipeline()
    watcher = FolderWatcher(settings=settings, pipeline_factory=lambda: pipeline)

    await watcher.process_once()
    await watcher.process_once()

    assert (settings.watch_output_dir / "redacted_bill.pdf").read_bytes() == b"existing"
    assert (settings.watch_output_dir / "redacted_bill_1.pdf").exists()


@pytest.mark.asyncio
async def test_disallowed_extension_is_ignored(tmp_path):
    settings = _settings(tmp_path, allowed_extensions="pdf")
    _touch(settings.watch_input_dir / "notes.txt")
    pipeline = _FakePipeline()
    watcher = FolderWatcher(settings=settings, pipeline_factory=lambda: pipeline)

    await watcher.process_once()
    processed = await watcher.process_once()

    assert processed == 0
    assert pipeline.calls == []
    assert (settings.watch_input_dir / "notes.txt").exists()


@pytest.mark.asyncio
async def test_missing_input_dir_is_a_noop(tmp_path):
    settings = _settings(tmp_path)  # directory never created
    pipeline = _FakePipeline()
    watcher = FolderWatcher(settings=settings, pipeline_factory=lambda: pipeline)

    processed = await watcher.process_once()
    assert processed == 0
