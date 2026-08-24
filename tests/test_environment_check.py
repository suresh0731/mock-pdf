"""Unit tests for the startup OCR engine-availability check (see
app/services/ocr/environment_check.py) -- the "explicit required-engine-set
check" that makes a machine missing an engine fail visibly instead of
silently producing different output than one with the full set installed.
"""

from __future__ import annotations

import logging

import pytest

from app.services.ocr import environment_check as ec


class _FakeSettings:
    def __init__(self, ocr_required_engines: str, ocr_strict_engine_check: bool):
        self.ocr_required_engines = ocr_required_engines
        self.ocr_strict_engine_check = ocr_strict_engine_check


def _patch_settings(monkeypatch, *, required: str, strict: bool):
    monkeypatch.setattr(ec, "get_settings", lambda: _FakeSettings(required, strict))


def _patch_availability(monkeypatch, **overrides: bool):
    checks = dict(ec._AVAILABILITY_CHECKS)
    for name, available in overrides.items():
        checks[name] = (lambda ok: (lambda: ok))(available)
    monkeypatch.setattr(ec, "_AVAILABILITY_CHECKS", checks)


# --- check_required_engines --------------------------------------------


def test_check_required_engines_all_available_is_ok(monkeypatch):
    _patch_settings(monkeypatch, required="tesseract,rapidocr", strict=False)
    _patch_availability(monkeypatch, tesseract=True, rapidocr=True)
    result = ec.check_required_engines()
    assert result.ok is True
    assert result.missing == []
    assert result.unknown == []
    assert set(result.available) == {"tesseract", "rapidocr"}


def test_check_required_engines_reports_missing_engine(monkeypatch):
    _patch_settings(monkeypatch, required="tesseract,rapidocr", strict=False)
    _patch_availability(monkeypatch, tesseract=True, rapidocr=False)
    result = ec.check_required_engines()
    assert result.ok is False
    assert result.missing == ["rapidocr"]
    assert result.available == ["tesseract"]


def test_check_required_engines_reports_unknown_engine_name(monkeypatch):
    _patch_settings(monkeypatch, required="tesseract,not_a_real_engine", strict=False)
    _patch_availability(monkeypatch, tesseract=True)
    result = ec.check_required_engines()
    assert result.ok is False
    assert result.unknown == ["not_a_real_engine"]


def test_check_required_engines_never_raises_on_unknown_or_missing(monkeypatch):
    _patch_settings(monkeypatch, required="bogus", strict=True)
    result = ec.check_required_engines()
    assert result.unknown == ["bogus"]


def test_check_required_engines_empty_setting_yields_trivially_ok_result(monkeypatch):
    _patch_settings(monkeypatch, required="", strict=False)
    result = ec.check_required_engines()
    assert result.required == []
    assert result.ok is True


def test_check_required_engines_ignores_blank_entries_and_whitespace(monkeypatch):
    _patch_settings(monkeypatch, required=" tesseract , , rapidocr ", strict=False)
    _patch_availability(monkeypatch, tesseract=True, rapidocr=True)
    result = ec.check_required_engines()
    assert result.required == ["tesseract", "rapidocr"]
    assert result.ok is True


# --- enforce_required_engines --------------------------------------------


def test_enforce_required_engines_strict_mode_raises_on_missing(monkeypatch):
    _patch_settings(monkeypatch, required="tesseract,rapidocr", strict=True)
    _patch_availability(monkeypatch, tesseract=True, rapidocr=False)
    with pytest.raises(RuntimeError, match="OCR engine availability check failed"):
        ec.enforce_required_engines()


def test_enforce_required_engines_warn_mode_logs_instead_of_raising(monkeypatch, caplog):
    _patch_settings(monkeypatch, required="tesseract,rapidocr", strict=False)
    _patch_availability(monkeypatch, tesseract=True, rapidocr=False)
    with caplog.at_level(logging.WARNING, logger="app.services.ocr.environment_check"):
        result = ec.enforce_required_engines()
    assert result.ok is False
    assert any("OCR engine availability check failed" in rec.message for rec in caplog.records)


def test_enforce_required_engines_all_available_logs_info_not_warning(monkeypatch, caplog):
    _patch_settings(monkeypatch, required="tesseract", strict=True)
    _patch_availability(monkeypatch, tesseract=True)
    with caplog.at_level(logging.INFO, logger="app.services.ocr.environment_check"):
        result = ec.enforce_required_engines()
    assert result.ok is True
    assert not any(rec.levelno >= logging.WARNING for rec in caplog.records)


def test_enforce_required_engines_returns_the_check_result(monkeypatch):
    _patch_settings(monkeypatch, required="tesseract", strict=False)
    _patch_availability(monkeypatch, tesseract=True)
    result = ec.enforce_required_engines()
    assert result.available == ["tesseract"]
