import sys
import types

from PIL import Image

import app.services.ocr.engines as engines
from app.services.ocr.engines import (
    _get_rapidocr_engine,
    _rapidocr_engines,
    _rapidocr_lang_params,
    ocr_image_rapidocr,
)


def test_rapidocr_lang_params_defaults_to_english_for_none():
    assert _rapidocr_lang_params(None) == ("en", "en")


def test_rapidocr_lang_params_defaults_to_english_for_english_only():
    assert _rapidocr_lang_params(["en"]) == ("en", "en")


def test_rapidocr_lang_params_prefers_indonesian_over_english():
    """Indonesian locale (LOCALE_LANG_MAP["id"] == ["en", "id"]) must select
    the id recognition model, not silently fall back to RapidOCR's
    English/Chinese default. Det and Rec must use the *same* lang_type:
    RapidOCR's model resolver validates ``Det.lang_type`` against its own
    per-language set, and the literal string ``"multi"`` (used here by a
    prior, buggy version of this function) is not a member of that set —
    passing it raises ``ValueError`` at engine construction for every
    Indonesian/Malay/Traditional-Chinese document, regardless of image
    content (see ``ocr_image_rapidocr``'s docstring)."""
    assert _rapidocr_lang_params(["en", "id"]) == ("id", "id")


def test_rapidocr_lang_params_prefers_malay_over_english():
    assert _rapidocr_lang_params(["en", "ms"]) == ("ms", "ms")


def test_rapidocr_lang_params_maps_chinese_variants():
    assert _rapidocr_lang_params(["en", "ch_sim"]) == ("ch", "ch")
    assert _rapidocr_lang_params(["en", "ch_tra"]) == ("chinese_cht", "chinese_cht")


def test_rapidocr_lang_params_unknown_language_falls_back_to_english():
    assert _rapidocr_lang_params(["fr"]) == ("en", "en")


def test_get_rapidocr_engine_caches_per_language_pair(monkeypatch):
    _rapidocr_engines.clear()
    created: list[dict] = []

    class FakeRapidOCR:
        def __init__(self, params=None):
            created.append(params)

    fake_module = types.ModuleType("rapidocr")
    fake_module.RapidOCR = FakeRapidOCR
    monkeypatch.setitem(sys.modules, "rapidocr", fake_module)

    engine_en = _get_rapidocr_engine("en", "en")
    engine_en_again = _get_rapidocr_engine("en", "en")
    engine_id = _get_rapidocr_engine("id", "id")

    assert engine_en is engine_en_again
    assert engine_en is not engine_id
    assert len(created) == 2
    assert created[0] == {"Det.lang_type": "en", "Rec.lang_type": "en"}
    assert created[1] == {"Det.lang_type": "id", "Rec.lang_type": "id"}

    _rapidocr_engines.clear()


class _FakeEmptyRapidResult:
    """Mirrors RapidOCR's real default-constructed ``RapidOCROutput()`` for
    a page with zero detected text: ``__len__`` is the documented "is this
    actually empty" check, but ``word_results`` is a non-empty *sentinel*
    tuple (``(("", 1.0, None),)``), not ``()``."""

    word_results = (("", 1.0, None),)

    def __len__(self):
        return 0


def test_ocr_image_rapidocr_handles_empty_detection_without_raising(monkeypatch):
    """Regression test: a page/image with no detected text must come back
    as an empty-but-successful result, not raise ``ValueError``. Blindly
    flattening ``word_results``'s empty-result sentinel tuple as if it
    were one real "line" of word tuples used to unpack the sentinel's
    first element (``""``, a 0-length string) into ``(word_text, score,
    box)`` and raise ``ValueError: not enough values to unpack`` — making
    every page with no text on it look like an engine crash."""

    class FakeEngine:
        def __call__(self, *args, **kwargs):
            return _FakeEmptyRapidResult()

    monkeypatch.setattr(engines, "_get_rapidocr_engine", lambda *a, **k: FakeEngine())

    text, conf, words = ocr_image_rapidocr(Image.new("RGB", (10, 10), "white"))

    assert text == ""
    assert conf == 0.0
    assert words == []
