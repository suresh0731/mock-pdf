import sys
import types

from app.services.ocr.engines import (
    _get_rapidocr_engine,
    _rapidocr_engines,
    _rapidocr_lang_params,
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
