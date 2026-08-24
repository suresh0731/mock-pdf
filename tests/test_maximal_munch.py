"""Unit tests for the lookahead/lookbehind maximal-munch window-extension
probe (see ``app.pipeline.redact._resolve_maximal_munch_window`` and its
helpers), which resolves prefix-collision ambiguities like "Maksima" vs.
"Maksima Plus" using OCR geometry.
"""

from __future__ import annotations

from app.models.pii_chunk import BBox
from app.pipeline.redact import (
    _build_window_candidates,
    _nearest_extension_word,
    _resolve_maximal_munch_window,
)
from app.services.ocr.ensemble_types import EnsembleWord


def _word(text: str, x: int, y: int, w: int = 30, h: int = 10) -> EnsembleWord:
    return EnsembleWord(
        text=text,
        bbox=BBox(x=x, y=y, w=w, h=h),
        ocr_confidence=0.9,
        engine_agreement=1.0,
        engines=["tesseract"],
    )


class _FakeStore:
    """Minimal stand-in exposing only ``best_match_score``, keyed by the
    exact joined window text the probe is expected to score."""

    def __init__(self, scores: dict[str, float]):
        self._scores = scores

    def best_match_score(self, normalized: str, entity_type: str, field_role: str | None = None) -> float:
        return self._scores.get(normalized, 0.0)


# --- _nearest_extension_word ---------------------------------------------


def test_nearest_extension_word_picks_same_row_right_neighbor():
    bbox = BBox(x=0, y=20, w=30, h=10)
    right = _word("Plus", 40, 20)
    below = _word("Other", 0, 40)
    assert _nearest_extension_word(bbox, [right, below], excluded_ids=set()) is right


def test_nearest_extension_word_falls_back_to_below_when_no_row_neighbor():
    bbox = BBox(x=0, y=20, w=30, h=10)
    below = _word("Plus", 0, 40)
    assert _nearest_extension_word(bbox, [below], excluded_ids=set()) is below


def test_nearest_extension_word_ignores_words_to_the_left():
    bbox = BBox(x=100, y=20, w=30, h=10)
    left = _word("Before", 0, 20)
    assert _nearest_extension_word(bbox, [left], excluded_ids=set()) is None


def test_nearest_extension_word_excludes_ids():
    bbox = BBox(x=0, y=20, w=30, h=10)
    right = _word("Plus", 40, 20)
    assert _nearest_extension_word(bbox, [right], excluded_ids={id(right)}) is None


def test_nearest_extension_word_returns_none_when_no_candidates():
    bbox = BBox(x=0, y=20, w=30, h=10)
    assert _nearest_extension_word(bbox, [], excluded_ids=set()) is None


def test_nearest_extension_word_prefers_nearest_of_several_row_candidates():
    bbox = BBox(x=0, y=20, w=30, h=10)
    near = _word("Near", 40, 20)
    far = _word("Far", 200, 20)
    assert _nearest_extension_word(bbox, [far, near], excluded_ids=set()) is near


# --- _build_window_candidates --------------------------------------------


def test_build_window_candidates_empty_input_returns_empty():
    assert _build_window_candidates([], []) == []


def test_build_window_candidates_no_extension_available_returns_original_only():
    words = [_word("Maksima", 0, 20)]
    assert _build_window_candidates(words, words) == [words]


def test_build_window_candidates_extends_with_adjacent_word():
    maksima = _word("Maksima", 0, 20)
    plus = _word("Plus", 40, 20)
    windows = _build_window_candidates([maksima], [maksima, plus])
    assert [maksima] in windows
    assert [maksima, plus] in windows


def test_build_window_candidates_stops_when_max_extra_reached():
    words = [_word(f"w{i}", i * 40, 20) for i in range(6)]
    windows = _build_window_candidates([words[0]], words, max_extra=2)
    lengths = sorted(len(w) for w in windows)
    assert lengths == [1, 2, 3]


def test_build_window_candidates_includes_trimmed_variant_for_multiword_input():
    a = _word("A", 0, 20)
    b = _word("B", 40, 20)
    windows = _build_window_candidates([a, b], [a, b])
    assert [a] in windows


def test_build_window_candidates_no_trim_for_single_word_input():
    a = _word("A", 0, 20)
    windows = _build_window_candidates([a], [a])
    assert [] not in windows


# --- _resolve_maximal_munch_window ----------------------------------------


def test_resolve_prefers_longer_window_within_margin():
    maksima = _word("Maksima", 0, 20)
    plus = _word("Plus", 100, 20)
    store = _FakeStore({"maksima": 0.8, "maksima plus": 1.0})
    words, text = _resolve_maximal_munch_window(
        [maksima], [maksima, plus], "ORG", None, store
    )
    assert text == "Maksima Plus"
    assert words == [maksima, plus]


def test_resolve_picks_decisively_better_shorter_window_over_longer():
    maksima = _word("Maksima", 0, 20)
    plus = _word("Plus", 100, 20)
    store = _FakeStore({"maksima": 1.0, "maksima plus": 0.5})
    words, text = _resolve_maximal_munch_window(
        [maksima], [maksima, plus], "ORG", None, store
    )
    assert text == "Maksima"
    assert words == [maksima]


def test_resolve_drops_spurious_trailing_word_via_trim():
    a = _word("A", 0, 20)
    b = _word("B", 40, 20)
    store = _FakeStore({"a b": 0.4, "a": 0.95})
    words, text = _resolve_maximal_munch_window([a, b], [a, b], "ORG", None, store)
    assert text == "A"
    assert words == [a]


def test_resolve_falls_back_to_original_when_no_windows():
    words, text = _resolve_maximal_munch_window([], [], "ORG", None, _FakeStore({}))
    assert words == []
    assert text == ""


def test_resolve_ties_break_toward_longer_window():
    maksima = _word("Maksima", 0, 20)
    plus = _word("Plus", 100, 20)
    store = _FakeStore({"maksima": 1.0, "maksima plus": 1.0})
    words, text = _resolve_maximal_munch_window(
        [maksima], [maksima, plus], "ORG", None, store
    )
    assert text == "Maksima Plus"
