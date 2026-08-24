"""Tests for gridline/rule-line removal preprocessing (OCR artifact fix)."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from app.services.preprocess.line_removal import strip_table_lines


def _blank_page(size: tuple[int, int] = (400, 300)) -> Image.Image:
    return Image.new("RGB", size, color="white")


def test_strip_table_lines_removes_long_horizontal_line() -> None:
    image = _blank_page()
    draw = ImageDraw.Draw(image)
    # A long thin horizontal rule, well past the default min_line_length_px.
    draw.line([(20, 150), (300, 150)], fill="black", width=1)

    cleaned = strip_table_lines(image)
    arr = np.array(cleaned)

    # Sample exactly the line's own row so a thin residual can't be diluted
    # away by surrounding whitespace in the mean.
    line_row = arr[150, 20:300]
    assert line_row.mean() > 200  # inpainted back toward white background


def test_strip_table_lines_removes_thin_one_pixel_line() -> None:
    """Real scanner gridlines are often exactly 1px thick — must still be caught."""
    image = _blank_page()
    draw = ImageDraw.Draw(image)
    draw.line([(20, 150), (300, 150)], fill="black", width=1)

    cleaned = strip_table_lines(image)
    arr = np.array(cleaned)
    assert arr[150, 160].tolist() != [0, 0, 0]


def test_strip_table_lines_removes_long_vertical_line() -> None:
    image = _blank_page()
    draw = ImageDraw.Draw(image)
    draw.line([(200, 20), (200, 280)], fill="black", width=1)

    cleaned = strip_table_lines(image)
    arr = np.array(cleaned)

    line_col = arr[20:280, 200]
    assert line_col.mean() > 200


def test_strip_table_lines_preserves_short_text_like_glyphs() -> None:
    image = _blank_page()
    draw = ImageDraw.Draw(image)
    # A small filled block standing in for a character glyph — far shorter
    # than min_line_length_px in both dimensions, so it must survive.
    draw.rectangle([50, 50, 62, 65], fill="black")

    cleaned = strip_table_lines(image)
    arr = np.array(cleaned)

    glyph_region = arr[50:66, 50:63]
    assert glyph_region.mean() < 100  # still mostly black, i.e. preserved


def test_strip_table_lines_preserves_glyph_next_to_a_line() -> None:
    """The exact failure mode reported: a table border must not eat into text."""
    image = _blank_page((400, 200))
    draw = ImageDraw.Draw(image)
    draw.line([(0, 100), (400, 100)], fill="black", width=1)
    # A short, narrow glyph-like mark sitting well clear of the rule line —
    # a wide filled block would itself look like a horizontal line to the
    # morphological-opening detector, so this stays glyph-sized on purpose.
    draw.rectangle([150, 60, 162, 78], fill="black")

    cleaned = strip_table_lines(image)
    arr = np.array(cleaned)

    word_region = arr[60:78, 150:162]
    assert word_region.mean() < 100
    assert arr[100, 200].tolist() != [0, 0, 0]


def test_strip_table_lines_blank_page_returns_unchanged() -> None:
    image = _blank_page()
    cleaned = strip_table_lines(image)
    arr = np.array(cleaned)
    assert arr.mean() > 250  # still essentially all-white, nothing erased


def test_strip_table_lines_returns_rgb_regardless_of_input_mode() -> None:
    grayscale = Image.new("L", (100, 100), color=255)
    cleaned = strip_table_lines(grayscale)
    assert cleaned.mode == "RGB"


def test_strip_table_lines_mostly_foreground_image_is_skipped() -> None:
    """A near-solid-black page must not be treated as 'all lines' and erased."""
    image = Image.new("RGB", (200, 200), color="black")
    cleaned = strip_table_lines(image)
    arr = np.array(cleaned)
    # Skipped (mask_fraction > 0.5 guard) means original content is kept as-is.
    assert arr.mean() < 50


def test_strip_table_lines_respects_min_line_length_threshold() -> None:
    image = _blank_page()
    draw = ImageDraw.Draw(image)
    # A short stroke below the custom min_line_length_px must be preserved.
    draw.line([(50, 50), (65, 50)], fill="black", width=1)

    cleaned = strip_table_lines(image, min_line_length_px=40)
    arr = np.array(cleaned)
    stroke_row = arr[50, 50:66]
    assert stroke_row.mean() < 200
