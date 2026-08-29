"""Unit tests for vector-native redaction of digital PDF pages.

Uses real ``fitz`` pages (not doubles) since the whole point of this
module is genuine PDF content mutation (add_redact_annot/apply_
redactions/insert_text) — a fake object couldn't stand in for that.
"""

from types import SimpleNamespace

import fitz

from app.services.redact.pdf_native_redactor import (
    redact_image_regions,
    redact_text_regions,
)

_DPI = 200
_SCALE = _DPI / 72.0  # pixels -> points is the inverse of this


def _region(*, x: int, y: int, w: int, h: int, mock_value: str = "", assignment_source: str = "auto"):
    return SimpleNamespace(
        padded_bbox=SimpleNamespace(x=x, y=y, w=w, h=h),
        mock_value=mock_value,
        assignment_source=assignment_source,
    )


def _pixel_rect(rect: fitz.Rect) -> tuple[int, int, int, int]:
    """A fitz.Rect (points) back to the pixel-space box a region's
    padded_bbox would have produced at ``_DPI`` — inverse of
    ``pdf_native_redactor._bbox_to_rect``, for building test regions
    from a rect ``page.search_for`` already found in point space."""
    return (
        int(rect.x0 * _SCALE),
        int(rect.y0 * _SCALE),
        int((rect.x1 - rect.x0) * _SCALE),
        int((rect.y1 - rect.y0) * _SCALE),
    )


def _new_page(width: float = 300.0, height: float = 150.0) -> fitz.Page:
    doc = fitz.open()
    return doc.new_page(width=width, height=height)


def test_redact_text_regions_removes_underlying_text():
    page = _new_page()
    page.insert_text((10, 30), "Name: John Doe SSN 123-45-6789")
    rect = page.search_for("123-45-6789")[0]
    px, py, pw, ph = _pixel_rect(rect)
    region = _region(x=px, y=py, w=pw, h=ph, mock_value="XXX-XX-0000")

    redact_text_regions(page, [region], _DPI)

    text = page.get_text()
    assert "123-45-6789" not in text
    assert "Name: John Doe SSN" in text


def test_redact_text_regions_draws_mock_value_back_in():
    page = _new_page()
    page.insert_text((10, 30), "Name: John Doe")
    rect = page.search_for("John Doe")[0]
    px, py, pw, ph = _pixel_rect(rect)
    region = _region(x=px, y=py, w=pw, h=ph, mock_value="PERSON_01")

    redact_text_regions(page, [region], _DPI)

    text = page.get_text()
    assert "John Doe" not in text
    assert "PERSON_01" in text


def test_redact_text_regions_skips_empty_mock_value():
    page = _new_page()
    page.insert_text((10, 30), "Secret Value")
    rect = page.search_for("Secret Value")[0]
    px, py, pw, ph = _pixel_rect(rect)
    region = _region(x=px, y=py, w=pw, h=ph, mock_value="")

    redact_text_regions(page, [region], _DPI)

    assert "Secret Value" not in page.get_text()


def test_redact_text_regions_ignores_brand_regions():
    page = _new_page()
    page.insert_text((10, 30), "Keep Me")
    brand_only = [_region(x=0, y=0, w=10, h=10, assignment_source="brand", mock_value="IMAGE")]

    redact_text_regions(page, brand_only, _DPI)

    assert "Keep Me" in page.get_text()


def test_redact_text_regions_noop_for_empty_list():
    page = _new_page()
    page.insert_text((10, 30), "Untouched")
    redact_text_regions(page, [], _DPI)
    assert "Untouched" in page.get_text()


def test_redact_image_regions_ignores_non_brand_regions():
    page = _new_page()
    page.insert_text((10, 30), "Not Brand")
    text_only = [_region(x=0, y=0, w=200, h=100, assignment_source="auto")]

    redact_image_regions(page, text_only, _DPI)

    assert "Not Brand" in page.get_text()


def test_redact_image_regions_blanks_only_covered_pixels_of_larger_image():
    """A background image spanning the full page must survive outside
    the redacted zone — PDF_REDACT_IMAGE_PIXELS, not the default
    PDF_REDACT_IMAGE_REMOVE, which would delete the whole image object.
    """
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    # A solid red image covering the whole page.
    import io as _io

    from PIL import Image as _Image

    buf = _io.BytesIO()
    _Image.new("RGB", (100, 100), (255, 0, 0)).save(buf, format="PNG")
    page.insert_image(fitz.Rect(0, 0, 200, 200), stream=buf.getvalue())

    # Redact only the top-left quadrant (brand/picture zone).
    zone = _region(x=0, y=0, w=int(100 * _SCALE), h=int(100 * _SCALE), assignment_source="brand")
    redact_image_regions(page, [zone], _DPI)

    pix = page.get_pixmap(colorspace=fitz.csRGB)
    # Inside the redacted zone: white.
    inside = pix.pixel(10, 10)
    # Outside the redacted zone: the original image still shows through.
    outside = pix.pixel(150, 150)
    assert inside == (255, 255, 255)
    assert outside != (255, 255, 255)


def test_redact_image_regions_draws_mock_value_when_present():
    page = _new_page()
    zone = _region(x=10, y=10, w=200, h=60, mock_value="FOOTER", assignment_source="brand")
    redact_image_regions(page, [zone], _DPI)
    assert "FOOTER" in page.get_text()


def test_phases_run_in_order_text_then_image_on_same_page():
    """Applying the two phases sequentially (as the renderer does) must
    not have either phase disturb regions the other phase owns."""
    page = _new_page()
    page.insert_text((10, 30), "Jane Roe")
    text_rect = page.search_for("Jane Roe")[0]
    tx, ty, tw, th = _pixel_rect(text_rect)
    text_region = _region(x=tx, y=ty, w=tw, h=th, mock_value="PERSON_02")
    brand_region = _region(x=0, y=100, w=200, h=40, mock_value="", assignment_source="brand")

    redact_text_regions(page, [text_region, brand_region], _DPI)
    redact_image_regions(page, [text_region, brand_region], _DPI)

    text = page.get_text()
    assert "Jane Roe" not in text
    assert "PERSON_02" in text
