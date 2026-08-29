"""Unit tests for the painted PDF renderer (white fill + black mock)."""

import logging
from types import SimpleNamespace

import fitz
from PIL import Image

from app.services.redact.pdf_renderer import (
    PageRenderInput,
    _regions_in_draw_order,
    _sanitize_filename,
    render_redacted_pdf,
)

_DPI = 200
_SCALE = _DPI / 72.0


def _pixel_rect(rect: fitz.Rect) -> tuple[int, int, int, int]:
    return (
        int(rect.x0 * _SCALE),
        int(rect.y0 * _SCALE),
        int((rect.x1 - rect.x0) * _SCALE),
        int((rect.y1 - rect.y0) * _SCALE),
    )


def _page(w: int = 120, h: int = 80, color: str = "white") -> Image.Image:
    return Image.new("RGB", (w, h), color)


def _region(
    *,
    page: int = 0,
    x: int,
    y: int,
    w: int,
    h: int,
    mock_value: str = "XXX",
    assignment_source: str = "auto",
    **extra: object,
) -> SimpleNamespace:
    return SimpleNamespace(
        page=page,
        padded_bbox=SimpleNamespace(x=x, y=y, w=w, h=h),
        mock_value=mock_value,
        assignment_source=assignment_source,
        **extra,
    )


def _pdf_to_image(pdf_bytes: bytes, page: int = 0) -> Image.Image:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pix = doc[page].get_pixmap(colorspace=fitz.csRGB, alpha=False)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc.close()


def test_render_paints_white_fill_over_original_color():
    page = _page()
    page.paste((255, 0, 0), [10, 20, 90, 44])
    region = _region(x=10, y=20, w=80, h=24)
    pdf = render_redacted_pdf([page], [region], "doc.pdf")
    img = _pdf_to_image(pdf)
    pixel = img.getpixel((12, 22))
    assert all(channel > 230 for channel in pixel)


def test_render_paints_black_mock_in_box():
    page = _page()
    page.paste((255, 0, 0), [10, 20, 90, 44])
    region = _region(x=10, y=20, w=80, h=24)
    pdf = render_redacted_pdf([page], [region], "doc.pdf")
    img = _pdf_to_image(pdf)
    has_black = False
    for px in range(10, 90):
        for py in range(20, 44):
            r, g, b = img.getpixel((px, py))
            if r < 25 and g < 25 and b < 25:
                has_black = True
                break
        if has_black:
            break
    assert has_black


def test_paint_white_only_when_mock_value_missing():
    page = _page()
    page.paste((255, 0, 0), [10, 20, 90, 44])
    region = SimpleNamespace(
        page=0,
        padded_bbox=SimpleNamespace(x=10, y=20, w=80, h=24),
    )
    pdf = render_redacted_pdf([page], [region], "doc.pdf")
    img = _pdf_to_image(pdf)
    pixel = img.getpixel((12, 22))
    assert all(channel > 230 for channel in pixel)


def test_tiny_bbox_does_not_crash():
    page = _page()
    region = _region(x=0, y=0, w=1, h=1, mock_value="XXX")
    pdf = render_redacted_pdf([page], [region], "doc.pdf")
    doc = fitz.open(stream=pdf, filetype="pdf")
    try:
        assert len(doc) == 1
    finally:
        doc.close()


def test_zero_area_bbox_skipped():
    page = _page()
    zero_w = _region(x=10, y=10, w=0, h=20)
    zero_h = _region(x=10, y=10, w=20, h=0)
    pdf = render_redacted_pdf([page], [zero_w, zero_h], "doc.pdf")
    doc = fitz.open(stream=pdf, filetype="pdf")
    try:
        assert len(doc) == 1
    finally:
        doc.close()


def test_patch_box_has_no_black_outline():
    """Table gridlines stay visible: the painted patch is a white fill
    with no hairline border sitting on the cell rule."""
    page = _page(color="gray")
    region = _region(x=20, y=20, w=40, h=20, mock_value="")
    pdf = render_redacted_pdf([page], [region], "doc.pdf")
    img = _pdf_to_image(pdf)
    edge = img.getpixel((20, 20))
    assert all(channel > 230 for channel in edge)
    brand = _region(x=0, y=0, w=10, h=10, assignment_source="brand")
    text = _region(x=10, y=0, w=10, h=10, assignment_source="auto")
    brand2 = _region(x=20, y=0, w=10, h=10, assignment_source="brand")
    ordered = _regions_in_draw_order([brand, text, brand2])
    assert ordered == [text, brand, brand2]


def test_brand_drawn_last_over_overlapping_text():
    page = _page()
    text = _region(x=10, y=10, w=60, h=28, mock_value="XXX", assignment_source="auto")
    brand = _region(x=30, y=10, w=60, h=28, mock_value="", assignment_source="brand")
    pdf = render_redacted_pdf([page], [text, brand], "doc.pdf")
    img = _pdf_to_image(pdf)
    # Overlap is x=30..70, y=10..38; center ~ (50, 24). The brand region has
    # no mock text, so it repaints a plain white rect over any black "XXX"
    # ink from the text region drawn underneath it.
    pixel = img.getpixel((50, 24))
    assert all(channel > 230 for channel in pixel)


def test_empty_redactions_preserves_original_page():
    page = _page(80, 50, "red")
    pdf = render_redacted_pdf([page], [], "doc.pdf")
    img = _pdf_to_image(pdf)
    pixel = img.getpixel((40, 25))
    assert pixel[0] > 230
    assert pixel[1] < 25
    assert pixel[2] < 25


def test_multi_page_count_matches():
    pages = [_page(color="red"), _page(color="blue")]
    pdf = render_redacted_pdf(pages, [], "doc.pdf")
    doc = fitz.open(stream=pdf, filetype="pdf")
    try:
        assert len(doc) == 2
    finally:
        doc.close()


def test_logs_omit_source_text(caplog):
    page = _page()
    region = _region(
        x=10,
        y=20,
        w=80,
        h=24,
        mock_value="XXX",
        source_text="S1234567A",
    )
    with caplog.at_level(logging.INFO, logger="app.services.redact.pdf_renderer"):
        render_redacted_pdf([page], [region], "invoice.pdf")
    log_text = caplog.text
    assert "source_text" not in log_text
    assert "S1234567A" not in log_text
    assert "filename_hash" in log_text


def test_sanitize_filename_strips_paths_and_truncates():
    assert "/" not in _sanitize_filename("../../etc/passwd")
    assert "\\" not in _sanitize_filename("../../etc/passwd")
    long_name = "a" * 300
    sanitized_long = _sanitize_filename(long_name)
    assert len(sanitized_long) <= 255
    assert _sanitize_filename("") == "document"
    assert _sanitize_filename(".") == "document"


def test_pdf_metadata_title_is_sanitized():
    page = _page()
    pdf = render_redacted_pdf([page], [], "..\\secret\\invoice.pdf")
    doc = fitz.open(stream=pdf, filetype="pdf")
    try:
        assert doc.metadata["title"] == "invoice.pdf"
    finally:
        doc.close()


def test_default_jpeg_output_is_much_smaller_than_png():
    """Regression guard for the 200KB-in/22MB-out size blowup: a
    lossless-PNG-embedded page is many times larger than the same page
    saved as JPEG, so the default must stay "jpeg"."""
    page = _page(800, 1000, "white")
    region = _region(x=100, y=100, w=300, h=40)
    jpeg_pdf = render_redacted_pdf([page], [region], "doc.pdf", image_format="jpeg")
    png_pdf = render_redacted_pdf([page], [region], "doc.pdf", image_format="png")
    assert len(jpeg_pdf) < len(png_pdf)


def test_png_format_still_supported_for_lossless_output():
    page = _page()
    page.paste((255, 0, 0), [10, 20, 90, 44])
    region = _region(x=10, y=20, w=80, h=24)
    pdf = render_redacted_pdf([page], [region], "doc.pdf", image_format="png")
    img = _pdf_to_image(pdf)
    pixel = img.getpixel((12, 22))
    assert all(channel > 230 for channel in pixel)


def test_unknown_image_format_falls_back_to_jpeg(caplog):
    page = _page()
    with caplog.at_level(logging.WARNING, logger="app.services.redact.pdf_renderer"):
        pdf = render_redacted_pdf([page], [], "doc.pdf", image_format="bmp")
    assert "unknown_redact_image_format" in caplog.text
    doc = fitz.open(stream=pdf, filetype="pdf")
    try:
        assert len(doc) == 1
    finally:
        doc.close()


def test_jpeg_quality_affects_output_size():
    page = _page(800, 1000, "white")
    region = _region(x=100, y=100, w=300, h=40)
    low_quality = render_redacted_pdf([page], [region], "doc.pdf", jpeg_quality=10)
    high_quality = render_redacted_pdf([page], [region], "doc.pdf", jpeg_quality=95)
    assert len(low_quality) < len(high_quality)


def _digital_fitz_page(text: str, width: float = 300.0, height: float = 150.0) -> fitz.Page:
    src_doc = fitz.open()
    page = src_doc.new_page(width=width, height=height)
    page.insert_text((10, 30), text)
    return page


def test_digital_page_kind_stays_vector_with_selectable_text():
    """A "digital" PageRenderInput must come out as a real vector page:
    non-redacted text stays selectable, the redacted span is gone, and
    the mock value is drawn back in as real text — not a rasterized
    JPEG page like the scanned path produces."""
    fitz_page = _digital_fitz_page("Account Name: John Doe")
    rect = fitz_page.search_for("John Doe")[0]
    px, py, pw, ph = _pixel_rect(rect)
    region = _region(page=0, x=px, y=py, w=pw, h=ph, mock_value="PERSON_01")

    page_input = PageRenderInput(image=_page(), page_kind="digital", fitz_page=fitz_page, dpi=_DPI)
    pdf = render_redacted_pdf([page_input], [region], "doc.pdf")

    doc = fitz.open(stream=pdf, filetype="pdf")
    try:
        assert len(doc) == 1
        text = doc[0].get_text()
        assert "Account Name" in text
        assert "John Doe" not in text
        assert "PERSON_01" in text
    finally:
        doc.close()


def test_scanned_page_kind_still_flattens_to_raster():
    """A "scanned" PageRenderInput (or one with no fitz_page) must keep
    getting the paint-then-flatten treatment even when passed through
    the new PageRenderInput shape."""
    image = _page(200, 100, "white")
    image.paste((255, 0, 0), [10, 20, 90, 44])
    region = _region(page=0, x=10, y=20, w=80, h=24)

    page_input = PageRenderInput(image=image, page_kind="scanned", fitz_page=None)
    pdf = render_redacted_pdf([page_input], [region], "doc.pdf")

    img = _pdf_to_image(pdf)
    pixel = img.getpixel((12, 22))
    assert all(channel > 230 for channel in pixel)


def test_mixed_digital_and_scanned_pages_assemble_in_correct_order():
    """Page 0 digital + page 1 scanned must come out as [vector page,
    image page] in that exact order — the per-page branching this
    module exists to add."""
    fitz_page = _digital_fitz_page("Digital Page One")
    digital_input = PageRenderInput(image=_page(), page_kind="digital", fitz_page=fitz_page, dpi=_DPI)
    scanned_input = PageRenderInput(image=_page(color="blue"), page_kind="scanned", fitz_page=None)

    pdf = render_redacted_pdf([digital_input, scanned_input], [], "doc.pdf")

    doc = fitz.open(stream=pdf, filetype="pdf")
    try:
        assert len(doc) == 2
        # Page 0 stayed real vector content: its original text survives
        # verbatim (nothing was redacted on it) and it carries no raster
        # image the way a flattened page always would.
        assert "Digital Page One" in doc[0].get_text()
        assert len(doc[0].get_images()) == 0
        # Page 1 is the flattened raster page: no extractable text, one
        # embedded image.
        assert doc[1].get_text().strip() == ""
        assert len(doc[1].get_images()) == 1
    finally:
        doc.close()


def test_vector_redaction_failure_falls_back_to_raster_for_that_page(caplog):
    """If a "digital" page's fitz_page can't actually be redacted as
    vector content (missing .parent/.number — should never happen for a
    real fitz.Page, but must not crash the whole render if it somehow
    does), that one page still comes out as a raster page rather than
    aborting the document."""
    image = _page(color="green")
    broken_fitz_page = SimpleNamespace()  # no .parent / .number
    page_input = PageRenderInput(image=image, page_kind="digital", fitz_page=broken_fitz_page, dpi=_DPI)

    with caplog.at_level(logging.WARNING, logger="app.services.redact.pdf_renderer"):
        pdf = render_redacted_pdf([page_input], [], "doc.pdf")

    assert "vector_redaction_failed" in caplog.text
    doc = fitz.open(stream=pdf, filetype="pdf")
    try:
        assert len(doc) == 1
        assert len(doc[0].get_images()) == 1
    finally:
        doc.close()
