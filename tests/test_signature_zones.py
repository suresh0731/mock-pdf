"""Unit tests for ink-gap signature-zone detection — pure OCR-word
geometry, no Docling/OpenCV/ML model. Fixtures mirror the signature-block
templates in ``tests/test_field_extractor.py``.
"""

from __future__ import annotations

from app.models.pii_chunk import BBox
from app.services.ocr.ensemble_types import EnsembleWord
from app.services.pii.brand_zones import BrandZone
from app.services.pii.signature_zones import detect_signature_zones

WordSpec = tuple[str, int, int, int, int]


def _words(specs: list[WordSpec]) -> list[EnsembleWord]:
    words: list[EnsembleWord] = []
    offset = 0
    for text, x, y, w, h in specs:
        words.append(
            EnsembleWord(
                text=text,
                bbox=BBox(x=x, y=y, w=w, h=h),
                ocr_confidence=0.9,
                engine_agreement=1.0,
                engines=["tesseract"],
                char_start=offset,
                char_end=offset + len(text),
            )
        )
        offset += len(text) + 1
    return words


def _xywh(zone: BrandZone) -> tuple[int, int, int, int]:
    return (zone.bbox.x, zone.bbox.y, zone.bbox.w, zone.bbox.h)


# Content row far above the signature block, so it never bounds the gap
# above the signatory name.
_HEADER = [("Invoice", 0, 0, 60, 20), ("Number", 65, 0, 60, 20)]


def test_ink_gap_above_signatory_name_capped_by_row_height():
    """No content directly above the name — gap is capped, not unbounded."""
    specs = [*_HEADER, ("Wahyu", 0, 940, 50, 20), ("Wijaya", 55, 940, 60, 20)]
    words = _words(specs)
    zones = detect_signature_zones(words, page_w=720, page_h=1040, page=0)
    above = [z for z in zones if z.bbox.y + z.bbox.h == 940]
    assert len(above) == 1
    assert above[0].zone == "signature"
    assert above[0].label == "SIGNATURE"
    # Capped at 6x the 20px anchor row height.
    assert above[0].bbox.h == 120


def test_ink_gap_above_bounded_by_nearest_content():
    """A closer line ('Hormat kami') sitting close above bounds the gap
    tightly instead of using the full cap."""
    specs = [
        *_HEADER,
        ("Hormat", 0, 900, 60, 20),
        ("kami", 65, 900, 40, 20),
        ("Wahyu", 0, 940, 50, 20),
        ("Wijaya", 55, 940, 60, 20),
    ]
    words = _words(specs)
    zones = detect_signature_zones(words, page_w=720, page_h=1040, page=0)
    above = [z for z in zones if z.bbox.y + z.bbox.h == 940]
    assert len(above) == 1
    # Gap runs from the closer row's bottom (920) to the name's top (940).
    assert above[0].bbox.y == 920
    assert above[0].bbox.h == 20


def test_ink_gap_below_signatory_name_when_nothing_follows():
    """Org name with nothing printed below it — e.g. 'Acknowledged by' with
    only a signature and blank page beneath. Gap below is still capped."""
    specs = [
        *_HEADER,
        ("PT", 0, 900, 30, 20),
        ("Sinarmas", 35, 900, 70, 20),
        ("Life", 110, 900, 40, 20),
        ("Insurance", 155, 900, 80, 20),
    ]
    words = _words(specs)
    zones = detect_signature_zones(words, page_w=720, page_h=1200, page=0)
    below = [z for z in zones if z.bbox.y == 920]
    assert len(below) == 1
    assert below[0].bbox.h == 120


def test_ink_gap_skipped_when_flush_against_neighboring_content():
    """Name row sits right up against the row above it — no room for ink,
    so no zone is painted there."""
    specs = [
        *_HEADER,
        ("Some", 0, 918, 50, 20),
        ("Label", 55, 918, 50, 20),
        ("Wahyu", 0, 940, 50, 20),
        ("Wijaya", 55, 940, 60, 20),
    ]
    words = _words(specs)
    zones = detect_signature_zones(words, page_w=720, page_h=1040, page=0)
    above = [z for z in zones if z.bbox.y + z.bbox.h == 940]
    assert above == []


def test_ink_gap_skipped_when_covered_by_existing_zone():
    """Docling already caught this signature as a picture block — the
    ink-gap zone that would otherwise be painted above the name is
    dropped so it isn't redacted twice."""
    specs = [*_HEADER, ("Wahyu", 0, 940, 50, 20), ("Wijaya", 55, 940, 60, 20)]
    words = _words(specs)
    existing = [
        BrandZone(zone="picture", page=0, bbox=BBox(x=0, y=820, w=720, h=120), label="IMAGE")
    ]
    zones = detect_signature_zones(
        words, page_w=720, page_h=1040, page=0, existing_zones=existing
    )
    above = [z for z in zones if z.bbox.y + z.bbox.h == 940]
    assert above == []


def test_two_stacked_signatories_each_bound_the_others_shared_gap():
    """Two signatories stacked one above the other: the blank strip between
    them is claimed by both (the first's 'below' gap and the second's
    'above' gap) — safe, harmless overlap rather than a missed signature."""
    specs = [
        *_HEADER,
        ("Indriati", 0, 940, 70, 20),
        ("Kusuma", 75, 940, 60, 20),
        ("Fitri", 0, 1000, 40, 20),
        ("Wahyuni", 45, 1000, 60, 20),
    ]
    words = _words(specs)
    zones = detect_signature_zones(words, page_w=720, page_h=1200, page=0)
    between = [z for z in zones if z.bbox.y == 960 and z.bbox.y + z.bbox.h == 1000]
    assert len(between) == 2
    assert all(z.bbox.h == 40 for z in between)


def test_disabled_returns_empty():
    specs = [*_HEADER, ("Wahyu", 0, 940, 50, 20), ("Wijaya", 55, 940, 60, 20)]
    words = _words(specs)
    assert detect_signature_zones(words, page_w=720, page_h=1040, page=0, enabled=False) == []


def test_no_words_returns_empty():
    assert detect_signature_zones([], page_w=720, page_h=1040, page=0) == []


def test_invalid_page_size_returns_empty():
    specs = [*_HEADER, ("Wahyu", 0, 940, 50, 20), ("Wijaya", 55, 940, 60, 20)]
    words = _words(specs)
    assert detect_signature_zones(words, page_w=0, page_h=1040, page=0) == []
    assert detect_signature_zones(words, page_w=720, page_h=0, page=0) == []


def test_no_signatory_anchor_returns_empty():
    """No bottom-of-page signature block at all — nothing to anchor on."""
    words = _words(_HEADER)
    assert detect_signature_zones(words, page_w=720, page_h=40, page=0) == []


def test_zones_never_overlap_the_anchor_words_own_box():
    specs = [*_HEADER, ("Wahyu", 0, 940, 50, 20), ("Wijaya", 55, 940, 60, 20)]
    words = _words(specs)
    zones = detect_signature_zones(words, page_w=720, page_h=1040, page=0)
    for zone in zones:
        assert zone.bbox.y + zone.bbox.h <= 940 or zone.bbox.y >= 960


# --- Native PDF signature graphics (digital pages) --------------------


def _page_with_png(rect: tuple[float, float, float, float], *, width: float = 300, height: float = 400):
    import io

    import fitz
    from PIL import Image as _Image

    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    buf = io.BytesIO()
    _Image.new("RGB", (80, 40), (0, 0, 180)).save(buf, format="PNG")
    page.insert_image(fitz.Rect(*rect), stream=buf.getvalue())
    return page


def test_native_bottom_image_is_signature_zone_even_without_name_anchor():
    """A signature/stamp image on a digital page must be boxed even when
    no signatory name was found (or stamp text would have collapsed the
    ink-gap). dpi=72 so pixel space == PDF points."""
    page = _page_with_png((40, 320, 160, 380))
    zones = detect_signature_zones([], page_w=300, page_h=400, page=0, fitz_page=page, dpi=72)
    assert len(zones) == 1
    assert zones[0].zone == "signature"
    assert zones[0].bbox.y >= 300
    assert zones[0].bbox.h >= 40


def test_native_top_logo_image_is_not_a_signature_zone():
    page = _page_with_png((40, 20, 160, 60))
    zones = detect_signature_zones([], page_w=300, page_h=400, page=0, fitz_page=page, dpi=72)
    assert zones == []


def test_native_full_page_image_is_not_a_signature_zone():
    page = _page_with_png((0, 0, 300, 400), width=300, height=400)
    zones = detect_signature_zones([], page_w=300, page_h=400, page=0, fitz_page=page, dpi=72)
    assert zones == []


def test_native_stamp_annot_is_signature_zone():
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=300, height=400)
    page.add_stamp_annot(fitz.Rect(50, 50, 180, 120))
    zones = detect_signature_zones([], page_w=300, page_h=400, page=0, fitz_page=page, dpi=72)
    assert len(zones) == 1
    assert zones[0].zone == "signature"


def test_native_ink_annot_is_signature_zone():
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=300, height=400)
    page.add_ink_annot([[(40, 300), (70, 310), (110, 305), (150, 325), (180, 315)]])
    zones = detect_signature_zones([], page_w=300, page_h=400, page=0, fitz_page=page, dpi=72)
    assert len(zones) == 1
    assert zones[0].zone == "signature"


def test_stamp_text_does_not_drop_native_signature_image():
    """Words sitting on the stamp (native-extracted seal text) would
    shrink the ink-gap to nothing; the image zone must still fire."""
    specs = [
        *_HEADER,
        ("BNI", 70, 330, 40, 16),
        ("Isobelle", 40, 370, 70, 16),
        ("Natali", 115, 370, 50, 16),
    ]
    page = _page_with_png((40, 300, 180, 365))
    zones = detect_signature_zones(
        _words(specs), page_w=300, page_h=400, page=0, fitz_page=page, dpi=72
    )
    image_zones = [z for z in zones if z.bbox.y < 370 and z.bbox.h >= 40]
    assert image_zones, "expected the native signature image to be zoned"
