import logging

from app.models.pii_chunk import BBox
from app.services.pii.brand_zones import (
    BrandZone,
    detect_brand_zones,
    detect_picture_zones,
    reconcile_picture_zones_with_text,
)
from app.services.structure.docling_adapter import DocBlock


def _block(block_type: str, bbox: BBox, text: str) -> DocBlock:
    return DocBlock(block_id="dl-0", block_type=block_type, bbox=bbox, text=text)


def _xywh(zone: BrandZone) -> tuple[int, int, int, int]:
    return (zone.bbox.x, zone.bbox.y, zone.bbox.w, zone.bbox.h)


def test_detect_default_zones_720x1100():
    zones = detect_brand_zones(720, 1100, page=0)
    assert len(zones) == 1
    footer = zones[0]
    assert footer.zone == "footer"
    assert footer.label == "FOOTER"
    assert footer.page == 0
    assert _xywh(footer) == (0, 968, 720, 132)


def test_detect_custom_percents_as_args():
    zones = detect_brand_zones(1000, 1000, page=0, footer_bottom_pct=0.05)
    footer = zones[0]
    assert _xywh(footer) == (0, 950, 1000, 50)


def test_brand_zone_is_not_mock_entry():
    zones = detect_brand_zones(720, 1100, page=0)
    assert zones
    for zone in zones:
        assert isinstance(zone, BrandZone)
        assert not hasattr(zone, "source_text")
        assert not hasattr(zone, "mapping_id")


def test_detect_unions_footer_block_into_footer():
    footer_block = _block("footer", BBox(x=50, y=950, w=200, h=80), "addr")
    zones = detect_brand_zones(720, 1100, page=0, blocks=[footer_block])
    footer = next(z for z in zones if z.zone == "footer")
    assert _xywh(footer) == (0, 950, 720, 150)


def test_detect_unions_overlapping_picture_into_footer():
    """A picture that pokes above the footer seed's top edge grows the
    zone upward to cover it (footer already spans the full page width)."""
    picture = _block("picture", BBox(x=500, y=900, w=180, h=100), "BRAND_MARK")
    zones = detect_brand_zones(720, 1100, page=0, blocks=[picture])
    footer = next(z for z in zones if z.zone == "footer")
    assert _xywh(footer) == (0, 900, 720, 200)


def test_detect_ignores_paragraph_overlap():
    paragraph = _block("paragraph", BBox(x=500, y=900, w=200, h=100), "body")
    zones = detect_brand_zones(720, 1100, page=0, blocks=[paragraph])
    footer = next(z for z in zones if z.zone == "footer")
    assert _xywh(footer) == (0, 968, 720, 132)


def test_detect_ignores_header_overlap():
    """Header blocks no longer feed into any zone now that the logo seed
    (the only zone a header could plausibly overlap) has been removed —
    even one that spatially overlaps the footer seed is ignored."""
    header = _block("header", BBox(x=400, y=1000, w=320, h=140), "letterhead")
    zones = detect_brand_zones(720, 1100, page=0, blocks=[header])
    footer = next(z for z in zones if z.zone == "footer")
    assert _xywh(footer) == (0, 968, 720, 132)


def test_detect_clamps_union_to_page_bounds():
    picture = _block("picture", BBox(x=700, y=1050, w=80, h=80), "mark")
    zones = detect_brand_zones(720, 1100, page=0, blocks=[picture])
    for zone in zones:
        box = zone.bbox
        assert box.x >= 0
        assert box.y >= 0
        assert box.x + box.w <= 720
        assert box.y + box.h <= 1100


def test_detect_none_blocks_equals_empty_list():
    none_zones = detect_brand_zones(720, 1100, page=0, blocks=None)
    empty_zones = detect_brand_zones(720, 1100, page=0, blocks=[])
    assert none_zones == empty_zones


def test_detect_omits_footer_when_patch_footer_false():
    zones = detect_brand_zones(720, 1100, page=0, patch_footer=False)
    assert zones == []


def test_detect_returns_empty_for_zero_page_size():
    assert detect_brand_zones(0, 1100, page=0) == []
    assert detect_brand_zones(720, 0, page=1) == []


def test_detect_returns_empty_for_negative_page_size():
    assert detect_brand_zones(-10, 1100, page=0) == []


def test_detect_logs_omit_block_text(caplog):
    secret = "John Smith NRIC S1234567A"
    picture = _block("picture", BBox(x=500, y=20, w=180, h=150), secret)
    with caplog.at_level(logging.DEBUG, logger="app.services.pii.brand_zones"):
        detect_brand_zones(720, 1100, page=0, blocks=[picture])
    assert "John Smith" not in caplog.text
    assert "S1234567A" not in caplog.text
    assert secret not in caplog.text


# --- detect_picture_zones: position-agnostic generic image/graphic redaction ---


def test_detect_picture_zones_mid_page_block_gets_own_zone():
    """A picture sitting nowhere near the logo/footer seed rectangles
    (e.g. a mid-page stamp or signature scan) still gets its own zone —
    the whole point of not being tied to one template's fixed layout."""
    mid_picture = _block("picture", BBox(x=100, y=500, w=200, h=150), "stamp")
    zones = detect_picture_zones(720, 1100, page=0, blocks=[mid_picture])
    assert len(zones) == 1
    zone = zones[0]
    assert zone.zone == "picture"
    assert zone.label == "IMAGE"
    assert zone.page == 0
    assert _xywh(zone) == (100, 500, 200, 150)


def test_detect_picture_zones_skips_tiny_blocks_below_min_area():
    tiny_icon = _block("picture", BBox(x=100, y=500, w=10, h=10), "icon")
    zones = detect_picture_zones(720, 1100, page=0, blocks=[tiny_icon], min_area_pct=0.0015)
    assert zones == []


def test_detect_picture_zones_keeps_blocks_at_or_above_min_area():
    block = _block("picture", BBox(x=100, y=500, w=40, h=40), "icon")
    zones = detect_picture_zones(720, 1100, page=0, blocks=[block], min_area_pct=0.0015)
    assert len(zones) == 1


def test_detect_picture_zones_dedupes_against_existing_zone():
    """A picture already unioned into the footer zone by detect_brand_zones
    must not also get painted a second time as its own picture zone."""
    footer_zone = BrandZone(zone="footer", page=0, bbox=BBox(x=0, y=900, w=720, h=200), label="FOOTER")
    picture_inside_footer = _block("picture", BBox(x=500, y=950, w=100, h=100), "brand mark")
    zones = detect_picture_zones(
        720, 1100, page=0, blocks=[picture_inside_footer], existing_zones=[footer_zone]
    )
    assert zones == []


def test_detect_picture_zones_keeps_block_not_covered_by_existing_zone():
    footer_zone = BrandZone(zone="footer", page=0, bbox=BBox(x=0, y=968, w=720, h=132), label="FOOTER")
    mid_picture = _block("picture", BBox(x=100, y=500, w=200, h=150), "stamp")
    zones = detect_picture_zones(
        720, 1100, page=0, blocks=[mid_picture], existing_zones=[footer_zone]
    )
    assert len(zones) == 1


def test_detect_picture_zones_disabled_returns_empty():
    block = _block("picture", BBox(x=100, y=500, w=200, h=150), "stamp")
    assert detect_picture_zones(720, 1100, page=0, blocks=[block], enabled=False) == []


def test_detect_picture_zones_ignores_non_picture_blocks():
    paragraph = _block("paragraph", BBox(x=100, y=500, w=200, h=150), "body")
    assert detect_picture_zones(720, 1100, page=0, blocks=[paragraph]) == []


def test_detect_picture_zones_returns_empty_for_invalid_page_size():
    block = _block("picture", BBox(x=100, y=500, w=200, h=150), "stamp")
    assert detect_picture_zones(0, 1100, page=0, blocks=[block]) == []
    assert detect_picture_zones(720, 0, page=0, blocks=[block]) == []


def test_detect_picture_zones_none_blocks_equals_empty_list():
    assert detect_picture_zones(720, 1100, page=0, blocks=None) == []
    assert detect_picture_zones(720, 1100, page=0, blocks=[]) == []


def test_detect_picture_zones_clamps_to_page_bounds():
    off_page = _block("picture", BBox(x=650, y=-20, w=150, h=80), "mark")
    zones = detect_picture_zones(720, 1100, page=0, blocks=[off_page])
    assert len(zones) == 1
    box = zones[0].bbox
    assert box.x >= 0
    assert box.y >= 0
    assert box.x + box.w <= 720
    assert box.y + box.h <= 1100


def test_detect_picture_zones_logs_omit_block_text(caplog):
    secret = "John Smith NRIC S1234567A"
    block = _block("picture", BBox(x=100, y=500, w=200, h=150), secret)
    with caplog.at_level(logging.DEBUG, logger="app.services.pii.brand_zones"):
        detect_picture_zones(720, 1100, page=0, blocks=[block])
    assert secret not in caplog.text


def test_detect_picture_zones_skips_picture_inside_table_cell():
    """Bank-column logos sit in a cell; the word patch already covers them."""
    cell = _block("cell", BBox(x=400, y=200, w=80, h=24), "")
    logo = _block("picture", BBox(x=410, y=204, w=40, h=16), "scb-logo")
    zones = detect_picture_zones(720, 1100, page=0, blocks=[cell, logo])
    assert zones == []


def test_detect_picture_zones_keeps_picture_outside_cells():
    cell = _block("cell", BBox(x=400, y=200, w=80, h=24), "")
    stamp = _block("picture", BBox(x=100, y=500, w=200, h=150), "stamp")
    zones = detect_picture_zones(720, 1100, page=0, blocks=[cell, stamp])
    assert len(zones) == 1
    assert _xywh(zones[0]) == (100, 500, 200, 150)


# --- reconcile_picture_zones_with_text -----------------------------------


def _picture_zone(x: int, y: int, w: int, h: int) -> BrandZone:
    return BrandZone(zone="picture", page=0, bbox=BBox(x=x, y=y, w=w, h=h), label="IMAGE")


def test_reconcile_leaves_footer_unchanged():
    footer = BrandZone(zone="footer", page=0, bbox=BBox(x=0, y=968, w=720, h=132), label="FOOTER")
    word = BBox(x=10, y=980, w=80, h=16)
    assert reconcile_picture_zones_with_text([footer], [word]) == [footer]


def test_reconcile_skips_picture_mostly_overlapping_word_patch():
    """Tiny bank logo sitting on DSDC_Bank — IMAGE would paint over the mock."""
    picture = _picture_zone(10, 10, 30, 16)
    word = BBox(x=8, y=8, w=80, h=20)
    assert reconcile_picture_zones_with_text([picture], [word]) == []


def test_reconcile_keeps_picture_when_word_is_inside_it():
    """OCR of logo text is a small word inside a large stamp — IMAGE covers it."""
    picture = _picture_zone(100, 500, 200, 150)
    word_inside = BBox(x=140, y=560, w=40, h=16)
    result = reconcile_picture_zones_with_text([picture], [word_inside])
    assert len(result) == 1
    assert _xywh(result[0]) == (100, 500, 200, 150)


def test_reconcile_clips_picture_off_adjacent_word_patch():
    """Signature IMAGE overlapping the mocked name below it is shrunk up."""
    picture = _picture_zone(100, 400, 120, 80)
    name_below = BBox(x=100, y=460, w=120, h=20)
    result = reconcile_picture_zones_with_text([picture], [name_below])
    assert len(result) == 1
    box = result[0].bbox
    assert box.y == 400
    assert box.y + box.h <= name_below.y
    assert box.h == 60


def test_reconcile_empty_text_boxes_returns_zones_unchanged():
    picture = _picture_zone(100, 500, 200, 150)
    assert reconcile_picture_zones_with_text([picture], []) == [picture]
