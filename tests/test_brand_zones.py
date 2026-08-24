import logging

from app.models.pii_chunk import BBox
from app.services.pii.brand_zones import BrandZone, detect_brand_zones
from app.services.structure.docling_adapter import DocBlock


def _block(block_type: str, bbox: BBox, text: str) -> DocBlock:
    return DocBlock(block_id="dl-0", block_type=block_type, bbox=bbox, text=text)


def _xywh(zone: BrandZone) -> tuple[int, int, int, int]:
    return (zone.bbox.x, zone.bbox.y, zone.bbox.w, zone.bbox.h)


def test_detect_default_zones_720x1100():
    zones = detect_brand_zones(720, 1100, page=0)
    assert len(zones) == 2
    logo = next(z for z in zones if z.zone == "logo")
    footer = next(z for z in zones if z.zone == "footer")
    assert logo.label == "LOGO"
    assert footer.label == "FOOTER"
    assert logo.page == 0
    assert footer.page == 0
    assert _xywh(logo) == (518, 0, 202, 132)
    assert _xywh(footer) == (0, 968, 720, 132)


def test_detect_custom_percents_as_args():
    zones = detect_brand_zones(
        1000,
        1000,
        page=0,
        logo_top_pct=0.10,
        logo_right_pct=0.20,
        footer_bottom_pct=0.05,
    )
    logo = next(z for z in zones if z.zone == "logo")
    footer = next(z for z in zones if z.zone == "footer")
    assert _xywh(logo) == (800, 0, 200, 100)
    assert _xywh(footer) == (0, 950, 1000, 50)


def test_brand_zone_is_not_mock_entry():
    zones = detect_brand_zones(720, 1100, page=0)
    assert zones
    for zone in zones:
        assert isinstance(zone, BrandZone)
        assert not hasattr(zone, "source_text")
        assert not hasattr(zone, "mapping_id")


def test_detect_unions_overlapping_picture_into_logo():
    picture = _block(
        "picture",
        BBox(x=500, y=20, w=180, h=150),
        "BRAND_MARK",
    )
    zones = detect_brand_zones(720, 1100, page=0, blocks=[picture])
    logo = next(z for z in zones if z.zone == "logo")
    footer = next(z for z in zones if z.zone == "footer")
    assert _xywh(logo) == (500, 0, 220, 170)
    assert _xywh(footer) == (0, 968, 720, 132)


def test_detect_unions_footer_block_into_footer():
    footer_block = _block("footer", BBox(x=50, y=950, w=200, h=80), "addr")
    zones = detect_brand_zones(720, 1100, page=0, blocks=[footer_block])
    logo = next(z for z in zones if z.zone == "logo")
    footer = next(z for z in zones if z.zone == "footer")
    assert _xywh(footer) == (0, 950, 720, 150)
    assert _xywh(logo) == (518, 0, 202, 132)


def test_detect_unions_header_into_logo():
    header = _block("HEADER", BBox(x=400, y=0, w=320, h=140), "letterhead")
    zones = detect_brand_zones(720, 1100, page=0, blocks=[header])
    logo = next(z for z in zones if z.zone == "logo")
    assert _xywh(logo) == (400, 0, 320, 140)


def test_detect_ignores_paragraph_overlap():
    paragraph = _block("paragraph", BBox(x=500, y=0, w=200, h=120), "body")
    zones = detect_brand_zones(720, 1100, page=0, blocks=[paragraph])
    logo = next(z for z in zones if z.zone == "logo")
    assert _xywh(logo) == (518, 0, 202, 132)


def test_detect_clamps_union_to_page_bounds():
    picture = _block("picture", BBox(x=700, y=-20, w=80, h=80), "mark")
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


def test_detect_omits_logo_when_patch_logo_false():
    zones = detect_brand_zones(720, 1100, page=0, patch_logo=False, patch_footer=True)
    assert len(zones) == 1
    assert zones[0].zone == "footer"
    assert all(z.zone != "logo" for z in zones)


def test_detect_omits_footer_when_patch_footer_false():
    zones = detect_brand_zones(720, 1100, page=0, patch_footer=False)
    assert len(zones) == 1
    assert zones[0].zone == "logo"


def test_detect_both_toggles_off_returns_empty():
    zones = detect_brand_zones(720, 1100, page=0, patch_logo=False, patch_footer=False)
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
