import logging

import numpy as np
from PIL import Image, ImageDraw

from app.services.preprocess.canonical import canonicalize_page, detect_crop_offset

_DENY_TOKENS = ("S1234567A", "NRIC", "iVBORw", "\x89PNG")


def _checkerboard(width: int, height: int, tile: int = 8) -> Image.Image:
    ys, xs = np.indices((height, width))
    mask = ((xs // tile) + (ys // tile)) % 2 == 0
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[mask] = 255
    return Image.fromarray(arr, mode="RGB")


def _inset_content_image() -> Image.Image:
    img = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((30, 20, 109, 79), fill="black")
    return img


def _pii_sharp_image() -> Image.Image:
    img = Image.new("RGB", (400, 80), "white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 20), "NRIC S1234567A", fill="black")
    return img


def _assert_logs_deny_list(caplog) -> None:
    for record in caplog.records:
        message = record.getMessage()
        for token in _DENY_TOKENS:
            assert token not in message


def test_canonicalize_good_passthrough():
    image = _checkerboard(120, 80)
    result = canonicalize_page(image, 2)
    assert result.transform.preprocessing == "none"
    assert result.transform.blur_tier == "good"
    assert result.page_index == 2
    assert result.canonical_image.size == image.size
    assert result.canonical_image.tobytes() == result.original_image.tobytes()
    assert result.original_image is not image


def test_canonicalize_mild_sauvola_same_size_rgb(monkeypatch):
    monkeypatch.setattr(
        "app.services.preprocess.canonical.detect_blur_tier",
        lambda _image: ("mild", 72.0),
    )
    image = Image.new("RGB", (64, 48), (180, 180, 180))
    result = canonicalize_page(image, 0)
    assert result.transform.preprocessing == "sauvola_unsharp"
    assert result.canonical_image.mode == "RGB"
    assert result.canonical_image.size == image.size
    assert result.transform.blur_tier == "mild"
    assert result.transform.blur_variance == 72.0


def test_canonicalize_severe_sauvola_same_size_rgb(monkeypatch):
    monkeypatch.setattr(
        "app.services.preprocess.canonical.detect_blur_tier",
        lambda _image: ("severe", 12.0),
    )
    image = Image.new("RGB", (64, 48), (180, 180, 180))
    result = canonicalize_page(image, 1)
    assert result.canonical_image.mode == "RGB"
    assert result.canonical_image.size == image.size
    assert result.transform.blur_tier == "severe"
    assert result.transform.preprocessing == "sauvola_unsharp"


def test_detect_crop_offset_inset_content():
    assert detect_crop_offset(_inset_content_image()) == (30, 20)


def test_detect_crop_offset_blank_is_zero():
    assert detect_crop_offset(Image.new("RGB", (100, 100), "white")) == (0, 0)


def test_canonicalize_never_shifts_transform_despite_content_inset():
    # canonical_image is pixel-registered 1:1 with original_image (never
    # actually cropped to the content bbox), so the transform must stay
    # (0, 0) even when detect_crop_offset would report a large inset —
    # otherwise every redaction box gets double-shifted by the margin.
    result = canonicalize_page(_inset_content_image(), 0)
    assert result.transform.dx == 0
    assert result.transform.dy == 0


def test_sec001_canonicalize_logs_contain_no_image_or_pii(caplog):
    caplog.set_level(logging.INFO)
    canonicalize_page(_pii_sharp_image(), 0)
    _assert_logs_deny_list(caplog)
    joined = " ".join(record.getMessage() for record in caplog.records)
    assert "page_index" in joined
    assert "blur_tier" in joined
    assert "dx" in joined
    assert "dy" in joined


def _rotated_ruled_page(degrees: float = 6.0) -> Image.Image:
    img = Image.new("L", (400, 600), 255)
    draw = ImageDraw.Draw(img)
    for y in range(60, 540, 24):
        draw.line([(30, y), (370, y)], fill=0, width=2)
    rgb = img.convert("RGB")
    return rgb.rotate(degrees, expand=False, fillcolor=(255, 255, 255), resample=Image.BICUBIC)


def test_canonicalize_deskews_before_stripping_and_reports_angle():
    result = canonicalize_page(_rotated_ruled_page(6.0), 0)
    assert result.transform.skew_angle_deg != 0.0
    # original_image and canonical_image must still be pixel-registered
    # 1:1 after deskewing (dx/dy stay 0) — both were rotated together from
    # the same shared source image, not just one of them.
    assert result.original_image.size == result.canonical_image.size
    assert result.transform.dx == 0
    assert result.transform.dy == 0


def test_canonicalize_deskew_disabled_keeps_rotation():
    rotated = _rotated_ruled_page(6.0)
    result = canonicalize_page(rotated, 0, deskew=False)
    assert result.transform.skew_angle_deg == 0.0
    assert result.original_image.tobytes() == rotated.convert("RGB").tobytes()


def test_canonicalize_deskew_default_true_matches_settings_default():
    from app.config import Settings

    assert Settings(_env_file=None).deskew_enabled is True


def test_severe_emits_warning_without_payload(monkeypatch, caplog):
    monkeypatch.setattr(
        "app.services.preprocess.canonical.detect_blur_tier",
        lambda _image: ("severe", 12.0),
    )
    caplog.set_level(logging.WARNING)
    canonicalize_page(_pii_sharp_image(), 3)
    assert any(record.levelno >= logging.WARNING for record in caplog.records)
    _assert_logs_deny_list(caplog)
