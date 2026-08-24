import logging
from unittest.mock import MagicMock

import numpy as np
from PIL import Image, ImageDraw

from app.services.preprocess.blur import detect_blur_tier, get_padding_px

_DENY_TOKENS = ("S1234567A", "NRIC", "iVBORw", "\x89PNG")


def _checkerboard(width: int, height: int, tile: int = 8) -> Image.Image:
    ys, xs = np.indices((height, width))
    mask = ((xs // tile) + (ys // tile)) % 2 == 0
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[mask] = 255
    return Image.fromarray(arr, mode="RGB")


def _flat_image(size: int = 80) -> Image.Image:
    return Image.new("RGB", (size, size), (128, 128, 128))


def _pii_sharp_image() -> Image.Image:
    img = Image.new("RGB", (400, 80), "white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 20), "NRIC S1234567A", fill="black")
    return img


def _mock_laplacian(monkeypatch, variance: float) -> None:
    result = MagicMock()
    result.var.return_value = variance
    monkeypatch.setattr(
        "app.services.preprocess.blur.cv2.Laplacian",
        lambda *_args, **_kwargs: result,
    )


def _assert_logs_deny_list(caplog) -> None:
    for record in caplog.records:
        message = record.getMessage()
        for token in _DENY_TOKENS:
            assert token not in message


def test_detect_blur_tier_sharp_is_good():
    img = _checkerboard(120, 80)
    tier, variance = detect_blur_tier(img)
    assert tier == "good"
    assert variance >= 100


def test_detect_blur_tier_low_variance_is_severe():
    tier, variance = detect_blur_tier(_flat_image())
    assert tier == "severe"
    assert variance < 50


def test_detect_blur_tier_boundary_good(monkeypatch):
    _mock_laplacian(monkeypatch, 50.0)
    img = Image.new("RGB", (16, 16), (80, 80, 80))
    assert detect_blur_tier(img, threshold_good=50, threshold_mild=10) == (
        "good",
        50.0,
    )


def test_detect_blur_tier_boundary_mild(monkeypatch):
    _mock_laplacian(monkeypatch, 49.9)
    img = Image.new("RGB", (16, 16), (80, 80, 80))
    assert detect_blur_tier(img, threshold_good=50, threshold_mild=10) == (
        "mild",
        49.9,
    )


def test_detect_blur_tier_boundary_severe(monkeypatch):
    _mock_laplacian(monkeypatch, 9.9)
    img = Image.new("RGB", (16, 16), (80, 80, 80))
    assert detect_blur_tier(img, threshold_good=50, threshold_mild=10) == (
        "severe",
        9.9,
    )


def test_get_padding_px_defaults():
    assert get_padding_px("good") == 4
    assert get_padding_px("mild") == 8
    assert get_padding_px("severe") == 12


def test_get_padding_px_override():
    assert get_padding_px("mild", padding_px_mild=20) == 20


def test_sec001_blur_logs_contain_no_image_or_pii(caplog):
    caplog.set_level(logging.INFO)
    detect_blur_tier(_pii_sharp_image())
    _assert_logs_deny_list(caplog)
