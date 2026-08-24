import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from app.models.pii_chunk import BBox
from app.models.redact import PageTransform
from app.services.ocr.ensemble import align_word_boxes
from app.services.pii.coordinate_map import apply_padding, canonical_to_original
from app.services.pii.redaction_scorer import score_redaction
from app.services.preprocess.blur import detect_blur_tier, get_padding_px


def _sharp_image() -> Image.Image:
    img = Image.new("RGB", (400, 200), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 80), "INVOICE  NRIC S1234567A", fill="black")
    return img


def _blurry_image() -> Image.Image:
    img = _sharp_image()
    arr = np.array(img).astype(np.float32)
    blurred = arr.copy()
    for _ in range(8):
        blurred[1:-1, 1:-1] = (
            blurred[:-2, 1:-1]
            + blurred[2:, 1:-1]
            + blurred[1:-1, :-2]
            + blurred[1:-1, 2:]
        ) / 4
    return Image.fromarray(blurred.astype(np.uint8))


def test_detect_blur_tier_sharp():
    tier, variance = detect_blur_tier(_sharp_image())
    assert tier in ("good", "mild", "severe")
    assert variance >= 0


def test_padding_increases_by_tier():
    assert get_padding_px("severe") >= get_padding_px("mild") >= get_padding_px("good")


def test_canonical_to_original_translation():
    bbox = BBox(x=10, y=20, w=50, h=12)
    transform = PageTransform(dx=5, dy=7, blur_tier="good", blur_variance=100.0, preprocessing="none")
    mapped = canonical_to_original(bbox, transform)
    assert mapped.x == 15 and mapped.y == 27


def test_apply_padding_clamps_to_page():
    bbox = BBox(x=0, y=0, w=20, h=10)
    padded = apply_padding(bbox, "severe", page_w=30, page_h=20)
    assert padded.x == 0
    assert padded.y == 0
    assert padded.w <= 30
    assert padded.h <= 20


def test_align_word_boxes_agreement():
    engine_words = [
        (
            "tesseract",
            [{"text": "S1234567A", "x": 100, "y": 50, "w": 80, "h": 14, "confidence": 0.9}],
        ),
        (
            "easyocr",
            [{"text": "S1234567A", "x": 102, "y": 51, "w": 78, "h": 15, "confidence": 0.85}],
        ),
    ]
    aligned = align_word_boxes(engine_words, iou_threshold=0.5, page=0)
    assert len(aligned) == 1
    assert aligned[0].engine_agreement >= 0.5
    assert "S1234567A" in aligned[0].text


def test_score_redaction_weights():
    from app.services.ocr.ensemble_types import EnsembleWord

    words = [
        EnsembleWord(
            text="test",
            bbox=BBox(x=0, y=0, w=10, h=10),
            ocr_confidence=1.0,
            engine_agreement=1.0,
            engines=["tesseract"],
        )
    ]
    score, breakdown = score_redaction(1.0, words, None)
    assert score > 0.5
    assert breakdown.presidio == 1.0
