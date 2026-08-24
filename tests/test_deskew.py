from PIL import Image, ImageDraw

from app.services.preprocess.deskew import deskew_image, detect_skew_angle


def _ruled_page(width: int = 500, height: int = 700) -> Image.Image:
    """A page of evenly-spaced horizontal rules, like a ruled letter/table."""
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    for y in range(60, height - 60, 24):
        draw.line([(40, y), (width - 40, y)], fill=0, width=2)
    return img.convert("RGB")


def _rotated(image: Image.Image, degrees: float) -> Image.Image:
    return image.rotate(degrees, expand=False, fillcolor=(255, 255, 255), resample=Image.BICUBIC)


def test_detect_skew_angle_blank_page_is_zero():
    assert detect_skew_angle(Image.new("RGB", (200, 200), "white")) == 0.0


def test_detect_skew_angle_axis_aligned_page_is_near_zero():
    angle = detect_skew_angle(_ruled_page())
    assert abs(angle) < 0.3


def test_detect_skew_angle_recovers_positive_and_negative_rotation():
    page = _ruled_page()
    for introduced in (5.0, -5.0, 10.0, -10.0, 2.0, -2.0):
        skewed = _rotated(page, introduced)
        detected = detect_skew_angle(skewed)
        # PIL's Image.rotate(degrees) rotates counter-clockwise for positive
        # degrees; cv2.getRotationMatrix2D uses the same convention, so the
        # correction needed is the negative of what was introduced.
        assert abs(detected - (-introduced)) < 1.0, (introduced, detected)


def test_detect_skew_angle_rejects_out_of_range_estimate():
    # A tiny, oddly-shaped foreground blob can produce a nonsense large
    # "angle" reading — must not be trusted as real page skew.
    img = Image.new("RGB", (200, 200), "white")
    ImageDraw.Draw(img).line([(50, 50), (60, 190)], fill="black", width=1)
    angle = detect_skew_angle(img, max_correction_deg=5.0)
    assert angle == 0.0


def test_deskew_image_noop_below_min_correction():
    page = _ruled_page()
    corrected, angle = deskew_image(page)
    assert angle == 0.0
    assert corrected is page  # unchanged input returned as-is, not a copy


def test_deskew_image_straightens_rotated_page():
    page = _ruled_page()
    skewed = _rotated(page, 6.0)
    corrected, angle = deskew_image(skewed)
    assert angle != 0.0
    assert corrected.size == skewed.size
    residual = detect_skew_angle(corrected)
    assert abs(residual) < 1.0


def test_deskew_image_fills_new_corners_white_not_black():
    page = _ruled_page()
    skewed = _rotated(page, 8.0)
    corrected, _angle = deskew_image(skewed)
    # A rotated rectangle leaves triangular gaps at the corners of the
    # frame; these must be filled white (matching a scanned page's
    # background), never left black, which would look like a redaction
    # box or bleed into edge-adjacent OCR tokens.
    corner_pixel = corrected.convert("RGB").getpixel((2, 2))
    assert corner_pixel == (255, 255, 255)


def test_deskew_image_does_not_rotate_already_straight_page():
    page = _ruled_page()
    corrected, angle = deskew_image(page)
    assert angle == 0.0
    assert corrected.convert("RGB").tobytes() == page.convert("RGB").tobytes()
