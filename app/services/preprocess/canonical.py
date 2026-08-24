import logging
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from skimage.filters import threshold_sauvola

from app.models.redact import PageTransform, Preprocessing
from app.services.preprocess.blur import detect_blur_tier
from app.services.preprocess.deskew import deskew_image
from app.services.preprocess.line_removal import strip_table_lines

logger = logging.getLogger(__name__)


@dataclass
class CanonicalPage:
    """One page after blur-tier treatment with crop-offset metadata."""

    page_index: int
    original_image: Image.Image
    canonical_image: Image.Image
    transform: PageTransform


def detect_crop_offset(image: Image.Image) -> tuple[int, int]:
    """Content bbox origin vs page edges (dx, dy). Blank → (0, 0)."""
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return 0, 0
    x, y, _, _ = cv2.boundingRect(coords)
    return int(x), int(y)


def _apply_sauvola_unsharp(image: Image.Image) -> Image.Image:
    """Sauvola binarization plus light unsharp, returned as same-size RGB."""
    gray = np.array(image.convert("L"))
    thresh = threshold_sauvola(gray, window_size=25)
    binary = (gray > thresh).astype(np.uint8) * 255
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.4, blurred, -0.4, 0)
    combined = cv2.addWeighted(sharp, 0.35, binary, 0.65, 0)
    rgb = cv2.cvtColor(combined.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


def canonicalize_page(
    image: Image.Image,
    page_index: int,
    strip_gridlines: bool = True,
    *,
    deskew: bool = True,
) -> CanonicalPage:
    """Deskew + blur-tier treatment + crop-offset metadata in a canonical page.

    Args:
        image: Source page image. ``original_image`` is an untouched copy
            of this *after* deskewing (see ``deskew``), regardless of
            ``strip_gridlines``.
        page_index: Zero-based page index, copied onto the result.
        strip_gridlines: When True (default), table/rule lines are
            inpainted out of the OCR-facing (canonical) image before blur
            treatment, so gridlines don't bleed into adjacent OCR tokens.
        deskew: When True (default), correct small page rotation before
            any other processing (see ``preprocess/deskew.py``) — applied
            once, upstream of the ``original_image``/``canonical_image``
            split, so both stay pixel-registered 1:1 afterward exactly
            like today (see the Note below).

    Note:
        ``canonical_image`` is never actually cropped to the content
        bounding box — it stays full-page and pixel-registered 1:1 with
        ``original_image`` (same size, same origin; only pixel values
        change via deskewing / gridline stripping / sauvola). So the
        transform's ``dx``/``dy`` are always 0: OCR runs on
        ``canonical_image`` directly (see ``pipeline/redact.py``), and its
        coordinates are already in ``original_image`` space with no
        translation needed. Feeding ``detect_crop_offset``'s content-margin
        offset into the transform here would double-shift every redaction
        box by the page's margin width, which is exactly the "text leaks
        past the redaction box" symptom this must not reintroduce.
    """
    skew_angle = 0.0
    if deskew:
        image, skew_angle = deskew_image(image)

    tier, variance = detect_blur_tier(image)
    original_image = image.copy()
    working = strip_table_lines(image) if strip_gridlines else image

    if tier == "good":
        canonical_image = working.copy()
        preprocessing: Preprocessing = "none"
    else:
        canonical_image = _apply_sauvola_unsharp(working)
        preprocessing = "sauvola_unsharp"

    transform = PageTransform(
        dx=0,
        dy=0,
        blur_tier=tier,
        blur_variance=variance,
        preprocessing=preprocessing,
        skew_angle_deg=round(skew_angle, 3),
    )
    width, height = canonical_image.size
    logger.info(
        "page_index=%s blur_tier=%s blur_variance=%s skew_angle_deg=%s "
        "dx=%s dy=%s preprocessing=%s width=%s height=%s",
        page_index,
        transform.blur_tier,
        transform.blur_variance,
        transform.skew_angle_deg,
        transform.dx,
        transform.dy,
        transform.preprocessing,
        width,
        height,
    )
    if transform.blur_tier == "severe":
        logger.warning("blur_tier=severe page_index=%s", page_index)

    return CanonicalPage(
        page_index=page_index,
        original_image=original_image,
        canonical_image=canonical_image,
        transform=transform,
    )
