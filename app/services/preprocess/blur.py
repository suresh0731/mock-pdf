import logging
from typing import Literal

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

BlurTier = Literal["good", "mild", "severe"]

THRESHOLD_GOOD = 100.0
THRESHOLD_MILD = 50.0
PADDING_PX_GOOD = 4
PADDING_PX_MILD = 8
PADDING_PX_SEVERE = 12


def detect_blur_tier(
    image: Image.Image,
    threshold_good: float = THRESHOLD_GOOD,
    threshold_mild: float = THRESHOLD_MILD,
) -> tuple[BlurTier, float]:
    """Laplacian variance → (tier, raw_variance)."""
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if variance >= threshold_good:
        tier: BlurTier = "good"
    elif variance >= threshold_mild:
        tier = "mild"
    else:
        tier = "severe"

    logger.info("blur_tier=%s blur_variance=%s", tier, variance)
    if tier == "severe":
        logger.warning("blur_tier=%s blur_variance=%s", tier, variance)
    return tier, variance


def get_padding_px(
    tier: BlurTier,
    padding_px_good: int = PADDING_PX_GOOD,
    padding_px_mild: int = PADDING_PX_MILD,
    padding_px_severe: int = PADDING_PX_SEVERE,
) -> int:
    """Pixel padding for redaction expansion (module defaults 4/8/12)."""
    if tier == "good":
        return padding_px_good
    if tier == "mild":
        return padding_px_mild
    return padding_px_severe
