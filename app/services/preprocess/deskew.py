"""Rotation-skew correction for scanned/photographed pages.

Several `repii/` samples are visibly skewed (crooked scans, off-angle phone
photos of a printed letter) — straightening them before OCR/structure
extraction reduces how often multi-line table-cell stitching needs the
OCR-geometry column-band fallback, and improves basic word-box accuracy
everywhere else too.

Estimates skew via a projection-profile search: candidate rotations of the
page's binary foreground mask are scored by the variance of their row-sum
profile, and the angle whose profile is "sharpest" (i.e. text lines/table
rules stack into tight horizontal bands) wins. This is deterministic and
dependency-free (no ML weights, no randomness), consistent with this
pipeline's determinism goals.

An earlier version used OpenCV's ``minAreaRect`` over the whole foreground
pixel cloud instead. That measures the rotation of the *tightest enclosing
rectangle* around all ink combined, which is only a good proxy for text
skew when the page content itself densely fills a rectangle. Real business
letters don't: a left-aligned address block, a right-aligned logo, and a
full-width table combine into an irregular point cloud whose minimum-area
box can be tilted even when every line of text is perfectly horizontal —
confirmed empirically on ``repii/1000095684.jpg``, an axis-aligned scan with
a faint background watermark that ``minAreaRect`` mis-read as an ~8.5 degree
skew. The projection-profile approach instead directly measures "how
line-structured does the content look at this candidate angle", which is
robust to irregular layouts and faint background noise. The angle-sign
convention below is empirically verified (see ``tests/test_deskew.py``)
against known synthetic rotations in both directions.

Must run once, upstream of ``canonicalize_page``'s line-stripping/blur-tier
branch, so both ``original_image`` and ``canonical_image`` stay
pixel-registered 1:1 (the same invariant ``canonical.py`` already documents
for ``dx``/``dy``) — the rotation must never be applied to only one of
them, or OCR/structure word geometry would silently stop lining up with
the final rendered page.
"""

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Below this, the estimate is treated as scanner/camera jitter, not real
# skew — rotating (and re-interpolating) an already-straight page would
# only soften edges for no benefit.
_MIN_CORRECTION_DEG = 0.3
# Above this, a "detected skew" is far more likely a mis-fit on sparse/
# unusual page content (e.g. a mostly-blank page, or a stray large picture
# block) than genuine page rotation — blindly rotating a page by tens of
# degrees on a bad reading would do more harm than the skew itself.
_MAX_CORRECTION_DEG = 15.0
_MIN_FOREGROUND_PIXELS = 200
# The angle search rotates a downscaled binary mask many times; capping the
# working resolution keeps this fast (well under a second per page) without
# losing the coarse line/rule structure the profile score depends on.
_SEARCH_MAX_DIM = 900
_COARSE_STEP_DEG = 1.0
_FINE_WINDOW_DEG = 1.0
_FINE_STEP_DEG = 0.1


def _foreground_mask(image: Image.Image, max_dim: int = _SEARCH_MAX_DIM) -> np.ndarray:
    """Binary (0/255) foreground mask, downscaled for fast angle search."""
    gray = np.array(image.convert("L"))
    height, width = gray.shape
    scale = min(1.0, max_dim / max(height, width, 1))
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    inverted = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


def _rotate_mask(mask: np.ndarray, angle_deg: float) -> np.ndarray:
    height, width = mask.shape
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        mask,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _row_profile_variance(mask: np.ndarray) -> float:
    """How "line-structured" a mask is: high when ink concentrates into a
    few sharp horizontal bands (axis-aligned text lines/table rules), low
    when it's smeared evenly across rows (rotated content)."""
    row_sums = mask.sum(axis=1).astype(np.float64)
    return float(row_sums.var())


def _best_angle_in_range(mask: np.ndarray, low: float, high: float, step: float) -> float:
    best_angle, best_score = 0.0, -1.0
    steps = max(0, int(round((high - low) / step))) + 1
    for i in range(steps):
        angle = low + i * step
        score = _row_profile_variance(_rotate_mask(mask, angle))
        if score > best_score:
            best_angle, best_score = angle, score
    return best_angle


def detect_skew_angle(
    image: Image.Image,
    *,
    max_correction_deg: float = _MAX_CORRECTION_DEG,
) -> float:
    """Estimate the page's rotation skew in degrees.

    Positive = the page content is rotated clockwise and needs a
    counter-clockwise correction of this same magnitude (i.e. the caller
    passes this value straight to ``cv2.getRotationMatrix2D``, whose own
    positive-angle convention is counter-clockwise). This module's search
    self-verifies the convention: it scores candidate corrections with the
    exact same rotation helper `deskew_image` later applies, so there is no
    separate sign-flip step to get backwards.

    Args:
        image: Source page image (any mode; converted to grayscale
            internally).
        max_correction_deg: Reject (return ``0.0``) any estimate larger
            than this in magnitude.

    Returns:
        Estimated skew in degrees, or ``0.0`` for a blank/near-blank page
        (nothing to measure) or an out-of-range estimate.
    """
    mask = _foreground_mask(image)
    if cv2.countNonZero(mask) < _MIN_FOREGROUND_PIXELS:
        return 0.0

    coarse_angle = _best_angle_in_range(mask, -max_correction_deg, max_correction_deg, _COARSE_STEP_DEG)
    fine_low = max(-max_correction_deg, coarse_angle - _FINE_WINDOW_DEG)
    fine_high = min(max_correction_deg, coarse_angle + _FINE_WINDOW_DEG)
    fine_angle = _best_angle_in_range(mask, fine_low, fine_high, _FINE_STEP_DEG)

    if abs(fine_angle) > max_correction_deg:
        return 0.0
    return round(fine_angle, 2)


def deskew_image(
    image: Image.Image,
    *,
    min_correction_deg: float = _MIN_CORRECTION_DEG,
    max_correction_deg: float = _MAX_CORRECTION_DEG,
) -> tuple[Image.Image, float]:
    """Rotate `image` to correct small skew, filling new corners white.

    Args:
        image: Source page image (any mode; converted to RGB internally).
        min_correction_deg: Skip correction below this magnitude.
        max_correction_deg: Reject an out-of-range angle estimate (see
            ``detect_skew_angle``).

    Returns:
        ``(corrected_image, angle_deg)``. When no correction is applied,
        ``angle_deg`` is ``0.0`` and ``corrected_image`` is `image` itself
        (not a copy) so a no-op call is cheap.
    """
    angle = detect_skew_angle(image, max_correction_deg=max_correction_deg)
    if abs(angle) < min_correction_deg:
        return image, 0.0

    arr = np.array(image.convert("RGB"))
    height, width = arr.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        arr,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    logger.info("deskew applied angle_deg=%s", round(angle, 3))
    return Image.fromarray(rotated), angle
