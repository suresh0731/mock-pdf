"""Table/rule-line removal to reduce OCR gridline artifacts.

A long thin table border or letterhead rule can bleed into an adjacent OCR
token — e.g. a vertical border stroke read as a leading ``l`` glued onto
``PT`` (``"PT BNI"`` -> ``"lPT BNI"``). This module detects long thin
horizontal/vertical lines via morphological opening and inpaints over them
before OCR runs, leaving character strokes (which are short and non-linear)
intact. Runs before OCR on the canonical image only; never touches the
original scan used for the final redacted render (SEC-001: no PII in logs;
this module never touches text content at all).
"""

import cv2
import numpy as np
from PIL import Image


def strip_table_lines(
    image: Image.Image,
    *,
    min_line_length_px: int = 40,
    line_thickness_px: int = 3,
) -> Image.Image:
    """Erase long thin horizontal/vertical lines from a page image.

    Args:
        image: Source page image (any mode; converted to RGB internally).
        min_line_length_px: Minimum unbroken run length (in its own
            direction) to treat a stroke as a table/rule line rather than a
            character glyph.
        line_thickness_px: Expected max visual thickness of a rule line, in
            pixels. Used only to size the dilation/inpaint halo around a
            detected line (so anti-aliased edges are covered too) — not to
            gate detection, since a real scanned gridline is often only
            1-2px thick and a detection kernel taller/wider than that would
            erode it to nothing before it could ever be found.

    Returns:
        Same-size RGB image with detected lines inpainted out, or the
        original image (as RGB) unchanged if no lines are found.
    """
    rgb = image.convert("RGB")
    arr = np.array(rgb)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # A 1-row/1-col cross-axis keeps detection thickness-agnostic: any
    # unbroken run of >= min_line_length_px black pixels in a single row
    # (or column) counts as part of a line, regardless of how many rows
    # (or columns) the actual rule is drawn across.
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_length_px, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_line_length_px))
    horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
    vert_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)
    lines_mask = cv2.bitwise_or(horiz_lines, vert_lines)

    # A real table/rule line is a small fraction of the page. A mask that
    # covers most of the image means Otsu misclassified a low-contrast or
    # near-uniform region as foreground, not that we found actual lines —
    # inpainting that would erase real content, so skip it.
    mask_fraction = float(np.count_nonzero(lines_mask)) / lines_mask.size
    if mask_fraction == 0.0 or mask_fraction > 0.5:
        return rgb

    dilate_kernel = np.ones((line_thickness_px, line_thickness_px), np.uint8)
    dilated_mask = cv2.dilate(lines_mask, dilate_kernel, iterations=1)
    cleaned = cv2.inpaint(arr, dilated_mask, 3, cv2.INPAINT_TELEA)
    return Image.fromarray(cleaned)
