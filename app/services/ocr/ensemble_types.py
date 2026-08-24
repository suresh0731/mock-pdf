from dataclasses import dataclass, field
from typing import TypedDict

from app.models.pii_chunk import BBox


class EngineWordDict(TypedDict, total=False):
    """Canned per-engine word box used by align_word_boxes and unit tests."""

    text: str
    x: int
    y: int
    w: int
    h: int
    confidence: float


@dataclass
class EnsembleWord:
    """Aligned word produced by IoU clustering across OCR engines."""

    text: str
    bbox: BBox
    ocr_confidence: float
    engine_agreement: float
    engines: list[str] = field(default_factory=list)
    page: int = 0
    char_start: int = 0
    char_end: int = 0

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "x": self.bbox.x,
            "y": self.bbox.y,
            "w": self.bbox.w,
            "h": self.bbox.h,
            "confidence": self.ocr_confidence,
        }
