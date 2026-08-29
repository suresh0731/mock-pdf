from dataclasses import dataclass, field

from app.models.redact import PageTransform
from app.services.ocr.ensemble_types import EnsembleWord
from app.services.preprocess.canonical import CanonicalPage
from app.services.structure.docling_adapter import DocBlock


@dataclass
class PageProcessState:
    canonical: CanonicalPage
    merged_text: str
    ensemble_words: list[EnsembleWord] = field(default_factory=list)
    blocks: list[DocBlock] = field(default_factory=list)
    word_context: dict = field(default_factory=dict)
    # "digital" (native PDF text layer, OCR skipped) or "scanned" (ensemble
    # OCR ran) — see app/services/ocr/native_text.py. Surfaced on
    # PageAuditSummary for operator visibility on mixed documents.
    page_kind: str = "scanned"

    @property
    def page_index(self) -> int:
        return self.canonical.page_index

    @property
    def transform(self) -> PageTransform:
        return self.canonical.transform
