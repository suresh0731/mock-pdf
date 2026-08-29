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
    # "digital" (native PDF text layer, OCR skipped), "scanned" (ensemble
    # OCR ran and found text), or "blank" (OCR found nothing and no
    # native text layer either — Settings.ocr_blank_page_skip_enabled;
    # see app/pipeline/redact.py._build_page_states). See
    # app/services/ocr/native_text.py for "digital" vs "scanned".
    # Surfaced on PageAuditSummary for operator visibility.
    page_kind: str = "scanned"
    # Source fitz.Page for this page, when the input was a real PDF that
    # PyMuPDF itself rendered (see page_renderer.RenderedPage.fitz_page) —
    # None for non-PDF input (jpg/png/tiff) and whenever the pdf2image
    # fallback path was used. Only consulted for "digital" pages: the
    # renderer (app/services/redact/pdf_renderer.py) clones this page and
    # redacts the clone as real vector content instead of flattening it
    # to an image like a "scanned" page — see pdf_native_redactor.py.
    fitz_page: object | None = None
    # Render DPI this page's canonical image was produced at (RedactOptions
    # .dpi), needed to convert a pixel-space padded_bbox back into PDF
    # point space (72/inch) when redacting fitz_page directly.
    dpi: int = 200
    # Keeps fitz_page's parent fitz.Document alive for as long as this
    # state is referenced — mirrors page_renderer.RenderedPage._doc's own
    # comment: a fitz.Page is only valid while its parent Document hasn't
    # been garbage collected, and page_states outlives the RenderedPage
    # list _build_page_states constructed it from.
    _fitz_doc: object | None = field(default=None, repr=False, compare=False)

    @property
    def page_index(self) -> int:
        return self.canonical.page_index

    @property
    def transform(self) -> PageTransform:
        return self.canonical.transform
