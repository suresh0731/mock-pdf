from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.models.pii_chunk import BBox


class CustomRedactTerm(BaseModel):
    search_value: str
    mock_label: str = "CUSTOM"


class RedactOptions(BaseModel):
    locale: str | None = None
    languages: list[str] | None = None
    auto_detect: bool = True
    dpi: int = Field(default=200, ge=100, le=300)
    include_audit_inline: bool = False
    strict_pii: bool = False
    custom_redactions: list[CustomRedactTerm] = Field(default_factory=list)
    patch_footer: bool = True
    patch_images: bool = True
    patch_signatures: bool = True
    # Whitelist of OCR engines to run ("tesseract"/"easyocr"/"rapidocr").
    # None/empty runs every available engine (auto/ensemble) — the default.
    ocr_engines: list[str] | None = None


BlurTier = Literal["good", "mild", "severe"]
Preprocessing = Literal["none", "sauvola_unsharp"]


class PageTransform(BaseModel):
    dx: int = 0
    dy: int = 0
    blur_tier: BlurTier = "good"
    blur_variance: float = 0.0
    preprocessing: Preprocessing = "none"
    # Rotation correction applied before any other preprocessing (see
    # app/services/preprocess/deskew.py). 0.0 means no correction was
    # needed/applied. Audit-only metadata — never fed back into dx/dy,
    # since the rotation is baked directly into original_image/
    # canonical_image pixels rather than tracked as a coordinate offset.
    skew_angle_deg: float = 0.0


class ConfidenceBreakdown(BaseModel):
    presidio: float
    ocr: float
    engine_agreement: float
    structural_context: float


class StructuralContext(BaseModel):
    block_id: str | None = None
    block_type: str
    block_label: str | None = None
    table_column: str | None = None
    table_row: int | None = None
    join_iou: float = 0.0


class RedactionRegion(BaseModel):
    region_id: str
    page: int
    entity_type: str
    canonical_bbox: BBox
    original_bbox: BBox
    padded_bbox: BBox
    redaction_confidence: float
    confidence_breakdown: ConfidenceBreakdown
    structural_context: StructuralContext | None = None
    blur_tier: BlurTier
    engines_seen: list[str] = Field(default_factory=list)
    mock_value: str = ""
    mapping_id: str | None = None
    assignment_source: Literal["auto", "user", "brand"] = "auto"


PageKind = Literal["digital", "scanned", "blank"]


class PageAuditSummary(BaseModel):
    page: int
    blur_tier: BlurTier
    blur_variance: float
    transform: PageTransform
    ensemble_word_count: int
    docling_block_count: int
    redaction_count: int
    # "digital": native PDF text layer used, OCR skipped for this page.
    # "scanned": ensemble OCR ran and found text. "blank": OCR found
    # nothing and no native text layer either (Settings.
    # ocr_blank_page_skip_enabled) — 0 redactions possible on this page.
    # See app/services/ocr/native_text.py and app/pipeline/redact.py.
    page_kind: PageKind = "scanned"


class RedactAuditResponse(BaseModel):
    request_id: str
    filename: str
    page_count: int
    processing_ms: int
    created_at: datetime
    summary: dict
    pages: list[PageAuditSummary]
    redactions: list[RedactionRegion]
