from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env.local", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "PII Redact Service"
    debug: bool = False

    tesseract_cmd: str | None = None
    poppler_path: str | None = None

    shard_base_path: Path = Path("./data")
    admin_auth_enabled: bool = False
    api_key: str = ""

    default_locale: str | None = None
    auto_detect_language: bool = True
    tesseract_langs: str = "eng"

    recognizers_dir: Path = Path("data/recognizers/custom")

    max_upload_mb: int = 25
    allowed_extensions: str = "jpg,jpeg,png,pdf,tiff,tif"

    redact_pipeline_enabled: bool = True
    mock_dictionary_path: Path = Path("data/mock-dictionary/mappings.json")
    logo_zone_top_pct: float = 0.12
    logo_zone_right_pct: float = 0.28
    footer_zone_bottom_pct: float = 0.12
    blur_threshold_good: float = 100.0
    blur_threshold_mild: float = 50.0
    padding_px_good: int = 4
    padding_px_mild: int = 8
    padding_px_severe: int = 12
    padding_multiplier_mild: float = 1.25
    padding_multiplier_severe: float = 1.5
    ensemble_iou_threshold: float = 0.5
    max_cluster_width_px: int = 0
    max_concurrent_pages: int = 2
    max_pages_per_document: int = 50
    docling_enabled: bool = True
    audit_retention_days: int = 90
    redact_dpi: int = 200
    max_concurrent_ocr: int = 3

    presidio_enabled: bool = False
    field_detection_enabled: bool = True
    # Restricts field-anchored detection to values already in the mock
    # dictionary — no brand-new auto-created entries for text nothing
    # matches. Table-cell OCR is noisy enough that free-running detection
    # spawned dozens of near-duplicate auto entries for the same real
    # value; curate the dictionary (or override in the UI/CSV) instead of
    # letting unknown text through.
    restrict_to_known_mappings: bool = True
    # Curated, git-tracked seed of known name/account-number -> mock rows
    # (see data/mock-dictionary/mappings.seed.json), loaded once at startup
    # without clobbering existing mappings (app/services/pii/seed_loader.py).
    # Cloning the repo always starts from these identical curated mappings;
    # mock_dictionary_path above is the separate, git-ignored, machine-local
    # runtime cache (hit_count/timestamps/auto-learned entries).
    mock_seed_path: Path | None = Path("data/mock-dictionary/mappings.seed.json")
    fuzzy_match_threshold: float = 0.85
    strip_gridlines_enabled: bool = True
    # Corrects small page rotation (crooked scans/off-angle phone photos)
    # before any other preprocessing — see app/services/preprocess/deskew.py.
    # Deterministic (OpenCV minAreaRect, no ML weights). Applied once to the
    # shared source image so original_image/canonical_image stay
    # pixel-registered 1:1 exactly like today.
    deskew_enabled: bool = True
    # Uses table-cell bounding boxes (Docling and/or img2table, already
    # computed for the structural-confidence join) as a row-merge signal so
    # a multi-line-wrapped table cell isn't fragmented into several
    # low-confidence pieces by pure geometry row-grouping. Purely
    # additive/optional — set to false to fall back to v1's geometry-only
    # row grouping if this ever over-merges on some other document.
    docling_cell_stitch_enabled: bool = True
    # img2table (OpenCV border/line detection, no ML weights — deterministic
    # by construction) supplements Docling's TableFormer for table/cell
    # geometry. See app/services/structure/table_geometry.py: where an
    # img2table table overlaps a Docling table region, its cells replace
    # Docling's own (TableFormer is a learned model that docling-project/
    # docling#2081 confirms can drop/merge cell text on OCR-driven scanned
    # tables); Docling tables img2table doesn't corroborate keep their own
    # cells unchanged, so coverage never regresses relative to Docling alone.
    img2table_enabled: bool = True
    # Minimum IoU between a Docling table region and an img2table table for
    # img2table's cell geometry to be trusted as a replacement for that
    # region's Docling cells.
    img2table_min_table_iou: float = 0.3
    # RapidOCR (ONNX PP-OCR) tends to read table/grid layouts more reliably
    # than the other engines. Only used by the legacy/debug multi-engine
    # path (``ocr_ensemble_mode="multi"`` or an explicit ``engine_filter``):
    # when a word-cluster tie falls inside a Docling-detected table/cell
    # region, prefer this engine's reading over the default
    # highest-confidence tie-break. Set enabled=False to fall back to
    # confidence-only tie-break.
    ocr_table_bias_enabled: bool = True
    ocr_table_bias_engine: str = "rapidocr"
    ocr_table_bias_min_overlap: float = 0.5
    # Last-resort fail-safe (see app/pipeline/redact.py's
    # _apply_spillover_safety_net): a name-shaped OCR word that ends up in
    # no redaction at all (dropped by stitching, unmatched by the fuzzy/
    # maximal-munch resolution) but sits immediately next to an
    # already-redacted span gets absorbed into that span's box rather than
    # left exposed as bare, readable text. Never creates a new redaction or
    # dictionary entry — purely extends an existing one's bounds.
    spillover_safety_net_enabled: bool = True

    # --- Deterministic OCR engine policy -----------------------------------
    # Default path: exactly one engine's output is used per page, chosen
    # from this explicit, logged order — never a cross-engine vote. This
    # removes the biggest source of cross-machine drift (whichever subset of
    # engines happened to install/import successfully). RapidOCR is primary
    # because it ships its models inside the wheel (no first-run network
    # download) and gives true per-word geometry via return_word_box=True.
    ocr_primary_engine: str = "rapidocr"
    ocr_fallback_engines: str = "tesseract,easyocr"
    # "single" (default, deterministic) or "multi" (legacy parallel
    # majority-vote across all available engines — debugging/comparison
    # only; also engaged automatically when a caller passes an explicit
    # engine_filter).
    ocr_ensemble_mode: str = "single"

    # Startup engine-availability check (see
    # app/services/ocr/environment_check.py). Comma-separated engine names
    # that must be available for the process to be considered healthy.
    # Only the primary engine is required by default: ocr_fallback_engines
    # is deliberately a *fallback*, only invoked if the primary is
    # unavailable/fails outright, so a machine missing tesseract/easyocr
    # still runs the default deterministic single-engine path correctly.
    # Widen this (e.g. "rapidocr,tesseract") if a deployment wants to
    # guarantee the fallback path also works before serving traffic.
    ocr_required_engines: str = "rapidocr"
    # When True, a missing required engine raises at startup (fail fast).
    # When False (default, friendlier for partial local dev setups), it
    # logs a loud warning instead. Set True for CI/production so a machine
    # silently missing an engine can never serve traffic.
    ocr_strict_engine_check: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
