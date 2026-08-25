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

    max_upload_mb: int = 25
    allowed_extensions: str = "jpg,jpeg,png,pdf,tiff,tif"

    redact_pipeline_enabled: bool = True
    mock_dictionary_path: Path = Path("data/mock-dictionary/mappings.json")
    footer_zone_bottom_pct: float = 0.12
    # Generic image/graphic redaction: unlike the fixed logo/footer percent
    # zones above (tuned for one template's chrome placement), this covers
    # every Docling-detected "picture" block anywhere on the page — see
    # app/services/pii/brand_zones.py's detect_picture_zones(). Global
    # default toggle; RedactOptions.patch_images overrides per-request.
    patch_images_enabled: bool = True
    # Minimum picture-block area (fraction of page area) to qualify for a
    # generic image redaction zone — filters bullet icons/checkbox glyphs
    # Docling may still classify as "picture".
    image_zone_min_area_pct: float = 0.0015
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
    # Output page images are rasterized at redact_dpi and re-encoded once the
    # redaction boxes are painted on (see app/services/redact/pdf_renderer.py)
    # — the source PDF's original vector text/compressed images are gone
    # either way, since burning boxes into pixels is what guarantees the
    # underlying PII can't leak back out. "jpeg" (default) keeps the
    # resulting PDF close to the original file size; a scanned/photographed
    # page's noise compresses far worse under lossless "png" (often 5-10x
    # larger) for no visible quality gain on a document redaction.
    redact_output_image_format: str = "jpeg"
    redact_output_jpeg_quality: int = 85

    field_detection_enabled: bool = True
    # Scans the page's full OCR text for every value already curated in the
    # mock dictionary (see app/services/pii/custom_redact.py's
    # find_term_spans, the same mechanism used for a per-request custom
    # term) and redacts any hit — independent of document layout/labels, so
    # a known name/org is caught anywhere it appears (transaction
    # narration, address block, ...) even on a document shape
    # field_extractor.py doesn't recognize. Can never discover brand-new
    # unseen text on its own (that's what field-anchored detection and
    # per-request custom terms are for); it only ever matches text that's
    # already in the dictionary.
    dictionary_scan_enabled: bool = True
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

    # --- Folder-watch ingestion (extension, alongside UI/API upload) -------
    # Polls `watch_input_dir` for new files and redacts them one at a time
    # (never in parallel — see app/services/watch/folder_watcher.py), using
    # the exact same RedactPipeline the UI/API already call. Disabled by
    # default so it never changes existing upload-only behavior; opt in
    # with WATCH_ENABLED=true.
    watch_enabled: bool = False
    watch_input_dir: Path = Path("data/watch/input")
    watch_output_dir: Path = Path("data/watch/output")
    # Seconds between directory scans. Also the minimum time a file's
    # size/mtime must stay unchanged before it's picked up (protects
    # against reading a file that's still being copied into the folder).
    watch_poll_seconds: float = 5.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
