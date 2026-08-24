import asyncio
import logging
from collections import Counter

from PIL import Image

from app.config import get_settings
from app.models.pii_chunk import BBox
from app.services.ocr.engine import EnginePageResult
from app.services.ocr.engines import (
    easyocr_available,
    ocr_image_easyocr,
    ocr_image_rapidocr,
    ocr_image_tesseract,
    rapidocr_available,
    tesseract_available,
)
from app.services.ocr.ensemble_types import EnsembleWord

logger = logging.getLogger(__name__)


def _is_valid_word(word: object) -> bool:
    """Return True when word is a dict with non-blank text and positive box."""
    if not isinstance(word, dict):
        return False
    text = word.get("text")
    if not isinstance(text, str) or not text.strip():
        return False
    for key in ("x", "y", "w", "h"):
        if key not in word:
            return False
        try:
            val = int(word[key])
        except (TypeError, ValueError):
            return False
        if key in ("w", "h") and val <= 0:
            return False
    return True


def _iou(a: BBox, b: BBox) -> float:
    """Return intersection-over-union of two axis-aligned boxes."""
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def _overlap_ratio(word: BBox, region: BBox) -> float:
    """Fraction of ``word``'s area covered by ``region``.

    Intentionally not IoU: a word is typically much smaller than the
    table/cell region it sits inside, so IoU would stay near zero even when
    the word is fully contained.
    """
    wx2, wy2 = word.x + word.w, word.y + word.h
    rx2, ry2 = region.x + region.w, region.y + region.h
    ix1, iy1 = max(word.x, region.x), max(word.y, region.y)
    ix2, iy2 = min(wx2, rx2), min(wy2, ry2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area = word.w * word.h
    return inter / area if area > 0 else 0.0


def _in_table_region(bbox: BBox, table_regions: list[BBox], min_overlap: float) -> bool:
    """True when ``bbox`` mostly overlaps any region in ``table_regions``."""
    return any(_overlap_ratio(bbox, region) >= min_overlap for region in table_regions)


def _word_bbox(word: dict) -> BBox:
    """Build a BBox from a validated engine word dict."""
    return BBox(x=int(word["x"]), y=int(word["y"]), w=int(word["w"]), h=int(word["h"]))


def _union_bbox(boxes: list[BBox]) -> BBox:
    """Axis-aligned union of one or more boxes."""
    return BBox(
        x=min(b.x for b in boxes),
        y=min(b.y for b in boxes),
        w=max(b.x + b.w for b in boxes) - min(b.x for b in boxes),
        h=max(b.y + b.h for b in boxes) - min(b.y for b in boxes),
    )


def _median_word_width(flat: list[tuple[str, dict]]) -> float:
    """Median positive word width; fallback 50.0 when none exist."""
    widths = [int(w["w"]) for _, w in flat if int(w.get("w", 0)) > 0]
    if not widths:
        return 50.0
    widths.sort()
    return float(widths[len(widths) // 2])


def _word_confidence(word: dict) -> float:
    """Coerce optional confidence to float; invalid values become 0.0."""
    try:
        return float(word.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _majority_text(cluster: list[tuple[str, dict]], prefer_engine: str | None = None) -> str:
    """Majority of stripped text; ties go to the highest-confidence member.

    When tied and ``prefer_engine`` is set, that engine's text wins if it's
    among the tied candidates (used to bias table-region clusters toward a
    specific engine) — otherwise falls back to the confidence tie-break.
    """
    texts = [word["text"].strip() for _, word in cluster]
    counts = Counter(texts)
    max_votes = max(counts.values())
    tied = {text for text, count in counts.items() if count == max_votes}
    if len(tied) == 1:
        return next(iter(tied))
    if prefer_engine:
        preferred = {
            word["text"].strip()
            for engine, word in cluster
            if engine == prefer_engine and word["text"].strip() in tied
        }
        if len(preferred) == 1:
            return next(iter(preferred))
    return max(
        (
            (_word_confidence(word), word["text"].strip())
            for _, word in cluster
            if word["text"].strip() in tied
        ),
        key=lambda item: item[0],
    )[1]


def _format_engine_failure(engine: str, exc: BaseException) -> str:
    """Log-safe: engine name + exception class only. Never str(exc)."""
    return f"Engine {engine} failed: {type(exc).__name__}"


def align_word_boxes(
    engine_words: list[tuple[str, list[dict]]],
    iou_threshold: float = 0.5,
    page: int = 0,
    max_cluster_width_px: int = 0,
    table_regions: list[BBox] | None = None,
    prefer_engine_in_tables: str | None = None,
    table_min_overlap: float = 0.5,
) -> list[EnsembleWord]:
    """Cluster word boxes across engines. Pure: no settings, I/O, or GPU.

    Args:
        engine_words: (engine_name, words) pairs. Each word: text, x, y, w, h,
            optional confidence.
        iou_threshold: cluster if IoU(word, first-member bbox) >= this.
            Default 0.5.
        page: copied onto every EnsembleWord.
        max_cluster_width_px: split if union width would exceed this.
            0 → int(1.5 * median_w).
        table_regions: Optional table/cell bboxes (e.g. from Docling). When a
            cluster's union bbox mostly overlaps one of these, a tied cluster
            prefers ``prefer_engine_in_tables``'s text instead of the default
            highest-confidence tie-break. No effect on clusters where all
            engines already agree.
        prefer_engine_in_tables: Engine name to bias toward inside
            ``table_regions``. ``None`` (default) disables the bias entirely.
        table_min_overlap: Fraction of a cluster's area that must fall
            inside a table region to count as "in the table". Default 0.5.

    Returns:
        Reading-order EnsembleWord list with union bbox, agreement, char spans.
        Empty input / all-invalid words → []. Never raises for bad word dicts.
    """
    if not isinstance(engine_words, list) or not engine_words:
        return []

    flat: list[tuple[str, dict]] = []
    for item in engine_words:
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        engine, words = item
        if not isinstance(engine, str) or not isinstance(words, list):
            continue
        for word in words:
            if _is_valid_word(word):
                flat.append((engine, word))

    if not flat:
        return []

    engines_available = len({engine for engine, _ in flat})
    max_width = (
        max_cluster_width_px
        if max_cluster_width_px > 0
        else int(_median_word_width(flat) * 1.5)
    )

    flat.sort(key=lambda item: (int(item[1]["y"]), int(item[1]["x"])))

    clusters: list[list[tuple[str, dict]]] = []
    for engine, word in flat:
        bbox = _word_bbox(word)
        placed = False
        for cluster in clusters:
            rep = _word_bbox(cluster[0][1])
            if _iou(bbox, rep) >= iou_threshold:
                union = _union_bbox([bbox, rep])
                if union.w <= max_width:
                    cluster.append((engine, word))
                    placed = True
                    break
        if not placed:
            clusters.append([(engine, word)])

    ensemble: list[EnsembleWord] = []
    for cluster in clusters:
        engines = list(dict.fromkeys(engine for engine, _ in cluster))
        confs = [_word_confidence(word) for _, word in cluster]
        boxes = [_word_bbox(word) for _, word in cluster]
        agreement = len(engines) / max(engines_available, 1)
        union_bbox = _union_bbox(boxes)
        prefer_engine = (
            prefer_engine_in_tables
            if table_regions and _in_table_region(union_bbox, table_regions, table_min_overlap)
            else None
        )
        ensemble.append(
            EnsembleWord(
                text=_majority_text(cluster, prefer_engine=prefer_engine),
                bbox=union_bbox,
                ocr_confidence=sum(confs) / len(confs),
                engine_agreement=agreement,
                engines=engines,
                page=page,
            )
        )

    ensemble.sort(key=lambda word: (word.bbox.y, word.bbox.x))
    logger.debug(
        "align_word_boxes: cluster_count=%s engines_available=%s",
        len(ensemble),
        engines_available,
    )

    last = len(ensemble) - 1
    offset = 0
    for index, word in enumerate(ensemble):
        word.char_start = offset
        word.char_end = offset + len(word.text)
        offset = word.char_end + (0 if index == last else 1)
    return ensemble


def merged_text_from_words(words: list[EnsembleWord]) -> str:
    """Join aligned words with a single space (no trailing separator)."""
    return " ".join(word.text for word in words)


# Recognized engine names. Availability/runner functions are looked up by
# name from this module's globals() (below) rather than captured into a
# dict at import time, so tests can monkeypatch
# ``app.services.ocr.ensemble.tesseract_available``/``ocr_image_tesseract``
# (etc.) and have the single-engine and multi-engine paths both observe it.
_KNOWN_ENGINES = ("tesseract", "easyocr", "rapidocr")


def _availability_fn(name: str):
    return globals().get(f"{name}_available")


def _runner_fn(name: str):
    return globals().get(f"ocr_image_{name}")


def _engine_call_args(name: str, image: Image.Image, tess_lang: str, langs: list[str]) -> tuple:
    if name == "tesseract":
        return (image, tess_lang)
    if name == "easyocr":
        return (image, langs)
    return (image,)


def _parse_engine_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


async def _run_one_engine(
    loop: asyncio.AbstractEventLoop,
    name: str,
    image: Image.Image,
    tess_lang: str,
    langs: list[str],
) -> EnginePageResult | None:
    """Run a single named engine; return None on unavailability/failure.

    Never raises — callers decide whether a ``None`` result is fatal.
    """
    if name not in _KNOWN_ENGINES:
        logger.warning("Unknown OCR engine %r; skipping", name)
        return None
    availability = _availability_fn(name)
    runner = _runner_fn(name)
    if availability is None or runner is None:
        logger.warning("Unknown OCR engine %r; skipping", name)
        return None
    if not availability():
        return None
    try:
        text, conf, words = await loop.run_in_executor(
            None, runner, *_engine_call_args(name, image, tess_lang, langs)
        )
    except Exception as exc:  # noqa: BLE001 - engine failures must not abort the page
        logger.warning("%s", _format_engine_failure(name, exc))
        return None
    return EnginePageResult(engine=name, text=text, confidence=conf, words=words)


async def _ensemble_ocr_page_single(
    image: Image.Image,
    page: int,
    tess_lang: str,
    langs: list[str],
) -> tuple[str, list[EnsembleWord], list[EnginePageResult]]:
    """Deterministic default policy: one primary engine, documented fallback.

    Tries ``Settings.ocr_primary_engine`` then ``Settings.ocr_fallback_engines``
    in the configured order, stopping at the first engine that is available
    and returns at least one word. Exactly one engine's output is used per
    page — never a cross-engine vote — so the result no longer depends on
    which subset of engines happens to be installed on a given machine, only
    on the explicit, logged order in ``Settings``.
    """
    settings = get_settings()
    loop = asyncio.get_event_loop()
    order = [settings.ocr_primary_engine, *_parse_engine_list(settings.ocr_fallback_engines)]
    # De-dupe while preserving order (primary may also appear in fallback list).
    order = list(dict.fromkeys(order))

    attempted: list[str] = []
    for name in order:
        attempted.append(name)
        result = await _run_one_engine(loop, name, image, tess_lang, langs)
        if result is not None and result.words:
            if name != settings.ocr_primary_engine:
                logger.warning(
                    "Primary/earlier OCR engines unavailable or empty (%s); "
                    "fell back to %s",
                    attempted[:-1],
                    name,
                )
            aligned = align_word_boxes([(result.engine, result.words)], page=page)
            merged = merged_text_from_words(aligned) or result.text
            return merged, aligned, [result]

    raise RuntimeError(f"All configured OCR engines failed or unavailable: {order}")


async def _ensemble_ocr_page_multi(
    image: Image.Image,
    page: int,
    tess_lang: str,
    langs: list[str],
    table_regions: list[BBox] | None,
    engine_filter: list[str] | None,
) -> tuple[str, list[EnsembleWord], list[EnginePageResult]]:
    """Legacy parallel multi-engine vote (debug/comparison only, non-default).

    Runs every available (or ``engine_filter``-whitelisted) engine
    concurrently and majority-votes tied word clusters via
    :func:`align_word_boxes`. This is what previously ran unconditionally and
    is the direct source of "redacted here, missed there" cross-machine
    drift, since the result silently changes with whichever subset of
    engines happens to be installed. Kept opt-in
    (``Settings.ocr_ensemble_mode == "multi"``) or explicit
    (``engine_filter`` passed) for debugging/engine-comparison only.
    """
    settings = get_settings()
    allowed = set(engine_filter) if engine_filter else None

    loop = asyncio.get_event_loop()
    tasks = []
    names: list[str] = []
    for name in _KNOWN_ENGINES:
        if allowed is not None and name not in allowed:
            continue
        availability = _availability_fn(name)
        if availability is None or not availability():
            continue
        runner = _runner_fn(name)
        tasks.append(
            loop.run_in_executor(None, runner, *_engine_call_args(name, image, tess_lang, langs))
        )
        names.append(name)

    if not tasks:
        raise RuntimeError("No OCR engines available")

    raw = await asyncio.gather(*tasks, return_exceptions=True)
    engine_results: list[EnginePageResult] = []
    engine_words: list[tuple[str, list[dict]]] = []

    for name, item in zip(names, raw):
        if isinstance(item, Exception):
            logger.warning("%s", _format_engine_failure(name, item))
            continue
        text, conf, words = item
        engine_results.append(
            EnginePageResult(engine=name, text=text, confidence=conf, words=words)
        )
        engine_words.append((name, words))

    if not engine_results:
        raise RuntimeError("All OCR engines failed")

    prefer_engine = settings.ocr_table_bias_engine if settings.ocr_table_bias_enabled else None
    aligned = align_word_boxes(
        engine_words,
        page=page,
        table_regions=table_regions,
        prefer_engine_in_tables=prefer_engine,
        table_min_overlap=settings.ocr_table_bias_min_overlap,
    )
    merged = merged_text_from_words(aligned)
    if not merged.strip():
        merged = engine_results[0].text
    return merged, aligned, engine_results


async def ensemble_ocr_page(
    image: Image.Image,
    page: int,
    tess_lang: str,
    langs: list[str],
    table_regions: list[BBox] | None = None,
    engine_filter: list[str] | None = None,
) -> tuple[str, list[EnsembleWord], list[EnginePageResult]]:
    """Resolve page OCR via the deterministic single-engine default policy.

    Args:
        image: Canonical page image.
        page: Zero-based page index copied onto aligned words.
        tess_lang: Tesseract language code.
        langs: EasyOCR language list.
        table_regions: Only used by the legacy multi-engine path (see
            below) to bias tied word clusters toward
            ``Settings.ocr_table_bias_engine``. No effect on the default
            single-engine path, which never forms cross-engine clusters.
        engine_filter: Optional whitelist of engine names
            (``"tesseract"``/``"easyocr"``/``"rapidocr"``). Passing this
            explicitly opts into the legacy parallel multi-engine vote
            (debugging/engine-comparison only) regardless of
            ``Settings.ocr_ensemble_mode``.

    Returns:
        Merged text, aligned words, and per-engine page results.

    Raises:
        RuntimeError: If no engines are available or all engines fail.
    """
    settings = get_settings()
    if engine_filter or settings.ocr_ensemble_mode == "multi":
        return await _ensemble_ocr_page_multi(
            image, page, tess_lang, langs, table_regions, engine_filter
        )
    return await _ensemble_ocr_page_single(image, page, tess_lang, langs)
