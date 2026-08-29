import io
import logging
import shutil
from pathlib import Path

from PIL import Image

from app.config import get_settings
from app.models.pii_chunk import BBox

logger = logging.getLogger(__name__)


def configure_tesseract() -> bool:
    settings = get_settings()
    try:
        import pytesseract

        if settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
        version = pytesseract.get_tesseract_version()
        logger.info("Tesseract version: %s", version)
        return True
    except Exception as exc:
        logger.warning("Tesseract not available: %s", exc)
        return False


def tesseract_available() -> bool:
    settings = get_settings()
    if settings.tesseract_cmd and Path(settings.tesseract_cmd).exists():
        return True
    return shutil.which("tesseract") is not None


def ocr_image_tesseract(image: Image.Image, lang: str = "eng") -> tuple[str, float, list[dict]]:
    import pytesseract

    settings = get_settings()
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
    words = []
    texts = []
    confs = []
    n = len(data["text"])
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        conf = float(data["conf"][i]) if str(data["conf"][i]).isdigit() else 0.0
        words.append(
            {
                "text": text,
                "x": int(data["left"][i]),
                "y": int(data["top"][i]),
                "w": int(data["width"][i]),
                "h": int(data["height"][i]),
                "confidence": conf / 100.0,
            }
        )
        texts.append(text)
        confs.append(conf)
    full_text = " ".join(texts)
    avg_conf = (sum(c for c in confs if c >= 0) / len(confs) / 100.0) if confs else 0.0
    return full_text, avg_conf, words


def easyocr_available() -> bool:
    try:
        import easyocr  # noqa: F401

        return True
    except ImportError:
        return False


def ocr_image_easyocr(image: Image.Image, langs: list[str] | None = None) -> tuple[str, float, list[dict]]:
    import easyocr
    import numpy as np

    lang_list = []
    for lang in langs or ["en"]:
        if lang in ("eng", "en"):
            lang_list.append("en")
        elif lang in ("ch_sim", "chi_sim"):
            lang_list.append("ch_sim")
        elif lang in ("ch_tra", "chi_tra"):
            lang_list.append("ch_tra")
    if not lang_list:
        lang_list = ["en"]
    reader = easyocr.Reader(list(dict.fromkeys(lang_list)), gpu=False)
    results = reader.readtext(np.array(image))
    words = []
    texts = []
    confs = []
    for bbox_pts, text, conf in results:
        xs = [p[0] for p in bbox_pts]
        ys = [p[1] for p in bbox_pts]
        words.append(
            {
                "text": text,
                "x": int(min(xs)),
                "y": int(min(ys)),
                "w": int(max(xs) - min(xs)),
                "h": int(max(ys) - min(ys)),
                "confidence": float(conf),
            }
        )
        texts.append(text)
        confs.append(float(conf))
    full_text = " ".join(texts)
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return full_text, avg_conf, words


def rapidocr_available() -> bool:
    try:
        from rapidocr import RapidOCR  # noqa: F401

        return True
    except ImportError:
        return False


# RapidOCR's recognition model is single-language-per-instance (unlike
# Tesseract's `eng+ind`-style combined pass), so a distinct engine instance
# is cached per (Det.lang_type, Rec.lang_type) pair actually requested —
# see _rapidocr_lang_params. The app only ever resolves a handful of
# distinct locales (app/services/locale/resolver.py's LOCALE_LANG_MAP), so
# this cache stays small and bounded.
_rapidocr_engines: dict[tuple[str, str], object] = {}

# Maps the app's internal language codes (see LOCALE_LANG_MAP) to RapidOCR's
# Rec.lang_type. RapidOCR (>=3.9, PP-OCRv6) supports these as discrete
# recognition-model choices; "en"/"ch" are its own defaults and don't need
# an entry here.
_RAPIDOCR_REC_LANG = {
    "id": "id",
    "ms": "ms",
    "ch_sim": "ch",
    "ch_tra": "chinese_cht",
}


def _rapidocr_lang_params(langs: list[str] | None) -> tuple[str, str]:
    """Map resolved locale languages to RapidOCR's (Det, Rec) lang_type.

    RapidOCR can only recognize one language per instance, so this picks a
    single representative language per document locale rather than
    combining languages the way Tesseract's ``eng+ind`` does. When a
    non-English language is present, it's preferred for ``Rec.lang_type``:
    PP-OCRv6's Latin-multilingual/CJK models still read English text
    reasonably well, whereas RapidOCR's *default* (English/Chinese-only)
    model does not read Indonesian/Malay text at all — see
    ``app/services/locale/resolver.py``'s ``LOCALE_LANG_MAP``.

    ``Det.lang_type`` must be one of RapidOCR's own supported per-language
    values (its model resolver validates against that exact set) — passing
    the literal string ``"multi"`` (a previous version of this function's
    attempt at "any non-English/non-Chinese language") is not itself a
    valid ``lang_type`` and always raises ``ValueError`` at engine
    construction, regardless of image content. Every language in
    ``_RAPIDOCR_REC_LANG`` is valid for *both* Det and Rec under the same
    name, so the same resolved language is used for both.

    Args:
        langs: Resolved language codes for this document (e.g. ``["en",
            "id"]``), or ``None``.

    Returns:
        ``(det_lang, rec_lang)``. Defaults to ``("en", "en")`` for an
        English-only or unresolved locale — identical to RapidOCR's
        pre-existing implicit default behavior for English documents.
    """
    for lang in langs or []:
        rec = _RAPIDOCR_REC_LANG.get(lang)
        if rec is not None:
            return rec, rec
    return "en", "en"


def _get_rapidocr_engine(det_lang: str, rec_lang: str):
    """Lazily construct one process-wide RapidOCR instance per language pair.

    RapidOCR's ONNX models load from files bundled inside the wheel (no
    first-run network download, unlike PaddleOCR), so each distinct
    ``(det_lang, rec_lang)`` configuration is safe to cache for the life of
    the process — constructing it per-call would reload the ONNX sessions
    on every page.
    """
    key = (det_lang, rec_lang)
    engine = _rapidocr_engines.get(key)
    if engine is None:
        from rapidocr import RapidOCR

        engine = RapidOCR(params={"Det.lang_type": det_lang, "Rec.lang_type": rec_lang})
        _rapidocr_engines[key] = engine
    return engine


def ocr_image_rapidocr(
    image: Image.Image, langs: list[str] | None = None
) -> tuple[str, float, list[dict]]:
    """Run RapidOCR (ONNX-converted PP-OCR models) and flatten to word dicts.

    Unlike PaddleOCR's ``predict()``/EasyOCR's ``readtext()`` (both of which
    only expose line-level boxes here), RapidOCR's ``return_word_box=True``
    returns true per-word geometry within each detected line — each line is
    a tuple of ``(text, score, quad_box)`` word tuples — so no downstream
    character-width splitting is needed for this engine's output (see
    ``_split_multiword_tokens`` in ``field_extractor.py``).

    Args:
        image: Page image to run OCR on.
        langs: Resolved locale languages (see ``resolve_languages``), used
            to pick RapidOCR's recognition model via
            :func:`_rapidocr_lang_params`. ``None``/unresolved defaults to
            English, matching prior behavior.
    """
    import numpy as np

    det_lang, rec_lang = _rapidocr_lang_params(langs)
    engine = _get_rapidocr_engine(det_lang, rec_lang)
    result = engine(np.array(image), return_word_box=True)
    words: list[dict] = []
    texts: list[str] = []
    confs: list[float] = []
    # A page/image with zero detected text is not an error — RapidOCR
    # returns a default-constructed RapidOCROutput() for it, whose
    # ``word_results`` is a *sentinel* ``(("", 1.0, None),)`` rather than
    # an empty tuple (RapidOCROutput.__len__ is the documented way to
    # detect this — see its docstring). Treating that sentinel as one
    # real "line" of word tuples below and unpacking its first element
    # (an empty string) as ``(word_text, score, box)`` raises
    # ``ValueError: not enough values to unpack`` — this used to look
    # like an engine failure/crash on every page with no text, when it
    # is actually just "no words found".
    if len(result) == 0:
        return "", 0.0, []
    line_results = getattr(result, "word_results", None) or ()
    for line in line_results:
        for word_text, score, box in line:
            word_text = (word_text or "").strip()
            if not word_text:
                continue
            xs = [point[0] for point in box]
            ys = [point[1] for point in box]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            words.append(
                {
                    "text": word_text,
                    "x": int(x_min),
                    "y": int(y_min),
                    "w": max(1, int(round(x_max - x_min))),
                    "h": max(1, int(round(y_max - y_min))),
                    "confidence": float(score),
                }
            )
            texts.append(word_text)
            confs.append(float(score))
    full_text = " ".join(texts)
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return full_text, avg_conf, words
