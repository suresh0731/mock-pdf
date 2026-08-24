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


_rapidocr_engine = None


def _get_rapidocr_engine():
    """Lazily construct a process-wide RapidOCR singleton.

    RapidOCR's ONNX models load from files bundled inside the wheel (no
    first-run network download, unlike PaddleOCR), so this is safe to cache
    once per process. Constructing it per-call would reload the ONNX
    sessions on every page.
    """
    global _rapidocr_engine
    if _rapidocr_engine is None:
        from rapidocr import RapidOCR

        _rapidocr_engine = RapidOCR()
    return _rapidocr_engine


def ocr_image_rapidocr(image: Image.Image) -> tuple[str, float, list[dict]]:
    """Run RapidOCR (ONNX-converted PP-OCR models) and flatten to word dicts.

    Unlike PaddleOCR's ``predict()``/EasyOCR's ``readtext()`` (both of which
    only expose line-level boxes here), RapidOCR's ``return_word_box=True``
    returns true per-word geometry within each detected line — each line is
    a tuple of ``(text, score, quad_box)`` word tuples — so no downstream
    character-width splitting is needed for this engine's output (see
    ``_split_multiword_tokens`` in ``field_extractor.py``).
    """
    import numpy as np

    engine = _get_rapidocr_engine()
    result = engine(np.array(image), return_word_box=True)
    words = []
    texts = []
    confs = []
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
