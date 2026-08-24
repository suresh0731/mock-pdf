import asyncio
import logging
from dataclasses import dataclass, field

from PIL import Image

from app.services.ocr.engines import (
    easyocr_available,
    ocr_image_easyocr,
    ocr_image_rapidocr,
    ocr_image_tesseract,
    rapidocr_available,
    tesseract_available,
)

logger = logging.getLogger(__name__)


@dataclass
class EnginePageResult:
    engine: str
    text: str
    confidence: float
    words: list[dict] = field(default_factory=list)


@dataclass
class PageOCRResult:
    text: str
    confidence: float
    words: list[dict]
    engine_results: list[EnginePageResult]


async def ocr_page(image: Image.Image, tess_lang: str, langs: list[str]) -> PageOCRResult:
    loop = asyncio.get_event_loop()
    tasks = []

    if tesseract_available():
        tasks.append(loop.run_in_executor(None, ocr_image_tesseract, image, tess_lang))
    if easyocr_available():
        tasks.append(loop.run_in_executor(None, ocr_image_easyocr, image, langs))
    if rapidocr_available():
        tasks.append(loop.run_in_executor(None, ocr_image_rapidocr, image))

    if not tasks:
        raise RuntimeError("No OCR engines available. Install Tesseract.")

    raw = await asyncio.gather(*tasks, return_exceptions=True)
    engine_names = []
    if tesseract_available():
        engine_names.append("tesseract")
    if easyocr_available():
        engine_names.append("easyocr")
    if rapidocr_available():
        engine_names.append("rapidocr")

    results: list[EnginePageResult] = []
    for name, item in zip(engine_names, raw):
        if isinstance(item, Exception):
            logger.warning("Engine %s failed: %s", name, item)
            continue
        text, conf, words = item
        results.append(EnginePageResult(engine=name, text=text, confidence=conf, words=words))

    if not results:
        raise RuntimeError("All OCR engines failed")

    merged = merge_engine_results(results)
    return merged


def merge_engine_results(results: list[EnginePageResult]) -> PageOCRResult:
    from app.services.ocr.text_merger import merge_texts

    texts = [r.text for r in results]
    merged_text = merge_texts(texts, backbone=results[0].text)
    backbone = next((r for r in results if r.engine == "tesseract"), results[0])
    return PageOCRResult(
        text=merged_text,
        confidence=sum(r.confidence for r in results) / len(results),
        words=backbone.words,
        engine_results=results,
    )
