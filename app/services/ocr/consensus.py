import asyncio
import logging
import re
from collections import Counter

from PIL import Image

from app.config import get_settings
from app.models.consensus import ConsensusResult, EngineResult
from app.services.ocr.engine import ocr_page
from app.services.ocr.engines import (
    easyocr_available,
    ocr_image_easyocr,
    ocr_image_rapidocr,
    ocr_image_tesseract,
    rapidocr_available,
    tesseract_available,
)
from app.services.ocr.validators import validate_entity

logger = logging.getLogger(__name__)


def _normalize(value: str, entity_type: str) -> str:
    v = value.strip()
    if entity_type.upper() in ("US_SSN", "SSN") and len(re.sub(r"\D", "", v)) == 9:
        d = re.sub(r"\D", "", v)
        return f"{d[:3]}-{d[3:5]}-{d[5:]}"
    return v


async def resolve_crop(
    crop: Image.Image,
    entity_type: str,
    tess_lang: str,
    langs: list[str],
    locale: str | None = None,
    stage: str = "B",
) -> ConsensusResult:
    loop = asyncio.get_event_loop()
    tasks = []
    names = []
    if tesseract_available():
        tasks.append(loop.run_in_executor(None, ocr_image_tesseract, crop, tess_lang))
        names.append("tesseract")
    if easyocr_available():
        tasks.append(loop.run_in_executor(None, ocr_image_easyocr, crop, langs))
        names.append("easyocr")
    if rapidocr_available():
        tasks.append(loop.run_in_executor(None, ocr_image_rapidocr, crop))
        names.append("rapidocr")

    raw = await asyncio.gather(*tasks, return_exceptions=True)
    engine_results: list[EngineResult] = []
    for name, item in zip(names, raw):
        if isinstance(item, Exception):
            continue
        text, conf, _ = item
        if text.strip():
            engine_results.append(EngineResult(engine=name, text=text.strip(), confidence=conf))

    return vote(engine_results, entity_type, locale, stage=stage)


def vote(
    engine_results: list[EngineResult],
    entity_type: str,
    locale: str | None = None,
    stage: str = "A",
) -> ConsensusResult:
    if not engine_results:
        return ConsensusResult(value=None, score=0.0, engines_agreed=0, engine_results=[], stage=stage)

    normalized = [_normalize(r.text, entity_type) for r in engine_results]
    counter = Counter(normalized)
    best_value, count = counter.most_common(1)[0]
    agreement = count / len(engine_results)
    validator_pass = 1.0 if validate_entity(entity_type, best_value, locale) else 0.0
    avg_conf = sum(r.confidence for r in engine_results) / len(engine_results)
    score = agreement * 0.5 + validator_pass * 0.3 + avg_conf * 0.2

    return ConsensusResult(
        value=best_value if validator_pass or agreement >= 0.67 else None,
        score=round(score, 4),
        engines_agreed=count,
        engine_results=engine_results,
        stage=stage,
        tier=1,
    )


async def resolve_from_stage_a(
    engine_texts: list[tuple[str, str, float]],
    entity_type: str,
    locale: str | None,
) -> ConsensusResult:
    results = [EngineResult(engine=n, text=t, confidence=c) for n, t, c in engine_texts if t.strip()]
    return vote(results, entity_type, locale, stage="A")
