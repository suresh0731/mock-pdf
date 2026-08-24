"""Legacy whole-page NER detector (Presidio + spaCy).

Disabled by default (``presidio_enabled=false``) in favor of the
field-anchored detector (``field_extractor.py``), which doesn't hallucinate
PERSON/ORG entities on noisy OCR text. Kept importable for a future opt-in
secondary/manual signal — never auto-redacts by default.

Presidio/spaCy are only imported lazily, inside the functions below, rather
than at module level: they are a heavy optional dependency chain (spaCy ->
thinc -> torch), and the rest of the pipeline must be able to start up and
run its default (field-anchored-only) path without them installed at all.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from app.config import get_settings

if TYPE_CHECKING:
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer

logger = logging.getLogger(__name__)

_analyzer: AnalyzerEngine | None = None


def _build_analyzer() -> AnalyzerEngine:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    global _analyzer
    if _analyzer is not None:
        return _analyzer

    settings = get_settings()
    try:
        provider = NlpEngineProvider(conf_file=None, nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        })
        nlp_engine = provider.create_engine()
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    except Exception as exc:
        logger.warning("spaCy model unavailable, using pattern-only analyzer: %s", exc)
        analyzer = AnalyzerEngine(supported_languages=["en"])

    for rec in _load_custom_recognizers(settings.recognizers_dir):
        analyzer.registry.add_recognizer(rec)

    for rec in _builtin_recognizers():
        analyzer.registry.add_recognizer(rec)

    _analyzer = analyzer
    return analyzer


def _builtin_recognizers() -> list[PatternRecognizer]:
    from presidio_analyzer import Pattern, PatternRecognizer

    return [
        PatternRecognizer(
            supported_entity="NRIC",
            patterns=[Pattern("nric", r"[STFGM]\d{7}[A-Z]", 0.85)],
        ),
        PatternRecognizer(
            supported_entity="MYKAD",
            patterns=[Pattern("mykad", r"\d{6}-\d{2}-\d{4}", 0.85)],
        ),
        PatternRecognizer(
            supported_entity="HKID",
            patterns=[Pattern("hkid", r"[A-Z]{1,2}\d{6}\([0-9A]\)", 0.85)],
        ),
        PatternRecognizer(
            supported_entity="NIK",
            patterns=[Pattern("nik", r"\d{16}", 0.8)],
        ),
    ]


def _load_custom_recognizers(path) -> list[PatternRecognizer]:
    import yaml
    from presidio_analyzer import Pattern, PatternRecognizer

    recognizers = []
    if not path.exists():
        return recognizers
    for file in path.glob("*.yaml"):
        try:
            data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
            patterns = [
                Pattern(p.get("name", "custom"), p["pattern"], p.get("score", 0.8))
                for p in data.get("patterns", [])
            ]
            if patterns:
                recognizers.append(
                    PatternRecognizer(
                        supported_entity=data.get("entity", "CUSTOM"),
                        patterns=patterns,
                    )
                )
        except Exception as exc:
            logger.warning("Failed loading recognizer %s: %s", file, exc)
    return recognizers


def detect_pii(text: str, locale: str | None = None) -> list[dict[str, Any]]:
    analyzer = _build_analyzer()
    results = analyzer.analyze(text=text, language="en")
    detections = []
    for r in results:
        snippet = text[r.start : r.end]
        detections.append(
            {
                "entity_type": r.entity_type,
                "start": r.start,
                "end": r.end,
                "score": r.score,
                "text": snippet,
            }
        )
    return detections
