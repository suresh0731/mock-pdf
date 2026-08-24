LOCALE_LANG_MAP: dict[str, list[str]] = {
    "sg": ["en"],
    "my": ["en", "ms"],
    "hk": ["en", "ch_tra", "ch_sim"],
    "id": ["en", "id"],
    "us": ["en"],
}


def resolve_languages(
    locale: str | None,
    languages: list[str] | None,
    auto_detect: bool,
    sample_text: str = "",
) -> tuple[str | None, list[str], str]:
    if languages:
        langs = languages
        loc = locale
    elif locale and locale in LOCALE_LANG_MAP:
        langs = LOCALE_LANG_MAP[locale]
        loc = locale
    elif auto_detect and sample_text.strip():
        try:
            from langdetect import detect

            code = detect(sample_text[:500])
            loc = _map_detected(code)
            langs = LOCALE_LANG_MAP.get(loc, ["eng"])
        except Exception:
            loc = locale
            langs = ["eng"]
    else:
        loc = locale
        langs = ["eng"]

    tess = "+".join(_to_tesseract(lang) for lang in langs)
    return loc, langs, tess


def _map_detected(code: str) -> str:
    mapping = {"en": "sg", "ms": "my", "id": "id", "zh-cn": "hk", "zh-tw": "hk"}
    return mapping.get(code, "sg")


def _to_tesseract(lang: str) -> str:
    mapping = {
        "en": "eng",
        "eng": "eng",
        "ms": "msa",
        "id": "ind",
        "ch_sim": "chi_sim",
        "ch_tra": "chi_tra",
    }
    return mapping.get(lang, lang)
