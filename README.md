# PII Redaction Service

Local-only redaction pipeline with a **NiceGUI** web portal (pure Python — no npm). Upload a bill, preview original vs redacted side-by-side, and add custom values to redact.

## Quick start (local — Windows)

1. Install [Python 3.11+](https://python.org), [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki), and optionally [Poppler](https://github.com/oschwartz10612/poppler-windows/releases) for PDFs.

2. Copy environment file:
```powershell
copy .env.local.example .env.local
```

3. Install exact-pinned dependencies (prefer the full transitive lockfile for byte-identical installs; `requirements.txt` alone still pins every *direct* dependency if you'd rather not use the lockfile):
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.lock.txt
python -m spacy download en_core_web_sm
# Exact pin (same model spaCy 3.8.x resolves to). `spacy download` does not
# accept pip-style `==` version pins — use the wheel URL instead:
# pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```
Regenerate `requirements.lock.txt` after any `requirements.txt` change with `uv pip compile requirements.txt -o requirements.lock.txt`, then re-run `pytest tests/regression/ -m slow` (the same-document-twice determinism check plus the `repii/` regression-coverage fixtures) before committing the bump.

`requirements.txt` includes `rapidocr`/`onnxruntime` alongside `easyocr` and `pytesseract` for OCR. RapidOCR's PP-OCR models ship inside the pip wheel — no first-run network download and no native `paddlepaddle`/AVX-CPU-instruction dependency, which used to be the most common source of one machine silently having a different OCR engine set than another. By default only one deterministic **primary** engine (`Settings.ocr_primary_engine`, default `rapidocr`) runs per page, with a documented fallback order (`Settings.ocr_fallback_engines`) tried only if the primary is unavailable or fails outright — never a cross-engine vote. A startup check (`app/services/ocr/environment_check.py`) compares actual engine availability against `Settings.ocr_required_engines` and fails fast (`Settings.ocr_strict_engine_check=true`) or logs a loud warning. Check `GET /v1/health` after startup to confirm which engines are actually available (`tesseract`/`easyocr`/`rapidocr`/`docling`/`opencv` → `"ok"` or `"missing"`); a `"missing"` Tesseract usually means the binary (not the `pytesseract` Python package) isn't installed or on `PATH` — see step 1.

### Model weight versions

Exact-pinning the libraries above also pins *which* model weights they fetch (each release requests a fixed, hardcoded model URL/revision), but two still hit the network on first run — vendor/back up these cache directories alongside the repo for fully offline, byte-identical model weights across machines:

| Engine | Model version source | Default cache dir (override via env var) |
|---|---|---|
| RapidOCR | Bundled inside the `rapidocr==3.9.2` wheel — **no network call, ever** | `.venv/Lib/site-packages/rapidocr/models/` |
| Tesseract | Whatever `tesseract --version` reports on `PATH`/`TESSERACT_CMD` — not pip-pinned; document your installed Tesseract version alongside this repo's `requirements.lock.txt` | n/a (native binary) |
| spaCy | `en_core_web_sm==3.8.0` (pinned explicitly above; must match the installed `spacy==3.8.13`'s expected model range) | `.venv/Lib/site-packages/en_core_web_sm/` |
| EasyOCR | Fixed detector (`craft_mlt_25k.pth`) + recognition (`english_g2.pth`) weights hardcoded per `easyocr==1.7.2` release | `%USERPROFILE%\.EasyOCR` (`EASYOCR_MODULE_PATH`) |
| Docling | Layout/table-structure weights hardcoded per `docling-ibm-models==3.14.0`, fetched from Hugging Face Hub | `%USERPROFILE%\.cache\huggingface` (`HF_HOME`) |

To fully vendor: after a clean install's first successful run, zip `%USERPROFILE%\.EasyOCR` and `%USERPROFILE%\.cache\huggingface`, commit them to shared storage (not this git repo — they're large binary blobs), and restore them (or point `EASYOCR_MODULE_PATH`/`HF_HOME` at a shared network path) on every machine instead of letting each one download independently.

4. Start the portal + API:
```powershell
.\scripts\dev.ps1
```

5. Open **http://127.0.0.1:8000/** (NiceGUI UI) · API docs: http://127.0.0.1:8000/docs

## Web UI (NiceGUI)

| Feature | Description |
|---------|-------------|
| Side-by-side preview | Original PDF left, redacted PDF right |
| Upload | PDF, PNG, JPG, TIFF |
| Auto PII redaction | Presidio + deterministic single-engine OCR (documented fallback order) |
| OCR engine picker | "Auto" uses the deterministic default engine order; pick Tesseract/EasyOCR/RapidOCR to force one — e.g. RapidOCR reads table-heavy statements more reliably than EasyOCR |
| Custom mock values | One per line; `value=MOCK_LABEL` for audit tags |
| Regenerate | Re-applies redactions using cached OCR (fast) |

## Mock dictionary: curated seed vs. runtime cache

`data/mock-dictionary/` splits into two files:

- `mappings.seed.json` — curated, **git-tracked** entries (`assignment_source: "user"` equivalents), reviewed like code. Loaded once at startup (`Settings.mock_seed_path`) without ever clobbering an existing mapping. Cloning the repo always starts from these identical mappings.
- `mappings.json` — the local, **git-ignored** runtime cache (`hit_count`, timestamps, auto-learned entries from live redaction runs). `Settings.mock_dictionary_path` points here; it's rewritten on every request and is expected to diverge machine-to-machine — that's fine, since `RESTRICT_TO_KNOWN_MAPPINGS=true` only needs the *seed* to already know a value for it to redact consistently everywhere.

To add a new curated mapping permanently, edit `mappings.seed.json` directly (or via `POST /v1/mappings/upload` CSV import → then copy the resulting rows into the seed file) rather than relying on an auto-learned local entry.

## Table structure detection: Docling + img2table

Table/cell geometry (used to stitch multi-line-wrapped cells and to score structural confidence) comes from two complementary sources, merged in `app/services/structure/table_geometry.py`:

- **Docling** (`TableFormer`, a learned model) — general paragraph/header/footer/table block detection.
- **img2table** (`Settings.img2table_enabled`, default on) — OpenCV border/line detection, no ML weights, so the same page always yields the same cell grid. Wherever an img2table table overlaps a Docling table region (IoU ≥ `Settings.img2table_min_table_iou`), its cells replace Docling's own — TableFormer is known to drop/merge cell text on OCR-driven scanned tables ([docling-project/docling#2081](https://github.com/docling-project/docling/issues/2081)), which img2table's direct gridline reading avoids. A Docling table img2table can't corroborate (e.g. broken/faint borders) keeps its own cells, so coverage never regresses relative to Docling alone.

Both run on `CanonicalPage.original_image` (not the line-stripped OCR `canonical_image`) since cell-boundary detection needs visible gridlines — the opposite of what OCR wants from `strip_table_lines`. The two images are pixel-registered 1:1, so bboxes from either transfer directly onto OCR word geometry with no translation.

## API endpoints

- `POST /v1/redact` — upload bill, receive redacted PDF
- `POST /v1/redact/regenerate/{session_id}` — re-redact with new custom terms
- `GET /v1/redact/audit/{request_id}` — audit metadata (no PII text)
- `GET /v1/health` — engine availability
- `POST /v1/ocr-only` — debug OCR + PII detection

## Project layout

- `app/ui/redact_portal.py` — NiceGUI web portal
- `app/pipeline/redact.py` — redaction orchestrator
- `app/services/preprocess/` — blur + canonical images
- `app/services/ocr/ensemble.py` — deterministic single-primary-engine OCR policy with a documented fallback order (`Settings.ocr_primary_engine`/`ocr_fallback_engines`); the legacy parallel multi-engine vote (word-cluster ties inside a Docling-detected table/cell region biased toward `Settings.ocr_table_bias_engine`, default RapidOCR) is still available opt-in (`Settings.ocr_ensemble_mode="multi"`) for debugging/engine comparison
- `app/services/ocr/environment_check.py` — startup engine-availability check (`Settings.ocr_required_engines`/`ocr_strict_engine_check`)
- `data/audit/requests/` — per-request audit JSON

## Tests

```powershell
pytest tests/
```

`tests/regression/` (real OCR/Docling against the `repii/` sample images, tens of seconds per image) is marked `slow` and excluded from the default run above (see `pytest.ini`). Run it explicitly — required before committing a dependency/OCR/extractor change:

```powershell
pytest tests/regression/ -m slow
```

It covers two things: `test_repii_coverage.py` asserts each sample image's redaction count per entity type never drops below an observed baseline (structural coverage, not exact OCR text), and `test_determinism.py` runs one sample through the pipeline twice with independent mock-dictionary stores and asserts identical redaction output, plus an explicit check that this machine actually has the OCR engines `Settings.ocr_required_engines` expects before trusting that comparison.
