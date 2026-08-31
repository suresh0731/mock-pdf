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
| Auto PII redaction | Field-anchored layout detection + dictionary-scan of curated mappings + deterministic single-engine OCR (documented fallback order) |
| OCR engine picker | "Auto" uses the deterministic default engine order; pick Tesseract/EasyOCR/RapidOCR to force one — e.g. RapidOCR reads table-heavy statements more reliably than EasyOCR |
| Custom mock values | One per line; `value=MOCK_LABEL` for audit tags |
| Cover signatures | Redacts the blank ink gap beside every detected signatory name/org row in the bottom-of-page signature block — see "Signature redaction" below |
| Regenerate | Re-applies redactions using cached OCR (fast) |

## Folder-watch ingestion (alongside UI upload)

An optional extension to the UI upload: point `WATCH_ENABLED=true` (see
`.env.local.example`) at an input directory and every file dropped into it
gets redacted automatically — no browser needed, and **no second process**.
Setting `WATCH_ENABLED=true` and starting the app normally
(`.\scripts\dev.ps1`) is enough — the watcher runs as a background task
*inside* that same `uvicorn`/FastAPI process (see `app/main.py`'s
`lifespan`), alongside the NiceGUI UI and API on the same port.

- Polls `WATCH_INPUT_DIR` (default `data/watch/input`) on an interval
  (`WATCH_POLL_SECONDS`, default 5s) and only picks up a file once its
  size/mtime has stayed unchanged across two scans (avoids reading a file
  mid-copy).
- Redacts **one file at a time, never in parallel** — the same
  `RedactPipeline` the UI/API already use.
- On success, writes `redacted_<name>.pdf` to `WATCH_OUTPUT_DIR` (default
  `data/watch/output`) and removes the original from the input directory.
- On any error, the original file is left untouched in the input directory
  and retried automatically on the next scan — nothing is ever moved on
  failure.

## Mock dictionary: curated seed vs. runtime cache

`data/mock-dictionary/` splits into two files:

- `mappings.seed.json` — curated, **git-tracked** entries (`assignment_source: "user"` equivalents), reviewed like code. Loaded once at startup (`Settings.mock_seed_path`) without ever clobbering an existing mapping. Cloning the repo always starts from these identical mappings.
- `mappings.json` — the local, **git-ignored** runtime cache (`hit_count`, timestamps, auto-learned entries from live redaction runs). `Settings.mock_dictionary_path` points here; it's rewritten on every request and is expected to diverge machine-to-machine — that's fine, since `RESTRICT_TO_KNOWN_MAPPINGS=true` only needs the *seed* to already know a value for it to redact consistently everywhere.

To add a new curated mapping permanently, edit `mappings.seed.json` directly (or via `POST /v1/mappings/upload` CSV import → then copy the resulting rows into the seed file) rather than relying on an auto-learned local entry.

## Debugging: comparing two machines via logs

When the same file redacts differently on two machines (e.g. a laptop vs. a
server), the fastest root-cause path is usually logs, not screenshots:

- **Startup**: every process logs one `"environment fingerprint"` line (OS,
  Python version, every OCR engine's availability, and installed versions of
  the libraries most likely to drift — OpenCV, Docling, RapidOCR,
  onnxruntime, EasyOCR, Tesseract binary, PyMuPDF). Diff this line between
  the two machines first — a mismatched `docling`/`opencv-python` version is
  the most common cause of two machines detecting different table/cell
  geometry (and therefore different padding) from an identical image.
- **Startup, config**: right after the fingerprint, one `"effective settings"`
  line dumps every resolved `Settings` field (padding px per blur tier,
  `restrict_to_known_mappings`, `img2table_min_table_iou`, every feature
  toggle, ...). `.env.local` is git-ignored and machine-local by design, so
  two machines can run identical code and identical package versions and
  still behave differently because one has an override the other doesn't —
  this line turns that into a one-line diff instead of re-deriving every
  knob's value by hand on both machines. `api_key` is redacted.
- **Per page**: one `"page processed"` line per page ties together
  `blur_tier`/`blur_variance`/`skew_angle_deg`, structure block counts by
  type, OCR word count, and which OCR engine actually produced those words
  (`ocr_engines_used`) — the direct inputs to the padding decision in
  `apply_padding`.
- **Per redaction** (`LOG_LEVEL=DEBUG`): `app.services.pii.coordinate_map`
  logs `"bbox padded and clamped"` for every redaction, showing whether the
  padded box was clamped against a table cell, a neighboring word, or
  neither, plus its final `x`/`y`/`w`/`h` — the exact "why is this box
  bigger/smaller/misplaced on the other machine" answer.

Logs render as one JSON object per line by default (`Settings.log_format`,
default `"json"`) so every field above is directly diffable/greppable
(`jq`, `grep`, plain `diff` of two saved log files) — set `log_format=plain`
for a human-readable line instead. Level is `Settings.log_level` (default
`"INFO"`; set `LOG_LEVEL=DEBUG` to also see the per-redaction padding line).

## Debugging: per-document OCR output dump

Every fuzzy-match decision (`Settings.fuzzy_dictionary_scan_threshold`, `fuzzy_match_threshold`, mock-dictionary resolution, ...) runs against exactly what OCR/native-text extraction produced — so when a value that should match doesn't, the fastest first check is whether OCR itself read a special/garbled character (a smart quote, a dropped diacritic, a digit misread as a letter, ...) rather than assuming the matching logic is at fault.

By default (`Settings.ocr_output_dump_enabled=true`), every redaction request writes one JSON file per document to `data/ocr-output/{request_id}.json` (git-ignored — it's real document text, same sensitivity as the substitution ledger, and it's never exposed via the audit response or UI). Each file has one entry per page:

```json
{
  "request_id": "req_...",
  "filename": "statement.pdf",
  "page_count": 2,
  "pages": [
    {
      "page_index": 0,
      "page_kind": "scanned",
      "word_count": 214,
      "merged_text": "... exact OCR/native text for this page ...",
      "words": [
        {"text": "Jane", "x": 120, "y": 88, "w": 40, "h": 14, "confidence": 0.97, "engine_agreement": 1.0, "engines": ["rapidocr"], "char_start": 0, "char_end": 4}
      ]
    }
  ]
}
```

`ensure_ascii=False` keeps the file human-readable while still JSON-escaping any control/non-printable character, so a genuinely corrupt OCR read (mojibake, a stray control byte) shows up as an explicit `\uXXXX` escape instead of an invisible/garbled terminal glyph. Set `OCR_OUTPUT_DUMP_ENABLED=false` (`.env.local`) to stop writing it once an issue is no longer being investigated, or on a high-volume deployment to save disk.

## Debugging: visualizing pipeline stages

To find exactly where a redaction goes wrong (a table header swallowed by a redaction box, a wrong mock value, PII that never got detected, an over/under-padded box, ...), `scripts/visualize_pipeline_stages.py` re-derives every intermediate artifact `RedactPipeline` produces for **one page** and writes each stage as its own numbered, annotated PNG instead of only the final redacted output. It calls the exact same functions `RedactPipeline` calls internally (`extract_field_candidates`, `_collect_redactions`, ...) — it never changes production behavior, only observes it.

| # | Stage (`pipeline_stages/N.png`) | What to look for |
|---|------|-------------------|
| 1 | Raw rendered input page | Sanity-check the render itself (size, dpi) |
| 2 | Preprocessed/canonical image | `page_kind` (`digital` vs `scanned`), blur tier, skew — what OCR actually reads |
| 3 | Structure blocks (Docling + img2table) | `0 structure blocks` = degraded table/cell detection downstream |
| 4 | OCR ensemble word boxes | Garbled/missing words; `single-engine only` words are lower-confidence gap candidates |
| 5 | Raw PII candidates (pre mock-resolution) | Every span *before* it's assigned a mock value, colored by detection source (field-anchored / dictionary-scan / dictionary-scan fuzzy / custom-term) |
| 6 | Accepted redactions (tight box) | What survived mock resolution + dedup, and which mock value it was assigned |
| 7 | Padding | Tight vs. painted box — a padded box that swallows a neighboring cell/word is visible here |
| 8 | Brand zones | `FOOTER`/`IMAGE`/`SIGNATURE` zones (logos, stamps, signature ink) painted on top of the PII redactions |
| 9 | Final output | The **actual** production render path — real vector-native redaction for `digital` pages, paint-then-flatten raster for `scanned` pages |

### Any PDF — digital (text-layer) *or* scanned/rasterized pages

The script always takes a PDF path — page selection goes through PyMuPDF's `doc.select` — so this is the same command whether the PDF has a real, selectable text layer (`page_kind=digital`, e.g. an invoice exported from software) or is just scanned/photographed pages saved as a PDF with no text layer at all (`page_kind=scanned`, e.g. a bank's scan-to-PDF export). Which one it is gets auto-detected per page by `app/services/ocr/native_text.py::classify_and_extract` (native text bypass) — you don't tell the script which; check stage 2's title banner (`page_kind=...`) to confirm what it detected:
```powershell
python scripts\visualize_pipeline_stages.py path\to\document.pdf --page 1
```
Images land in `pipeline_stages/` by default (override with `--out-dir`).

### Raw scanned images (JPG, PNG, TIFF — not already a PDF)

This is the one case that needs an extra step: the script requires a PDF as input (again, for `doc.select` page selection), so a loose image file first needs wrapping in a throwaway single-page PDF. This doesn't change anything the pipeline sees — `app/services/ocr/page_renderer.py`'s `load_pages` treats a rendered PDF-page pixmap and a directly-loaded image identically, and the wrapped page still classifies as `page_kind=scanned` end-to-end (there's no text layer either way, same as a scanned PDF above):
```powershell
python -c "from PIL import Image; Image.open('test-input\1000099358.jpg').convert('RGB').save('test-input\1000099358.pdf')"
python scripts\visualize_pipeline_stages.py test-input\1000099358.pdf --page 1
```
Delete the intermediate `.pdf` afterward — it's only a page-selection shim, not an input the pipeline needs kept around.

### Isolating dictionary problems from pipeline-logic problems

- `--mapping-csv path\to\mappings.csv` — import an additional `source_text,mock_value` CSV on top of the base dictionary before rendering, so a doc-specific test mapping (e.g. a client-supplied `mappings.csv`) drives the run.
- `--empty-dictionary` — start from an **empty** mock dictionary instead of seeding from the real, machine-local `data/mock-dictionary/mappings.json`. Combine with `--mapping-csv` to see redactions driven *only* by a known-good CSV, with zero interference from stale/auto-learned local entries.
- `--use-real-dictionary` (default) — seed from the real `mappings.json` (a read-only copy is used; the real file is never written to).

`mappings.json` only ever grows and is never overwritten by a re-import (see "Mock dictionary" above and `app/services/pii/mapping_csv.py`'s docstring), so a `--use-real-dictionary` run and an `--empty-dictionary --mapping-csv your.csv` run of the *same page* can legitimately disagree: if a stale or wrong entry for the same `source_text` already exists locally (e.g. from an earlier auto-learned/uncorrected run), the real-dictionary run keeps it, while the clean run shows what *should* happen per the CSV. When the two runs disagree, the root cause is almost always a stale local dictionary entry, not a bug in `extract_field_candidates`/`_collect_redactions` — check stage 6's mock value against the CSV first before digging into detection logic.

## Signature redaction: ink-gap geometry, not a picture/ML model

Handwritten signatures are redacted as a `SIGNATURE` brand zone
(`app/services/pii/signature_zones.py`), painted the same way as the
`FOOTER`/`IMAGE` zones above — but detected completely differently.
Docling's layout model (the same one behind `IMAGE` zones) is trained on
printed graphics (logos, photos, charts), so it only inconsistently
recognizes cursive ink as a `picture` block: verified directly against
`repii/` samples, where two pen signatures side by side on the same page
had one boxed by Docling and the other missed outright, purely because
one stroke happened to be denser/more graphic-shaped than the other.

Rather than add a second, similarly unreliable ML model just for
signatures, signature detection instead anchors on structure this
pipeline already computes: the printed signatory name/org row
`field_extractor.py` finds in the bottom-of-page signature block (e.g.
`Wahyu Wijaya` under a signature, or `PT Schroder Investment Management
Indonesia` under an "Acknowledged by" line with no name below it). The
redaction zone is the blank vertical gap immediately above and/or below
that anchor — capped to a plausible signature height and bounded by
whatever real OCR text sits nearest on either side, so it only ever
covers blank paper, never someone else's printed text. No new
dependency, no model weights to vendor, fully deterministic from the
same OCR word geometry already extracted for every other redaction.

One related fix this required: `_group_rows` (the same-line word
clustering every field-anchored detector shares) clusters purely by
vertical adjacency with no horizontal-continuity check, so two
independent signatories printed side by side on one baseline — a common
"Authorized by" layout — used to land in a single combined row. That
combined row can exceed the signature-block word-count guards and drop
*every* signatory on that line at once (confirmed directly: two names
plus a one-letter suffix hit 5 tokens against a 4-word cap). The
signature-block detectors now split a row by horizontal gap before
applying those guards (`_split_row_by_x_gap`) — scoped to just those two
detectors, since a wide same-row gap elsewhere (table columns,
label:value rows) is already handled correctly by that mode's own
column/zone logic.

Toggle: `Settings.patch_signatures_enabled` (global default) /
`RedactOptions.patch_signatures` (per-request, UI checkbox "Cover
signatures").

## Native-text bypass: forcing scanned/OCR-only processing

By default, a PDF page carrying a real, selectable text layer (`page_kind=digital`) skips OCR entirely — its text is read straight from the PDF, and it's redacted with the vector-native path (real PyMuPDF redaction annotations, output stays a true PDF page) instead of the raster paint-then-flatten path every scanned page uses. Classification is per-page and automatic; a mixed document (e.g. a copyable cover page followed by scanned pages) needs no flag.

Set `Settings.native_text_bypass_enabled=false` (`NATIVE_TEXT_BYPASS_ENABLED=false` in `.env.local`) to disable that entirely and force **every** page through the scanned/OCR path, regardless of how much real text layer it carries: no page is ever classified `digital`, `app/services/ocr/native_text.py::classify_and_extract` is never called, and the ensemble-OCR-failure fallback to native text (`app/pipeline/redact.py`) is also disabled. Use this when a deployment needs guaranteed OCR-only behavior — e.g. a document source whose embedded text layer is known to be unreliable, or a policy that every output page must go through the raster redaction path rather than the vector-native one.

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
