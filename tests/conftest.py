"""Shared pytest configuration.

Runs `tests/integration/` after every top-level `tests/*.py` module. Pytest's
default collection order is filesystem order, and `integration` sorts before
`test_*.py` alphabetically ('i' < 't') — so a plain `pytest tests/` collects
integration tests *first*. Several integration tests boot the real FastAPI
app (`create_app()` inside a `TestClient`), which runs the startup
OCR-engine-availability check (`app.services.ocr.environment_check`) and
imports the real `rapidocr`/`easyocr` packages as a side effect. Once
imported, they stay in `sys.modules` for the rest of the process — so a
later unit test asserting "the ensemble unit tests never load a real GPU-
capable engine" (`tests/test_ensemble.py::
test_ensemble_tests_do_not_load_gpu_engines`) would incorrectly fail purely
due to collection order, not anything the unit test itself did. Running
integration last avoids this cross-file leakage without weakening either
test's intent.
"""


def pytest_collection_modifyitems(items):
    items.sort(key=lambda item: "integration" in item.nodeid.replace("\\", "/").split("/"))
