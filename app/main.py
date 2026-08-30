import asyncio
import contextlib
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.mock_routes import router as mock_router
from app.api.routes import router
from app.config import get_settings
from app.logging_config import configure_logging
from app.services.ocr.engines import configure_tesseract
from app.services.ocr.environment_check import (
    enforce_required_engines,
    log_effective_settings,
    log_environment_fingerprint,
)
from app.services.watch.folder_watcher import FolderWatcher

configure_logging(get_settings().log_level, get_settings().log_format)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_tesseract()
    # Logged first and unconditionally (unlike enforce_required_engines,
    # which only checks Settings.ocr_required_engines) so a "why did this
    # machine behave differently" investigation always has a same-shaped
    # snapshot to diff against another machine's, even when every engine
    # required is present and nothing looks wrong at startup.
    log_environment_fingerprint()
    # .env.local is git-ignored/machine-local — a version/package match does
    # NOT mean a config match. Log every resolved knob so a padding/table/
    # restrict_to_known_mappings override on one machine is a log diff away.
    log_effective_settings()
    enforce_required_engines()
    settings.shard_base_path.mkdir(parents=True, exist_ok=True)
    (settings.shard_base_path / "audit" / "requests").mkdir(parents=True, exist_ok=True)
    settings.mock_dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    (settings.shard_base_path / "shards").mkdir(parents=True, exist_ok=True)

    # Opt-in folder-watch ingestion (WATCH_ENABLED=true) — additive to the
    # UI/API upload, off by default so it never changes existing behavior.
    watch_task: asyncio.Task | None = None
    if settings.watch_enabled:
        settings.watch_input_dir.mkdir(parents=True, exist_ok=True)
        settings.watch_output_dir.mkdir(parents=True, exist_ok=True)
        watch_task = asyncio.create_task(FolderWatcher(settings=settings).run_forever())

    yield

    if watch_task is not None:
        watch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watch_task


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    app.include_router(mock_router)
    return app


def create_app_with_ui() -> FastAPI:
    application = create_app()
    from app.ui.redact_portal import setup_ui

    setup_ui(application)
    return application


# `from app.main import create_app` must not mount NiceGUI (Story 10 e2e).
# uvicorn app.main:app still gets the portal outside pytest.
app = create_app() if "pytest" in sys.modules else create_app_with_ui()
