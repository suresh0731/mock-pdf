import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.mock_routes import router as mock_router
from app.api.routes import router
from app.config import get_settings
from app.services.ocr.engines import configure_tesseract
from app.services.ocr.environment_check import enforce_required_engines

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_tesseract()
    enforce_required_engines()
    settings.shard_base_path.mkdir(parents=True, exist_ok=True)
    settings.recognizers_dir.mkdir(parents=True, exist_ok=True)
    (settings.shard_base_path / "audit" / "requests").mkdir(parents=True, exist_ok=True)
    settings.mock_dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    (settings.shard_base_path / "shards").mkdir(parents=True, exist_ok=True)
    yield


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
