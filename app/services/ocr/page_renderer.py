import io
import logging
from pathlib import Path

from PIL import Image

from app.config import get_settings

logger = logging.getLogger(__name__)


def load_pages(file_bytes: bytes, filename: str, dpi: int = 200) -> list[Image.Image]:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _pdf_to_images(file_bytes, dpi=dpi)
    image = Image.open(io.BytesIO(file_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return [image]


def _pdf_to_images(file_bytes: bytes, dpi: int = 200) -> list[Image.Image]:
    settings = get_settings()
    try:
        import fitz

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            if img.mode != "RGB":
                img = img.convert("RGB")
            pages.append(img)
        if pages:
            return pages
    except Exception as exc:
        logger.warning("PyMuPDF render failed, trying pdf2image: %s", exc)

    from pdf2image import convert_from_bytes

    kwargs = {"dpi": dpi}
    if settings.poppler_path:
        kwargs["poppler_path"] = settings.poppler_path
    images = convert_from_bytes(file_bytes, **kwargs)
    return [img.convert("RGB") for img in images]
