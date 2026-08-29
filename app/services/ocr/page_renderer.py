import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class RenderedPage:
    """One rasterized page plus (PDF input only) its source ``fitz.Page``.

    ``fitz_page`` is ``None`` for non-PDF inputs (jpg/png/tiff) and
    whenever the pdf2image fallback path is used (PyMuPDF itself failed to
    open/render the file) — both cases are always treated as "scanned" by
    ``app.services.ocr.native_text``, identical to pre-native-text-bypass
    behavior, since there's no text layer to read natively either way.

    ``_doc`` keeps the owning ``fitz.Document`` alive for as long as this
    page is referenced: a ``fitz.Page`` is only valid while its parent
    Document hasn't been garbage collected, and without this reference the
    Document could be collected as soon as ``_pdf_to_images`` returns while
    the pipeline still holds onto ``fitz_page`` for native-text extraction.
    """

    image: Image.Image
    fitz_page: object | None = None
    _doc: object | None = field(default=None, repr=False, compare=False)


def load_pages(file_bytes: bytes, filename: str, dpi: int = 200) -> list[RenderedPage]:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _pdf_to_images(file_bytes, dpi=dpi)
    image = Image.open(io.BytesIO(file_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return [RenderedPage(image=image)]


def _pdf_to_images(file_bytes: bytes, dpi: int = 200) -> list[RenderedPage]:
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
            pages.append(RenderedPage(image=img, fitz_page=page, _doc=doc))
        if pages:
            return pages
    except Exception as exc:
        logger.warning("PyMuPDF render failed, trying pdf2image: %s", exc)

    from pdf2image import convert_from_bytes

    kwargs = {"dpi": dpi}
    if settings.poppler_path:
        kwargs["poppler_path"] = settings.poppler_path
    images = convert_from_bytes(file_bytes, **kwargs)
    return [RenderedPage(image=img.convert("RGB")) for img in images]
