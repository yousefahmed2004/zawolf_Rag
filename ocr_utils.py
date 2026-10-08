"""
OCR Fallback Module
====================

Pure-library OCR fallback: no Tesseract, no Poppler, no system-level
installs, and no dependency on Gemini or any other LLM (so this keeps
working even if the chat model in rag.py is swapped for a free/local
model later).

Two pip packages handle everything:

    pip install pymupdf easyocr

- pymupdf (imported as `fitz`)  -> rasterizes a PDF page straight to
  an image in memory. Ships as a self-contained wheel on Windows/Mac/
  Linux, so it replaces Poppler entirely — nothing else to install.

- easyocr                       -> the actual OCR engine, pure Python
  (PyTorch-based). Supports Arabic + English out of the box, so it
  replaces Tesseract entirely — no tesseract.exe, no PATH setup.

Notes:

- The first time OCR runs, EasyOCR downloads its recognition model
  weights (a few hundred MB) and caches them locally — this needs a
  one-time internet connection, after which it works fully offline.
- EasyOCR runs on CPU by default here (gpu=False) since this project
  doesn't assume a GPU is available. If you have a CUDA GPU, you can
  speed it up by setting OCR_USE_GPU=true in .env.
- Languages are configurable via OCR_LANGUAGES in .env (default
  "ar,en"). See EasyOCR's docs for supported language codes.

If these dependencies are missing, `OCR_AVAILABLE` will be False and
`ocr_pdf_page()` will raise a clear RuntimeError instead of silently
failing or crashing the app at import time.
"""

import os

try:
    import fitz  # PyMuPDF
    import numpy as np
    import easyocr

    OCR_AVAILABLE = True

except ImportError:
    OCR_AVAILABLE = False


# =========================
# Text-quality heuristic
# =========================
#
# Used to decide whether a page extracted by pypdf is "real" text
# or whether it's actually a scanned/image-based page (empty or
# near-empty / mostly non-alphabetic extraction artifacts).

MIN_TEXT_LENGTH = 30
MIN_ALPHA_RATIO = 0.3


def needs_ocr(extracted_text: str) -> bool:
    """
    Returns True if the text extracted by pypdf looks insufficient,
    meaning the page is likely scanned/image-based and should go
    through OCR instead.
    """

    stripped = (extracted_text or "").strip()

    if len(stripped) < MIN_TEXT_LENGTH:
        return True

    alpha_chars = sum(
        1 for ch in stripped if ch.isalpha()
    )

    alpha_ratio = alpha_chars / max(len(stripped), 1)

    return alpha_ratio < MIN_ALPHA_RATIO


# =========================
# EasyOCR reader (lazy singleton)
# =========================
#
# Loading EasyOCR's models takes a few seconds, so we load them once
# on first use rather than on every page/request.

_reader = None


def _get_reader():

    global _reader

    if _reader is None:

        langs = [
            lang.strip()
            for lang in os.getenv("OCR_LANGUAGES", "ar,en").split(",")
            if lang.strip()
        ]

        use_gpu = os.getenv("OCR_USE_GPU", "false").lower() == "true"

        _reader = easyocr.Reader(
            langs,
            gpu=use_gpu
        )

    return _reader


# =========================
# PDF page -> image (in-memory, no Poppler)
# =========================

def _pdf_page_to_numpy(pdf_path, page_number, dpi=200):

    doc = fitz.open(pdf_path)

    page = doc.load_page(page_number - 1)

    zoom = dpi / 72

    matrix = fitz.Matrix(zoom, zoom)

    pix = page.get_pixmap(matrix=matrix)

    image = np.frombuffer(
        pix.samples,
        dtype=np.uint8
    ).reshape(pix.height, pix.width, pix.n)

    doc.close()

    if pix.n == 4:
        image = image[:, :, :3]

    return image


# =========================
# OCR execution
# =========================

def ocr_pdf_page(
    pdf_path: str,
    page_number: int,
    dpi: int = 200
) -> str:
    """
    Rasterizes a single page of a PDF file on disk (1-indexed, matching
    pypdf's enumeration used in app.py) and runs OCR on it, returning
    the recognized text.
    """

    if not OCR_AVAILABLE:
        raise RuntimeError(
            "OCR fallback requires 'pymupdf' and 'easyocr'. "
            "Install with: pip install pymupdf easyocr"
        )

    image = _pdf_page_to_numpy(
        pdf_path,
        page_number,
        dpi=dpi
    )

    reader = _get_reader()

    # paragraph=True groups nearby text boxes together so multi-line
    # text stays readable instead of one word per line.
    text_lines = reader.readtext(
        image,
        detail=0,
        paragraph=True
    )

    return "\n".join(text_lines).strip()