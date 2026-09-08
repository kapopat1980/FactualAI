"""Image input via OCR (Section 5.2 of the paper).

Loads an image with OpenCV, converts to grayscale, histogram-equalizes for
lighting normalization, then extracts text with Tesseract (pytesseract).
Requires the `tesseract-ocr` system binary in addition to the Python
bindings; both are optional extras (`pip install factualai[ocr]`) so the
core package can be installed and tested without them.
"""
from __future__ import annotations

OCR_PREVIEW_CHARS = 300


def extract_text_from_image(image_path: str) -> str:
    """Run the OCR pipeline described in Section 5.2 and return the extracted
    text (full text; callers truncate to OCR_PREVIEW_CHARS for the API
    response, matching the paper's user-facing preview)."""
    import cv2
    import pytesseract

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return pytesseract.image_to_string(equalized)
