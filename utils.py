import logging
import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)


def preprocess_image(img):
    try:
        if not isinstance(img, np.ndarray):
            if isinstance(img, Image.Image):
                img = np.array(img)
            else:
                raise ValueError("Input must be PIL Image or numpy array")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Deskewing
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

        skew_angle = 0
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
            skew_angle = np.median(angles)
            if abs(skew_angle) >= 45:
                skew_angle = 0  # Avoid extreme rotations

        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
        deskewed = cv2.warpAffine(blurred, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(deskewed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Dilation (milder kernel)
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        # Return grayscale deskewed for OCR (binary can lose details)
        return deskewed  # Or dilated if binary preferred
    except cv2.error as e:
        raise RuntimeError(f"OpenCV error in preprocessing: {e}")
    except Exception as e:
        raise RuntimeError(f"Error in image preprocessing: {e}")

def normalize_ocr_char(c):
    """Map common OCR misreads to the intended digit."""
    _MAP = {'O': '0', 'o': '0', 'l': '1', 'I': '1', 'S': '5',
            'B': '8', 'Z': '2', 'z': '2', 'g': '9', 'D': '0'}
    return _MAP.get(c, c)


def fuzzy_store_match(text, known_stores):
    """Try to match OCR text to a known store number."""
    text = text.strip()
    if text in known_stores:
        return text
    normalized = ''.join(normalize_ocr_char(c) for c in text)
    if normalized in known_stores:
        return normalized
    return None


def parse_ocr_number(text):
    """Parse a number from OCR text, handling common misreads."""
    text = text.strip().replace(',', '').replace(' ', '')
    try:
        return float(text)
    except ValueError:
        pass
    normalized = ''.join(normalize_ocr_char(c) for c in text)
    normalized = normalized.replace(',', '').replace(' ', '')
    try:
        return float(normalized)
    except ValueError:
        return None


def _group_into_rows(results, y_threshold=30):
    """Group OCR results into rows by Y position."""
    sorted_results = sorted(results, key=lambda r: (r[0][0][1], r[0][0][0]))
    rows = []
    current_row = []
    prev_y = None
    for bbox, text, conf in sorted_results:
        y = bbox[0][1]
        if prev_y is None or abs(y - prev_y) < y_threshold:
            current_row.append((bbox, text.strip(), conf))
        else:
            if current_row:
                rows.append(sorted(current_row, key=lambda item: item[0][0][0]))
            current_row = [(bbox, text.strip(), conf)]
        prev_y = y
    if current_row:
        rows.append(sorted(current_row, key=lambda item: item[0][0][0]))
    return rows


def parse_single_dept_ocr(results, stores, min_conf=0.5, y_threshold=30):
    """Parse OCR results from a single-department image.

    Each row is expected to contain a store number and a cube value.
    Returns (store_cubes, skipped_rows) where store_cubes is a dict
    {store: cube_value} and skipped_rows is a list of unrecognized row texts.
    """
    results = [r for r in results if r[2] >= min_conf]
    rows = _group_into_rows(results, y_threshold)

    known_stores = set(stores)
    store_cubes = {}
    skipped_rows = []

    for row in rows:
        texts = [item[1] for item in row if item[1]]
        if not texts:
            continue

        store = None
        value = None
        for t in texts:
            if store is None:
                matched = fuzzy_store_match(t, known_stores)
                if matched is not None:
                    store = matched
                    continue
            if value is None:
                parsed = parse_ocr_number(t)
                if parsed is not None:
                    value = parsed

        if store is not None and value is not None:
            store_cubes[store] = store_cubes.get(store, 0) + value
        elif texts:
            skipped_rows.append(texts)

    return store_cubes, skipped_rows


def run_ocr(image_bytes, ocr_engine, is_pdf=False):
    """Run OCR on image bytes and return a list of (bbox, text, conf) tuples.

    Handles both single images and multi-page PDFs.  EasyOCR reader is
    cached via ``get_easyocr_reader`` so it's only loaded once per session.

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of the uploaded file.
    ocr_engine : str
        ``"EasyOCR"`` or ``"Tesseract"``.
    is_pdf : bool
        If True, ``image_bytes`` is a PDF and will be split into pages first.

    Returns
    -------
    list[tuple]
        Each element is ``(bbox, text, confidence)``.
    """
    import io
    import pytesseract
    from pdf2image import convert_from_bytes

    all_results = []

    if is_pdf:
        pages = convert_from_bytes(image_bytes)
    else:
        pages = [Image.open(io.BytesIO(image_bytes))]

    for page in pages:
        preprocessed_img = preprocess_image(page)
        if ocr_engine == "EasyOCR":
            reader = get_easyocr_reader()
            results = reader.readtext(preprocessed_img)
        else:
            tess_data = pytesseract.image_to_data(
                preprocessed_img, output_type=pytesseract.Output.DICT
            )
            results = []
            for j in range(len(tess_data["text"])):
                if float(tess_data["conf"][j]) > 0:
                    x = tess_data["left"][j]
                    y = tess_data["top"][j]
                    w = tess_data["width"][j]
                    h = tess_data["height"][j]
                    bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                    text = tess_data["text"][j]
                    conf = float(tess_data["conf"][j]) / 100.0
                    results.append((bbox, text, conf))
        all_results.extend(results)

    return all_results


def get_easyocr_reader():
    """Return a cached EasyOCR reader instance.

    Uses ``st.cache_resource`` when running inside Streamlit, otherwise
    creates a plain instance (useful for tests).
    """
    try:
        import streamlit as st

        @st.cache_resource
        def _load():
            import easyocr
            return easyocr.Reader(["en"])

        return _load()
    except Exception:
        import easyocr
        return easyocr.Reader(["en"])
