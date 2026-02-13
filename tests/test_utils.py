import sys
import types
import pytest
import numpy as np
from PIL import Image

# Stub out heavy optional imports that utils.py pulls in at module level
for mod_name in ("easyocr", "pytesseract", "pdf2image"):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

from utils import preprocess_image, normalize_ocr_char, fuzzy_store_match, parse_ocr_number, parse_single_dept_ocr


# --- preprocess_image tests ---

class TestPreprocessImage:
    def test_accepts_numpy_bgr(self):
        """BGR numpy array should be processed without error."""
        img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        result = preprocess_image(img)
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 2  # grayscale output

    def test_accepts_numpy_grayscale(self):
        """Grayscale numpy array should be processed without error."""
        img = np.random.randint(0, 255, (100, 200), dtype=np.uint8)
        result = preprocess_image(img)
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 2

    def test_accepts_pil_image(self):
        """PIL Image should be converted and processed."""
        pil_img = Image.fromarray(np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8))
        result = preprocess_image(pil_img)
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 2

    def test_output_dimensions_match_input(self):
        """Output should have the same height and width as input."""
        img = np.random.randint(0, 255, (150, 300, 3), dtype=np.uint8)
        result = preprocess_image(img)
        assert result.shape == (150, 300)

    def test_rejects_invalid_input(self):
        """Non-image input should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="Error in image preprocessing"):
            preprocess_image("not_an_image")

    def test_rejects_none(self):
        with pytest.raises(RuntimeError):
            preprocess_image(None)

    def test_white_image(self):
        """Solid white image should process without error."""
        img = np.ones((100, 200, 3), dtype=np.uint8) * 255
        result = preprocess_image(img)
        assert result.shape == (100, 200)

    def test_black_image(self):
        """Solid black image should process without error."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = preprocess_image(img)
        assert result.shape == (100, 200)


# --- Helper ---

def _make_result(x, y, text, conf=0.9):
    """Helper to build an OCR result entry: (bbox, text, confidence)."""
    bbox = [[x, y], [x + 50, y], [x + 50, y + 20], [x, y + 20]]
    return (bbox, text, conf)


# --- normalize_ocr_char tests ---

class TestNormalizeOcrChar:
    def test_maps_O_to_zero(self):
        assert normalize_ocr_char('O') == '0'

    def test_maps_lowercase_l_to_one(self):
        assert normalize_ocr_char('l') == '1'

    def test_passes_through_digit(self):
        assert normalize_ocr_char('7') == '7'

    def test_passes_through_unknown_letter(self):
        assert normalize_ocr_char('x') == 'x'


# --- fuzzy_store_match tests ---

class TestFuzzyStoreMatch:
    STORES = {"100", "200", "300"}

    def test_exact_match(self):
        assert fuzzy_store_match("100", self.STORES) == "100"

    def test_ocr_O_for_zero(self):
        assert fuzzy_store_match("1O0", self.STORES) == "100"

    def test_ocr_l_for_one(self):
        assert fuzzy_store_match("l00", self.STORES) == "100"

    def test_no_match(self):
        assert fuzzy_store_match("999", self.STORES) is None

    def test_strips_whitespace(self):
        assert fuzzy_store_match("  200 ", self.STORES) == "200"


# --- parse_ocr_number tests ---

class TestParseOcrNumber:
    def test_plain_integer(self):
        assert parse_ocr_number("500") == 500.0

    def test_float(self):
        assert parse_ocr_number("12.5") == 12.5

    def test_comma_separated(self):
        assert parse_ocr_number("1,500") == 1500.0

    def test_ocr_O_for_zero(self):
        assert parse_ocr_number("5O0") == 500.0

    def test_garbage_returns_none(self):
        assert parse_ocr_number("abc") is None

    def test_spaces_stripped(self):
        assert parse_ocr_number(" 300 ") == 300.0


# --- parse_single_dept_ocr tests ---

class TestParseSingleDeptOcr:
    STORES = ["100", "200", "300"]

    def test_basic_store_value_pairs(self):
        results = [
            _make_result(0, 0, "100"),
            _make_result(100, 0, "500"),
            _make_result(0, 50, "200"),
            _make_result(100, 50, "300"),
        ]
        cubes, skipped = parse_single_dept_ocr(results, self.STORES)
        assert cubes["100"] == 500.0
        assert cubes["200"] == 300.0
        assert "300" not in cubes  # no data row for store 300

    def test_fuzzy_store_matching(self):
        results = [
            _make_result(0, 0, "1O0"),   # O instead of 0
            _make_result(100, 0, "750"),
        ]
        cubes, skipped = parse_single_dept_ocr(results, self.STORES)
        assert cubes["100"] == 750.0

    def test_fuzzy_number_parsing(self):
        results = [
            _make_result(0, 0, "200"),
            _make_result(100, 0, "5O0"),  # O instead of 0 in number
        ]
        cubes, skipped = parse_single_dept_ocr(results, self.STORES)
        assert cubes["200"] == 500.0

    def test_low_confidence_filtered(self):
        results = [
            _make_result(0, 0, "100", conf=0.9),
            _make_result(100, 0, "500", conf=0.1),  # filtered
        ]
        cubes, skipped = parse_single_dept_ocr(results, self.STORES, min_conf=0.5)
        # store matched but no value above threshold
        assert "100" not in cubes

    def test_duplicate_stores_summed(self):
        results = [
            _make_result(0, 0, "100"),
            _make_result(100, 0, "200"),
            _make_result(0, 50, "100"),
            _make_result(100, 50, "300"),
        ]
        cubes, skipped = parse_single_dept_ocr(results, self.STORES)
        assert cubes["100"] == 500.0

    def test_skipped_rows_reported(self):
        results = [
            _make_result(0, 0, "HEADER"),
            _make_result(100, 0, "TEXT"),
            _make_result(0, 50, "100"),
            _make_result(100, 50, "400"),
        ]
        cubes, skipped = parse_single_dept_ocr(results, self.STORES)
        assert cubes["100"] == 400.0
        assert len(skipped) == 1
        assert "HEADER" in skipped[0]

    def test_empty_results(self):
        cubes, skipped = parse_single_dept_ocr([], self.STORES)
        assert cubes == {}
        assert skipped == []

    def test_unknown_store_skipped(self):
        results = [
            _make_result(0, 0, "999"),
            _make_result(100, 0, "500"),
        ]
        cubes, skipped = parse_single_dept_ocr(results, self.STORES)
        assert "999" not in cubes
