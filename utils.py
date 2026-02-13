import easyocr
import pytesseract
from PIL import Image
import pdf2image  # Unused here, but kept if needed
import numpy as np
import cv2
import pandas as pd

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

def parse_ocr_results(results, departments, stores, min_conf=0.5, y_threshold=30):
    try:
        # Filter low-confidence results
        results = [r for r in results if r[2] >= min_conf]
        
        # Sort by y (row), then x (column)
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
        
        data = []
        known_stores = set(stores)
        dept_map = {dept.lower(): dept for dept in departments}

        for row in rows:
            texts = [item[1] for item in row if item[1]]  # Skip empty
            if not texts:
                continue

            # Skip any row that looks like a header (contains dept names or "STORE")
            if any(t.lower() in dept_map or t.lower() == "store" for t in texts):
                continue
            
            potential_store = texts[0]
            if potential_store.isdigit() and 1 <= len(potential_store) <= 4 and potential_store in known_stores:
                row_dict = {"STORE": potential_store}
                value_idx = 1  # Start after store
                for dept in departments:  # Assign in order, skip non-numeric
                    while value_idx < len(texts) and not texts[value_idx].replace('.', '').replace(',', '').isdigit():
                        value_idx += 1
                    if value_idx < len(texts):
                        try:
                            row_dict[dept] = float(texts[value_idx].replace(',', ''))
                        except ValueError:
                            row_dict[dept] = 0.0
                        value_idx += 1
                    else:
                        row_dict[dept] = 0.0
                data.append(row_dict)
        
        # Sum duplicates if any
        if data:
            df = pd.DataFrame(data)
            df = df.groupby("STORE", as_index=False).sum(numeric_only=True)
            data = df.to_dict(orient='records')
        
        # Defaults for missing
        extracted = {d["STORE"]: d for d in data}
        final_data = []
        for store in stores:
            if store in extracted:
                row = extracted[store]
            else:
                row = {"STORE": store}
                for dept in departments:
                    row[dept] = 0.0
            final_data.append(row)
        
        return final_data
    except Exception as e:
        raise RuntimeError(f"Error parsing OCR results: {e}")
