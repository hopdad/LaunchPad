import easyocr
import pytesseract
from PIL import Image
import pdf2image
import numpy as np
import cv2

def preprocess_image(img):
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)
        
        skew_angle = np.median(angles)
        if abs(skew_angle) < 45:
            (h, w) = gray.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
            deskewed = cv2.warpAffine(blurred, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        else:
            deskewed = blurred
    else:
        deskewed = blurred
    
    thresh = cv2.adaptiveThreshold(deskewed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    return dilated

def parse_ocr_results(results, departments, stores):
    sorted_results = sorted(results, key=lambda r: (r[0][0][1], r[0][0][0]))
    
    rows = []
    current_row = []
    prev_y = None
    y_threshold = 30
    for bbox, text, conf in sorted_results:
        y = bbox[0][1]
        if prev_y is None or abs(y - prev_y) < y_threshold:
            current_row.append((bbox, text, conf))
        else:
            if current_row:
                rows.append(sorted(current_row, key=lambda item: item[0][0][0]))
            current_row = [(bbox, text, conf)]
        prev_y = y
    if current_row:
        rows.append(sorted(current_row, key=lambda item: item[0][0][0]))
    
    data = []
    known_stores = set(stores)
    dept_map = {dept.lower(): dept for dept in departments}
    current_store = None
    row_dict = {}
    
    for row in rows:
        texts = [item[1].strip() for item in row]
        if not texts:
            continue
        
        potential_store = texts[0]
        if potential_store.isdigit() and 1 <= len(potential_store) <= 4 and potential_store in known_stores:
            if current_store:
                data.append(row_dict)
            current_store = potential_store
            row_dict = {"STORE": current_store}
            dept_idx = 0
            for text in texts[1:]:
                if text.replace('.', '').isdigit() and dept_idx < len(departments):
                    row_dict[departments[dept_idx]] = float(text)
                    dept_idx += 1
                elif text.lower() in dept_map:
                    continue
        elif current_store:
            pass
    
    if current_store:
        data.append(row_dict)
    
    extracted = {d["STORE"]: d for d in data if "STORE" in d}
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
