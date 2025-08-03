# ocr_engine.py

import cv2
import numpy as np
import pytesseract
from PIL import Image
import imutils
import matplotlib.pyplot as plt
import time

# Configure Tesseract path (optional, for local setups)
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def preprocess_image_smooth(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    thresh = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    return dilated

def get_word_boxes(image, preprocessed):
    h_proj = np.sum(preprocessed, axis=1)
    lines = []
    threshold = np.max(h_proj) * 0.05
    line_start = None
    for i, val in enumerate(h_proj):
        if val > threshold and line_start is None:
            line_start = i
        elif val <= threshold and line_start is not None:
            lines.append((line_start, i))
            line_start = None

    word_boxes = []
    for y1, y2 in lines:
        line_img = preprocessed[y1:y2, :]
        v_proj = np.sum(line_img, axis=0)
        word_start = None
        for j, val in enumerate(v_proj):
            if val > 0 and word_start is None:
                word_start = j
            elif val == 0 and word_start is not None:
                x1, x2 = word_start, j
                word_boxes.append((x1, y1, x2 - x1, y2 - y1))
                word_start = None
    return word_boxes

def extract_characters(word_img):
    gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    char_regions = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w > 5 and h > 10:
            char_regions.append((x, y, w, h))

    char_regions = sorted(char_regions, key=lambda b: b[0])
    characters = []
    for (x, y, w, h) in char_regions:
        char_img = binary[y:y+h, x:x+w]
        char_img = cv2.copyMakeBorder(char_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
        char_img = cv2.cvtColor(char_img, cv2.COLOR_GRAY2BGR)
        config = r'--oem 3 --psm 10 -l eng+script/Handwriting'
        char = pytesseract.image_to_string(char_img, config=config).strip()
        if char:
            characters.append({
                'char': char[0],
                'position': (x, y, w, h),
                'image': char_img
            })
    return characters

def predict_word(image_box, char_preds):
    from_chars = ''.join([c['char'] for c in char_preds])
    config = r'--oem 3 --psm 8 -l eng+script/Handwriting'
    word_text_direct = pytesseract.image_to_string(image_box, config=config).strip()
    gray = cv2.cvtColor(image_box, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    word_text_thresh = pytesseract.image_to_string(bin_img, config=config).strip()
    final = max([from_chars, word_text_direct, word_text_thresh], key=len)
    return final

def visualize_word_prediction(original, box, char_preds, final_word):
    vis = original.copy()
    x, y, w, h = box
    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
    for c in char_preds:
        cx, cy, cw, ch = c['position']
        cv2.rectangle(vis, (x+cx, y+cy), (x+cx+cw, y+cy+ch), (255, 0, 0), 1)
        cv2.putText(vis, c['char'], (x+cx, y+cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    cv2.putText(vis, final_word, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    return vis

def process_image(image_path):
    original = cv2.imread(image_path)
    preprocessed = preprocess_image_smooth(original)
    word_boxes = get_word_boxes(original, preprocessed)

    all_predictions = []
    for i, box in enumerate(word_boxes):
        x, y, w, h = box
        word_img = original[y:y+h, x:x+w]
        chars_info = extract_characters(word_img)
        final_word = predict_word(word_img, chars_info)
        vis = visualize_word_prediction(original, box, chars_info, final_word)
        all_predictions.append({
            'word': final_word,
            'box': box,
            'characters': chars_info,
            'visualization': vis
        })
    return all_predictions, original
