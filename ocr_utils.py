import cv2
import pytesseract
import numpy as np
import time
import os
from PIL import Image

# Make sure pytesseract is correctly linked
pytesseract.pytesseract.tesseract_cmd = "tesseract"  # Only works if tesseract is in PATH

def process_image(image_path):
    try:
        # 1. Read and preprocess the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Unable to read the image. Check the format.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        time.sleep(0.5)  # Simulated progress step

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        time.sleep(0.5)

        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        time.sleep(0.5)

        # 2. OCR at word level
        data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)

        words_with_chars = []

        for i in range(len(data['text'])):
            word = data['text'][i].strip()
            if word:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                word_crop = gray[y:y + h, x:x + w]

                chars = []
                try:
                    boxes = pytesseract.image_to_boxes(word_crop)
                    for b in boxes.splitlines():
                        b = b.strip().split(' ')
                        if len(b) >= 1:
                            char = b[0]
                            chars.append({'char': char})
                except:
                    chars = [{'char': c} for c in word]

                words_with_chars.append({'word': word, 'characters': chars})

        time.sleep(0.5)
        return words_with_chars

    except Exception as e:
        print(f"[ERROR] {e}")
        return [{"word": "[OCR Failed]", "characters": []}]
