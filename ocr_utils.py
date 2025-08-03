import cv2
import pytesseract
import numpy as np
import time

# Optional: Set path if Tesseract is not in PATH
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def process_image(image_path, progress_callback=None):
    try:
        # STEP 1: Load and preprocess image
        if progress_callback: progress_callback(0, "Reading Image...")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Unable to read image.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        time.sleep(0.3)
        if progress_callback: progress_callback(30, "Applying Blur...")
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        time.sleep(0.3)

        if progress_callback: progress_callback(60, "Applying Threshold...")
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        time.sleep(0.3)

        if progress_callback: progress_callback(80, "Running OCR...")
        data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
        result = []

        for i in range(len(data['text'])):
            word = data['text'][i].strip()
            if word:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                word_crop = gray[y:y + h, x:x + w]
                characters = []

                try:
                    boxes = pytesseract.image_to_boxes(word_crop)
                    for b in boxes.strip().splitlines():
                        char_data = b.split(' ')
                        if len(char_data) >= 1:
                            characters.append(char_data[0])
                except Exception:
                    characters = list(word)

                result.append({
                    "word": word,
                    "characters": characters,
                    "position": f"{x}x{y}"
                })

        if progress_callback: progress_callback(100, "Done.")
        return result

    except Exception as e:
        return [{"word": "[OCR Failed]", "characters": [], "position": ""}]
