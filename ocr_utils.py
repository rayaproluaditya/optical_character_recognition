# ocr_utils.py for Streamlit
import cv2
import pytesseract
import numpy as np
import time
from PIL import Image

# Make sure pytesseract is correctly linked
pytesseract.pytesseract.tesseract_cmd = "tesseract"  # Requires tesseract in PATH

def process_image_streamlit(image):
    try:
        # Convert the uploaded file to OpenCV format
        image = np.array(Image.open(image).convert("RGB"))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 1. Grayscale conversion
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        time.sleep(0.5)

        # 2. Apply Gaussian Blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        time.sleep(0.5)

        # 3. Thresholding
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        time.sleep(0.5)

        # 4. OCR data extraction
        data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)

        words_data = []

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
                            chars.append(char)
                except:
                    chars = list(word)

                words_data.append({
                    'word': word,
                    'characters': ''.join(chars),
                    'position': f"{x}x{y}"
                })

        time.sleep(0.5)
        return words_data

    except Exception as e:
        print(f"[ERROR] {e}")
        return [{"word": "[OCR Failed]", "characters": "", "position": "-"}]
