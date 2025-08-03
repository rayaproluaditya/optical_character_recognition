import cv2
import pytesseract
import numpy as np
import time
import os

# Ensure tesseract is accessible
pytesseract.pytesseract.tesseract_cmd = "tesseract"

def process_image(image_path):
    try:
        # Step 1: Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Unable to read the image. Check the file format.")

        # Step 2: Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        time.sleep(0.5)  # simulate progress

        # Step 3: Apply Gaussian Blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        time.sleep(0.5)

        # Step 4: Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        time.sleep(0.5)

        # Step 5: OCR text extraction (straightforward full text)
        text = pytesseract.image_to_string(thresh)
        time.sleep(0.5)

        # Clean up and return final prediction
        cleaned_text = text.strip()
        return cleaned_text if cleaned_text else "[No readable text found]"

    except Exception as e:
        print(f"[ERROR] {e}")
        return "[OCR Failed]"
