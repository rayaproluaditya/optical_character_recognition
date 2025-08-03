# utils/ocr_engine.py

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import pytesseract
import cv2
import numpy as np

# Load TrOCR model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def read_text_trocr(image_pil):
    """
    Extract text from PIL image using TrOCR (AI-based).
    """
    pixel_values = processor(images=image_pil, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text.strip()

def read_text_tesseract(image_cv2):
    """
    Extract text from OpenCV image using Tesseract (traditional).
    """
    gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip()
