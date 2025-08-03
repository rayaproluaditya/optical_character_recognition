import streamlit as st
from PIL import Image
import numpy as np
import cv2
import time

from utils.ocr_engine import read_text_trocr, read_text_tesseract, preprocess_image_smooth, get_word_boxes

st.set_page_config(page_title="OCR Extractor", layout="centered")

st.title("🧠 Smart OCR Extractor")
st.markdown("Upload a handwritten or typed image and extract text using AI.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    pil_image = Image.open(uploaded_file).convert("RGB")
    st.image(pil_image, caption="Uploaded Image", use_column_width=True)

    # Convert PIL image to OpenCV (numpy) format
    image = np.array(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    method = st.radio("Choose OCR method:", ("TrOCR (AI-based)", "Tesseract (Traditional)"))

    if st.button("🔍 Extract Text"):
        with st.spinner("Analyzing..."):
            time.sleep(1)

            if method == "TrOCR (AI-based)":
                result = read_text_trocr(pil_image)
            else:
                result = read_text_tesseract(image)

        st.success("✅ Text Extracted:")
        st.code(result, language="text")
