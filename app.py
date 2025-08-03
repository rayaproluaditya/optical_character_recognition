import streamlit as st
from PIL import Image
import numpy as np
import cv2
import time

from utils.ocr_engine import read_text_trocr, read_text_tesseract

# Streamlit page configuration
st.set_page_config(
    page_title="OCR Extractor",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Main Title
st.title("🧠 Smart OCR Extractor")
st.markdown("""
Upload a **handwritten or printed image** and extract text using:
- 🤖 AI-based TrOCR (by Microsoft)
- 🔍 Tesseract OCR (Traditional)
""")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])

# Proceed only if a file is uploaded
if uploaded_file:
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error("❌ Error opening the image file. Please upload a valid image.")
    else:
        st.image(pil_image, caption="📸 Uploaded Image", use_column_width=True)

        # Convert PIL image to OpenCV (numpy) format
        image_cv2 = np.array(pil_image)
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

        # OCR Method Selection
        method = st.radio("🛠 Choose OCR Method:", ("TrOCR (AI-based)", "Tesseract (Traditional)"))

        # Button to trigger OCR
        if st.button("🔍 Extract Text"):
            with st.spinner("🧠 Extracting text..."):
                time.sleep(1)
                try:
                    if method == "TrOCR (AI-based)":
                        result = read_text_trocr(pil_image)
                    else:
                        result = read_text_tesseract(image_cv2)
                    st.success("✅ Text Extracted:")
                    st.code(result, language="text")
                except Exception as e:
                    st.error(f"❌ OCR failed: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by R. Aditya Prakash")
