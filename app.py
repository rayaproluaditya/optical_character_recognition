import streamlit as st
from PIL import Image
from utils.ocr_engine import read_text_trocr, read_text_tesseract
import time

st.set_page_config(page_title="OCR Extractor", layout="centered")

st.title("🧠 Smart OCR Extractor")
st.markdown("Upload a handwritten or typed image and extract text using AI.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    method = st.radio("Choose OCR method:", ("TrOCR (AI-based)", "Tesseract (Traditional)"))

    if st.button("🔍 Extract Text"):
        with st.spinner("Analyzing..."):
            time.sleep(1)
            if method == "TrOCR (AI-based)":
                result = read_text_trocr(image)
            else:
                result = read_text_tesseract(image)
        st.success("✅ Text Extracted:")
        st.code(result, language="text")
