import streamlit as st
from ocr_utils import process_image
from PIL import Image
import time
import os

st.set_page_config(page_title="OCR App", page_icon="📄")

st.title("📄 Optical Character Recognition App")

st.markdown("Upload an image to extract text using our OCR engine (TrOCR / Tesseract).")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    with st.spinner("Uploading and processing image..."):
        # Save uploaded file
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("✅ Image uploaded successfully!")

        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Process image
        with st.spinner("Running OCR..."):
            start_time = time.time()
            predictions = process_image(file_path)
            processing_time = round(time.time() - start_time, 2)

        st.markdown("### 📋 Extracted Text:")
        st.code(predictions, language='text')

        st.markdown(f"⏱️ **Processing Time:** `{processing_time} seconds`")
