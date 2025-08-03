import streamlit as st
from ocr_utils import process_image
import os
import time

st.set_page_config(page_title="OCR App", layout="centered")
st.title("📝 OCR Character Extractor")

uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    file_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(file_path, caption="Uploaded Image", use_column_width=True)

    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(percent, msg):
        progress_bar.progress(percent)
        status_text.text(f"Step {int(percent / 33.4) + 1}: {msg}")

    st.write("🔄 Processing image...")
    start = time.time()
    predictions = process_image(file_path, progress_callback=update_progress)
    end = time.time()
    duration = round(end - start, 2)

    st.success(f"✅ OCR Completed in {duration} seconds")
    st.markdown("### Step 3: Recognition Results")

    if predictions and predictions[0]["word"] == "[OCR Failed]":
        st.error("❌ OCR failed to extract text. Try another image.")
    else:
        st.markdown("#### Extracted Table")
        st.write("Word | Characters | Position")
        st.write("--- | --- | ---")
        for item in predictions:
            word = item['word']
            chars = ', '.join(item['characters'])
            pos = item['position']
            st.write(f"{word} | {chars} | {pos}")
