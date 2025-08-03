from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from ocr_utils import process_image
import time

UPLOAD_FOLDER = 'uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            start_time = time.time()
            predictions = process_image(filepath)
            processing_time = round(time.time() - start_time, 2)

            return render_template("index.html", predictions=predictions, processing_time=processing_time, uploaded=True, filename=filename)

    return render_template("index.html", uploaded=False)

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
