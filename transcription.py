import os
import cv2
import string
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np

# 英字（大文字・小文字）、数字、句読点を含むリスト
classes = list(string.ascii_letters + string.digits + string.punctuation)
image_size = (28, 28)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = load_model('./model_transcription.h5')

def dynamic_resize(roi):
    return cv2.resize(roi, image_size)

def enhance_text(roi):
    edges = cv2.Canny(roi, 50, 150)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    return dilated

def detect_text_regions(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = [cv2.boundingRect(contour) for contour in contours]
    regions.sort(key=lambda x: x[0])
    return gray, regions

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # ファイルがアップロードされたかチェック
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # ユーザがファイルを選択しなかった場合
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        # ファイルが許可された拡張子を持っているかチェック
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # ここでファイルをモデルに入力して予測する処理
            gray, regions = detect_text_regions(file_path)
            recognized_text = ""
            for rect in regions:
                x, y, w, h = rect
                roi = gray[y: y+h, x: x+w]
                roi = dynamic_resize(roi)
                roi = enhance_text(roi)
                roi = np.expand_dims(roi, axis=[0, -1])
                pred = model.predict(roi)
                class_index = np.argmax(pred)
                recognized_text += classes[class_index]

            return render_template("index.html", answer=f"{recognized_text}")

    return render_template("index.html", answer="")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0', port=port)
