"""
app.py
------
Flask backend for Face Mask Detection System.

Endpoints:
    GET  /                    - Main web page
    POST /api/predict/image   - Predict from uploaded image
    GET  /api/video_feed      - MJPEG webcam stream with detection
    GET  /api/metrics         - Model performance metrics
    POST /api/video/start     - Start webcam
    POST /api/video/stop      - Stop webcam
"""

import os
import io
import cv2
import pickle
import base64
import threading
import numpy as np
from flask import Flask, request, jsonify, render_template, Response
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

app = Flask(__name__)

# ── Globals ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "model/mask_detector.keras"
LB_PATH      = "model/label_binarizer.pkl"
METRICS_PATH = "model/metrics.pkl"
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
IMG_SIZE = 224

model    = None
lb       = None
metrics  = {}

camera        = None
camera_lock   = threading.Lock()
camera_active = False

# ── Load model ─────────────────────────────────────────────────────────────────
def load_assets():
    global model, lb, metrics
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run  python train_model.py  first.")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LB_PATH, "rb") as f:
        lb = pickle.load(f)
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "rb") as f:
            metrics = pickle.load(f)
    print("✅ Model loaded.")

load_assets()

# ── Helpers ────────────────────────────────────────────────────────────────────
def predict_face_roi(face_roi):
    """Run mask classifier on a cropped face region."""
    face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)
    preds = model.predict(face, verbose=0)[0]
    idx   = np.argmax(preds)
    label = lb.classes_[idx]
    conf  = float(preds[idx])
    return label, conf

def draw_detections(frame):
    """Detect faces and draw bounding boxes with mask labels."""
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    stats = {"with_mask": 0, "without_mask": 0, "total": len(faces)}
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue
        label, conf = predict_face_roi(face_roi)
        is_mask = label == "with_mask"
        color   = (0, 200, 80) if is_mask else (0, 60, 220)
        text    = f"{'MASK' if is_mask else 'NO MASK'} {conf*100:.0f}%"
        stats[label] += 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(frame, (x, y-28), (x+w, y), color, -1)
        cv2.putText(frame, text, (x+4, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame, stats

def gen_frames():
    """Generator for MJPEG webcam stream."""
    global camera, camera_active
    while camera_active:
        with camera_lock:
            if camera is None or not camera.isOpened():
                break
            success, frame = camera.read()
        if not success:
            break
        frame, _ = draw_detections(frame)
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buffer.tobytes() + b"\r\n")

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict/image", methods=["POST"])
def predict_image():
    """Detect masks in an uploaded image file or base64 string."""
    try:
        if "image" in request.files:
            file  = request.files["image"]
            img   = Image.open(file.stream).convert("RGB")
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        elif request.is_json:
            data    = request.get_json()
            b64     = data.get("image", "").split(",")[-1]
            img_bytes = base64.b64decode(b64)
            img     = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            frame   = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        else:
            return jsonify({"error": "No image provided"}), 400

        result_frame, stats = draw_detections(frame)

        _, buffer  = cv2.imencode(".jpg", result_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64_result = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "image":        "data:image/jpeg;base64," + b64_result,
            "total_faces":  stats["total"],
            "with_mask":    stats["with_mask"],
            "without_mask": stats["without_mask"],
            "safe":         stats["without_mask"] == 0 and stats["total"] > 0,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/video/start", methods=["POST"])
def start_video():
    global camera, camera_active
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera_active = True
    return jsonify({"status": "started"})


@app.route("/api/video/stop", methods=["POST"])
def stop_video():
    global camera, camera_active
    camera_active = False
    with camera_lock:
        if camera:
            camera.release()
            camera = None
    return jsonify({"status": "stopped"})


@app.route("/api/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/metrics")
def get_metrics():
    return jsonify(metrics)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
