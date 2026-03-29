# MaskGuard – Face Mask Detection System
## MobileNetV2 Transfer Learning · OpenCV · Flask · TensorFlow

---

## Project Structure

```
mask-detector/
├── app.py              ← Flask server + REST API + webcam stream
├── train_model.py      ← Download dataset + train MobileNetV2
├── requirements.txt    ← Python dependencies
├── run.sh              ← Mac/Linux one-command setup
├── run.bat             ← Windows one-command setup
├── model/              ← Auto-created after training
│   ├── mask_detector.keras   ← Trained model
│   ├── label_binarizer.pkl   ← Label encoder
│   └── metrics.pkl           ← Training metrics
├── templates/
│   └── index.html      ← Main webpage
└── static/
    ├── css/style.css   ← Futuristic dark theme
    └── js/main.js      ← Frontend logic
```

---

## Step-by-Step Setup

### Step 1 – Download all files
Put all files in a folder called `mask-detector` maintaining the structure above.

### Step 2 – Run (ONE command)

**Mac/Linux:**
```bash
cd mask-detector
bash run.sh
```

**Windows:**
```
Double-click run.bat
OR
cd mask-detector
run.bat
```

That's it! The script will:
- Install all dependencies
- Download the Face Mask Dataset from GitHub (~45MB)
- Train MobileNetV2 on the dataset (~10-15 mins first time)
- Open the browser automatically

---

## Manual Setup (if run.sh fails)

```bash
# 1. Create virtual env
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install flask opencv-python numpy tensorflow Pillow requests

# 3. Train model (only once)
python train_model.py

# 4. Start server
python app.py
```

Open: http://localhost:5000

---

## Features

### Image Detection
- Upload any JPG/PNG/WEBP image
- Detects multiple faces simultaneously
- Shows bounding boxes with MASK/NO MASK labels
- Confidence percentage per detection
- Safe/Unsafe verdict banner

### Live Webcam Detection
- Real-time MJPEG stream from your webcam
- Green box = wearing mask
- Red box = not wearing mask
- Works with multiple people in frame

### Metrics Dashboard
- Validation accuracy chart
- Training vs validation accuracy per epoch
- Dataset statistics

---

## API Endpoints

### POST /api/predict/image
```bash
curl -X POST http://localhost:5000/api/predict/image \
  -F "image=@/path/to/photo.jpg"
```
Response:
```json
{
  "image": "data:image/jpeg;base64,...",
  "total_faces": 2,
  "with_mask": 1,
  "without_mask": 1,
  "safe": false
}
```

### GET /api/video_feed
Returns MJPEG stream for webcam detection.

### GET /api/metrics
Returns training metrics JSON.

---

## Model Architecture

```
Input (224×224×3)
    ↓
MobileNetV2 (ImageNet pretrained, frozen)
    ↓
AveragePooling2D (7×7)
    ↓
Flatten
    ↓
Dense(128, ReLU) + Dropout(0.5)
    ↓
Dense(2, Softmax)
    ↓
Output: [with_mask_prob, without_mask_prob]
```

**Why MobileNetV2?**
- Lightweight (3.4M params) — runs on CPU
- Pre-trained on ImageNet — great feature extraction
- Fast inference — suitable for real-time detection

---

## Interview Questions & Answers

**Q: Why MobileNetV2 over VGG or ResNet?**
A: MobileNetV2 uses depthwise separable convolutions which are 8-9x faster with similar accuracy. Better for real-time edge deployment.

**Q: What is Transfer Learning?**
A: We freeze the ImageNet-pretrained base layers and only train the classification head. This needs less data and trains faster.

**Q: How does Haar Cascade detect faces?**
A: It uses a sliding window with trained Haar features and Adaboost classifier to detect face-like patterns in images.

**Q: How would you deploy this in production?**
A: Replace Flask dev server with Gunicorn + Nginx. Containerize with Docker. Use ONNX runtime for faster inference.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `No module named tensorflow` | `pip install tensorflow` |
| `Model not found` | Run `python train_model.py` first |
| `Webcam not accessible` | Allow camera permission in browser/OS |
| `Dataset download fails` | Check internet; try manual download from GitHub |
| `Port 5000 in use` | Change to `app.run(port=5001)` in app.py |
| Slow training | Normal for CPU — use Google Colab for GPU |
