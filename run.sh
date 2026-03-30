#!/bin/bash
# ─────────────────────────────────────────────────────
#  MaskGuard – Face Mask Detection System
#  One command setup & run: bash run.sh
# ─────────────────────────────────────────────────────
set -e

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   MaskGuard – Face Mask Detection System     ║"
echo "║   MobileNetV2 · OpenCV · Flask · TensorFlow  ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

echo "▶ [1/6] Checking Python..."
if ! command -v python3 &>/dev/null; then
    echo "❌ Python3 not found. Install from https://www.python.org"
    exit 1
fi
echo "✅ $(python3 --version)"

echo ""
echo "▶ [2/6] Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Created."
else
    echo "✅ Already exists."
fi
source venv/bin/activate

echo ""
echo "▶ [3/6] Installing dependencies (first time ~5 mins)..."
pip install --upgrade pip --quiet
pip install flask opencv-python numpy tensorflow Pillow requests --quiet
echo "✅ All dependencies installed."

echo ""
echo "▶ [4/6] Training model (first time ~10-15 mins)..."
if [ ! -f "model/mask_detector.keras" ]; then
    python3 train_model.py
else
    echo "✅ Model already trained, skipping."
fi

echo ""
echo "▶ [5/6] Opening browser..."
sleep 2
open "http://127.0.0.1:8080" 2>/dev/null || xdg-open "http://127.0.0.1:8080" 2>/dev/null || true

echo ""
echo "▶ [6/6] Starting server..."
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  ✅ MaskGuard is ready!                      ║"
echo "║  🌐 http://127.0.0.1:8080                    ║"
echo "║  🎥 Webcam tab for live detection            ║"
echo "║  🛑 Stop: Press Ctrl+C                       ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
python3 app.py
