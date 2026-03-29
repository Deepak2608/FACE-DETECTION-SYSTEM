#!/bin/bash
set -e

cd "$(dirname "$0")"

# Remove old venv if it exists
if [ -d "venv" ]; then
    rm -rf venv
fi

# Create fresh venv
echo "Creating virtual environment..."
python3 -m venv venv

# Activate and install packages
echo "Installing packages..."
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install "numpy<2" opencv-python flask tensorflow scikit-learn pillow requests

echo "✅ Setup complete! Now running app..."
python app.py
