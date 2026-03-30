@echo off
echo.
echo ╔══════════════════════════════════════════════╗
echo ║   MaskGuard – Face Mask Detection System     ║
echo ║   MobileNetV2 · OpenCV · Flask · TensorFlow  ║
echo ╚══════════════════════════════════════════════╝
echo.

echo [1/6] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 ( echo Python not found. Install from python.org & pause & exit /b 1 )
echo  Python found!

echo.
echo [2/6] Setting up virtual environment...
if not exist "venv" ( python -m venv venv & echo  Created. ) else ( echo  Already exists. )
call venv\Scripts\activate.bat

echo.
echo [3/6] Installing dependencies...
pip install --upgrade pip --quiet
pip install flask opencv-python numpy tensorflow Pillow requests --quiet
echo  Done!

echo.
echo [4/6] Training model (first time ~10-15 mins)...
if not exist "model\mask_detector.keras" (
    python train_model.py
) else (
    echo  Model already trained.
)

echo.
echo [5/6] Opening browser...
timeout /t 2 /nobreak >nul
start http://127.0.0.1:8080

echo.
echo [6/6] Starting server...
echo.
echo  MaskGuard ready at http://127.0.0.1:8080
echo  Press Ctrl+C to stop
echo.
python app.py
pause
