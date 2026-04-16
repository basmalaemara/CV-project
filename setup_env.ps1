# Setup Environment for Hand Gesture Recognition
Write-Host "--- Creating Virtual Environment ---" -ForegroundColor Cyan
python -m venv venv

Write-Host "--- Activating Environment and Installing Requirements ---" -ForegroundColor Cyan
./venv/Scripts/python.exe -m pip install --upgrade pip
./venv/Scripts/python.exe -m pip install -r requirements.txt

Write-Host "--- Verifying Installations ---" -ForegroundColor Cyan
./venv/Scripts/python.exe -c "import mediapipe as mp; print('MediaPipe version:', mp.__version__)"
./venv/Scripts/python.exe -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU devices:', tf.config.list_physical_devices('GPU'))"

Write-Host "`nSetup Complete! To activate manually, run: .\venv\Scripts\Activate.ps1" -ForegroundColor Green
