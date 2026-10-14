# Automated Thread-Break and Slot-Number Detection

This project contains separate webcam applications for thread-break detection,
YOLO-based slot-number detection, and OCR-based slot-number detection.

## Project layout

```text
app/                 Python applications and shared OCR helper
archive/             Preserved legacy detector
config/              Optional camera-specific settings
models/
  keras/             Keras classifier and class labels
  yolo/              YOLO thread-break and slot-number weights
requirements.txt     Python dependencies
```

## Setup

From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The OCR applications also require the Tesseract executable. Install Tesseract
for Windows and add it to `PATH`, or set `TESSERACT_CMD` to the full path of
`tesseract.exe`.

## Run an application

Run commands from the project root:

```powershell
.\.venv\Scripts\python.exe app\thread_camera_detection.py
.\.venv\Scripts\python.exe app\thread_break_detection_ocr.py
.\.venv\Scripts\python.exe app\number_camera_detection.py
.\.venv\Scripts\python.exe app\slot_number_camera_detection.py
.\.venv\Scripts\python.exe app\model_exp1.py
```

Press `q` to close a webcam application. The automatic OCR slot-number
detector scans bright display or label regions without requiring manual
selection.

To select a different webcam, the slot-number detector accepts `--camera 1`
and `--ocr-interval 1` (seconds between scans):

```powershell
.\.venv\Scripts\python.exe app\slot_number_camera_detection.py --camera 0 --ocr-interval 0.5
```

The OCR helper can be checked independently:

```powershell
.\.venv\Scripts\python.exe app\slot_number_ocr.py
```

## Models and local settings

YOLO weights are stored in `models\yolo\`; the Keras model and its labels are
stored in `models\keras\`. Applications resolve model paths from the project
directory, so they can be launched from the project root.

The combined thread/OCR application can optionally read
`config\spindle_regions.json` for fixed camera-to-slot mappings. This local
camera configuration is ignored by Git. Copy
`config\spindle_regions.example.json` to `config\spindle_regions.json` and
edit its coordinates for a fixed camera setup.
