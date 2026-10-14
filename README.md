# Automated Thread-Break and Slot-Number Detection

This project contains webcam applications for thread-break detection and
printed slot-number detection. The combined detector uses YOLO for thread
breaks and Tesseract OCR for slot numbers.

## Project layout

```text
app/
  combined_thread_slot_detection.py  Combined YOLO + OCR application
  model_exp1.py                      Keras webcam classifier
  number_camera_detection.py         YOLO slot-class detector
  slot_number_camera_detection.py   Thread detection + automatic slot OCR
  slot_number_ocr.py                 Shared Tesseract OCR functions
  thread_break_detection_ocr.py     Legacy combined detector with OCR
  thread_camera_detection.py         Thread-break detector only (no OCR)
archive/
  legacy_thread_camera_detection.py Preserved older thread detector
config/
  spindle_regions.example.json      Example fixed-camera slot mapping
models/
  keras/                             Keras model and class labels
  yolo/                              Thread-break and slot-class weights
requirements.txt                     Python package dependencies
```

## Setup

Run these commands in PowerShell from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The OCR applications also need the Tesseract executable installed on Windows.
Add it to `PATH`, or set `TESSERACT_CMD` to the full path of `tesseract.exe`.

## Run the applications

Run commands from the project root:

```powershell
# Thread-break YOLO detection only
.\.venv\Scripts\python.exe app\thread_camera_detection.py

# Thread-break YOLO detection plus automatic OCR slot-number detection
.\.venv\Scripts\python.exe app\slot_number_camera_detection.py

# Alternative combined YOLO and OCR implementation
.\.venv\Scripts\python.exe app\combined_thread_slot_detection.py

# Slot/spindle class detection using the slot YOLO model
.\.venv\Scripts\python.exe app\number_camera_detection.py

# Keras webcam image classifier
.\.venv\Scripts\python.exe app\model_exp1.py
```

Press `q` to close a webcam window; the slot OCR application also accepts
`Esc`. Its camera index and OCR interval can be selected on the command line:

```powershell
.\.venv\Scripts\python.exe app\slot_number_camera_detection.py --camera 0 --ocr-interval 0.5
```

The OCR helper can be checked without opening the camera:

```powershell
.\.venv\Scripts\python.exe app\slot_number_ocr.py
```

## Detection details

- `thread_camera_detection.py` uses `models\yolo\thread_break.pt` and does not
  run OCR.
- `slot_number_camera_detection.py` runs the same thread-break model and
  automatically finds bright regions for printed-number OCR. Tesseract reads
  the digits; when a thread break is detected, the application attempts to
  associate it with a nearby OCR slot reading.
- `combined_thread_slot_detection.py` is an additional combined application
  using the thread-break model and the same automatic OCR slot-number logic.
- `number_camera_detection.py` uses `models\yolo\slot_number.pt` to detect
  trained slot classes (`slot_01` through `slot_04`). Those are model classes,
  not OCR transcription of arbitrary printed digits.
- `slot_number_ocr.py` provides shared digit preprocessing, Tesseract OCR,
  drawing, and thread-to-slot association functions.

YOLO weights are stored in `models\yolo\`. The Keras model and its labels are
stored in `models\keras\`. The scripts resolve model paths relative to the
project, so they can be launched from the project root.

The legacy `thread_break_detection_ocr.py` application supports optional
fixed-camera slot mappings at `config\spindle_regions.json`. To configure it,
copy `config\spindle_regions.example.json` to that filename and edit the slot
and coordinates. The local configuration file is ignored by Git.
