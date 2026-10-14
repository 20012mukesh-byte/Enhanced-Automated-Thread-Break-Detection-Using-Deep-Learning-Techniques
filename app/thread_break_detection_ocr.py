import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from slot_number_ocr import (
    TesseractNotFoundError,
    associate_thread_to_slot,
    configure_tesseract,
    detect_slot_numbers,
    draw_slot_readings,
    expand_roi,
    tesseract_install_help,
)

# ---------------- CONFIGURATION ----------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
THREAD_MODEL_PATH = PROJECT_ROOT / "models" / "yolo" / "thread_break.pt"
# Optional label-region helper trained in this project (classes slot_01..slot_04).
# Used only to crop likely plate areas for OCR — class names are not treated as
# printed slot numbers.
SLOT_REGION_MODEL_PATH = PROJECT_ROOT / "models" / "yolo" / "slot_number.pt"
CAMERA_INDEX = 0
CONFIDENCE = 0.40
IMAGE_SIZE = 640
OCR_EVERY_N_FRAMES = 6
UNKNOWN_SLOT = "SLOT UNKNOWN"
# Optional JSON list of {"slot": "125", "x1": 0, "y1": 0, "x2": 100, "y2": 200}
# for a fixed camera view. Leave missing/empty to rely on live OCR only.
SPINDLE_REGIONS_PATH = PROJECT_ROOT / "config" / "spindle_regions.json"

# ---------------- LOAD MODEL ----------------
print("Loading Thread Break Detection Model...")

if not THREAD_MODEL_PATH.is_file():
    print(f"Error: thread model not found: {THREAD_MODEL_PATH}")
    raise SystemExit(1)

try:
    thread_model = YOLO(str(THREAD_MODEL_PATH))
    print(f"Thread model loaded from: {THREAD_MODEL_PATH}")
    print("Thread classes:", thread_model.names)
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Make sure the model file exists at: {THREAD_MODEL_PATH}")
    raise SystemExit(1)

slot_region_model = None
if SLOT_REGION_MODEL_PATH.is_file():
    try:
        slot_region_model = YOLO(str(SLOT_REGION_MODEL_PATH))
        print(f"Slot-region model loaded from: {SLOT_REGION_MODEL_PATH}")
        print("Slot-region classes:", slot_region_model.names)
        print("Note: these class names locate plates; OCR reads the printed number.")
    except Exception as e:
        print(f"Slot-region model could not be loaded ({e}). OCR will use the full frame.")
        slot_region_model = None

ocr_ready = False
try:
    tesseract_path = configure_tesseract()
    ocr_ready = True
    print(f"Tesseract OCR ready: {tesseract_path}")
except TesseractNotFoundError as error:
    print(str(error))
    print("Thread-break detection will still run. Slot numbers will show as SLOT UNKNOWN.")

fixed_regions = []
if SPINDLE_REGIONS_PATH.is_file():
    try:
        loaded = json.loads(SPINDLE_REGIONS_PATH.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            fixed_regions = loaded
            print(f"Loaded {len(fixed_regions)} fixed spindle region(s) from {SPINDLE_REGIONS_PATH.name}")
    except (OSError, json.JSONDecodeError) as error:
        print(f"Could not read {SPINDLE_REGIONS_PATH.name}: {error}")

# ---------------- OPEN WEBCAM ----------------
camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    print("Error: Webcam could not be opened.")
    print("Check camera permissions or try CAMERA_INDEX = 1.")
    raise SystemExit(1)

print("Webcam started.")
print("Press 'q' to quit.")

previous_time = time.time()
frame_index = 0
cached_slots = []

# ---------------- REAL-TIME DETECTION ----------------
try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read webcam frame.")
            break

        results = thread_model.predict(
            source=frame,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            verbose=False,
        )

        result = results[0]
        annotated_frame = result.plot()

        thread_breaks = []
        detected_classes = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = thread_model.names[class_id]
            detected_classes.append(class_name.lower())
            if class_name.lower() == "thread_break":
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                thread_breaks.append((x1, y1, x2, y2))

        run_ocr = ocr_ready and (
            frame_index % OCR_EVERY_N_FRAMES == 0
            or (bool(thread_breaks) and not cached_slots)
        )
        if run_ocr:
            slot_readings = []
            try:
                rois = []
                if slot_region_model is not None:
                    region_results = slot_region_model.predict(
                        source=frame,
                        conf=CONFIDENCE,
                        imgsz=IMAGE_SIZE,
                        verbose=False,
                    )
                    for region_box in region_results[0].boxes:
                        rx1, ry1, rx2, ry2 = region_box.xyxy[0].tolist()
                        width = max(1.0, rx2 - rx1)
                        height = max(1.0, ry2 - ry1)
                        rois.append(
                            (
                                int(rx1 - 0.15 * width),
                                int(ry1 - 0.15 * height),
                                int(rx2 + 0.15 * width),
                                int(ry2 + 0.15 * height),
                            )
                        )

                for break_box in thread_breaks:
                    rois.append(expand_roi(break_box, frame.shape))

                seen = set()
                for roi in rois:
                    key = tuple(roi)
                    if key in seen:
                        continue
                    seen.add(key)
                    slot_readings.extend(detect_slot_numbers(frame, roi=roi))

                if not slot_readings:
                    slot_readings = detect_slot_numbers(frame)

                # Drop duplicate OCR hits of the same number in nearly the same place.
                unique = []
                for reading in slot_readings:
                    duplicate = False
                    for existing in unique:
                        if reading.text != existing.text:
                            continue
                        cx, cy = reading.center
                        ex, ey = existing.center
                        if abs(cx - ex) < 25 and abs(cy - ey) < 25:
                            duplicate = True
                            break
                    if not duplicate:
                        unique.append(reading)
                cached_slots = unique
            except TesseractNotFoundError:
                ocr_ready = False
                cached_slots = []
                print(tesseract_install_help())

        draw_slot_readings(annotated_frame, cached_slots)

        if "thread_break" in detected_classes:
            if not thread_breaks:
                status = f"THREAD BREAK - {UNKNOWN_SLOT}"
            else:
                labels = []
                for break_box in thread_breaks:
                    slot = associate_thread_to_slot(
                        break_box,
                        cached_slots,
                        fixed_regions=fixed_regions,
                    )
                    x1, y1, x2, y2 = (int(v) for v in break_box)
                    if slot is None:
                        label = UNKNOWN_SLOT
                    else:
                        label = f"SLOT {slot.text}"
                    labels.append(label)
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                unique_labels = list(dict.fromkeys(labels))
                if len(unique_labels) == 1:
                    status = f"THREAD BREAK - {unique_labels[0]}"
                else:
                    status = "THREAD BREAK - " + ", ".join(unique_labels)
            status_color = (0, 0, 255)

        elif "not_thread_break" in detected_classes:
            status = "THREAD NORMAL"
            status_color = (0, 255, 0)

        else:
            status = "THREAD NOT DETECTED"
            status_color = (0, 165, 255)

        current_time = time.time()
        elapsed = current_time - previous_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        previous_time = current_time
        frame_index += 1

        cv2.putText(
            annotated_frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
        )

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        if not ocr_ready:
            cv2.putText(
                annotated_frame,
                "OCR unavailable - install Tesseract",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
            )

        cv2.imshow("Thread Break Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Webcam closed. Detection stopped.")
