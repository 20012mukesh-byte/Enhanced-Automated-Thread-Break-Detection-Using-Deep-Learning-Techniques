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
)

# ---------------- CONFIGURATION ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "yolo" / "thread_break.pt"
CAMERA_INDEX = 0
CONFIDENCE = 0.40
IMAGE_SIZE = 640
SLOT_OCR_INTERVAL = 15

# ---------------- LOAD MODEL ----------------
print("Loading Thread Break Detection Model...")

try:
    configure_tesseract()
except TesseractNotFoundError as error:
    print(str(error))
    raise SystemExit(1)

if not MODEL_PATH.is_file():
    print(f"Error: thread-break model file not found: {MODEL_PATH}")
    raise SystemExit(1)

try:
    model = YOLO(str(MODEL_PATH))
    print(f"Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Make sure the model file exists in: {BASE_DIR}")
    raise SystemExit(1)

# ---------------- OPEN WEBCAM ----------------
camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    print("Error: Webcam could not be opened.")
    print("Check camera permissions or try CAMERA_INDEX = 1.")
    raise SystemExit(1)

print("Webcam started.")
print("Press 'q' to quit.")

previous_time = time.time()
frame_count = 0
slot_readings = []

# ---------------- REAL-TIME DETECTION ----------------
try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read webcam frame.")
            break

        if frame_count % SLOT_OCR_INTERVAL == 0:
            slot_readings = detect_slot_numbers(frame)
        frame_count += 1

        # Run YOLO prediction
        results = model.predict(
            source=frame,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            verbose=False
        )

        result = results[0]

        # Draw bounding boxes and labels
        annotated_frame = result.plot()

        detected_classes = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            normalized_class = class_name.lower()
            detected_classes.append(normalized_class)

            if normalized_class == "thread_break":
                thread_box = tuple(float(value) for value in box.xyxy[0].tolist())
                slot = associate_thread_to_slot(thread_box, slot_readings)
                slot_label = f"SLOT {slot.text}" if slot else "SLOT UNKNOWN"
                x1, y1 = int(thread_box[0]), int(thread_box[1])
                cv2.putText(
                    annotated_frame,
                    slot_label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255) if slot else (0, 165, 255),
                    2,
                )

        draw_slot_readings(annotated_frame, slot_readings)

        # Determine status from detected classes
        if "thread_break" in detected_classes:
            status = "THREAD BREAK DETECTED"
            status_color = (0, 0, 255)

        elif "not_thread_break" in detected_classes:
            status = "THREAD NORMAL"
            status_color = (0, 255, 0)

        else:
            status = "THREAD NOT DETECTED"
            status_color = (0, 165, 255)

        # Calculate FPS
        current_time = time.time()
        elapsed = current_time - previous_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        previous_time = current_time

        # Display status and FPS
        cv2.putText(
            annotated_frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2
        )

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.imshow("Thread Break Detection", annotated_frame)

        # Press q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Webcam closed. Detection stopped.")