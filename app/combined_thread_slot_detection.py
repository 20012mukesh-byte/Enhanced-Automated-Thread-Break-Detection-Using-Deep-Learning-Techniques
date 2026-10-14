"""Run thread-break detection and automatic slot-number OCR on one webcam."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from slot_number_camera_detection import find_number_regions
from slot_number_ocr import (
    SlotReading,
    TesseractNotFoundError,
    associate_thread_to_slot,
    configure_tesseract,
    detect_slot_numbers,
    draw_slot_readings,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
THREAD_MODEL_PATH = PROJECT_ROOT / "models" / "yolo" / "thread_break.pt"
CAMERA_INDEX = 0
THREAD_CONFIDENCE = 0.40
IMAGE_SIZE = 640
OCR_INTERVAL = 0.5


def load_slot_readings(frame: np.ndarray) -> list[SlotReading]:
    readings = []
    for x1, y1, x2, y2 in find_number_regions(frame):
        readings.extend(detect_slot_numbers(frame, roi=(x1, y1, x2, y2)))

    unique_readings = []
    for reading in readings:
        center_x, center_y = reading.center
        duplicate = any(
            reading.text == existing.text
            and abs(center_x - existing.center[0]) < 25
            and abs(center_y - existing.center[1]) < 25
            for existing in unique_readings
        )
        if not duplicate:
            unique_readings.append(reading)
    return unique_readings


def main() -> int:
    print("Loading thread-break model and configuring slot-number OCR...")
    try:
        if not THREAD_MODEL_PATH.is_file():
            raise FileNotFoundError(f"Thread-break model not found: {THREAD_MODEL_PATH}")
        thread_model = YOLO(str(THREAD_MODEL_PATH))
        tesseract_path = configure_tesseract()
    except Exception as error:
        print(f"Error setting up combined detection: {error}")
        return 1

    print(f"Thread classes: {thread_model.names}")
    print(f"Tesseract OCR ready: {tesseract_path}")

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        camera.release()
        print(f"Error: webcam {CAMERA_INDEX} could not be opened.")
        print("Check camera permissions or change CAMERA_INDEX in this file.")
        return 1

    print("Webcam started. Press 'q' or Esc to quit.")
    previous_time = time.monotonic()
    next_ocr_time = 0.0
    slot_readings = []

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Error: could not read a webcam frame.")
                break

            thread_result = thread_model.predict(
                source=frame,
                conf=THREAD_CONFIDENCE,
                imgsz=IMAGE_SIZE,
                verbose=False,
            )[0]

            now = time.monotonic()
            if now >= next_ocr_time:
                try:
                    slot_readings = load_slot_readings(frame)
                except TesseractNotFoundError as error:
                    print(str(error))
                    break
                next_ocr_time = now + OCR_INTERVAL

            display = thread_result.plot()
            thread_labels = []
            for box in thread_result.boxes:
                class_name = str(thread_model.names[int(box.cls[0])]).lower()
                if class_name != "thread_break":
                    continue
                thread_box = tuple(float(value) for value in box.xyxy[0].tolist())
                slot = associate_thread_to_slot(thread_box, slot_readings)
                label = f"SLOT {slot.text}" if slot is not None else "SLOT UNKNOWN"
                thread_labels.append(label)
                x1, y1 = int(thread_box[0]), int(thread_box[1])
                cv2.putText(
                    display,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255) if slot is not None else (0, 165, 255),
                    2,
                )
            draw_slot_readings(display, slot_readings)

            thread_classes = {
                str(thread_model.names[int(box.cls[0])]).lower()
                for box in thread_result.boxes
            }
            if "thread_break" in thread_classes:
                thread_status = "THREAD BREAK"
                thread_color = (0, 0, 255)
            elif "not_thread_break" in thread_classes:
                thread_status = "THREAD NORMAL"
                thread_color = (0, 255, 0)
            else:
                thread_status = "THREAD NOT DETECTED"
                thread_color = (0, 165, 255)

            if slot_readings:
                detected_slots = ", ".join(reading.text for reading in slot_readings)
                slot_status = f"SLOT NUMBERS: {detected_slots}"
            else:
                slot_status = "SLOT NUMBERS: NOT DETECTED"

            if thread_labels:
                thread_status += " - " + ", ".join(dict.fromkeys(thread_labels))

            now = time.monotonic()
            elapsed = now - previous_time
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            previous_time = now

            cv2.putText(
                display,
                thread_status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                thread_color,
                2,
            )
            cv2.putText(
                display,
                slot_status,
                (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 0, 255),
                2,
            )
            cv2.putText(
                display,
                f"FPS: {fps:.1f}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )
            cv2.imshow("Combined Thread and Slot Detection", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam closed. Combined detection stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())