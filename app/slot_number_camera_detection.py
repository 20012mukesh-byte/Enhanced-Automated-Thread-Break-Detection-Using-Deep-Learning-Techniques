"""Detect printed slot numbers from a webcam using Tesseract OCR."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

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
THREAD_CONFIDENCE = 0.40
IMAGE_SIZE = 640


def find_number_regions(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Find large bright display/label areas to OCR without manual selection."""
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    kernel_size = max(9, int(min(height, width) * 0.035))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    connected = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        connected,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    minimum_area = height * width * 0.025
    candidates = []
    for contour in contours:
        x, y, region_width, region_height = cv2.boundingRect(contour)
        area = region_width * region_height
        aspect_ratio = region_width / max(1, region_height)
        if (
            area < minimum_area
            or region_width < 60
            or region_height < 45
            or not 0.4 <= aspect_ratio <= 2.2
        ):
            continue
        candidates.append((area, (x, y, x + region_width, y + region_height)))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [region for _, region in candidates[:5]]


def read_slot_numbers(frame: np.ndarray) -> list[SlotReading]:
    """OCR slot numbers from automatically located bright regions."""
    readings = []
    for roi in find_number_regions(frame):
        readings.extend(detect_slot_numbers(frame, roi=roi))

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument(
        "--ocr-interval",
        type=float,
        default=0.5,
        help="seconds between OCR scans (default: 0.5)",
    )
    args = parser.parse_args()

    if args.ocr_interval <= 0:
        parser.error("--ocr-interval must be greater than zero")

    if not THREAD_MODEL_PATH.is_file():
        print(f"Error: thread-break model not found: {THREAD_MODEL_PATH}")
        return 1

    try:
        thread_model = YOLO(str(THREAD_MODEL_PATH))
        executable = configure_tesseract()
    except Exception as error:
        print(f"Error setting up detection: {error}")
        return 1

    print(f"Thread-break model loaded: {THREAD_MODEL_PATH}")
    print(f"Thread classes: {thread_model.names}")
    print(f"Tesseract OCR ready: {executable}")
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        camera.release()
        print(f"Error: webcam {args.camera} could not be opened.")
        print("Check camera permissions or try another index, such as --camera 1.")
        return 1

    print("Webcam started. Press 'q' or Esc to quit.")
    slot_readings = []
    next_ocr_time = 0.0
    previous_time = time.monotonic()

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

            current_time = time.monotonic()
            if current_time >= next_ocr_time:
                try:
                    slot_readings = read_slot_numbers(frame)
                except TesseractNotFoundError as error:
                    print(str(error))
                    break
                next_ocr_time = current_time + args.ocr_interval

            display = thread_result.plot()
            thread_labels = []
            thread_classes = set()
            for box in thread_result.boxes:
                class_name = str(thread_model.names[int(box.cls[0])]).lower()
                thread_classes.add(class_name)
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

            if "thread_break" in thread_classes:
                thread_status = "THREAD BREAK DETECTED"
                thread_color = (0, 0, 255)
            elif "not_thread_break" in thread_classes:
                thread_status = "THREAD NORMAL"
                thread_color = (0, 255, 0)
            else:
                thread_status = "THREAD NOT DETECTED"
                thread_color = (0, 165, 255)

            if thread_labels:
                thread_status += " - " + ", ".join(dict.fromkeys(thread_labels))

            if slot_readings:
                numbers = ", ".join(reading.text for reading in slot_readings)
                slot_status = f"SLOT NUMBERS: {numbers}"
                slot_color = (0, 255, 0)
            else:
                slot_status = "SLOT NUMBERS: SEARCHING..."
                slot_color = (0, 165, 255)

            elapsed = current_time - previous_time
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            previous_time = current_time

            cv2.putText(
                display,
                thread_status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                thread_color,
                2,
            )
            cv2.putText(
                display,
                slot_status,
                (20, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                slot_color,
                2,
            )
            cv2.putText(
                display,
                f"FPS: {fps:.1f} | Q/Esc: quit",
                (20, 104),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )
            cv2.imshow("Thread and Slot Detection", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam closed. Slot number detection stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
