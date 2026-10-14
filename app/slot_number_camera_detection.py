"""Detect printed slot numbers from a webcam using Tesseract OCR."""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from slot_number_ocr import (
    TesseractNotFoundError,
    configure_tesseract,
    detect_slot_numbers,
    draw_slot_readings,
)


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

    try:
        executable = configure_tesseract()
    except TesseractNotFoundError as error:
        print(str(error))
        return 1

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

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Error: could not read a webcam frame.")
                break

            current_time = time.monotonic()
            if current_time >= next_ocr_time:
                try:
                    readings = []
                    for roi in find_number_regions(frame):
                        readings.extend(detect_slot_numbers(frame, roi=roi))
                    slot_readings = readings
                except TesseractNotFoundError as error:
                    print(str(error))
                    break
                next_ocr_time = current_time + args.ocr_interval

            display = frame.copy()
            draw_slot_readings(display, slot_readings)

            if slot_readings:
                numbers = ", ".join(reading.text for reading in slot_readings)
                status = f"SLOT NUMBER DETECTED: {numbers}"
                color = (0, 255, 0)
            else:
                status = "SEARCHING FOR SLOT NUMBER..."
                color = (0, 165, 255)

            cv2.putText(
                display,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
            cv2.putText(
                display,
                "Automatic detection | Q/Esc: quit",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Slot Number Detection", display)

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
