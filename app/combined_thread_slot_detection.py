"""Run thread-break detection and automatic slot-number OCR on one webcam."""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from slot_number_ocr import (
    SlotReading,
    associate_thread_to_slot,
    draw_slot_readings,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
THREAD_MODEL_PATH = PROJECT_ROOT / "models" / "yolo" / "thread_break.pt"
SLOT_MODEL_PATH = PROJECT_ROOT / "models" / "yolo" / "slot_number.pt"
CAMERA_INDEX = 0
THREAD_CONFIDENCE = 0.40
SLOT_CONFIDENCE = 0.40
IMAGE_SIZE = 320
SLOT_IMAGE_SIZE = 640
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
SLOT_INTERVAL = 0.5
DEVICE = "0" if torch.cuda.is_available() else "cpu"
HALF_PRECISION = DEVICE != "cpu"


def load_slot_readings(frame: np.ndarray, slot_model: YOLO) -> list[SlotReading]:
    result = slot_model.predict(
        source=frame,
        conf=SLOT_CONFIDENCE,
        imgsz=SLOT_IMAGE_SIZE,
        device=DEVICE,
        half=HALF_PRECISION,
        verbose=False,
    )[0]

    readings = []
    for box in result.boxes:
        class_name = str(slot_model.names[int(box.cls[0])]).lower()
        if not class_name.startswith("slot_"):
            continue
        text = class_name.removeprefix("slot_")
        if not text.isdigit():
            continue
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
        readings.append(
            SlotReading(
                text=text,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=float(box.conf[0]),
            )
        )
    return readings


def main() -> int:
    print("Loading thread-break and slot-number models...")
    try:
        if not THREAD_MODEL_PATH.is_file():
            raise FileNotFoundError(f"Thread-break model not found: {THREAD_MODEL_PATH}")
        if not SLOT_MODEL_PATH.is_file():
            raise FileNotFoundError(f"Slot-number model not found: {SLOT_MODEL_PATH}")
        thread_model = YOLO(str(THREAD_MODEL_PATH))
        slot_model = YOLO(str(SLOT_MODEL_PATH))
    except Exception as error:
        print(f"Error setting up combined detection: {error}")
        return 1

    print(f"Thread classes: {thread_model.names}")
    print(f"Slot classes: {slot_model.names}")
    print(
        f"Inference device: {DEVICE} | image size: {IMAGE_SIZE} | "
        f"half precision: {'on' if HALF_PRECISION else 'off'}"
    )

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        camera.release()
        print(f"Error: webcam {CAMERA_INDEX} could not be opened.")
        print("Check camera permissions or change CAMERA_INDEX in this file.")
        return 1

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_camera_fps = camera.get(cv2.CAP_PROP_FPS)
    print(
        f"Webcam started: {actual_width}x{actual_height} at "
        f"{actual_camera_fps:.1f} FPS (requested {CAMERA_FPS}). "
        "Press 'q' or Esc to quit."
    )

    frame_times: deque[float] = deque()
    next_slot_time = 0.0
    slot_readings = []
    slot_future: Future[list[SlotReading]] | None = None

    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="slot-detection") as slot_executor:
            while True:
                success, frame = camera.read()
                if not success:
                    print("Error: could not read a webcam frame.")
                    break

                thread_result = thread_model.predict(
                    source=frame,
                    conf=THREAD_CONFIDENCE,
                    imgsz=IMAGE_SIZE,
                    device=DEVICE,
                    half=HALF_PRECISION,
                    verbose=False,
                )[0]

                now = time.monotonic()
                if slot_future is not None and slot_future.done():
                    try:
                        slot_readings = slot_future.result()
                    except Exception as error:
                        print(f"Error during slot-number detection: {error}")
                        break
                    slot_future = None

                if slot_future is None and now >= next_slot_time:
                    slot_future = slot_executor.submit(load_slot_readings, frame.copy(), slot_model)
                    next_slot_time = now + SLOT_INTERVAL

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

                frame_times.append(now)
                while frame_times and now - frame_times[0] > 1.0:
                    frame_times.popleft()
                fps = (
                    (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                    if len(frame_times) > 1
                    else 0.0
                )

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