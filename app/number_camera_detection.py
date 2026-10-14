import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# ---------------- CONFIGURATION ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "yolo" / "slot_number.pt"
CAMERA_INDEX = 0
CONFIDENCE = 0.40
IMAGE_SIZE = 640


def digit_from_class_name(class_name: str) -> str | None:
    """Return a single digit if the class looks like 0-9."""
    name = class_name.strip().lower()
    if name.isdigit() and len(name) == 1:
        return name
    for token in name.replace("-", " ").replace("_", " ").split():
        if token.isdigit() and len(token) == 1:
            return token
    return None


def assemble_number(boxes, names) -> str:
    """Build a number by reading detected digits left to right."""
    digits = []
    for box in boxes:
        class_id = int(box.cls[0])
        class_name = names[class_id]
        digit = digit_from_class_name(class_name)
        if digit is None:
            continue
        x1, _, _, _ = box.xyxy[0].tolist()
        digits.append((x1, digit))

    digits.sort(key=lambda item: item[0])
    return "".join(digit for _, digit in digits)


print("Loading number detection model...")

if not MODEL_PATH.is_file():
    print(f"Error: slot-number model file not found: {MODEL_PATH}")
    raise SystemExit(1)

try:
    model = YOLO(str(MODEL_PATH))
    print(f"Model loaded successfully from: {MODEL_PATH}")
    print("Classes:", model.names)
except Exception as error:
    print(f"Error loading model: {error}")
    raise SystemExit(1)

camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    print("Error: Webcam could not be opened.")
    print("Check camera permissions or try CAMERA_INDEX = 1.")
    raise SystemExit(1)

print("Webcam started.")
print("Press 'q' to quit.")

previous_time = time.time()

try:
    while True:
        success, frame = camera.read()
        if not success:
            print("Error: Could not read webcam frame.")
            break

        results = model.predict(
            source=frame,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            verbose=False,
        )
        result = results[0]
        annotated_frame = result.plot()

        detected_number = assemble_number(result.boxes, model.names)
        detected_labels = [model.names[int(box.cls[0])] for box in result.boxes]

        current_time = time.time()
        elapsed = current_time - previous_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        previous_time = current_time

        if detected_number:
            status = f"NUMBER: {detected_number}"
            status_color = (0, 255, 0)
        elif detected_labels:
            status = "DETECTED: " + ", ".join(detected_labels)
            status_color = (0, 255, 255)
        else:
            status = "NO NUMBER DETECTED"
            status_color = (0, 165, 255)

        cv2.putText(
            annotated_frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
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

        cv2.imshow("Number Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Webcam closed. Detection stopped.")
