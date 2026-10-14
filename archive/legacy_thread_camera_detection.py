import cv2
import time
from pathlib import Path
from ultralytics import YOLO

# ---------------- CONFIGURATION ----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolo" / "thread_break.pt"
CAMERA_INDEX = 0
CONFIDENCE = 0.40
IMAGE_SIZE = 640

# ---------------- LOAD MODEL ----------------
print("Loading Thread Break Detection Model...")

try:
    model = YOLO(str(MODEL_PATH))
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Make sure the model file exists at: {MODEL_PATH}")
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

# ---------------- REAL-TIME DETECTION ----------------
try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read webcam frame.")
            break

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
            detected_classes.append(class_name.lower())

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