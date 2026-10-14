"""Digit OCR for printed spindle/slot numbers.

Reuses the OpenCV + Tesseract approach from
https://github.com/bucks-hub/realtime-text-detection-ocr
(digit-only config, bounding boxes, webcam-friendly crops)
without copying that author's hard-coded Tesseract path.
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import pytesseract
except ImportError:  # pragma: no cover - dependency missing until installed
    pytesseract = None


# Typical Windows install locations (UB Mannheim / installer defaults).
_WINDOWS_TESSERACT_CANDIDATES = (
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Tesseract-OCR" / "tesseract.exe",
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Tesseract-OCR" / "tesseract.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Tesseract-OCR" / "tesseract.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
    Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe",
)

DIGIT_OCR_CONFIG = r"--oem 3 --psm 6 outputbase digits"
DIGIT_OCR_FALLBACK_CONFIG = r"--oem 3 --psm 13 outputbase digits"
MIN_OCR_CONFIDENCE = 55.0
MIN_SLOT_DIGITS = 1
MAX_SLOT_DIGITS = 4
# Group digits on the same line if their gap is smaller than this multiple of height.
DIGIT_GAP_RATIO = 0.85
# Horizontal uniqueness: another slot this close (relative) is treated as ambiguous.
AMBIGUITY_DISTANCE_RATIO = 1.65
COLUMN_OVERLAP_MIN = 0.12


@dataclass(frozen=True)
class SlotReading:
    text: str
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> int:
        return max(1, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(1, self.y2 - self.y1)


class TesseractNotFoundError(RuntimeError):
    """Raised when the Tesseract executable cannot be located."""


def tesseract_install_help() -> str:
    return (
        "Tesseract OCR is not installed or was not found on PATH.\n"
        "Windows (recommended):\n"
        "  1. Download the installer from https://github.com/UB-Mannheim/tesseract/wiki\n"
        "  2. Install it (default folder is C:\\Program Files\\Tesseract-OCR).\n"
        "  3. Tick 'Add to PATH' during setup, or set TESSERACT_CMD to tesseract.exe.\n"
        "Alternative: choco install tesseract\n"
        "macOS: brew install tesseract\n"
        "Linux: sudo apt install tesseract-ocr\n"
        "Then restart the terminal / VS Code so PATH is updated."
    )


def find_tesseract_executable() -> Path | None:
    env_path = os.environ.get("TESSERACT_CMD", "").strip()
    if env_path:
        candidate = Path(env_path)
        if candidate.is_file():
            return candidate

    which = shutil.which("tesseract")
    if which:
        return Path(which)

    if os.name == "nt":
        for candidate in _WINDOWS_TESSERACT_CANDIDATES:
            if candidate.is_file():
                return candidate
    return None


def configure_tesseract() -> Path:
    if pytesseract is None:
        raise TesseractNotFoundError(
            "Python package pytesseract is missing. Install with:\n"
            "  .\\.venv\\Scripts\\pip.exe install pytesseract Pillow"
        )

    executable = find_tesseract_executable()
    if executable is None:
        raise TesseractNotFoundError(tesseract_install_help())

    pytesseract.pytesseract.tesseract_cmd = str(executable)
    tessdata_dir = executable.parent / "tessdata"
    if (tessdata_dir / "eng.traineddata").is_file():
        os.environ["TESSDATA_PREFIX"] = str(tessdata_dir)
    return executable


def preprocess_for_digits(bgr_image: np.ndarray) -> np.ndarray:
    """Light preprocessing that helps printed digits without heavy filters."""
    if bgr_image is None or bgr_image.size == 0:
        return np.zeros((32, 32), dtype=np.uint8)

    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY) if bgr_image.ndim == 3 else bgr_image
    height, width = gray.shape[:2]
    min_side = min(height, width)
    if min_side < 48:
        scale = 48.0 / max(min_side, 1)
        gray = cv2.resize(
            gray,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
    elif min_side > 720:
        scale = 720.0 / min_side
        gray = cv2.resize(
            gray,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Tesseract usually prefers dark text on a light background.
    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)
    return thresh


def _parse_data_rows(data: dict) -> list[dict]:
    rows = []
    n = len(data.get("text", []))
    for i in range(n):
        text = str(data["text"][i]).strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < 0 or not text:
            continue
        rows.append(
            {
                "text": text,
                "conf": conf,
                "left": int(data["left"][i]),
                "top": int(data["top"][i]),
                "width": int(data["width"][i]),
                "height": int(data["height"][i]),
            }
        )
    return rows


def _digits_only(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit())


def _group_digit_boxes(rows: list[dict], x_offset: int, y_offset: int) -> list[SlotReading]:
    digit_boxes = []
    for row in rows:
        digits = _digits_only(row["text"])
        if not digits:
            continue
        if row["conf"] < MIN_OCR_CONFIDENCE:
            continue
        digit_boxes.append(
            {
                "text": digits,
                "conf": row["conf"],
                "x1": row["left"] + x_offset,
                "y1": row["top"] + y_offset,
                "x2": row["left"] + row["width"] + x_offset,
                "y2": row["top"] + row["height"] + y_offset,
            }
        )

    digit_boxes.sort(key=lambda item: (item["y1"], item["x1"]))
    used = [False] * len(digit_boxes)
    grouped: list[SlotReading] = []

    for i, box in enumerate(digit_boxes):
        if used[i]:
            continue
        cluster = [box]
        used[i] = True
        cy = (box["y1"] + box["y2"]) / 2.0
        height = max(1, box["y2"] - box["y1"])
        for j in range(i + 1, len(digit_boxes)):
            if used[j]:
                continue
            other = digit_boxes[j]
            other_cy = (other["y1"] + other["y2"]) / 2.0
            other_h = max(1, other["y2"] - other["y1"])
            if abs(other_cy - cy) > 0.55 * max(height, other_h):
                continue
            gap = other["x1"] - cluster[-1]["x2"]
            max_gap = DIGIT_GAP_RATIO * max(height, other_h)
            if 0 <= gap <= max_gap or other["x1"] <= cluster[-1]["x2"]:
                cluster.append(other)
                used[j] = True
            elif gap > max_gap * 2:
                break

        cluster.sort(key=lambda item: item["x1"])
        text = "".join(item["text"] for item in cluster)
        text = _digits_only(text)
        if not (MIN_SLOT_DIGITS <= len(text) <= MAX_SLOT_DIGITS):
            continue
        conf = float(sum(item["conf"] for item in cluster) / len(cluster))
        grouped.append(
            SlotReading(
                text=text,
                x1=min(item["x1"] for item in cluster),
                y1=min(item["y1"] for item in cluster),
                x2=max(item["x2"] for item in cluster),
                y2=max(item["y2"] for item in cluster),
                confidence=conf,
            )
        )
    return grouped


def detect_slot_numbers(
    bgr_image: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
) -> list[SlotReading]:
    """Return printed digit groups found in an image or ROI."""
    if pytesseract is None:
        return []
    if bgr_image is None or bgr_image.size == 0:
        return []

    x_offset = 0
    y_offset = 0
    crop = bgr_image
    if roi is not None:
        x1, y1, x2, y2 = roi
        height, width = bgr_image.shape[:2]
        x1 = max(0, min(width, int(x1)))
        x2 = max(0, min(width, int(x2)))
        y1 = max(0, min(height, int(y1)))
        y2 = max(0, min(height, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return []
        crop = bgr_image[y1:y2, x1:x2]
        x_offset, y_offset = x1, y1

    processed = preprocess_for_digits(crop)
    for config in (DIGIT_OCR_CONFIG, DIGIT_OCR_FALLBACK_CONFIG):
        try:
            data = pytesseract.image_to_data(
                processed,
                config=config,
                output_type=pytesseract.Output.DICT,
            )
        except pytesseract.TesseractNotFoundError as error:
            raise TesseractNotFoundError(tesseract_install_help()) from error
        except pytesseract.TesseractError:
            return []

        readings = _group_digit_boxes(_parse_data_rows(data), x_offset, y_offset)
        if readings:
            return readings

    return []


def expand_roi(
    box: tuple[float, float, float, float],
    frame_shape: tuple[int, ...],
    pad_x_ratio: float = 0.35,
    pad_y_ratio: float = 1.35,
    prefer_below: bool = True,
) -> tuple[int, int, int, int]:
    """Expand a thread-break box toward the likely label plate area."""
    x1, y1, x2, y2 = box
    height, width = frame_shape[:2]
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    pad_x = box_w * pad_x_ratio
    pad_y = box_h * pad_y_ratio
    rx1 = int(max(0, x1 - pad_x))
    rx2 = int(min(width, x2 + pad_x))
    if prefer_below:
        ry1 = int(max(0, y1 - box_h * 0.25))
        ry2 = int(min(height, y2 + pad_y))
    else:
        ry1 = int(max(0, y1 - pad_y))
        ry2 = int(min(height, y2 + box_h * 0.25))
    return rx1, ry1, rx2, ry2


def _horizontal_overlap_fraction(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    overlap = min(a[2], b[2]) - max(a[0], b[0])
    if overlap <= 0:
        return 0.0
    width = min(max(1, a[2] - a[0]), max(1, b[2] - b[0]))
    return overlap / width


def associate_thread_to_slot(
    thread_box: tuple[float, float, float, float],
    slots: list[SlotReading],
    fixed_regions: list[dict] | None = None,
) -> SlotReading | None:
    """Map one thread-break box to a slot number without guessing across spindles.

    A slot is accepted only when it shares the same vertical column as the
    detection (or a configured spindle region). If two slots are similarly
    plausible, return None so the UI can show SLOT UNKNOWN.
    """
    tx1, ty1, tx2, ty2 = (int(v) for v in thread_box)
    thread_rect = (tx1, ty1, tx2, ty2)
    tcx = (tx1 + tx2) / 2.0
    tcy = (ty1 + ty2) / 2.0
    thread_w = max(1, tx2 - tx1)

    if fixed_regions:
        matching_regions = []
        for region in fixed_regions:
            rx1, ry1, rx2, ry2 = (int(region[k]) for k in ("x1", "y1", "x2", "y2"))
            if rx1 <= tcx <= rx2 and ry1 <= tcy <= ry2:
                matching_regions.append(region)
        if len(matching_regions) == 1:
            slot_text = str(matching_regions[0].get("slot", "")).strip()
            if slot_text.isdigit():
                region = matching_regions[0]
                return SlotReading(
                    text=slot_text,
                    x1=int(region["x1"]),
                    y1=int(region["y1"]),
                    x2=int(region["x2"]),
                    y2=int(region["y2"]),
                    confidence=100.0,
                )
        if len(matching_regions) > 1:
            return None

    column_candidates: list[tuple[float, SlotReading]] = []
    for slot in slots:
        overlap = _horizontal_overlap_fraction(
            thread_rect,
            (slot.x1, slot.y1, slot.x2, slot.y2),
        )
        scx, scy = slot.center
        horiz = abs(scx - tcx)
        same_column = overlap >= COLUMN_OVERLAP_MIN or horiz <= 0.75 * max(thread_w, slot.width)
        if not same_column:
            continue
        dist = ((scx - tcx) ** 2 + (scy - tcy) ** 2) ** 0.5
        column_candidates.append((dist, slot))

    if not column_candidates:
        return None

    column_candidates.sort(key=lambda item: item[0])
    best_dist, best_slot = column_candidates[0]
    for dist, other in column_candidates[1:]:
        if other.text == best_slot.text:
            continue
        if dist <= best_dist * AMBIGUITY_DISTANCE_RATIO:
            return None
        other_overlap = _horizontal_overlap_fraction(
            thread_rect,
            (other.x1, other.y1, other.x2, other.y2),
        )
        best_overlap = _horizontal_overlap_fraction(
            thread_rect,
            (best_slot.x1, best_slot.y1, best_slot.x2, best_slot.y2),
        )
        if other_overlap >= COLUMN_OVERLAP_MIN and abs(other_overlap - best_overlap) < 0.15:
            return None
    return best_slot


def draw_slot_readings(frame: np.ndarray, slots: list[SlotReading]) -> None:
    for slot in slots:
        cv2.rectangle(frame, (slot.x1, slot.y1), (slot.x2, slot.y2), (255, 180, 0), 2)
        cv2.putText(
            frame,
            slot.text,
            (slot.x1, max(20, slot.y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 180, 0),
            2,
        )


def diagnose(image_path: Path | None = None) -> int:
    print("Checking Tesseract OCR setup...")
    try:
        executable = configure_tesseract()
    except TesseractNotFoundError as error:
        print(str(error))
        return 1

    print(f"Tesseract executable: {executable}")
    try:
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract version: {version}")
    except Exception as error:
        print(f"Tesseract was found but did not run: {error}")
        return 1

    if image_path is not None:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: could not read image: {image_path}")
            return 2

        readings = detect_slot_numbers(image)
        if not readings:
            print("No slot numbers detected.")
            return 0

        for reading in readings:
            print(
                f"Detected slot {reading.text} "
                f"(confidence: {reading.confidence:.1f}%, "
                f"box: {reading.x1},{reading.y1},{reading.x2},{reading.y2})"
            )
        return 0

    sample = np.full((80, 220), 255, dtype=np.uint8)
    cv2.putText(sample, "125", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3)
    sample_bgr = cv2.cvtColor(sample, cv2.COLOR_GRAY2BGR)
    readings = detect_slot_numbers(sample_bgr)
    texts = [item.text for item in readings]
    print(f"Synthetic OCR test (expected 125): {texts or ['(no digits read)']}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read slot numbers from an image.")
    parser.add_argument(
        "--image",
        type=Path,
        help="image file to scan; without this option, run the synthetic OCR self-test",
    )
    args = parser.parse_args()
    raise SystemExit(diagnose(args.image))
