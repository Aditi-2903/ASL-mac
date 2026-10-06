import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

HANDSHAPE_HINTS = {
    "A": "Closed fist, thumb resting upright against the side of the index finger.",
    "B": "Flat hand, four fingers together pointing up, thumb folded across the palm.",
    "C": "Whole hand curved into a 'C' shape, like holding an invisible cup.",
    "D": "Index finger straight up, thumb touches curled middle fingertip, ring/pinky curled.",
    "F": "Thumb and index fingertip form a circle; middle, ring, pinky extended and spread.",
    "I": "Pinky extended straight up, other fingers curled, thumb pressed against the curl.",
    "L": "Thumb and index finger extended at a right angle, other three fingers curled in.",
    "V": "Index and middle fingers extended in a V, ring/pinky curled, thumb tucked.",
    "W": "Index, middle, ring fingers extended and spread, pinky and thumb curled together.",
    "Y": "Thumb and pinky extended out, three middle fingers curled into the palm.",
}


def flatten_landmarks(hand_landmarks):
    return [coord for lm in hand_landmarks for coord in (lm.x, lm.y, lm.z)]


def draw_landmarks(frame, landmarks):
    h, w, _ = frame.shape
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--letters", type=str, default="A,B,C,L,Y")
    parser.add_argument("--samples-per-letter", type=int, default=40)
    parser.add_argument("--out", type=str, default="asl_landmarks_mac.csv")
    parser.add_argument("--images-dir", type=str, default="training_images")
    parser.add_argument("--rotate180", action="store_true", help="Set this on the Pi, leave off on Mac")
    args = parser.parse_args()

    letters = [l.strip() for l in args.letters.split(",") if l.strip()]
    images_dir = Path(args.images_dir)

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"Could not open camera at index {args.device}.")
        return

    print("Warming up camera exposure...")
    for _ in range(20):
        cap.read()

    out_path = Path(args.out)
    write_header = not out_path.exists()
    f = open(out_path, "a", newline="")
    writer = csv.writer(f)
    if write_header:
        header = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
        writer.writerow(header)

    frame_counter = 0

    for letter in letters:
        hint = HANDSHAPE_HINTS.get(letter)
        print(f"\n=== Letter '{letter}' ===")
        if hint:
            print(f"  Shape: {hint}")
        input("Hold the shape steady, press Enter to start capturing...")
        collected = 0
        print(f"Capturing {args.samples_per_letter} samples for '{letter}'...")

        letter_dir = images_dir / letter
        letter_dir.mkdir(parents=True, exist_ok=True)

        while collected < args.samples_per_letter:
            ret, frame = cap.read()
            if not ret:
                continue
            if args.rotate180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, frame_counter)
            frame_counter += 1

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                writer.writerow([letter] + flatten_landmarks(landmarks))
                collected += 1

                annotated = frame.copy()
                draw_landmarks(annotated, landmarks)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                img_path = letter_dir / f"{letter}_{ts}.jpg"
                cv2.imwrite(str(img_path), annotated)

                if collected % 10 == 0:
                    print(f"  {collected}/{args.samples_per_letter}")
            time.sleep(0.05)

        print(f"Done with '{letter}'.")

    f.close()
    cap.release()
    landmarker.close()
    print(f"\nSaved dataset to {out_path.resolve()}")
    print(f"Saved training images to {images_dir.resolve()}/<letter>/")


if __name__ == "__main__":
    main()
