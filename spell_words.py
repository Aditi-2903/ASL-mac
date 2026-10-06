import time
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
CLF_PATH = Path(__file__).parent / "asl_letter_model_mac.joblib"
CAPTURES_DIR = Path(__file__).parent / "captures"

STABLE_FRAMES = 10
MIN_CONFIDENCE = 0.6
WORD_GAP_SECONDS = 1.5  # no-hand-detected gap that counts as "end of word"


def normalize(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    coords = coords - coords[0]
    ref = coords[9]
    angle_current = np.arctan2(ref[1], ref[0])
    angle_target = np.arctan2(-1.0, 0.0)
    rot = angle_target - angle_current
    cos_r, sin_r = np.cos(rot), np.sin(rot)
    rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    coords[:, :2] = coords[:, :2] @ rot_matrix.T
    scale = np.linalg.norm(coords[9])
    if scale < 1e-6:
        scale = 1.0
    return (coords / scale).flatten().reshape(1, -1)


def draw_landmarks(frame, landmarks):
    h, w, _ = frame.shape
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)


def main():
    clf = joblib.load(CLF_PATH)
    CAPTURES_DIR.mkdir(exist_ok=True)

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    print("Warming up camera exposure...")
    for _ in range(20):
        cap.read()

    frame_counter = 0
    last_letter = None
    stable_count = 0
    printed_this_hold = False

    word_buffer = []
    last_hand_seen_time = None
    word_finalized = True  # True = waiting for a new word to start

    print("Running. Hold a letter to spell; pause (drop your hand) to end a word. Ctrl+C to stop.\n")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, frame_counter)

        if result.hand_landmarks:
            last_hand_seen_time = time.time()
            word_finalized = False

            landmarks = result.hand_landmarks[0]
            features = normalize(landmarks)
            pred = clf.predict(features)[0]
            conf = clf.predict_proba(features).max()

            if pred == last_letter and conf >= MIN_CONFIDENCE:
                stable_count += 1
            else:
                stable_count = 0
                printed_this_hold = False
            last_letter = pred

            if stable_count >= STABLE_FRAMES and not printed_this_hold:
                printed_this_hold = True
                word_buffer.append(pred)
                print(f"  + {pred}   (word so far: {''.join(word_buffer)})")

                letter_dir = CAPTURES_DIR / pred
                letter_dir.mkdir(exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = letter_dir / f"{pred}_{ts}_conf{conf:.2f}.jpg"
                annotated = frame.copy()
                draw_landmarks(annotated, landmarks)
                cv2.putText(annotated, f"{pred} ({conf:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imwrite(str(filename), annotated)
        else:
            last_letter = None
            stable_count = 0
            printed_this_hold = False

            if (
                word_buffer
                and not word_finalized
                and last_hand_seen_time is not None
                and (time.time() - last_hand_seen_time) > WORD_GAP_SECONDS
            ):
                word = "".join(word_buffer)
                print(f"\n>>> WORD: {word}\n")
                word_buffer = []
                word_finalized = True

        frame_counter += 1
        time.sleep(0.03)


if __name__ == "__main__":
    main()
