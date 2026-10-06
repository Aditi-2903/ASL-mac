import time
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
CLF_PATH = Path(__file__).parent / "asl_letter_model_mac.joblib"

STABLE_FRAMES = 10
MIN_CONFIDENCE = 0.6


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
    classes = clf.classes_

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    print("Warming up camera exposure...")
    for _ in range(20):
        cap.read()

    frame_counter = 0
    last_letter = None
    stable_count = 0

    print("Press 'q' in the window to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, frame_counter)
        frame_counter += 1

        display = frame.copy()

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            draw_landmarks(display, landmarks)

            features = normalize(landmarks)
            proba = clf.predict_proba(features)[0]
            pred = classes[np.argmax(proba)]
            conf = proba.max()

            if pred == last_letter and conf >= MIN_CONFIDENCE:
                stable_count += 1
            else:
                stable_count = 0
            last_letter = pred

            locked = stable_count >= STABLE_FRAMES
            color = (0, 255, 0) if locked else (0, 165, 255)
            label = f"{pred} ({conf:.2f})" + ("  LOCKED" if locked else "")
            cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # per-class probability bars, so you can see what it's torn between
            for i, (c, p) in enumerate(zip(classes, proba)):
                y = 60 + i * 22
                bar_len = int(p * 150)
                cv2.rectangle(display, (10, y), (10 + bar_len, y + 15), (255, 200, 0), -1)
                cv2.putText(display, f"{c}:{p:.2f}", (165, y + 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            last_letter = None
            stable_count = 0
            cv2.putText(display, "no hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("ASL live debug", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
