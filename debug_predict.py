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


def main():
    clf = joblib.load(CLF_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    print("Warming up...")
    for _ in range(20):
        cap.read()

    frame_counter = 0
    print("Ctrl+C to stop.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, frame_counter)
        frame_counter += 1

        if result.hand_landmarks:
            features = normalize(result.hand_landmarks[0])
            proba = clf.predict_proba(features)[0]
            classes = clf.classes_
            pred = classes[np.argmax(proba)]
            conf = proba.max()
            all_probs = "  ".join(f"{c}={p:.2f}" for c, p in zip(classes, proba))
            print(f"pred={pred} conf={conf:.2f}  |  {all_probs}")
        else:
            print("no hand detected")

        time.sleep(0.2)


if __name__ == "__main__":
    main()
