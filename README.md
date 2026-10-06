# ASL-mac

A static ASL fingerspelling classifier built on MediaPipe hand landmarks, developed and iterated on macOS before porting to a Raspberry Pi 5 edge deployment.

## Approach
Each frame's 21 hand landmarks (x, y, z) are translated to the wrist, rotation-normalized (so in-plane hand tilt doesn't affect classification), and scale-normalized (so distance from the camera doesn't matter). A RandomForestClassifier is trained on these normalized features to classify the static handshape.

## Setup
python3 -m venv asl-env
source asl-env/bin/activate
pip install -r requirements.txt
curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

## Scripts
- `collect_letters.py` — walks through a list of letters, capturing labeled landmark samples into `asl_landmarks_mac.csv`
- `train_classifier.py` — trains the RandomForest on the CSV, prints a classification report + confusion matrix
- `live_classify.py` / `live_display.py` — real-time classification from the webcam (terminal output vs. a live OpenCV window with per-class probability bars)
- `debug_predict.py` / `debug_predict_avg.py` — diagnostic tools for watching per-class confidence live while adjusting hand position
- `spell_words.py` — buffers locked letters into words, using a pause in hand detection as a word boundary

## Current letters
A, B, C, D, F, I, L, V, W, Y (static holds only — J and Z require motion and aren't handled by this approach)

## Known limitations / future work
- Rotation-normalization only corrects in-plane tilt, not full 3D viewing angle (e.g. camera positioned above/beside vs. facing the signer)
- G/D, H/N/U, and K/P share near-identical handshapes in ASL, distinguished mainly by hand orientation — conflicts with the rotation-normalization approach and needs dedicated handling
- A/S/T form a tight closed-fist cluster differentiated only by subtle thumb placement
- No handling yet for doubled letters in a word, or continuous/connected fingerspelling
