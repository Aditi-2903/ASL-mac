import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def normalize(row):
    coords = row.reshape(21, 3)
    coords = coords - coords[0]  # translate: wrist to origin

    # rotate so wrist->middle_mcp always points "up" in the image
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
    return (coords / scale).flatten()


def main():
    df = pd.read_csv("asl_landmarks_mac.csv")
    labels = df["label"].values
    X_raw = df.drop(columns=["label"]).values
    X = np.array([normalize(r) for r in X_raw])

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred, labels=sorted(set(labels))))

    joblib.dump(clf, "asl_letter_model_mac.joblib")
    print("Saved model to asl_letter_model_mac.joblib")


if __name__ == "__main__":
    main()
