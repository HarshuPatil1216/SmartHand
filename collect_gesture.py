import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import sys
import time

DATA_PATH = "dataset/gestures.csv"
MODEL_PATH = "models/hand_landmarker.task"

TARGET_ADDITIONAL = 200

VALID_LABELS = [
    "fist",
    "open_palm",
    "two_fingers",
    "pinch",
    "three_fingers"
]

if len(sys.argv) < 2:
    print("Usage: python collect_gesture.py gesture_name")
    print("Available gestures:", ", ".join(VALID_LABELS))
    raise SystemExit

LABEL = sys.argv[1].lower()

if LABEL not in VALID_LABELS:
    print("Invalid gesture:", LABEL)
    print("Choose from:", ", ".join(VALID_LABELS))
    raise SystemExit

existing_data = pd.read_csv(DATA_PATH)

existing_columns = existing_data.columns.tolist()
feature_columns = [c for c in existing_columns if c != "label"]

current_count = int((existing_data["label"] == LABEL).sum())

print("\nGesture:", LABEL)
print("Existing samples:", current_count)
print("New samples to collect:", TARGET_ADDITIONAL)
print("Expected total:", current_count + TARGET_ADDITIONAL)

def extract_features(hand):

    landmarks = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand],
        dtype=np.float32
    )

    landmarks = landmarks - landmarks[0].copy()

    scale = np.max(
        np.linalg.norm(landmarks[:, :2], axis=1)
    )

    if scale < 1e-6:
        return None

    landmarks = landmarks / scale

    return landmarks.flatten().tolist()


options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam not detected!")
    raise SystemExit

saved_count = 0
frame_count = 0
last_save_time = 0

print("\n--------------------------------")
print("DATA COLLECTION STARTED")
print("Gesture:", LABEL)
print("SPACE = Save sample")
print("Q = Exit")
print("--------------------------------")

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:

    while saved_count < TARGET_ADDITIONAL:

        success, frame = cap.read()

        if not success:
            print("Camera reading failed!")
            break

        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = frame_count * 33
        frame_count += 1

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        features = None

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            for lm in hand:
                cx = int(lm.x * w)
                cy = int(lm.y * h)

                cv2.circle(
                    frame,
                    (cx, cy),
                    4,
                    (0, 255, 0),
                    -1
                )

            features = extract_features(hand)

        cv2.putText(
            frame,
            f"Gesture: {LABEL}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Saved: {saved_count}/{TARGET_ADDITIONAL}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "SPACE = SAVE | Q = EXIT",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        if features is not None:
            cv2.putText(
                frame,
                "HAND DETECTED",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                frame,
                "NO HAND DETECTED",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        cv2.imshow("Gesture Dataset Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Collection stopped by user.")
            break

        if key == 32:

            current_time = time.time()

            if current_time - last_save_time < 0.35:
                continue

            if features is None:
                print("No hand detected. Try again.")
                continue

            row = features + [LABEL]

            sample = pd.DataFrame(
                [row],
                columns=existing_columns
            )

            sample.to_csv(
                DATA_PATH,
                mode="a",
                header=False,
                index=False
            )

            saved_count += 1
            last_save_time = current_time

            print(
                f"Saved sample: {saved_count}/{TARGET_ADDITIONAL}"
            )

cap.release()
cv2.destroyAllWindows()

print("\nCollection finished!")
print("Gesture:", LABEL)
print("New samples saved:", saved_count)
