
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

DATA_PATH = "dataset/gestures.csv"
MODEL_PATH = "models/hand_landmarker.task"

# Read existing dataset columns
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("Existing gestures.csv not found!")

existing_columns = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()
feature_columns = [c for c in existing_columns if c != "label"]

print("Feature columns:", len(feature_columns))

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

def extract_features(hand):

    landmarks = np.array([
        [lm.x, lm.y, lm.z] for lm in hand
    ], dtype=np.float32)

    landmarks = landmarks - landmarks[0].copy()

    scale = np.max(
        np.linalg.norm(landmarks[:, :2], axis=1)
    )

    if scale < 1e-6:
        return None

    landmarks = landmarks / scale
    return landmarks.flatten().tolist()


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise SystemExit("Webcam not detected!")

frame_count = 0
collected = 0
target = 100

print("\nRIGHT CLICK DATA COLLECTION")
print("Show THREE FINGERS: Index + Middle + Ring")
print("Press SPACE to save one sample")
print("Press Q to exit")

try:

    with HandLandmarker.create_from_options(options) as landmarker:

        while collected < target:

            success, frame = cap.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = frame_count * 33
            frame_count += 1

            result = landmarker.detect_for_video(
                mp_image, timestamp_ms
            )

            features = None

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]
                features = extract_features(hand)

                height, width, _ = frame.shape

                for lm in hand:
                    cv2.circle(
                        frame,
                        (int(lm.x * width), int(lm.y * height)),
                        4,
                        (0, 255, 0),
                        -1
                    )

            cv2.putText(
                frame,
                "Gesture: THREE FINGERS",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Samples: {collected}/{target}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "SPACE = Save | Q = Exit",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow("SmartHand - Right Click Dataset", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == 32:

                if features is None:
                    print("No hand detected. Try again.")
                    continue

                row = dict(zip(feature_columns, features))
                row["label"] = "three_fingers"

                pd.DataFrame([row]).to_csv(
                    DATA_PATH,
                    mode="a",
                    header=False,
                    index=False
                )

                collected += 1
                print(f"Saved sample: {collected}/{target}")

finally:

    cap.release()
    cv2.destroyAllWindows()

print(f"\nCollection completed: {collected} samples")
