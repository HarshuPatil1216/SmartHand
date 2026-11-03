import cv2
import mediapipe as mp
import csv
import os
import time
import numpy as np

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Hand connections
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (5,9), (9,10), (10,11), (11,12),
    (9,13), (13,14), (14,15), (15,16),
    (13,17), (17,18), (18,19), (19,20),
    (0,17)
]

# Gesture labels and keyboard keys
GESTURES = {
    ord('1'): "open_palm",
    ord('2'): "fist",
    ord('3'): "index_point",
    ord('4'): "two_fingers",
    ord('5'): "pinch"
}

DATASET_PATH = "dataset/gestures.csv"
os.makedirs("dataset", exist_ok=True)

# Extract 63 normalized features
def extract_features(hand_landmarks):

    landmarks = np.array([
        [lm.x, lm.y, lm.z]
        for lm in hand_landmarks
    ], dtype=np.float32)

    # Wrist is landmark 0
    wrist = landmarks[0].copy()

    # Make wrist the origin
    landmarks = landmarks - wrist

    # Normalize using maximum 2D distance from wrist
    scale = np.max(np.linalg.norm(landmarks[:, :2], axis=1))

    if scale < 1e-6:
        return None

    landmarks = landmarks / scale

    return landmarks.flatten().tolist()


# MediaPipe configuration
options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Prepare CSV file
file_exists = os.path.exists(DATASET_PATH)

with open(DATASET_PATH, "a", newline="") as file:
    writer = csv.writer(file)

    if not file_exists or os.path.getsize(DATASET_PATH) == 0:
        writer.writerow(
            [f"feature_{i}" for i in range(63)] + ["label"]
        )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Webcam not detected!")
        raise SystemExit

    frame_count = 0
    sample_counts = {name: 0 for name in GESTURES.values()}

    print("\nSMART HAND - DATASET COLLECTION")
    print("--------------------------------")
    print("1 = Open Palm")
    print("2 = Fist")
    print("3 = Index Point")
    print("4 = Two Fingers")
    print("5 = Pinch")
    print("Q = Exit")
    print("--------------------------------")

    try:
        with HandLandmarker.create_from_options(options) as landmarker:

            while True:

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

                hand_detected = False

                if result.hand_landmarks:

                    hand_detected = True
                    hand = result.hand_landmarks[0]

                    height, width, _ = frame.shape
                    points = []

                    for lm in hand:
                        points.append((
                            int(lm.x * width),
                            int(lm.y * height)
                        ))

                    for start, end in HAND_CONNECTIONS:
                        cv2.line(
                            frame,
                            points[start],
                            points[end],
                            (0, 255, 0),
                            2
                        )

                    for point in points:
                        cv2.circle(
                            frame, point, 5, (0, 0, 255), -1
                        )

                cv2.putText(
                    frame,
                    "Press 1-5 to save gesture sample",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2
                )

                y_position = 65

                for key, name in GESTURES.items():
                    text = f"{chr(key)} - {name}: {sample_counts[name]}"
                    cv2.putText(
                        frame,
                        text,
                        (10, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2
                    )
                    y_position += 28

                cv2.imshow("SmartHand - Dataset Collection", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

                if key in GESTURES:

                    if not hand_detected:
                        print("No hand detected. Try again.")
                        continue

                    features = extract_features(
                        result.hand_landmarks[0]
                    )

                    if features is not None:
                        label = GESTURES[key]

                        writer.writerow(features + [label])
                        file.flush()

                        sample_counts[label] += 1

                        print(
                            f"Saved {label}: "
                            f"{sample_counts[label]} samples"
                        )

    finally:
        cap.release()
        cv2.destroyAllWindows()

print("\nDataset collection completed!")
print("Samples collected:", sample_counts)
print("Dataset saved at:", DATASET_PATH)
