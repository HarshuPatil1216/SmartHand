
import cv2
import pandas as pd
import mediapipe as mp
import numpy as np
import joblib
import time
from collections import deque, Counter

# ==========================================
# 1. LOAD TRAINED ML MODEL
# ==========================================

saved_data = joblib.load("models/gesture_model.pkl")
model = saved_data["model"]

# Original feature column names from training
feature_columns = saved_data["feature_columns"]

print("Random Forest model loaded successfully!")

# ==========================================
# 2. MEDIAPIPE TASKS API
# ==========================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# ==========================================
# 3. FEATURE EXTRACTION
# ==========================================

def extract_features(hand_landmarks):

    landmarks = np.array([
        [lm.x, lm.y, lm.z]
        for lm in hand_landmarks
    ], dtype=np.float32)

    wrist = landmarks[0].copy()
    landmarks = landmarks - wrist

    scale = np.max(
        np.linalg.norm(landmarks[:, :2], axis=1)
    )

    if scale < 1e-6:
        return None

    landmarks = landmarks / scale

    return landmarks.flatten().tolist()


# ==========================================
# 4. MEDIAPIPE CONFIGURATION
# ==========================================

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

# ==========================================
# 5. OPEN WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam not detected!")
    raise SystemExit

prediction_history = deque(maxlen=7)

frame_count = 0
start_time = time.time()

print("SmartHand Real-Time Recognition Started!")
print("Press Q to exit.")

# ==========================================
# 6. REAL-TIME RECOGNITION
# ==========================================

try:

    with HandLandmarker.create_from_options(options) as landmarker:

        while True:

            success, frame = cap.read()

            if not success:
                print("Unable to capture frame!")
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

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

            predicted_label = "No Hand"
            confidence = 0.0

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                height, width, _ = frame.shape
                points = []

                for lm in hand:

                    x = int(lm.x * width)
                    y = int(lm.y * height)

                    points.append((x, y))

                # Draw hand connections
                for start, end in HAND_CONNECTIONS:

                    cv2.line(
                        frame,
                        points[start],
                        points[end],
                        (0, 255, 0),
                        2
                    )

                # Draw landmarks
                for point in points:

                    cv2.circle(
                        frame,
                        point,
                        5,
                        (0, 0, 255),
                        -1
                    )

                # Extract 63 features
                features = extract_features(hand)

                if features is not None:

                    # IMPORTANT:
                    # Use DataFrame with original feature names
                    input_data = pd.DataFrame(
                        [features],
                        columns=feature_columns
                    )

                    # Predict gesture
                    prediction = model.predict(
                        input_data
                    )[0]

                    probabilities = model.predict_proba(
                        input_data
                    )[0]

                    confidence = float(
                        np.max(probabilities)
                    ) * 100

                    prediction_history.append(prediction)

                    # Majority voting
                    stable_label = Counter(
                        prediction_history
                    ).most_common(1)[0][0]

                    predicted_label = stable_label

                else:

                    prediction_history.clear()

            else:

                prediction_history.clear()

            # ==================================
            # 7. DISPLAY GESTURE AND CONFIDENCE
            # ==================================

            cv2.rectangle(
                frame,
                (10, 10),
                (430, 115),
                (30, 30, 30),
                -1
            )

            cv2.putText(
                frame,
                "Gesture: " + predicted_label,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence:.1f}%",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            # ==================================
            # 8. FPS DISPLAY
            # ==================================

            elapsed = time.time() - start_time

            fps = (
                frame_count / elapsed
                if elapsed > 0
                else 0
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # ==================================
            # 9. SHOW WEBCAM
            # ==================================

            cv2.imshow(
                "SmartHand - Real-Time Gesture Recognition",
                frame
            )

            # Press Q to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:

    cap.release()
    cv2.destroyAllWindows()

    print("SmartHand recognition stopped successfully.")
