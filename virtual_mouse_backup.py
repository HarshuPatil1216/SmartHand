
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pyautogui
import joblib
import time
from collections import deque, Counter

# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

bundle = joblib.load("models/gesture_model.pkl")

model = bundle["model"]
feature_columns = bundle["feature_columns"]

print("Gesture model loaded successfully!")

# ==========================================
# 2. MEDIAPIPE SETUP
# ==========================================

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1
)

# ==========================================
# 3. MOUSE SETTINGS
# ==========================================

screen_width, screen_height = pyautogui.size()

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

mouse_active = False

# Cursor smoothing
smoothening = 0.12
dead_zone = 3

prev_x = 0
prev_y = 0
cursor_initialized = False

# Average recent index finger positions
cursor_history = deque(maxlen=5)

# Gesture smoothing
gesture_history = deque(maxlen=5)

# Click cooldown
last_click_time = 0
click_cooldown = 0.6

# Scroll settings
previous_scroll_y = None
scroll_accumulator = 0
scroll_threshold = 8

# ==========================================
# 4. FEATURE EXTRACTION
# ==========================================

def extract_features(hand_landmarks):

    landmarks = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
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


# ==========================================
# 5. START WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam not detected!")
    raise SystemExit

frame_count = 0

print("\nSMART VIRTUAL MOUSE STARTED")
print("--------------------------------")
print("Open Palm     : Activate")
print("Fist          : Pause")
print("Index Point   : Cursor Movement")
print("Two Fingers   : Scroll")
print("Pinch         : Left Click")
print("Three Fingers : Right Click")
print("Q             : Exit")
print("--------------------------------")

# ==========================================
# 6. MAIN LOOP
# ==========================================

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:

    while True:

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

        gesture = "No Hand"
        confidence = 0.0

        # ==================================
        # 7. HAND DETECTION
        # ==================================

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

            if features is not None:

                input_data = pd.DataFrame(
                    [features],
                    columns=feature_columns
                )

                prediction = model.predict(input_data)[0]

                probabilities = model.predict_proba(input_data)[0]

                confidence = float(np.max(probabilities))

                if confidence >= 0.55:

                    gesture_history.append(prediction)

                    gesture = Counter(
                        gesture_history
                    ).most_common(1)[0][0]

                else:
                    gesture = "uncertain"

                # ==================================
                # 8. ACTIVATE MOUSE
                # ==================================

                if gesture == "open_palm":

                    mouse_active = True
                    previous_scroll_y = None
                    scroll_accumulator = 0

                # ==================================
                # 9. PAUSE MOUSE
                # ==================================

                elif gesture == "fist":

                    mouse_active = False
                    previous_scroll_y = None
                    scroll_accumulator = 0

                # ==================================
                # 10. CURSOR MOVEMENT
                # ==================================

                elif gesture == "index_point" and mouse_active:

                    # Index fingertip
                    index_x = int(hand[8].x * w)
                    index_y = int(hand[8].y * h)

                    # Average recent fingertip positions
                    cursor_history.append(
                        (index_x, index_y)
                    )

                    avg_x = int(np.mean(
                        [p[0] for p in cursor_history]
                    ))

                    avg_y = int(np.mean(
                        [p[1] for p in cursor_history]
                    ))

                    # Use central camera region
                    target_x = np.interp(
                        avg_x,
                        [w * 0.15, w * 0.85],
                        [0, screen_width - 1]
                    )

                    target_y = np.interp(
                        avg_y,
                        [h * 0.15, h * 0.85],
                        [0, screen_height - 1]
                    )

                    target_x = np.clip(
                        target_x, 0, screen_width - 1
                    )

                    target_y = np.clip(
                        target_y, 0, screen_height - 1
                    )

                    # Avoid initial cursor jump
                    if not cursor_initialized:

                        prev_x = target_x
                        prev_y = target_y
                        cursor_initialized = True

                    # Ignore tiny movements
                    if abs(target_x - prev_x) > dead_zone:

                        prev_x += (
                            target_x - prev_x
                        ) * smoothening

                    if abs(target_y - prev_y) > dead_zone:

                        prev_y += (
                            target_y - prev_y
                        ) * smoothening

                    pyautogui.moveTo(
                        int(prev_x),
                        int(prev_y)
                    )

                # ==================================
                # 11. TWO FINGERS - SMOOTH SCROLL
                # ==================================

                elif gesture == "two_fingers" and mouse_active:

                    # Middle point between index and middle tips
                    current_scroll_y = (
                        hand[8].y + hand[12].y
                    ) / 2

                    if previous_scroll_y is not None:

                        difference = (
                            previous_scroll_y - current_scroll_y
                        ) * h

                        scroll_accumulator += difference

                        # Scroll after enough movement
                        if abs(scroll_accumulator) >= scroll_threshold:

                            scroll_steps = int(
                                scroll_accumulator / scroll_threshold
                            )

                            scroll_steps = int(np.clip(
                                scroll_steps, -5, 5
                            ))

                            if scroll_steps != 0:

                                pyautogui.scroll(scroll_steps)

                                scroll_accumulator -= (
                                    scroll_steps * scroll_threshold
                                )

                    previous_scroll_y = current_scroll_y

                else:

                    previous_scroll_y = None
                    scroll_accumulator = 0

                # ==================================
                # 12. PINCH - LEFT CLICK
                # ==================================

                if gesture == "pinch" and mouse_active:

                    current_time = time.time()

                    if current_time - last_click_time > click_cooldown:

                        pyautogui.click()

                        last_click_time = current_time

                        cv2.putText(
                            frame,
                            "LEFT CLICK",
                            (30, 160),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )

                # ==================================
                # 13. THREE FINGERS - RIGHT CLICK
                # ==================================

                elif gesture == "three_fingers" and mouse_active:

                    current_time = time.time()

                    if current_time - last_click_time > click_cooldown:

                        pyautogui.rightClick()

                        last_click_time = current_time

                        cv2.putText(
                            frame,
                            "RIGHT CLICK",
                            (30, 160),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 0, 255),
                            2
                        )

        # ==================================
        # 14. DISPLAY INFORMATION
        # ==================================

        status = "ACTIVE" if mouse_active else "PAUSED"

        status_color = (
            (0, 255, 0) if mouse_active
            else (0, 0, 255)
        )

        cv2.putText(
            frame,
            f"Gesture: {gesture}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence * 100:.1f}%",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Mouse: {status}",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2
        )

        cv2.imshow(
            "Smart Hand Gesture Virtual Mouse",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# ==========================================
# 15. CLEANUP
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("Virtual Mouse stopped successfully!")
