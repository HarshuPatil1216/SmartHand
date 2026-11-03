import cv2
import mediapipe as mp
import time

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

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam not detected!")
    exit()

frame_count = 0
start_time = time.time()

try:
    with HandLandmarker.create_from_options(options) as landmarker:

        print("SmartHand Hand Detection Started!")
        print("Show your hand in front of webcam.")
        print("Press Q to exit.")

        while True:

            success, frame = cap.read()

            if not success:
                print("Unable to capture frame!")
                break

            # Mirror webcam display
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

            if result.hand_landmarks:

                for hand_index, hand in enumerate(result.hand_landmarks):

                    height, width, _ = frame.shape
                    points = []

                    # Extract 21 hand landmarks
                    for landmark in hand:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
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

                    # Draw landmark points
                    for point in points:
                        cv2.circle(
                            frame,
                            point,
                            5,
                            (0, 0, 255),
                            -1
                        )

                    # Correct mirrored handedness label
                    label = "Unknown"

                    if result.handedness and hand_index < len(result.handedness):
                        original_label = (
                            result.handedness[hand_index][0].category_name
                        )

                        if original_label == "Left":
                            label = "Right"
                        elif original_label == "Right":
                            label = "Left"

                    # Display corrected label
                    cv2.putText(
                        frame,
                        label + " Hand",
                        (points[0][0], points[0][1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 0, 0),
                        2
                    )

                cv2.putText(
                    frame,
                    "Hand Detected!",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "Show your hand",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            # Calculate FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.imshow(
                "SmartHand - Hand Detection",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Hand detection stopped successfully.")
