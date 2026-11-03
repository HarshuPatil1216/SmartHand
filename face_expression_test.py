import cv2
import time
from collections import deque, Counter
from facial_fer_model import FacialExpressionRecog

# Model paths
FACE_MODEL = "models/face_detection_yunet_2023mar.onnx"
EXPRESSION_MODEL = "models/facial_expression_recognition_mobilefacenet_2022july.onnx"

# Load face detector
face_detector = cv2.FaceDetectorYN.create(
    FACE_MODEL,
    "",
    (640, 480),
    0.7,
    0.3,
    5000
)

# Load expression recognition model
expression_model = FacialExpressionRecog(
    modelPath=EXPRESSION_MODEL
)

expression_history = deque(maxlen=7)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam not detected!")
    raise SystemExit

print("SmartHand Facial Expression Recognition Started")
print("Press Q to exit")

try:
    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        face_detector.setInputSize((width, height))

        # Detect face
        _, faces = face_detector.detect(frame)

        expression = "No Face Detected"

        if faces is not None:

            # Process the largest detected face
            face = max(
                faces,
                key=lambda f: f[2] * f[3]
            )

            x, y, w, h = face[:4].astype(int)

            # Predict facial expression
            prediction = expression_model.infer(
                frame,
                face[:-1]
            )

            expression_id = int(prediction[0])

            expression = expression_model.getDesc(
                expression_id
            )

            expression_history.append(expression)

            # Smooth predictions
            stable_expression = Counter(
                expression_history
            ).most_common(1)[0][0]

            expression = stable_expression

            # Draw face rectangle
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        else:
            expression_history.clear()

        # Display result
        cv2.rectangle(
            frame,
            (10, 10),
            (460, 85),
            (30, 30, 30),
            -1
        )

        cv2.putText(
            frame,
            "Facial Expression: " + expression,
            (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "SmartHand - Facial Expression Recognition",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

    print("Facial expression recognition stopped.")
