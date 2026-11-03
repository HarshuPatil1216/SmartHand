import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam not detected!")
    exit()

print("SmartHand Webcam Started!")
print("Press Q to close webcam.")

while True:
    success, frame = cap.read()

    if not success:
        print("Unable to capture frame")
        break

    frame = cv2.flip(frame, 1)

    cv2.imshow("SmartHand - Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam closed successfully.")
