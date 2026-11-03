# SmartHand – System Architecture

## 1. Project Overview

SmartHand is an AI-powered virtual mouse system that enables touchless computer interaction using hand gestures. It also includes facial expression recognition using computer vision.

## 2. Input Module

A webcam captures real-time video frames. OpenCV processes the frames before sending them to the hand and face detection modules.

## 3. Hand Gesture Recognition Module

- MediaPipe detects 21 hand landmarks.
- Feature extraction generates 63 normalized features.
- A trained Random Forest classifier predicts the hand gesture.
- Six gestures are used to control mouse operations.

## 4. Facial Expression Recognition Module

- YuNet detects faces.
- Facial landmarks are used for face alignment.
- MobileFaceNet classifies facial expressions into seven categories.

## 5. Mouse Control Module

PyAutoGUI performs mouse operations based on recognized gestures:

- Cursor movement
- Left click
- Right click
- Scrolling
- Mouse activation and pause

## 6. Dashboard Module

Streamlit provides a graphical dashboard containing project information, gesture details, facial expression categories, model performance and application start/stop controls.

## 7. Technologies Used

Python, OpenCV, MediaPipe, NumPy, Pandas, Scikit-learn, Random Forest, PyAutoGUI, Streamlit and ONNX models.

## 8. Model Performance

The current hand gesture classifier achieved 86.98% accuracy on its existing test split of 361 samples.