# SmartHand -- Interview & Viva Preparation

**Project:** SmartHand -- AI-Powered Virtual Mouse Using Hand Gesture
Recognition and Facial Expression Recognition\
**Repository:** https://github.com/HarshuPatil1216/SmartHand

## 1. Project overview

SmartHand is a touch-free human-computer interaction system. It captures
webcam frames, detects hand landmarks with MediaPipe, classifies
gestures using a trained Random Forest model, and performs mouse actions
using PyAutoGUI. A separate facial-expression module detects faces with
YuNet and classifies visible facial-expression categories using a
MobileFaceNet-based ONNX model. A Streamlit dashboard presents project
and model visualizations.

### Main modules

1.  Webcam capture and frame processing
2.  Hand landmark detection and feature extraction
3.  Gesture classification
4.  Gesture smoothing and mouse automation
5.  Face detection and facial-expression classification
6.  Streamlit dashboard

## 2. Technologies and libraries

  -----------------------------------------------------------------------
  Technology / library                Purpose
  ----------------------------------- -----------------------------------
  Python 3.13                         Main programming language

  OpenCV (`cv2`)                      Webcam capture, frame processing,
                                      display and DNN inference

  MediaPipe Tasks -- Hand Landmarker  Detects 21 hand landmarks

  NumPy                               Numerical operations and feature
                                      arrays

  Pandas                              Dataset handling

  scikit-learn                        Random Forest, train/test split and
                                      evaluation metrics

  PyAutoGUI                           Cursor movement, clicking and
                                      scrolling

  ONNX models                         Face detector and facial-expression
                                      model files

  Streamlit                           Interactive dashboard

  Plotly                              Interactive charts, where used

  Matplotlib                          Confusion-matrix visualizations

  joblib / pickle                     Model persistence, depending on the
                                      model-loading code
  -----------------------------------------------------------------------

**Note:** Verify exact package versions against the current
`requirements.txt`. ONNX is a model format, not the
gesture-classification algorithm.

### Important model files

-   `models/gesture_model.pkl` -- trained gesture classifier/model
    bundle.
-   `models/hand_landmarker.task` -- MediaPipe hand landmark model.
-   `models/face_detection_yunet_2023mar.onnx` -- YuNet face detector.
-   `models/facial_expression_recognition_mobilefacenet_2022july.onnx`
    -- facial-expression model.
-   `models/confusion_matrix.png` -- gesture-model confusion matrix.

## 3. Algorithms and models

### Random Forest -- hand gesture classification

Random Forest is a supervised ensemble algorithm. It trains multiple
decision trees and combines their predictions; for classification, the
final class is selected by majority voting.

**Project configuration** - `RandomForestClassifier` - 300 estimators
(decision trees) - `class_weight='balanced'` - 63 normalized input
features - Six gesture classes - Stratified 80:20 train/test split

**Why Random Forest?** The input consists of structured numerical
landmark features. Random Forest can learn nonlinear patterns and
combines multiple trees, making it a practical classifier for this
multi-class task.

### MediaPipe Hand Landmarker

MediaPipe detects 21 hand landmarks. Each landmark has X, Y and Z
coordinates. MediaPipe performs landmark detection; it is not the Random
Forest classifier.

### Facial expression recognition

-   **YuNet:** detects the face region.
-   **MobileFaceNet-based ONNX model:** classifies the face into one of
    seven visible expression categories.
-   **OpenCV DNN:** loads and runs the ONNX model.

Expression labels are visual categories, not definitive measurements of
someone's internal emotional state.

## 4. Feature engineering

-   21 hand landmarks are detected.
-   Each landmark has three coordinates: X, Y and Z.
-   Total feature count: `21 × 3 = 63`.
-   Features are normalized relative to wrist landmark 0, using a scale
    based on the maximum 2D landmark distance.

**Why normalize?** It reduces the effect of hand position and apparent
size, helping the classifier focus more on relative hand shape.

### Gesture classes and actions

  Gesture           Action
  ----------------- ------------------------
  `open_palm`       Activate virtual mouse
  `fist`            Pause virtual mouse
  `index_point`     Move cursor
  `two_fingers`     Scroll
  `pinch`           Left click
  `three_fingers`   Right click

### Prediction stability

-   Gesture prediction history: maximum 5 predictions.
-   Gesture confidence threshold: 0.55.
-   Cursor smoothing and dead-zone handling.
-   Facial-expression prediction history: up to 7 predictions.

## 5. Dataset and evaluation

### Dataset distribution

-   Total: **1,801 samples**
-   `fist`: 301
-   `open_palm`: 300
-   `index_point`: 300
-   `two_fingers`: 300
-   `pinch`: 300
-   `three_fingers`: 300

### Training and testing

-   Stratified 80:20 train/test split.
-   Test set: 361 samples.
-   Stratification preserves approximately the same class proportions in
    train and test sets.

### Result

# 86.98% gesture-model accuracy

This is the accuracy on a randomly selected, stratified held-out test
split of 361 samples. It does not guarantee the same performance for
every new user, camera, lighting condition or independent recording
session. A stronger evaluation would hold out complete recording
sessions or users.

### Per-class precision and recall

  Gesture           Precision   Recall
  --------------- ----------- --------
  fist                   0.85     0.90
  index_point            0.78     0.87
  open_palm              0.92     0.95
  pinch                  0.88     0.85
  three_fingers          0.88     0.85
  two_fingers            0.94     0.80

### Evaluation terms

-   **Accuracy:** proportion of all predictions that are correct.
-   **Precision:** among predictions for a class, the proportion that
    are correct.
-   **Recall:** among actual samples of a class, the proportion
    correctly identified.
-   **F1-score:** harmonic mean of precision and recall.
-   **Confusion matrix:** compares actual and predicted classes.

## 6. End-to-end workflow

1.  OpenCV captures frames from the webcam.
2.  MediaPipe detects the hand and its 21 landmarks.
3.  The application converts landmarks into 63 normalized features.
4.  Random Forest predicts one of six gestures.
5.  Prediction history and confidence threshold help stabilize the
    decision.
6.  The recognized gesture is mapped to a PyAutoGUI mouse operation.
7.  Separately, YuNet detects a face and the ONNX expression model
    predicts a visible expression category.
8.  Streamlit presents project information and model visualizations.

## 7. Facial-expression categories

1.  Angry
2.  Disgust
3.  Fearful
4.  Happy
5.  Neutral
6.  Sad
7.  Surprised

## 8. Viva / interview questions and answers

### Q1. What is SmartHand?

SmartHand is an AI-powered virtual mouse that uses webcam-based hand
gestures to control mouse operations. It also includes facial-expression
recognition and a Streamlit dashboard.

### Q2. What problem does it solve?

It demonstrates a touch-free way to perform basic mouse operations and
an alternative interaction concept.

### Q3. Which programming language did you use?

Python 3.13.

### Q4. Which algorithm is used for gesture classification?

Random Forest Classifier with 300 estimators.

### Q5. Why did you choose Random Forest?

The input is structured numerical landmark data. Random Forest combines
decision trees and can learn nonlinear patterns, making it a practical
multi-class classifier.

### Q6. What is MediaPipe?

MediaPipe is a framework for perception pipelines. Here, its Hand
Landmarker detects 21 hand landmarks.

### Q7. Is MediaPipe the classification algorithm?

No. MediaPipe detects landmarks. The extracted features are passed to
the trained Random Forest model for gesture classification.

### Q8. How many landmarks are detected?

21 landmarks per detected hand.

### Q9. How many input features are used?

63: 21 landmarks multiplied by three coordinates (X, Y and Z).

### Q10. Why normalize landmarks?

Features are calculated relative to the wrist and normalized by scale.
This reduces sensitivity to hand location and apparent size.

### Q11. How many gestures are recognized?

Six: open palm, fist, index point, two fingers, pinch and three fingers.

### Q12. How does cursor movement work?

The index-point gesture controls cursor movement. The detected hand
position is mapped to screen coordinates, with smoothing to reduce
jitter.

### Q13. How do left and right clicks work?

Pinch performs a left click; the three-finger gesture performs a right
click.

### Q14. How does scrolling work?

The two-finger gesture activates scrolling through PyAutoGUI.

### Q15. What is PyAutoGUI?

A Python automation library used to control pointer movement, clicks and
scrolling.

### Q16. What is Random Forest?

An ensemble method that combines predictions from multiple decision
trees. Classification uses majority voting.

### Q17. What does `n_estimators=300` mean?

The Random Forest contains 300 decision trees.

### Q18. What does `class_weight='balanced'` mean?

It adjusts class weights inversely according to class frequencies to
help account for class imbalance.

### Q19. What is a train-test split?

It separates data into training samples used to fit the model and test
samples used to evaluate it.

### Q20. Why use stratified splitting?

It preserves approximately the same class distribution in both subsets.

### Q21. What is your model accuracy?

86.98% on a stratified held-out test set of 361 samples.

### Q22. Does that guarantee the same real-world accuracy?

No. The split is sample-based and may contain samples similar to
training samples. Independent sessions and users are needed for stronger
generalization testing.

### Q23. What is a confusion matrix?

A table comparing actual class labels with predicted labels, showing
correct and incorrect classifications.

### Q24. What is precision?

The proportion of predictions for a class that are actually correct.

### Q25. What is recall?

The proportion of actual samples of a class that the model correctly
identifies.

### Q26. How do you reduce cursor shaking?

Cursor smoothing and a dead zone reduce small unwanted movements.

### Q27. Why maintain prediction history?

It reduces rapid switching between gesture labels due to noisy frames.

### Q28. What is the confidence threshold?

A minimum prediction-confidence value required before accepting a
gesture. The configured value is 0.55.

### Q29. Which face detector is used?

YuNet, loaded from an ONNX model file.

### Q30. Which facial-expression model is used?

A MobileFaceNet-based facial-expression model in ONNX format.

### Q31. Which expression categories are recognized?

Angry, disgust, fearful, happy, neutral, sad and surprised.

### Q32. What is ONNX?

Open Neural Network Exchange, a format for representing machine-learning
models for use across compatible tools and runtimes.

### Q33. What is OpenCV DNN?

OpenCV's Deep Neural Network module, which can load and run supported
neural-network models such as ONNX models.

### Q34. Why Streamlit?

It provides a quick way to build an interactive Python dashboard for
project information, metrics and visualizations.

### Q35. What is Plotly used for?

Interactive charts and visualizations in the dashboard, where
configured.

### Q36. What are the limitations?

Performance depends on lighting, camera position, hand visibility, user
differences and similarity between gestures. Facial-expression
predictions can be affected by face angle, lighting and occlusion.

### Q37. What happens if no hand is detected?

The application cannot extract landmarks for that frame, so it cannot
make a reliable gesture-based decision from that frame.

### Q38. Can it work without a physical mouse?

It can issue mouse-control actions through PyAutoGUI. The computer still
needs a supported operating environment and input setup; the project
provides virtual mouse control.

### Q39. What are future improvements?

More diverse training data, independent-user testing, customizable
gestures, better cursor stability, voice commands and improved
accessibility.

### Q40. What did you learn?

Webcam-based computer vision, MediaPipe landmark detection, feature
engineering, supervised classification, model evaluation, ONNX
inference, Python automation, dashboard development and GitHub
documentation.

## 9. One-minute project explanation

"Good morning. My project is SmartHand, an AI-powered virtual mouse
using hand gesture recognition and facial-expression recognition. We
capture live video through a webcam using OpenCV. MediaPipe detects 21
hand landmarks, and we extract 63 normalized coordinate features. These
features are passed to a Random Forest classifier with 300 estimators,
which recognizes six hand gestures. We map these gestures to cursor
movement, left click, right click, scrolling and pause or activation
using PyAutoGUI. We also use YuNet for face detection and a
MobileFaceNet-based ONNX model to classify seven visible
facial-expression categories. Our gesture model achieved 86.98% accuracy
on a stratified held-out test set of 361 samples. We built a Streamlit
dashboard to present model information and visualizations. The project
demonstrates practical integration of machine learning, computer vision
and automation."

## 10. Resume project description

**SmartHand -- AI-Powered Virtual Mouse \| Python, OpenCV, MediaPipe,
scikit-learn, Random Forest, PyAutoGUI, ONNX, Streamlit**

-   Developed a webcam-based virtual mouse using MediaPipe hand
    landmarks and a Random Forest gesture classifier.
-   Extracted 63 normalized features from 21 hand landmarks to classify
    six hand gestures.
-   Integrated PyAutoGUI for cursor movement, clicking and scrolling,
    with prediction-history and cursor-smoothing techniques.
-   Added YuNet face detection and an ONNX-based facial-expression
    classification module.
-   Achieved 86.98% gesture-classification accuracy on a stratified
    held-out test set of 361 samples and built a Streamlit visualization
    dashboard.

## 11. Interview preparation tips

-   Explain the difference between MediaPipe landmark detection and
    Random Forest gesture classification.
-   Be ready to explain why there are 63 features:
    `21 landmarks × 3 coordinates`.
-   State accuracy with its test-set context: 86.98% on 361 held-out
    samples.
-   Do not claim independent-user or real-world accuracy unless
    separately tested.
-   Do not claim expression labels reveal actual emotions.
-   Be ready to open the code and explain feature extraction,
    prediction, gesture mapping and face-model inference.
-   Check the actual `requirements.txt` and imports before quoting exact
    package versions.
