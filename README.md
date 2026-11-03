\# 🖐️ SmartHand – AI-Powered Virtual Mouse



\*\*Virtual Mouse using Hand Gesture Recognition and Facial Expression Recognition\*\*



SmartHand is an AI-powered Human-Computer Interaction (HCI) project that enables users to control computer mouse operations using hand gestures through a webcam. It also detects facial-expression categories in real time using Computer Vision and Deep Learning models.



\## 🚀 Features



\### 1. Hand Gesture Recognition



SmartHand recognizes six hand gestures:



| Hand Gesture | Mouse Operation |

|---|---|

| Open Palm | Activate Mouse |

| Fist | Pause Mouse |

| Index Point | Move Cursor |

| Two Fingers | Scroll |

| Pinch | Left Click |

| Three Fingers | Right Click |



\### 2. Facial Expression Recognition



The system detects seven facial-expression categories:



\- Angry

\- Disgust

\- Fearful

\- Happy

\- Neutral

\- Sad

\- Surprised



\### 3. Interactive Dashboard



The Streamlit dashboard provides:



\- Project overview

\- Hand gesture information

\- Facial expression information

\- Model performance

\- Confusion matrix

\- Application start/stop controls



\## 🛠️ Technologies Used



\- Python

\- OpenCV

\- MediaPipe

\- NumPy

\- Pandas

\- Scikit-learn

\- Random Forest Classifier

\- PyAutoGUI

\- Streamlit

\- ONNX Models

\- YuNet Face Detector

\- MobileFaceNet Facial Expression Model



\## 🧠 Machine Learning Model



The hand gesture recognition module uses a Random Forest Classifier.



\*\*Feature Extraction:\*\*

\- 21 hand landmarks

\- X, Y and Z coordinates

\- 63 normalized features



\*\*Dataset:\*\*

\- Total samples: 1,801

\- Six gesture classes

\- 80% training data

\- 20% testing data



\*\*Model Accuracy: 86.98%\*\*



The accuracy is based on the current random train-test split and may differ on independent real-world testing.



\## 🏗️ Project Architecture



1\. Webcam captures real-time video.

2\. OpenCV processes video frames.

3\. MediaPipe detects hand landmarks.

4\. Feature extraction generates normalized hand features.

5\. Random Forest predicts the hand gesture.

6\. PyAutoGUI performs mouse operations.

7\. YuNet detects the face.

8\. MobileFaceNet classifies facial expressions.

9\. Streamlit provides the project dashboard.



\## 📂 Project Structure



```text

SmartHand/

│

├── src/

├── dataset/

│   └── gestures.csv

│

├── models/

│   ├── gesture\_model.pkl

│   ├── hand\_landmarker.task

│   ├── face\_detection\_yunet\_2023mar.onnx

│   ├── facial\_expression\_recognition\_mobilefacenet\_2022july.onnx

│   └── confusion\_matrix.png

│

├── docs/

│   └── architecture.md

│

├── screenshots/

│

├── app.py

├── virtual\_mouse.py

├── train\_model.py

├── collect\_data.py

├── facial\_fer\_model.py

├── face\_expression\_test.py

├── requirements.txt

└── README.md

```



\## ⚙️ Installation and Setup



\### Step 1: Clone Repository



```bash

git clone https://github.com/HarshuPatil1216/SmartHand.git

cd SmartHand

```



\### Step 2: Create Virtual Environment



```bash

python -m venv venv

```



\### Step 3: Activate Virtual Environment



Windows PowerShell:



```powershell

.\\venv\\Scripts\\Activate.ps1

```



\### Step 4: Install Dependencies



```bash

python -m pip install -r requirements.txt

```



Ensure that PyAutoGUI is installed:



```bash

python -m pip install pyautogui

```



\### Step 5: Run SmartHand Virtual Mouse



```bash

python virtual\_mouse.py

```



\### Step 6: Run Streamlit Dashboard



Open another PowerShell terminal in the project directory:



```bash

streamlit run app.py

```



\## 📊 Model Performance



The current hand gesture model achieved \*\*86.98% accuracy\*\* on its held-out test split of 361 samples.



The project also includes a confusion matrix for evaluating gesture classification.



\## 💡 Applications



\- Touchless computer interaction

\- Human-Computer Interaction

\- Accessibility-oriented interaction systems

\- Computer Vision applications

\- AI and Machine Learning demonstrations



\## ⚠️ Important Notes



\- A working webcam is required.

\- Good lighting improves hand and face detection.

\- Facial expression predictions represent visible expression categories, not a person's actual emotional state.

\- Use the virtual mouse carefully because recognized gestures can trigger real mouse clicks and scrolling.



\## 🔮 Future Enhancements



\- Improve gesture recognition accuracy.

\- Add more custom hand gestures.

\- Improve cursor stability and scrolling.

\- Add additional accessibility features.

\- Improve dashboard visualization.

\- Evaluate models using independent recording sessions.



\## 👨‍💻 Author



\*\*Harsh Patil\*\*



GitHub: \[HarshuPatil1216](https://github.com/HarshuPatil1216)



\## ⭐ Support



If you find this project interesting, consider giving the repository a star!



\---



\*\*SmartHand – Making Computer Interaction Touchless with AI.\*\*

