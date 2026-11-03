
import streamlit as st
import pandas as pd
import subprocess
import sys
from pathlib import Path
import time


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

VIRTUAL_MOUSE_FILE = BASE_DIR / "virtual_mouse.py"
MODEL_FILE = BASE_DIR / "models" / "gesture_model.pkl"
DATASET_FILE = BASE_DIR / "dataset" / "gestures.csv"
CONFUSION_MATRIX = BASE_DIR / "models" / "confusion_matrix.png"

st.set_page_config(
    page_title="SmartHand AI Dashboard",
    page_icon="🖐️",
    layout="wide"
)


# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}

h1, h2, h3 {
    color: #ffffff;
}

div[data-testid="stMetric"] {
    background-color: #1c2333;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #30394d;
}

div.stButton > button {
    border-radius: 10px;
    height: 48px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. SESSION STATE
# ==========================================

if "mouse_process" not in st.session_state:
    st.session_state.mouse_process = None


def is_running():

    process = st.session_state.mouse_process

    return process is not None and process.poll() is None


# ==========================================
# 4. SIDEBAR
# ==========================================

st.sidebar.title("🖐️ SmartHand")

st.sidebar.caption("AI Powered Human Computer Interaction")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Hand Gestures",
        "Facial Recognition",
        "Model Performance",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "SmartHand uses Machine Learning "
    "and Computer Vision to control "
    "a computer using hand gestures."
)


# ==========================================
# 5. DASHBOARD PAGE
# ==========================================

if page == "Dashboard":

    st.title("🖐️ SmartHand AI Dashboard")

    st.write(
        "Virtual Mouse using Hand Gesture Recognition "
        "and Facial Expression Recognition"
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Hand Gestures", "6")

    col2.metric("Expressions", "7")

    col3.metric("Model Accuracy", "86.98%")

    col4.metric("Input Device", "Webcam")

    st.divider()

    st.subheader("🚀 Virtual Mouse Control")

    st.write(
        "Start the SmartHand application to control "
        "your computer using hand gestures."
    )

    col_start, col_stop = st.columns(2)

    with col_start:

        if st.button(
            "▶ Start SmartHand",
            use_container_width=True
        ):

            if is_running():

                st.warning("SmartHand is already running.")

            elif not VIRTUAL_MOUSE_FILE.exists():

                st.error("virtual_mouse.py file not found.")

            elif not MODEL_FILE.exists():

                st.error("Gesture model not found.")

            else:

                try:

                    st.session_state.mouse_process = (
                        subprocess.Popen(
                            [
                                sys.executable,
                                str(VIRTUAL_MOUSE_FILE)
                            ],
                            cwd=str(BASE_DIR)
                        )
                    )

                    st.success("SmartHand started successfully!")

                except Exception as error:

                    st.error(f"Unable to start: {error}")

    with col_stop:

        if st.button(
            "⏹ Stop SmartHand",
            use_container_width=True
        ):

            if is_running():

                st.session_state.mouse_process.terminate()

                try:
                    st.session_state.mouse_process.wait(
                        timeout=3
                    )
                except subprocess.TimeoutExpired:
                    st.session_state.mouse_process.kill()

                st.session_state.mouse_process = None

                st.success("SmartHand stopped.")

            else:

                st.info("SmartHand is not running.")

    if is_running():

        st.success("🟢 SmartHand Status: RUNNING")

    else:

        st.warning("🔴 SmartHand Status: STOPPED")

    st.divider()

    st.subheader("System Features")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        ### ✋ Hand Gesture Recognition

        - Open Palm – Activate Mouse
        - Fist – Pause Mouse
        - Index Point – Move Cursor
        - Two Fingers – Scroll
        - Pinch – Left Click
        - Three Fingers – Right Click
        """)

    with col2:

        st.markdown("""
        ### 😊 Facial Expression Recognition

        - Angry
        - Disgust
        - Fearful
        - Happy
        - Neutral
        - Sad
        - Surprised
        """)


# ==========================================
# 6. HAND GESTURES PAGE
# ==========================================

elif page == "Hand Gestures":

    st.title("✋ Hand Gesture Recognition")

    st.write(
        "SmartHand recognizes six hand gestures "
        "to perform computer operations."
    )

    gestures = {
        "Open Palm": "Activate Virtual Mouse",
        "Fist": "Pause Virtual Mouse",
        "Index Point": "Move Cursor",
        "Two Fingers": "Scroll Screen",
        "Pinch": "Left Mouse Click",
        "Three Fingers": "Right Mouse Click"
    }

    for gesture, action in gestures.items():

        with st.container(border=True):

            st.subheader(gesture)

            st.write(f"Action: {action}")


# ==========================================
# 7. FACIAL RECOGNITION PAGE
# ==========================================

elif page == "Facial Recognition":

    st.title("😊 Facial Expression Recognition")

    st.write(
        "The facial expression module detects "
        "visible facial-expression categories."
    )

    expressions = [
        "Angry",
        "Disgust",
        "Fearful",
        "Happy",
        "Neutral",
        "Sad",
        "Surprised"
    ]

    col1, col2 = st.columns(2)

    for index, expression in enumerate(expressions):

        with col1 if index % 2 == 0 else col2:

            st.info(f"😊 {expression}")

    st.caption(
        "Note: Detected facial expressions do not "
        "necessarily reveal a person's actual mood."
    )


# ==========================================
# 8. MODEL PERFORMANCE PAGE
# ==========================================

elif page == "Model Performance":

    st.title("📊 Model Performance")

    st.write("Hand gesture classification model results.")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", "86.98%")

    col2.metric("Training Samples", "1440")

    col3.metric("Testing Samples", "361")

    st.divider()

    if CONFUSION_MATRIX.exists():

        st.subheader("Confusion Matrix")

        st.image(
            str(CONFUSION_MATRIX),
            caption="Gesture Recognition Confusion Matrix",
            use_container_width=True
        )

    else:

        st.warning("Confusion matrix image not found.")

    if DATASET_FILE.exists():

        st.subheader("Dataset Distribution")

        try:

            data = pd.read_csv(DATASET_FILE)

            if "label" in data.columns:

                counts = data["label"].value_counts()

                st.bar_chart(counts)

                st.write("Total Dataset Samples:", len(data))

            else:

                st.warning("Label column not found in dataset.")

        except Exception as error:

            st.error(f"Dataset reading error: {error}")

    else:

        st.warning("Dataset file not found.")


# ==========================================
# 9. ABOUT PROJECT PAGE
# ==========================================

elif page == "About Project":

    st.title("ℹ️ About SmartHand")

    st.markdown("""
    ## Project Title

    SmartHand – Virtual Mouse using Hand Gesture
    Recognition and Facial Expression Recognition.

    ## Technologies Used

    - Python
    - OpenCV
    - MediaPipe
    - NumPy
    - Pandas
    - Scikit-learn
    - Random Forest Classifier
    - PyAutoGUI
    - Streamlit
    - ONNX Models

    ## Project Objectives

    1. Develop a touchless computer interaction system.
    2. Control mouse operations using hand gestures.
    3. Recognize facial-expression categories.
    4. Integrate Machine Learning and Computer Vision.
    5. Provide an accessible and interactive interface.

    ## Applications

    - Touchless computer control
    - Human Computer Interaction
    - Accessibility support
    - Computer Vision research
    - AI/ML learning and demonstration
    """)


# ==========================================
# 10. FOOTER
# ==========================================

st.divider()

st.markdown(
    "<center>SmartHand AI Project | "
    "Hand Gesture Recognition + Facial Expression Recognition"
    "</center>",
    unsafe_allow_html=True
)
