
import streamlit as st
import pandas as pd
import plotly.express as px
import subprocess
import sys
from pathlib import Path


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "models" / "gesture_model.pkl"
DATASET_FILE = BASE_DIR / "dataset" / "gestures.csv"
CONFUSION_MATRIX = BASE_DIR / "models" / "confusion_matrix.png"
VIRTUAL_MOUSE_FILE = BASE_DIR / "virtual_mouse.py"

st.set_page_config(
    page_title="SmartHand | AI Analytics",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #080d1a 0%, #111827 100%);
    color: #f8fafc;
}

[data-testid="stSidebar"] {
    background-color: #0b1220;
    border-right: 1px solid #263247;
}

[data-testid="stSidebar"] * {
    color: #e2e8f0;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 4px;
}

.subtitle {
    color: #94a3b8;
    font-size: 15px;
    margin-bottom: 25px;
}

.metric-card {
    background: linear-gradient(145deg, #182438, #101827);
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #293b55;
    min-height: 125px;
}

.metric-label {
    color: #94a3b8;
    font-size: 14px;
}

.metric-value {
    color: #ffffff;
    font-size: 30px;
    font-weight: 800;
    margin-top: 8px;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 15px;
    margin-bottom: 12px;
}

.info-card {
    background-color: #172338;
    border: 1px solid #293b55;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 10px;
}

div.stButton > button {
    border-radius: 10px;
    min-height: 45px;
    font-weight: 700;
}

div[data-testid="stMetric"] {
    background: #172338;
    border: 1px solid #293b55;
    border-radius: 14px;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD DATASET
# ==========================================

@st.cache_data
def load_dataset():

    if DATASET_FILE.exists():

        return pd.read_csv(DATASET_FILE)

    return pd.DataFrame()


data = load_dataset()

dataset_available = (
    not data.empty and "label" in data.columns
)

if dataset_available:

    gesture_counts = data["label"].value_counts().reset_index()

    gesture_counts.columns = ["Gesture", "Samples"]

    total_samples = len(data)

    number_of_gestures = data["label"].nunique()

else:

    gesture_counts = pd.DataFrame(
        columns=["Gesture", "Samples"]
    )

    total_samples = 0

    number_of_gestures = 0


# ==========================================
# 4. SESSION STATE
# ==========================================

if "mouse_process" not in st.session_state:

    st.session_state.mouse_process = None


def is_running():

    process = st.session_state.mouse_process

    return process is not None and process.poll() is None


# ==========================================
# 5. COMMON HEADER
# ==========================================

def page_header(title, subtitle):

    st.markdown(
        f'<div class="main-title">{title}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="subtitle">{subtitle}</div>',
        unsafe_allow_html=True
    )


def metric_card(label, value, icon):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{icon} {label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# 6. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.markdown("# 🖐️ SmartHand")

st.sidebar.caption("AI POWERED INTERACTION SYSTEM")

st.sidebar.divider()

page = st.sidebar.radio(
    "MAIN MENU",
    [
        "Dashboard",
        "Hand Gestures",
        "Facial Recognition",
        "Model Analytics",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### SYSTEM STATUS")

if is_running():

    st.sidebar.success("● Virtual Mouse Running")

else:

    st.sidebar.warning("● Virtual Mouse Stopped")

st.sidebar.caption("SmartHand AI Project | 2026")


# ==========================================
# 7. DASHBOARD
# ==========================================

if page == "Dashboard":

    page_header(
        "🖐️ SmartHand AI Dashboard",
        "Intelligent Human-Computer Interaction using Machine Learning and Computer Vision"
    )

    # METRIC CARDS

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        metric_card(
            "Model Accuracy",
            "86.98%",
            "🎯"
        )

    with c2:

        metric_card(
            "Dataset Samples",
            f"{total_samples:,}",
            "📊"
        )

    with c3:

        metric_card(
            "Hand Gestures",
            "6",
            "✋"
        )

    with c4:

        metric_card(
            "Expressions",
            "7",
            "😊"
        )

    st.write("")

    # APPLICATION CONTROL

    st.markdown(
        '<div class="section-title">🚀 Virtual Mouse Control</div>',
        unsafe_allow_html=True
    )

    control1, control2, control3 = st.columns([1, 1, 2])

    with control1:

        if st.button(
            "▶ Start Application",
            use_container_width=True
        ):

            if is_running():

                st.info("Application is already running.")

            elif not VIRTUAL_MOUSE_FILE.exists():

                st.error("virtual_mouse.py not found.")

            elif not MODEL_FILE.exists():

                st.error("Gesture model not found.")

            else:

                try:

                    st.session_state.mouse_process = subprocess.Popen(
                        [sys.executable, str(VIRTUAL_MOUSE_FILE)],
                        cwd=str(BASE_DIR)
                    )

                    st.success("Application started.")

                except Exception as error:

                    st.error(str(error))

    with control2:

        if st.button(
            "⏹ Stop Application",
            use_container_width=True
        ):

            if is_running():

                process = st.session_state.mouse_process

                process.terminate()

                try:

                    process.wait(timeout=3)

                except subprocess.TimeoutExpired:

                    process.kill()

                st.session_state.mouse_process = None

                st.success("Application stopped.")

            else:

                st.info("Application is not running.")

    with control3:

        if is_running():

            st.success("🟢 SYSTEM ONLINE")

        else:

            st.warning("🔴 SYSTEM OFFLINE")

    st.divider()

    # DATA VISUALIZATIONS

    st.markdown(
        '<div class="section-title">📈 Gesture Dataset Analytics</div>',
        unsafe_allow_html=True
    )

    if dataset_available:

        chart1, chart2 = st.columns(2)

        with chart1:

            st.markdown("#### Gesture Distribution")

            donut = px.pie(
                gesture_counts,
                names="Gesture",
                values="Samples",
                hole=0.58,
                template="plotly_dark"
            )

            donut.update_layout(
                height=360,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend_title_text="Gestures",
                margin=dict(t=25, b=15, l=15, r=15)
            )

            donut.update_traces(
                textposition="inside",
                textinfo="label+percent"
            )

            st.plotly_chart(
                donut,
                use_container_width=True
            )

        with chart2:

            st.markdown("#### Samples per Gesture")

            bar = px.bar(
                gesture_counts,
                x="Gesture",
                y="Samples",
                text="Samples",
                template="plotly_dark"
            )

            bar.update_traces(
                textposition="outside"
            )

            bar.update_layout(
                height=360,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Gesture Class",
                yaxis_title="Number of Samples",
                margin=dict(t=25, b=15, l=15, r=15)
            )

            st.plotly_chart(
                bar,
                use_container_width=True
            )

    else:

        st.warning(
            "Dataset not available. Check dataset/gestures.csv."
        )

    st.divider()

    # FEATURE SUMMARY

    st.markdown(
        '<div class="section-title">⚙️ AI Modules</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="info-card">
            <h3>✋ Hand Gesture Recognition</h3>
            <p>MediaPipe landmark detection and Random Forest classification.</p>
            <ul>
                <li>Open Palm - Activate Mouse</li>
                <li>Fist - Pause Mouse</li>
                <li>Index Point - Cursor Movement</li>
                <li>Two Fingers - Scroll</li>
                <li>Pinch - Left Click</li>
                <li>Three Fingers - Right Click</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="info-card">
            <h3>😊 Facial Expression Recognition</h3>
            <p>YuNet face detection and MobileFaceNet expression classification.</p>
            <ul>
                <li>Angry</li>
                <li>Disgust</li>
                <li>Fearful</li>
                <li>Happy</li>
                <li>Neutral</li>
                <li>Sad</li>
                <li>Surprised</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# 8. HAND GESTURES
# ==========================================

elif page == "Hand Gestures":

    page_header(
        "✋ Hand Gesture Recognition",
        "Six recognized gestures for touchless computer control"
    )

    gesture_info = {
        "Open Palm": ("🖐️", "Activate Virtual Mouse"),
        "Fist": ("✊", "Pause Virtual Mouse"),
        "Index Point": ("☝️", "Move Cursor"),
        "Two Fingers": ("✌️", "Scroll Screen"),
        "Pinch": ("🤏", "Left Mouse Click"),
        "Three Fingers": ("🤟", "Right Mouse Click")
    }

    cols = st.columns(3)

    for index, (gesture, info) in enumerate(gesture_info.items()):

        icon, action = info

        with cols[index % 3]:

            st.markdown(
                f"""
                <div class="info-card">
                    <h2>{icon}</h2>
                    <h3>{gesture}</h3>
                    <p>{action}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    if dataset_available:

        st.subheader("Gesture Training Distribution")

        st.dataframe(
            gesture_counts,
            use_container_width=True,
            hide_index=True
        )


# ==========================================
# 9. FACIAL RECOGNITION
# ==========================================

elif page == "Facial Recognition":

    page_header(
        "😊 Facial Expression Recognition",
        "Computer Vision based facial-expression classification"
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

    cols = st.columns(4)

    icons = ["😠", "🤢", "😨", "😄", "😐", "😢", "😲"]

    for index, expression in enumerate(expressions):

        with cols[index % 4]:

            st.markdown(
                f"""
                <div class="info-card">
                    <h2>{icons[index]}</h2>
                    <h4>{expression}</h4>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.info(
        "Facial expression categories do not establish a person's actual mood or internal emotional state."
    )

    st.subheader("Technology Used")

    st.write(
        "YuNet Face Detector + MobileFaceNet ONNX Model + OpenCV DNN"
    )


# ==========================================
# 10. MODEL ANALYTICS
# ==========================================

elif page == "Model Analytics":

    page_header(
        "📊 Model Analytics",
        "Explore dataset statistics and gesture classifier performance"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Recorded Accuracy",
        "86.98%"
    )

    c2.metric(
        "Training Samples",
        "1,440"
    )

    c3.metric(
        "Testing Samples",
        "361"
    )

    st.caption(
        "These evaluation values are from the previously recorded random train-test split."
    )

    st.divider()

    st.subheader("Confusion Matrix")

    if CONFUSION_MATRIX.exists():

        st.image(
            str(CONFUSION_MATRIX),
            use_container_width=True
        )

    else:

        st.warning(
            "Confusion matrix image not found. Run train_model.py first."
        )

    st.divider()

    if dataset_available:

        st.subheader("Dataset Class Distribution")

        bar = px.bar(
            gesture_counts,
            x="Gesture",
            y="Samples",
            color="Gesture",
            text="Samples",
            template="plotly_dark"
        )

        bar.update_layout(
            height=420,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            bar,
            use_container_width=True
        )

        st.subheader("Dataset Preview")

        st.dataframe(
            data.head(10),
            use_container_width=True
        )

        st.write(
            f"Total dataset records: {len(data):,}"
        )

        st.write(
            f"Total feature columns: {len(data.columns) - 1}"
        )

    else:

        st.warning("Dataset file is missing or has no label column.")


# ==========================================
# 11. ABOUT PROJECT
# ==========================================

elif page == "About Project":

    page_header(
        "ℹ️ About SmartHand",
        "AI-powered touchless Human-Computer Interaction system"
    )

    st.markdown("""
    ### Project Overview

    SmartHand is an AI-powered virtual mouse application.
    It uses hand gestures to control computer operations
    and detects visible facial-expression categories.

    ### Technologies

    - Python
    - OpenCV
    - MediaPipe
    - NumPy
    - Pandas
    - Scikit-learn
    - Random Forest
    - PyAutoGUI
    - Streamlit
    - Plotly
    - ONNX
    - YuNet
    - MobileFaceNet

    ### Project Objectives

    1. Build a touchless mouse control system.
    2. Recognize hand gestures using Machine Learning.
    3. Detect facial-expression categories.
    4. Integrate Computer Vision and AI.
    5. Build an interactive analytics dashboard.

    ### Future Scope

    - Improve gesture classification.
    - Add new gestures.
    - Improve cursor smoothing.
    - Add accessibility features.
    - Perform independent real-world model evaluation.
    """)

    st.markdown(
        "**SmartHand – Making Computer Interaction Touchless with AI.**"
    )


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#94a3b8; padding:15px;">
        🖐️ <b>SmartHand AI</b><br>
        Hand Gesture Recognition + Facial Expression Recognition<br>
        Built with Python, Machine Learning and Computer Vision
    </div>
    """,
    unsafe_allow_html=True
)
