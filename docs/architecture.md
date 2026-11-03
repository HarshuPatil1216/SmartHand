# \# 🏗️ SmartHand – System Architecture

# 

# \## 1. Project Overview

# 

# SmartHand is an AI-powered virtual mouse system that uses hand gesture recognition and facial expression recognition to provide touch-free computer interaction.

# 

# The system captures real-time webcam input, detects hand landmarks, classifies gestures using a trained Machine Learning model, and performs mouse operations.

# 

# It also detects faces and recognizes visible facial-expression categories.

# 

# \## 2. System Architecture

# 

# ```text

# &#x20;             Webcam Input

# &#x20;                   |

# &#x20;                   v

# &#x20;            OpenCV Video Capture

# &#x20;                   |

# &#x20;         +---------+----------+

# &#x20;         |                    |

# &#x20;         v                    v

# &#x20;   Hand Detection        Face Detection

# &#x20;    (MediaPipe)             (YuNet)

# &#x20;         |                    |

# &#x20;         v                    v

# &#x20;  21 Hand Landmarks     Face Region Detection

# &#x20;         |                    |

# &#x20;         v                    v

# &#x20;  Feature Extraction    Facial Expression

# &#x20;    (63 Features)        Recognition Model

# &#x20;         |                    |

# &#x20;         v                    v

# &#x20;  Random Forest         Expression Category

# &#x20;  Classification          (7 Classes)

# &#x20;         |

# &#x20;         v

# &#x20;  Gesture Prediction

# &#x20;         |

# &#x20;         v

# &#x20;   Gesture Smoothing

# &#x20;         |

# &#x20;         v

# &#x20;   PyAutoGUI Actions

# &#x20;         |

# &#x20;         v

# &#x20;   Virtual Mouse Control

# 

# &#x20;         Streamlit Dashboard

# &#x20;         Model Visualization

# ```

# 

# \## 3. Hand Gesture Recognition

# 

# SmartHand uses MediaPipe Hand Landmarker to detect 21 hand landmarks.

# 

# Each landmark contains three coordinates:

# 

# \- X – Horizontal position

# \- Y – Vertical position

# \- Z – Depth information

# 

# The system extracts 63 features from the 21 landmarks.

# 

# These features are normalized relative to the wrist landmark to make gesture recognition less dependent on hand position and size.

# 

# \## 4. Machine Learning Model

# 

# The project uses a Random Forest Classifier for gesture classification.

# 

# \*\*Model Configuration:\*\*

# 

# | Parameter | Value |

# |---|---|

# | Algorithm | Random Forest Classifier |

# | Number of Estimators | 300 |

# | Input Features | 63 |

# | Gesture Classes | 6 |

# | Train-Test Split | 80:20 |

# | Class Weight | Balanced |

# | Test Accuracy | 86.98% |

# 

# The model predicts one of six supported hand gestures.

# 

# \## 5. Gesture-to-Mouse Mapping

# 

# | Gesture | Mouse Operation |

# |---|---|

# | Open Palm | Activate virtual mouse |

# | Fist | Pause virtual mouse |

# | Index Point | Move cursor |

# | Two Fingers | Scroll |

# | Pinch | Left click |

# | Three Fingers | Right click |

# 

# PyAutoGUI performs the corresponding mouse operations based on the recognized gesture.

# 

# \## 6. Facial Expression Recognition

# 

# The facial expression recognition module uses:

# 

# \- \*\*YuNet:\*\* Face detection.

# \- \*\*MobileFaceNet-based ONNX model:\*\* Facial expression classification.

# \- \*\*OpenCV DNN:\*\* Model inference.

# 

# The system recognizes seven visible facial-expression categories:

# 

# 1\. Angry

# 2\. Disgust

# 3\. Fearful

# 4\. Happy

# 5\. Neutral

# 6\. Sad

# 7\. Surprised

# 

# These are visual expression classifications and should not be interpreted as definitive measurements of internal emotions.

# 

# \## 7. Gesture Smoothing and Stability

# 

# To reduce sudden changes in predictions, SmartHand uses a short prediction history.

# 

# \- Gesture prediction history: 5 frames.

# \- Confidence threshold: 0.55.

# \- Cursor smoothing and dead-zone handling.

# \- Facial expression history smoothing: 7 predictions.

# 

# These techniques help improve the stability of real-time interaction.

# 

# \## 8. Dashboard and Visualization

# 

# The Streamlit dashboard provides a visual interface for exploring the project.

# 

# It includes model-related information and visualizations such as:

# 

# \- Gesture model performance.

# \- Confusion matrix.

# \- Dataset information.

# \- Model evaluation results.

# 

# Plotly and Matplotlib are used for visualizations.

# 

# \## 9. Technologies Used

# 

# | Technology | Purpose |

# |---|---|

# | Python | Main programming language |

# | OpenCV | Webcam and image processing |

# | MediaPipe | Hand landmark detection |

# | NumPy | Numerical computation |

# | Pandas | Dataset processing |

# | Scikit-learn | Machine Learning |

# | PyAutoGUI | Mouse automation |

# | ONNX | Facial expression model |

# | Streamlit | Dashboard |

# | Plotly | Interactive visualizations |

# | Matplotlib | Model evaluation plots |

# 

# \## 10. Model Evaluation

# 

# The Random Forest gesture classifier achieved \*\*86.98% accuracy\*\* on a randomly selected, stratified held-out test set containing 361 samples.

# 

# The confusion matrix is used to analyze correct and incorrect predictions across the six gesture classes.

# 

# \*\*Evaluation limitation:\*\* Since the test split is sample-based, it may not fully represent performance across completely independent recording sessions or different users.

# 

# \## 11. Limitations

# 

# \- Performance can vary with lighting conditions.

# \- Hand visibility and camera position affect detection.

# \- Gesture recognition may vary across users and recording sessions.

# \- Unintended gestures can sometimes trigger mouse operations.

# \- Facial expression classification may be affected by lighting, face angle, and occlusion.

# \- Facial expression categories do not establish a person's actual emotional state.

# 

# \## 12. Future Enhancements

# 

# \- Improve accuracy using more diverse gesture datasets.

# \- Add customizable gesture controls.

# \- Improve cursor movement stability.

# \- Add voice command integration.

# \- Enhance accessibility features.

# \- Support additional gestures and interaction modes.

# 

# \---

# 

# \*\*SmartHand – AI-Powered Touch-Free Human-Computer Interaction\*\*

# 

# Developed using Python, Machine Learning and Computer Vision.

