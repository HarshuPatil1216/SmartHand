import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# 1. Load dataset
DATASET_PATH = "dataset/gestures.csv"

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Separate features and labels
X = df.drop("label", axis=1)
y = df["label"]

print("\nGesture distribution:")
print(y.value_counts())

# 3. Split dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 4. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

# 5. Train model
print("\nTraining Random Forest model...")
model.fit(X_train, y_train)

print("Model training completed!")

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\n-----------------------------")
print("MODEL EVALUATION")
print("-----------------------------")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# 8. Confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

print("\nConfusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

display.plot(cmap="Blues", xticks_rotation=45)
plt.title("SmartHand Gesture Classification")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
plt.show()

# 9. Save trained model
os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "feature_columns": list(X.columns)
    },
    "models/gesture_model.pkl"
)

print("\nModel saved successfully!")
print("Location: models/gesture_model.pkl")
print("\nDAY 4 MODEL TRAINING COMPLETED!")
