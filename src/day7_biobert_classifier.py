import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.multiclass import unique_labels

# ===============================
# PATHS (AUTO-SAFE)
# ===============================
POSSIBLE_DATA_PATHS = [
    "data/processed/medical_text_cleaned.csv",
    "data/processed/cleaned_medical_transcriptions.csv",
    "data/cleaned_medical_transcriptions.csv",
    "cleaned_medical_transcriptions.csv"
]

EMBEDDINGS_PATH = "artifacts/biobert/biobert_embeddings.npy"
MODEL_DIR = "models"

MODEL_PATH = os.path.join(MODEL_DIR, "biobert_specialty_classifier.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "biobert_label_encoder.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# LOAD DATA (SAFE)
# ===============================
print("🔄 Loading data...")

DATA_PATH = None
for path in POSSIBLE_DATA_PATHS:
    if os.path.exists(path):
        DATA_PATH = path
        break

if DATA_PATH is None:
    raise FileNotFoundError("❌ Cleaned CSV file not found in expected locations.")

df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded data from: {DATA_PATH}")
print("📌 Available columns:", df.columns.tolist())

LABEL_COL = "medical_specialty"
y = df[LABEL_COL].astype(str)

# ===============================
# LOAD EMBEDDINGS
# ===============================
X = np.load(EMBEDDINGS_PATH)
print(f"✅ BioBERT embeddings loaded: {X.shape}")

# ===============================
# LABEL ENCODING
# ===============================
encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)
print(f"🏷 Total classes: {len(encoder.classes_)}")

# ===============================
# TRAIN-TEST SPLIT (SAFE)
# ===============================
# Ensure test set >= number of classes
test_size = max(0.2, len(np.unique(y_enc)) / len(y_enc))
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=test_size, random_state=42, stratify=None
)
print("✅ Train-test split completed")

# ===============================
# MODEL TRAINING
# ===============================
print("⚙ Training BioBERT classifier (Logistic Regression)...")

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)
print("✅ Model training completed")

# ===============================
# EVALUATION (SAFE)
# ===============================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy:.4f}\n")

# Only include classes present in y_test
labels_in_test = unique_labels(y_test)
print("📊 Classification Report:\n")
print(
    classification_report(
        y_test,
        y_pred,
        labels=labels_in_test,
        target_names=encoder.inverse_transform(labels_in_test)
    )
)

# ===============================
# SAVE MODEL
# ===============================
joblib.dump(model, MODEL_PATH)
joblib.dump(encoder, ENCODER_PATH)

print("\n💾 Model & encoder saved successfully")
print(f"📁 Model: {MODEL_PATH}")
print(f"📁 Encoder: {ENCODER_PATH}")
