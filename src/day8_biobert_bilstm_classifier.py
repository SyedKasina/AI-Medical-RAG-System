# src/day8_biobert_bilstm_classifier.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "data/cleaned_medical_transcriptions.csv"
EMBED_PATH = "artifacts/biobert/biobert_embeddings.npy"
MODEL_DIR = "models"
EPOCHS = 8
BATCH_SIZE = 16  # smaller batch for tiny dataset
LR = 1e-3
HIDDEN_DIM = 128

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
print("🔄 Loading data...")
df = pd.read_csv(DATA_PATH)
X_embed = np.load(EMBED_PATH)
print(f"✅ BioBERT embeddings: {X_embed.shape}")

labels = df["medical_specialty"].astype(str)

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(labels)
num_classes = len(encoder.classes_)
print(f"🏷 Total classes: {num_classes}")

# -----------------------------
# TRAIN-TEST SPLIT (SAFE)
# -----------------------------
n_samples = X_embed.shape[0]
n_classes = num_classes

# Calculate safe test size
test_size = max(0.2, n_classes / n_samples)
if n_samples * test_size < n_classes:
    test_size = n_classes / n_samples  # ensure at least 1 sample per class in test

# Attempt stratified split; fallback if impossible
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X_embed, y, test_size=test_size, random_state=42, stratify=y
    )
except ValueError:
    print("⚠ Dataset too small for stratified split. Using random split without stratify.")
    X_train, X_test, y_train, y_test = train_test_split(
        X_embed, y, test_size=test_size, random_state=42, stratify=None
    )

print(f"✅ Train-test split completed: Train={len(X_train)}, Test={len(X_test)}")

# -----------------------------
# CONVERT TO TENSORS
# -----------------------------
# LSTM expects (batch, seq_len, embed_dim). Use seq_len=1 for embeddings
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)

# -----------------------------
# MODEL
# -----------------------------
class BioBERT_BiLSTM(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_forward = h_n[0]
        h_backward = h_n[1]
        h = torch.cat((h_forward, h_backward), dim=1)
        return self.fc(h)

model = BioBERT_BiLSTM(embed_dim=768, hidden_dim=HIDDEN_DIM, num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# TRAINING
# -----------------------------
print("🚀 Training Bi-LSTM model...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

# -----------------------------
# EVALUATION
# -----------------------------
model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        outputs = model(xb)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.numpy())
        all_true.extend(yb.numpy())

acc = accuracy_score(all_true, all_preds)
print(f"\n🎯 Accuracy: {acc:.4f}\n")
print("📊 Classification Report:\n")
print(classification_report(all_true, all_preds, target_names=encoder.classes_))

# -----------------------------
# SAVE MODEL & ENCODER
# -----------------------------
torch.save(model.state_dict(), f"{MODEL_DIR}/biobert_bilstm_model.pt")
joblib.dump(encoder, f"{MODEL_DIR}/biobert_bilstm_label_encoder.pkl")

print("\n✅ Bi-LSTM model & encoder saved successfully")
