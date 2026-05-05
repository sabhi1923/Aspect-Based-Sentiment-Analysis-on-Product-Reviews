import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("data/Laptop_Train_v2.csv")

# Use ALL required columns for ABSA
df = df[["Sentence", "Aspect Term", "polarity"]]
df = df.dropna()

df["Sentence"] = df["Sentence"].astype(str)
df["Aspect Term"] = df["Aspect Term"].astype(str)

# Remove conflict (optional but recommended)
df["polarity"] = df["polarity"].str.lower().str.strip()
df = df[df["polarity"] != "conflict"]

# Label mapping
label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

df["label"] = df["polarity"].map(label_map)
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
train_s, val_s, train_a, val_a, train_l, val_l = train_test_split(
    df["Sentence"].tolist(),
    df["Aspect Term"].tolist(),
    df["label"].tolist(),
    test_size=0.15,
    random_state=42,
    stratify=df["label"]
)

# -----------------------------
# LOAD MODEL + TOKENIZER
# -----------------------------
MODEL_PATH = "models/bert_explicit_absa"

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

model = BertForSequenceClassification.from_pretrained(
    MODEL_PATH
)

# -----------------------------
# DATASET CLASS (ABSA)
# -----------------------------
class ABSADataset(Dataset):

    def __init__(self, sentences, aspects, labels):
        self.encodings = tokenizer(
            sentences,
            aspects,   # 🔥 IMPORTANT (sentence + aspect)
            truncation=True,
            padding=True,
            max_length=128
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# -----------------------------
# CREATE DATASET
# -----------------------------
val_dataset = ABSADataset(val_s, val_a, val_l)

# -----------------------------
# PREDICTION
# -----------------------------
trainer = Trainer(model=model)

predictions = trainer.predict(val_dataset)

y_pred = predictions.predictions.argmax(axis=1)
y_true = predictions.label_ids

# -----------------------------
# METRICS
# -----------------------------
print("\n✅ ABSA MODEL EVALUATION RESULTS")
print("="*50)

print("\nAccuracy:")
print(accuracy_score(y_true, y_pred))

print("\nClassification Report (Precision, Recall, F1-score):")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Negative", "Neutral", "Positive"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))