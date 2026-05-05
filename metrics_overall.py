import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("Laptop_Train_v2.csv")

df = df[["Sentence", "polarity"]]
df = df.dropna(subset=["Sentence"])
df["Sentence"] = df["Sentence"].astype(str)
label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
    "conflict": 1
}

df["polarity"] = df["polarity"].str.lower().str.strip()
df["label"] = df["polarity"].map(label_map)
df["label"] = df["label"].astype(int)

# -----------------------------
# Train/Test Split
# -----------------------------

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Sentence"].tolist(),
    df["label"].tolist(),
    test_size=0.15,
    random_state=42
)

# -----------------------------
# Load Tokenizer + Model
# -----------------------------

tokenizer = BertTokenizer.from_pretrained("models/overall_sentiment_model")

model = BertForSequenceClassification.from_pretrained(
    "models/overall_sentiment_model"
)

# -----------------------------
# Dataset Class
# -----------------------------

class ReviewDataset(Dataset):

    def __init__(self, texts, labels):

        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=96
        )

        self.labels = labels

    def __getitem__(self, idx):

        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.labels)


val_dataset = ReviewDataset(val_texts, val_labels)

# -----------------------------
# Predict
# -----------------------------

trainer = Trainer(model=model)

predictions = trainer.predict(val_dataset)

y_pred = predictions.predictions.argmax(axis=1)
y_true = predictions.label_ids

# -----------------------------
# Metrics
# -----------------------------

print("\nAccuracy:", accuracy_score(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))