import os
import pandas as pd
import torch
from torch.optim import AdamW
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# BASE PATH — Single source of truth
# ============================================================
BASE_PATH          = r"C:\Users\Lenovo\Downloads\asp\asp"
TRAIN_CSV          = f"{BASE_PATH}/Laptop_Train_v2.csv"
MODEL_SAVE_PATH    = f"{BASE_PATH}/models/bert_explicit_absa"
OVERALL_SAVE_PATH  = f"{BASE_PATH}/models/overall_sentiment_model"

# ============================================================
# SETTINGS
# ============================================================
BERT_MODEL_NAME = "bert-base-uncased"
MAX_LEN         = 128
BATCH_SIZE      = 16
EPOCHS          = 4
LEARNING_RATE   = 2e-5
RANDOM_SEED     = 42

# Polarity → numeric label
LABEL_MAP = {
    "negative": 0,
    "neutral":  1,
    "positive": 2,
    "conflict": 1   # treat conflict as neutral
}

# ============================================================
# DEVICE
# ============================================================
device = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available()
                       else "cpu")
print(f"Using device: {device}")

# ============================================================
# DATASET CLASS
# ============================================================
class ABSADataset(Dataset):
    """
    Each sample = (sentence, aspect_term) → polarity label
    Tokenized as: [CLS] sentence [SEP] aspect_term [SEP]
    """
    def __init__(self, sentences, aspects, labels, tokenizer, max_len):
        self.sentences  = sentences
        self.aspects    = aspects
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_len    = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.sentences[idx],
            self.aspects[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "token_type_ids": encoding["token_type_ids"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long)
        }


class OverallSentimentDataset(Dataset):
    """
    For overall sentiment model — uses full sentence only (no aspect).
    """
    def __init__(self, sentences, labels, tokenizer, max_len):
        self.sentences = sentences
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.sentences[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "token_type_ids": encoding["token_type_ids"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ============================================================
# LOAD & CLEAN DATA
# ============================================================
def load_data(csv_path: str):
    print(f"\n📂 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"   Raw rows: {len(df)}")
    print(f"   Columns : {list(df.columns)}")

    # Standardise column names (lowercase, strip spaces)
    df.columns = df.columns.str.lower().str.strip()

    # Rename if needed
    rename_map = {}
    if "sentence"     not in df.columns and "text"    in df.columns:
        rename_map["text"]    = "sentence"
    if "aspect term"  not in df.columns and "aspect"  in df.columns:
        rename_map["aspect"]  = "aspect term"
    if "polarity"     not in df.columns and "sentiment" in df.columns:
        rename_map["sentiment"] = "polarity"

    df.rename(columns=rename_map, inplace=True)

    # Keep only needed columns
    required = ["sentence", "aspect term", "polarity"]
    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in CSV. "
                f"Found: {list(df.columns)}"
            )

    df = df[required].copy()

    # Drop nulls
    df.dropna(subset=required, inplace=True)

    # Normalise polarity labels
    df["polarity"] = df["polarity"].str.lower().str.strip()
    df = df[df["polarity"].isin(LABEL_MAP.keys())]
    df["label"] = df["polarity"].map(LABEL_MAP)

    print(f"   Clean rows: {len(df)}")
    print(f"   Label dist:\n{df['polarity'].value_counts().to_string()}")

    return df


# ============================================================
# TRAINING LOOP
# ============================================================
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss   = 0
    correct      = 0
    total        = 0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds       = torch.argmax(outputs.logits, dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total * 100
    return avg_loss, accuracy


def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    correct    = 0
    total      = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )

            total_loss += outputs.loss.item()
            preds       = torch.argmax(outputs.logits, dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = correct / total * 100
    return avg_loss, accuracy, all_preds, all_labels


# ============================================================
# TRAIN & SAVE — ABSA MODEL
# ============================================================
def train_absa_model(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  TRAINING ABSA MODEL (sentence + aspect → polarity)")
    print("="*60)

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model     = BertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=3
    ).to(device)

    # Split
    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=RANDOM_SEED,
        stratify=df["label"]
    )
    print(f"   Train: {len(train_df)} | Val: {len(val_df)}")

    train_dataset = ABSADataset(
        train_df["sentence"].tolist(),
        train_df["aspect term"].tolist(),
        train_df["label"].tolist(),
        tokenizer, MAX_LEN
    )
    val_dataset = ABSADataset(
        val_df["sentence"].tolist(),
        val_df["aspect term"].tolist(),
        val_df["label"].tolist(),
        tokenizer, MAX_LEN
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        val_loss, val_acc, preds, labels_true = eval_epoch(
            model, val_loader, device
        )

        print(f"\n  Epoch {epoch}/{EPOCHS}")
        print(f"    Train  Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"    Val    Loss: {val_loss:.4f}   | Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best model
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            model.save_pretrained(MODEL_SAVE_PATH)
            tokenizer.save_pretrained(MODEL_SAVE_PATH)
            print(f"    ✅ Best model saved → {MODEL_SAVE_PATH}")

    # Final classification report
    label_names = ["negative", "neutral", "positive"]
    print("\n  📊 Classification Report (Validation):")
    print(classification_report(labels_true, preds, target_names=label_names))

    print(f"\n  ✅ ABSA model training complete. Best Val Acc: {best_val_acc:.2f}%")


# ============================================================
# TRAIN & SAVE — OVERALL SENTIMENT MODEL
# ============================================================
def train_overall_model(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  TRAINING OVERALL SENTIMENT MODEL (sentence → polarity)")
    print("="*60)

    # Deduplicate sentences — take majority polarity per sentence
    sentence_labels = (
        df.groupby("sentence")["label"]
        .agg(lambda x: x.mode()[0])
        .reset_index()
    )
    print(f"   Unique sentences: {len(sentence_labels)}")

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model     = BertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=3
    ).to(device)

    train_df, val_df = train_test_split(
        sentence_labels, test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=sentence_labels["label"]
    )
    print(f"   Train: {len(train_df)} | Val: {len(val_df)}")

    train_dataset = OverallSentimentDataset(
        train_df["sentence"].tolist(),
        train_df["label"].tolist(),
        tokenizer, MAX_LEN
    )
    val_dataset = OverallSentimentDataset(
        val_df["sentence"].tolist(),
        val_df["label"].tolist(),
        tokenizer, MAX_LEN
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        val_loss, val_acc, preds, labels_true = eval_epoch(
            model, val_loader, device
        )

        print(f"\n  Epoch {epoch}/{EPOCHS}")
        print(f"    Train  Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"    Val    Loss: {val_loss:.4f}   | Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(OVERALL_SAVE_PATH, exist_ok=True)
            model.save_pretrained(OVERALL_SAVE_PATH)
            tokenizer.save_pretrained(OVERALL_SAVE_PATH)
            print(f"    ✅ Best model saved → {OVERALL_SAVE_PATH}")

    label_names = ["negative", "neutral", "positive"]
    print("\n  📊 Classification Report (Validation):")
    print(classification_report(labels_true, preds, target_names=label_names))

    print(f"\n  ✅ Overall model training complete. Best Val Acc: {best_val_acc:.2f}%")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    print("\n🚀 Starting ABSA Training Pipeline")
    print(f"   Base path : {BASE_PATH}")
    print(f"   Train CSV : {TRAIN_CSV}")

    # Load data
    df = load_data(TRAIN_CSV)

    # Train ABSA model (sentence + aspect → polarity)
    train_absa_model(df)

    # Train overall sentiment model (sentence → polarity)
    train_overall_model(df)

    print("\n" + "="*60)
    print("  🎉 ALL MODELS TRAINED SUCCESSFULLY!")
    print(f"  ABSA Model    → {MODEL_SAVE_PATH}")
    print(f"  Overall Model → {OVERALL_SAVE_PATH}")
    print("="*60)
