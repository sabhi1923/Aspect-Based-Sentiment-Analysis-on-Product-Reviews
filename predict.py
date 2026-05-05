# /Users/hdass2/Downloads/asp/predict.py

import os
import re
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from rapidfuzz import fuzz

# ============================================================
# BASE PATH — All files live here
# ============================================================
BASE_PATH          = r"C:\Users\Lenovo\Downloads\asp\asp"
MODEL_PATH         = f"{BASE_PATH}/models/bert_explicit_absa"
OVERALL_MODEL_PATH = f"{BASE_PATH}/models/overall_sentiment_model"
TRAIN_CSV          = f"{BASE_PATH}/Laptop_Train_v2.csv"



# ============================================================
# ASPECT CATEGORY MAP
# ============================================================
ASPECT_CATEGORY_MAP = {
    # Camera
    "camera":           "Camera",
    "cam":              "Camera",
    "built-in camera":  "Camera",
    "built in camera":  "Camera",
    "webcam":           "Camera",
    "photo quality":    "Camera",
    "video quality":    "Camera",
    "image":            "Camera",
    "picture":          "Camera",
    "photo":            "Camera",

    # Battery
    "battery":          "Battery",
    "battery life":     "Battery",
    "battery timer":    "Battery",
    "charge":           "Battery",
    "charging":         "Battery",
    "charged":          "Battery",

    # Performance
    "performance":      "Performance",
    "speed":            "Performance",
    "processor":        "Performance",
    "cpu":              "Performance",
    "ram":              "Performance",
    "memory":           "Performance",
    "processing power": "Performance",
    "runs":             "Performance",
    "boot up":          "Performance",
    "startup":          "Performance",
    "gaming":           "Performance",
    "graphics":         "Performance",
    "graphics card":    "Performance",
    "gpu":              "Performance",

    # Display
    "display":              "Display",
    "screen":               "Display",
    "resolution":           "Display",
    "brightness":           "Display",
    "screen resolution":    "Display",
    "led backlit display":  "Display",

    # Keyboard
    "keyboard":         "Keyboard",
    "keys":             "Keyboard",
    "backlit keyboard": "Keyboard",
    "key broad":        "Keyboard",
    "key pad":          "Keyboard",

    # Build Quality
    "build quality":    "Build Quality",
    "design":           "Build Quality",
    "build":            "Build Quality",
    "body":             "Build Quality",
    "frame":            "Build Quality",
    "material":         "Build Quality",
    "weight":           "Build Quality",
    "size":             "Build Quality",
    "portability":      "Build Quality",

    "case":             "case",
    "cover":            "case",
    "protective case":  "case",

    # Software / OS
    "operating system": "Software",
    "software":         "Software",
    "os":               "Software",
    "windows":          "Software",
    "mac os":           "Software",
    "osx":              "Software",
    "applications":     "Software",
    "programs":         "Software",

    # Touchpad
    "touchpad":         "Touchpad",
    "trackpad":         "Touchpad",
    "touch pad":        "Touchpad",
    "touch-pad":        "Touchpad",
    "mousepad":         "Touchpad",

    # Audio
    "speakers":         "Audio",
    "sound":            "Audio",
    "audio":            "Audio",
    "microphone":       "Audio",
    "volume":           "Audio",

    # Connectivity
    "wifi":             "Connectivity",
    "wireless":         "Connectivity",
    "bluetooth":        "Connectivity",
    "usb":              "Connectivity",
    "usb ports":        "Connectivity",
    "ethernet":         "Connectivity",
    "hdmi port":        "Connectivity",

    # Storage
    "storage":          "Storage",
    "hard drive":       "Storage",
    "ssd":              "Storage",
    "hard disk":        "Storage",
    "hdd":              "Storage",
    "hard drive space": "Storage",

    # Price / Value
    "price":            "Price/Value",
    "value":            "Price/Value",
    "cost":             "Price/Value",
    "price tag":        "Price/Value",

    # Support / Warranty
    "warranty":         "Support/Warranty",
    "customer service": "Support/Warranty",
    "tech support":     "Support/Warranty",
    "support":          "Support/Warranty",
    "service":          "Support/Warranty",

    # Cooling
    "fan":              "Cooling",
    "cooling system":   "Cooling",
    "cooling pad":      "Cooling",
    "heat":             "Cooling",

    # Ports
    "ports":            "Ports",
    "usb port":         "Ports",
    "hdmi":             "Ports",
    "vga port":         "Ports",
}

# ============================================================
# KEYWORD TRIGGERS FOR EACH CATEGORY
# ============================================================
CATEGORY_KEYWORDS = {
    "Camera": [
        "camera", "cam", "photo", "picture", "image",
        "webcam", "selfie", "lens", "megapixel", "shoot",
        "photography", "video quality", "photo quality"
    ],
    "Battery": [
        "battery", "charge", "charging", "drain", "drains",
        "battery life", "hours", "backup", "unplugged", "power"
    ],
    "Performance": [
        "performance", "speed", "fast", "slow", "processor",
        "cpu", "ram", "memory", "lag", "freeze", "hang",
        "boot", "startup", "gaming", "game", "processing",
        "graphics", "gpu", "render", "runs", "running",
        "performs"
    ],
    "Display": [
        "display", "screen", "resolution", "brightness",
        "pixel", "hd", "4k", "retina", "panel", "led",
        "backlight", "glare", "color", "vivid", "sharp",
    ],
    "Keyboard": [
        "keyboard", "keys", "key", "typing", "keypad",
        "backlit keyboard", "key broad", "type", "keypress"
    ],
    "Build Quality": [
        "build", "design", "body", "frame", "material",
        "weight", "heavy", "light", "portable", "slim",
        "thin", "size", "durability", "sturdy", "solid"
    ],
    "Case": [
        "case", "cover", "protective case"
    ],
    "Software": [
        "software", "os", "operating system", "windows",
        "macos", "app", "application", "program", "bloat",
        "driver", "update", "crash", "freeze", "install"
    ],
    "Touchpad": [
        "touchpad", "trackpad", "touch pad", "mousepad",
        "cursor", "pointer", "scroll", "gesture", "click pad"
    ],
    "Audio": [
        "speaker", "sound", "audio", "music", "volume",
        "headphone", "bass", "microphone", "mic", "noise"
    ],
    "Connectivity": [
        "wifi", "wireless", "bluetooth", "usb", "ethernet",
        "hdmi", "network", "signal", "connect", "internet"
    ],
    "Storage": [
        "storage", "hard drive", "hdd", "ssd", "disk",
        "space", "memory card", "gb", "tb", "capacity"
    ],
    "Price/Value": [
        "price", "cost", "value", "worth", "expensive",
        "cheap", "affordable", "budget", "money", "pricing"
    ],
    "Support/Warranty": [
        "warranty", "support", "customer service", "repair",
        "service", "tech support", "helpdesk", "return"
    ],
    "Cooling": [
        "fan", "cooling", "heat", "hot", "overheat",
        "temperature", "thermal", "ventilation", "warm"
    ],
    "Ports": [
        "port", "hdmi", "usb", "vga", "thunderbolt",
        "jack", "slot", "connector", "io", "interface"
    ],
}

# ============================================================
# LABEL MAPPING
# ============================================================
id2label = {0: "negative", 1: "neutral", 2: "positive"}

# ============================================================
# GLOBAL MODEL VARIABLES
# Loaded lazily — only when predict_review() is first called
# ============================================================
_model            = None
_tokenizer        = None
_overall_model    = None
_overall_tokenizer = None


def _check_model_exists(path: str, name: str):
    """Raise a clear error if model folder is missing."""
    config_file = os.path.join(path, "config.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"\n❌ {name} not found at: {path}\n"
            f"   Please run  'python train.py'  first to train and save the model.\n"
        )


def _load_models():
    """
    Load both models into global variables.
    Called once on first prediction — not at import time.
    """
    global _model, _tokenizer, _overall_model, _overall_tokenizer

    if _model is not None:
        return  # already loaded

    # ---- Check folders exist ---- #
    _check_model_exists(MODEL_PATH,         "ABSA model")
    _check_model_exists(OVERALL_MODEL_PATH, "Overall sentiment model")

    print(f"⏳ Loading ABSA model from        : {MODEL_PATH}")
    _model     = BertForSequenceClassification.from_pretrained(
        MODEL_PATH, local_files_only=True
    )
    _tokenizer = BertTokenizer.from_pretrained(
        MODEL_PATH, local_files_only=True
    )
    _model.eval()
    print("✅ ABSA model loaded")

    print(f"⏳ Loading Overall model from      : {OVERALL_MODEL_PATH}")
    _overall_model     = BertForSequenceClassification.from_pretrained(
        OVERALL_MODEL_PATH, local_files_only=True
    )
    _overall_tokenizer = BertTokenizer.from_pretrained(
        OVERALL_MODEL_PATH, local_files_only=True
    )
    _overall_model.eval()
    print("✅ Overall sentiment model loaded\n")


# ============================================================
# UTILITIES
# ============================================================

def normalize_text(text: str) -> str:
    """Lowercase and strip extra spaces."""
    return text.lower().strip()


def word_boundary_match(keyword: str, text: str) -> bool:
    """
    Exact whole-word match.
    Prevents 'play' matching inside 'display'.
    """
    pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
    return bool(re.search(pattern, text.lower()))


def fuzzy_keyword_match(keyword: str, text: str, threshold: int = 82) -> bool:
    """
    Handles typos like 'camra' → 'camera', 'battry' → 'battery'.
    1. Tries exact word-boundary match first (fast).
    2. Falls back to fuzzy ngram match (handles typos).
    """
    keyword_lower = keyword.lower()

    # Fast path — exact match
    if word_boundary_match(keyword_lower, text):
        return True

    # Fuzzy path — build ngrams same length as keyword
    words  = text.lower().split()
    n      = len(keyword_lower.split())
    ngrams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]

    for ngram in ngrams:
        if fuzz.ratio(keyword_lower, ngram) >= threshold:
            return True

    return False


def get_sentiment_from_model(review: str, aspect: str) -> dict:
    """
    Run BERT ABSA model for a (review, aspect) pair.
    Returns {negative, neutral, positive} as percentages.
    """
    inputs = _tokenizer(
        review,
        aspect,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    with torch.no_grad():
        outputs = _model(**inputs)

    probs = F.softmax(outputs.logits, dim=1).squeeze()
    return {
        "negative": round(probs[0].item() * 100, 2),
        "neutral":  round(probs[1].item() * 100, 2),
        "positive": round(probs[2].item() * 100, 2),
    }


def get_dominant_sentiment(scores: dict) -> tuple:
    """Return (label, confidence%) for the highest scoring sentiment."""
    label = max(scores, key=scores.get)
    return label, scores[label]


def aggregate_category_scores(all_scores: list) -> dict:
    """Average multiple aspect model outputs for the same category."""
    if not all_scores:
        return {"negative": 0.0, "neutral": 0.0, "positive": 0.0}

    n = len(all_scores)
    return {
        "negative": round(sum(s["negative"] for s in all_scores) / n, 2),
        "neutral":  round(sum(s["neutral"]  for s in all_scores) / n, 2),
        "positive": round(sum(s["positive"] for s in all_scores) / n, 2),
    }


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict_review(review_text: str) -> tuple:
    """
    Given a review text returns:
      - category_results : {category: {negative, neutral, positive}}
      - overall_result   : {negative, neutral, positive}

    Steps:
      1. Load models (only on first call)
      2. Fuzzy-match keywords per category against review text
      3. Run ABSA BERT model per matched keyword
      4. Average scores within same category
      5. Overall = average across all detected categories
      6. Fallback to overall-sentiment model if no aspects found
    """

    # ---- Step 1: Load models (lazy — only once) ---- #
    _load_models()

    # ---- Step 2: Detect categories present in review ---- #
    category_scores: dict = {}

    for category, keywords in CATEGORY_KEYWORDS.items():

        matched_keywords = [
            kw for kw in keywords
            if fuzzy_keyword_match(kw, review_text, threshold=82)
        ]

        if not matched_keywords:
            continue

        # Prefer longer (more specific) keywords first
        # e.g. "battery life" preferred over "battery"
        matched_keywords = sorted(matched_keywords, key=len, reverse=True)

        # Deduplicate by first word to avoid redundant calls
        seen_roots      = set()
        unique_keywords = []
        for kw in matched_keywords:
            root = kw.split()[0]
            if root not in seen_roots:
                unique_keywords.append(kw)
                seen_roots.add(root)

        # ---- Step 3: Run ABSA model ---- #
        scores_for_category = []
        for kw in unique_keywords[:2]:   # max 2 per category
            scores = get_sentiment_from_model(review_text, kw)
            scores_for_category.append(scores)

        # ---- Step 4: Average within category ---- #
        category_scores[category] = aggregate_category_scores(
            scores_for_category
        )

    # ---- Step 5: Compute overall sentiment ---- #
    if category_scores:
        n = len(category_scores)
        overall_result = {
            "negative": round(
                sum(v["negative"] for v in category_scores.values()) / n, 2
            ),
            "neutral": round(
                sum(v["neutral"]  for v in category_scores.values()) / n, 2
            ),
            "positive": round(
                sum(v["positive"] for v in category_scores.values()) / n, 2
            ),
        }

    else:
        # ---- Step 6: Fallback to overall sentiment model ---- #
        inputs = _overall_tokenizer(
            review_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        with torch.no_grad():
            outputs = _overall_model(**inputs)

        probs = F.softmax(outputs.logits, dim=1).squeeze()
        overall_result = {
            "negative": round(probs[0].item() * 100, 2),
            "neutral":  round(probs[1].item() * 100, 2),
            "positive": round(probs[2].item() * 100, 2),
        }

    return category_scores, overall_result
