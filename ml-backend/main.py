from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import io
import joblib
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import urlparse
import pandas as pd
import tldextract
import re
import requests
import os
import torch
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
from io import BytesIO


app = FastAPI()

# ----------------------------
# Paths + URLs
# ----------------------------
PT_MODEL_URL = "https://github.com/Aishusrinath/fake-job-detector/releases/download/v1.0/email_resnet18_best.pt"
PT_MODEL_PATH = "email_resnet18_best.pt"

NEWS_API_KEY = "66a21a149d374c229abc8dfec6dd54a3"


DEVICE = torch.device("cpu")
CLASS_NAMES = ["scam", "legit"]   # order from training
# ----------------------------
# Download .pt model
def download_model():
    """Downloads .pt file from GitHub Releases if not present."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("Model downloaded.")
    else:
        print("Model already exists.")

# IMAGE TRANSFORMS
# ================================
eval_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

# ----------------------------
# Load PyTorch Image Model
# ----------------------------
def load_image_model():
    try:
        model = models.resnet18(weights=None)
    except:
        model = models.resnet18(pretrained=False)

    in_features = model.fc.in_features

    # MUST MATCH TRAINING EXACTLY:
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, len(CLASS_NAMES))
    )

    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=True)

    model.eval()
    model.to(DEVICE)

    print("✔ Model loaded correctly.")
    return model


image_model = None



# ----------------------------
# STARTUP EVENT
# ----------------------------
@app.on_event("startup")
def startup_event():
    global image_model, job_model, job_vectorizer, url_model

    download_model()
    image_model = load_image_model()

    job_model = joblib.load("fake_job_model.pkl")
    job_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    url_model = joblib.load("phishing_model.pkl")

# ================================
# PREDICT FUNCTION
# ================================
def run_prediction(img: Image.Image):
    img = img.convert("RGB")
    x = eval_tfms(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = image_model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy().ravel()

    idx = probs.argmax()
    return CLASS_NAMES[idx], float(probs[idx]), {
        c: float(p) for c, p in zip(CLASS_NAMES, probs)
    }
# ----------------------------
# IMAGE PREDICTION ENDPOINT
# ----------------------------
@app.post("/predict-image")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    img = Image.open(io.BytesIO(content))

    label, confidence, distribution = run_prediction(img)

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": distribution
    }


# ----------------------------
# URL + JOB PREDICTORS (Your code unchanged)
# ----------------------------
class JobRequest(BaseModel):
    text: str


class URLRequest(BaseModel):
    url: str


@app.get("/")
def home():
    return {"message": "API Running"}


@app.get("/news")
def get_news():
    url = f"https://newsapi.org/v2/everything?q=phishing+Canada&language=en&pageSize=5&apiKey={NEWS_API_KEY}"
    resp = requests.get(url)
    data = resp.json()
    return {"articles": data.get("articles", [])}


@app.post("/predict_job")
def predict(data: JobRequest):
    X = job_vectorizer.transform([data.text])
    pred = job_model.predict(X)[0]
    return {"prediction": "FAKE" if pred == 1 else "REAL"}


def extract_features(url):
    features = {}
    parsed = urlparse(url)
    ext = tldextract.extract(url)

    features["url_length"] = len(url)
    features["num_dots"] = url.count(".")
    features["num_hyphens"] = url.count("-")
    features["num_digits"] = sum(c.isdigit() for c in url)
    features["num_special_chars"] = sum(not c.isalnum() for c in url)
    features["has_ip"] = bool(re.search(r"\d+\.\d+\.\d+\.\d+", url))
    features["has_at"] = "@" in url
    features["has_https"] = parsed.scheme == "https"
    features["contains_percent"] = "%" in url
    features["contains_question"] = "?" in url
    features["contains_equal"] = "=" in url

    suspicious_keywords = ["login", "verify", "secure", "update", "account", "bank"]
    features["keyword_flag"] = any(k in url.lower() for k in suspicious_keywords)

    features["tld"] = ext.suffix
    features["tld_len"] = len(ext.suffix)

    return features







@app.post("/predict_url")
def predict_url(item: URLRequest):
    feature_df = pd.DataFrame([extract_features(item.url)])
    feature_df = pd.get_dummies(feature_df, columns=["tld"], drop_first=True)

    # Align training columns
    for col in url_model.feature_names_in_:
        if col not in feature_df.columns:
            feature_df[col] = 0

    feature_df = feature_df[url_model.feature_names_in_]
    prediction = url_model.predict(feature_df)[0]

    return {
        "url": item.url,
        "prediction": "Phishing 🚨" if prediction == 1 else "Legitimate ✅",
    }



