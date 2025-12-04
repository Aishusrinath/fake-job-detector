from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
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
import io

app = FastAPI()

# CORS — allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://fake-job-detect-front.onrender.com",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"], 
)

# --------------------------------
# CONSTANTS
# --------------------------------
PT_MODEL_URL = "https://github.com/Aishusrinath/fake-job-detector/releases/download/v1.0/email_resnet18_best.pt"
PT_MODEL_PATH = "email_resnet18_best.pt"
NEWS_API_KEY = "66a21a149d374c229abc8dfec6dd54a3"

DEVICE = torch.device("cpu")
CLASS_NAMES = ["scam", "legit"]

# --------------------------------
# LAZY LOAD MODELS
# --------------------------------
image_model = None
job_model = None
job_vectorizer = None
url_model = None


def download_model():
    if not os.path.exists(PT_MODEL_PATH):
        print("Downloading email model...")
        r = requests.get(PT_MODEL_URL)
        with open(PT_MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("Model downloaded.")


def get_image_model():
    global image_model
    if image_model is None:
        download_model()
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, len(CLASS_NAMES))
        )
        state = torch.load(PT_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state, strict=True)
        model.eval()
        model.to(DEVICE)
        image_model = model
        print("✔ Image model loaded lazily")
    return image_model


def get_job_model():
    global job_model, job_vectorizer
    if job_model is None:
        job_model = joblib.load("fake_job_model.pkl")
        job_vectorizer = joblib.load("tfidf_vectorizer.pkl")
        print("✔ Job detector loaded lazily")
    return job_model, job_vectorizer


def get_url_model():
    global url_model
    if url_model is None:
        url_model = joblib.load("phishing_model.pkl")
        print("✔ URL model loaded lazily")
    return url_model


# --------------------------------
# IMAGE PREDICTION
# --------------------------------
eval_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def run_prediction(img: Image.Image):
    model = get_image_model()
    img = img.convert("RGB")
    x = eval_tfms(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy().ravel()

    idx = probs.argmax()
    return CLASS_NAMES[idx], float(probs[idx]), {
        c: float(p) for c, p in zip(CLASS_NAMES, probs)
    }


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content))
    except Exception:
        return {"error": "Invalid image"}

    label, conf, dist = run_prediction(img)
    return {"label": label, "confidence": conf, "probabilities": dist}


# --------------------------------
# NEWS ENDPOINT
# --------------------------------
@app.get("/")
def home():
    return {"message": "API Running"}


@app.get("/news")
def get_news():
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q=phishing+Canada&language=en&pageSize=5&apiKey={NEWS_API_KEY}"
    )
    r = requests.get(url)
    data = r.json()
    return {"articles": data.get("articles", [])}


# --------------------------------
# JOB DESCRIPTION CHECKER
# --------------------------------
class JobRequest(BaseModel):
    text: str


@app.post("/predict_job")
def predict_job(data: JobRequest):
    job_model, vectorizer = get_job_model()
    X = vectorizer.transform([data.text])
    pred = job_model.predict(X)[0]
    return {"prediction": "FAKE" if pred == 1 else "REAL"}


# --------------------------------
# URL CHECKER
# --------------------------------
class URLRequest(BaseModel):
    url: str


def extract_features(url):
    parsed = urlparse(url)
    ext = tldextract.extract(url)

    return {
        "url_length": len(url),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": sum(not c.isalnum() for c in url),
        "has_ip": bool(re.search(r"\d+\.\d+\.\d+\.\d+", url)),
        "has_at": "@" in url,
        "has_https": parsed.scheme == "https",
        "contains_percent": "%" in url,
        "contains_question": "?" in url,
        "contains_equal": "=" in url,
        "keyword_flag": any(k in url.lower() for k in
                            ["login", "verify", "secure", "update", "account", "bank"]),
        "tld": ext.suffix,
        "tld_len": len(ext.suffix),
    }


@app.post("/predict_url")
def predict_url(req: URLRequest):
    model = get_url_model()

    # Load the exact training columns
    columns = joblib.load("columns.pkl")

    # Extract URL features
    features = extract_features(req.url)
    df = pd.DataFrame([features])

    # Create TLD dummies exactly like training
    df = pd.get_dummies(df, columns=["tld"], drop_first=True)

    # Add missing columns
    for col in columns:
        if col not in df.columns:
            df[col] = 0

    # Make sure only the training columns are used & in correct order
    df = df[columns]

    # Prediction
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0]

    return {
        "url": req.url,
        "label": "phishing" if pred == 1 else "legitimate",
        "confidence": float(prob[pred]),
        "probabilities": {
            "legitimate": float(prob[0]),
            "phishing": float(prob[1])
        }
    }





