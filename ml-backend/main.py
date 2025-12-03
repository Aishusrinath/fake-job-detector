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
import torchvision.transforms as transforms
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


# ----------------------------
# Download .pt model
# ----------------------------
def download_pt_model():
    if not os.path.exists(PT_MODEL_PATH):
        print("Downloading image classifier model (.pt)...")
        resp = requests.get(PT_MODEL_URL)
        with open(PT_MODEL_PATH, "wb") as f:
            f.write(resp.content)
        print("Model download complete.")
    else:
        print(".pt model already exists locally.")


# ----------------------------
# Load PyTorch Image Model
# ----------------------------
def load_image_model():
    model = models.resnet18(weights=None)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # CHANGE if you have different classes

    state_dict = torch.load(PT_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()
    return model


# ----------------------------
# Preprocess Image
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess(image: Image.Image):
    return transform(image)


# ----------------------------
# STARTUP EVENT
# ----------------------------
@app.on_event("startup")
def startup_event():
    global image_model, job_model, job_vectorizer, url_model

    download_pt_model()
    image_model = load_image_model()

    job_model = joblib.load("fake_job_model.pkl")
    job_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    url_model = joblib.load("phishing_model.pkl")


# ----------------------------
# IMAGE PREDICTION ENDPOINT
# ----------------------------
@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = image_model(tensor)
        pred_class = torch.argmax(output, dim=1).item()

    return {"filename": file.filename, "prediction": int(pred_class)}


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

