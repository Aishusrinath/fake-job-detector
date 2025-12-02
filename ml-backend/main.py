from fastapi import FastAPI
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
from torchvision import transforms
from PIL import Image
from fastapi import UploadFile, File
from io import BytesIO


app = FastAPI()


@app.on_event("startup")
def load_pt_image_model():
    download_pt_model()
    global image_model

    image_model = torch.load(PT_MODEL_PATH, map_location="cpu")
    image_model.eval()

    # Define your preprocessing transform
    global preprocess
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),   # adjust for your model
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # change if your training used different values
            std=[0.229, 0.224, 0.225]
        )
    ])

    print("Image classifier loaded.")



NEWS_API_KEY = "66a21a149d374c229abc8dfec6dd54a3"

# Enable CORS so frontend can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model + vectorizer
# Fake Job Detector
job_model = joblib.load("fake_job_model.pkl")
job_vectorizer = joblib.load("tfidf_vectorizer.pkl")


# URL Detector Model
url_model = joblib.load("phishing_model.pkl")

# url_model = joblib.load("url_model.pkl")
# url_vectorizer = joblib.load("url_vectorizer.pkl")

# ----------------------------
# Google Drive Download
# ----------------------------
PT_MODEL_URL = "https://drive.google.com/file/d/15Vqn8-rAIIQhEdxYUcgwx1jC6znrmXx_/view?usp=sharing"
PT_MODEL_PATH = "email_model.pt"   # rename as needed

def download_pt_model():
    if not os.path.exists(PT_MODEL_PATH):
        print("Downloading image classifier model (.pt)...")
        resp = requests.get(PT_MODEL_URL)
        with open(PT_MODEL_PATH, "wb") as f:
            f.write(resp.content)
        print("Model download complete.")
    else:
        print(".pt model already exists locally.")








# Feature extraction function
def extract_features(url):
    features = {}
    parsed = urlparse(url)
    ext = tldextract.extract(url)
    
    features['url_length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_special_chars'] = sum(not c.isalnum() for c in url)
    features['has_ip'] = bool(re.search(r"\d+\.\d+\.\d+\.\d+", url))
    features['has_at'] = '@' in url
    features['has_https'] = parsed.scheme == 'https'
    features['contains_percent'] = '%' in url
    features['contains_question'] = '?' in url
    features['contains_equal'] = '=' in url
    
    suspicious_keywords = ['login', 'verify', 'secure', 'update', 'account', 'bank']
    features['keyword_flag'] = any(k in url.lower() for k in suspicious_keywords)
    
    features['tld'] = ext.suffix
    features['tld_len'] = len(ext.suffix)
    
    return features



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
    if data.get("status") == "ok":
        return {"articles": data.get("articles", [])}
    else:
        return {"articles": []}

@app.post("/predict_job")
def predict(data: JobRequest):
    # Transform text
    X = job_vectorizer.transform([data.text])

    # Predict
    pred = job_model.predict(X)[0]

    return {
        "prediction": "FAKE" if pred == 1 else "REAL"
    }


# URL / Phishing Detection
# -------------------------------
@app.post("/predict_url")
def predict(item: URLRequest):
    feature_df = pd.DataFrame([extract_features(item.url)])
    feature_df = pd.get_dummies(feature_df, columns=['tld'], drop_first=True)
    
    # Align columns with training data
    for col in url_model.feature_names_in_:
        if col not in feature_df.columns:
            feature_df[col] = 0
    feature_df = feature_df[url_model.feature_names_in_]
    
    prediction = url_model.predict(feature_df)[0]
    return {"url": item.url, "prediction": "Phishing 🚨" if prediction == 1 else "Legitimate ✅"}


@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    # Read image bytes
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    # Preprocess
    tensor = preprocess(image).unsqueeze(0)

    # Inference
    with torch.no_grad():
        output = image_model(tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    return {
        "filename": file.filename,
        "prediction": int(predicted_class)
    }















