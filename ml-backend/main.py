# from fastapi import FastAPI, UploadFile, File
# from pydantic import BaseModel
# import io
# import joblib
# from fastapi.middleware.cors import CORSMiddleware
# from urllib.parse import urlparse
# import pandas as pd
# import tldextract
# import re
# import requests
# import os
# import torch
# from torchvision import models, transforms
# import torch.nn as nn
# from PIL import Image
# from io import BytesIO


# app = FastAPI()



# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # ----------------------------
# # Paths + URLs
# # ----------------------------
# PT_MODEL_URL = "https://github.com/Aishusrinath/fake-job-detector/releases/download/v1.0/email_resnet18_best.pt"
# PT_MODEL_PATH = "email_resnet18_best.pt"

# NEWS_API_KEY = "66a21a149d374c229abc8dfec6dd54a3"


# DEVICE = torch.device("cpu")
# CLASS_NAMES = ["scam", "legit"]   # order from training
# # ----------------------------
# # Download .pt model
# def download_model():
#     """Downloads .pt file from GitHub Releases if not present."""
#     if not os.path.exists(PT_MODEL_PATH):
#         print("Downloading model...")
#         r = requests.get(PT_MODEL_URL, stream=True)
#         with open(PT_MODEL_PATH, "wb") as f:
#             f.write(r.content)
#         print("Model downloaded.")
#     else:
#         print("Model already exists.")

# # IMAGE TRANSFORMS
# # ================================
# eval_tfms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485,0.456,0.406],
#                          std=[0.229,0.224,0.225]),
# ])

# # ----------------------------
# # Load PyTorch Image Model
# # ----------------------------
# def load_image_model():
#     try:
#         model = models.resnet18(weights=None)
#     except:
#         model = models.resnet18(pretrained=False)

#     in_features = model.fc.in_features

#     # MUST MATCH TRAINING EXACTLY:
#     model.fc = nn.Sequential(
#         nn.Dropout(0.2),
#         nn.Linear(in_features, len(CLASS_NAMES))
#     )

#     state = torch.load(PT_MODEL_PATH, map_location=DEVICE)
#     model.load_state_dict(state, strict=True)

#     model.eval()
#     model.to(DEVICE)

#     print("✔ Model loaded correctly.")
#     return model


# image_model = None



# # ----------------------------
# # STARTUP EVENT
# # ----------------------------
# @app.on_event("startup")
# def startup_event():
#     global image_model, job_model, job_vectorizer, url_model

#     download_model()
#     image_model = load_image_model()

#     job_model = joblib.load("fake_job_model.pkl")
#     job_vectorizer = joblib.load("tfidf_vectorizer.pkl")
#     url_model = joblib.load("phishing_model.pkl")

# # ================================
# # PREDICT FUNCTION
# # ================================
# def run_prediction(img: Image.Image):
#     img = img.convert("RGB")
#     x = eval_tfms(img).unsqueeze(0).to(DEVICE)

#     with torch.no_grad():
#         logits = image_model(x)
#         probs = torch.softmax(logits, dim=1).cpu().numpy().ravel()

#     idx = probs.argmax()
#     return CLASS_NAMES[idx], float(probs[idx]), {
#         c: float(p) for c, p in zip(CLASS_NAMES, probs)
#     }
# # ----------------------------
# # IMAGE PREDICTION ENDPOINT
# # ----------------------------
# @app.post("/predict-image")
# async def predict(file: UploadFile = File(...)):
#     content = await file.read()
#     try:
#         img = Image.open(io.BytesIO(content))
#     except Exception as e:
#         return {"error": "Invalid image file", "details": str(e)}

#     label, confidence, distribution = run_prediction(img)

#     return {
#         "label": label,
#         "confidence": confidence,
#         "probabilities": distribution
#     }


# # ----------------------------
# # URL + JOB PREDICTORS (Your code unchanged)
# # ----------------------------
# class JobRequest(BaseModel):
#     text: str


# class URLRequest(BaseModel):
#     url: str


# @app.get("/")
# def home():
#     return {"message": "API Running"}


# @app.get("/news")
# def get_news():
#     url = f"https://newsapi.org/v2/everything?q=phishing+Canada&language=en&pageSize=5&apiKey={NEWS_API_KEY}"
#     resp = requests.get(url)
#     data = resp.json()
#     return {"articles": data.get("articles", [])}


# @app.post("/predict_job")
# def predict(data: JobRequest):
#     X = job_vectorizer.transform([data.text])
#     pred = job_model.predict(X)[0]
#     return {"prediction": "FAKE" if pred == 1 else "REAL"}


# def extract_features(url):
#     features = {}
#     parsed = urlparse(url)
#     ext = tldextract.extract(url)

#     features["url_length"] = len(url)
#     features["num_dots"] = url.count(".")
#     features["num_hyphens"] = url.count("-")
#     features["num_digits"] = sum(c.isdigit() for c in url)
#     features["num_special_chars"] = sum(not c.isalnum() for c in url)
#     features["has_ip"] = bool(re.search(r"\d+\.\d+\.\d+\.\d+", url))
#     features["has_at"] = "@" in url
#     features["has_https"] = parsed.scheme == "https"
#     features["contains_percent"] = "%" in url
#     features["contains_question"] = "?" in url
#     features["contains_equal"] = "=" in url

#     suspicious_keywords = ["login", "verify", "secure", "update", "account", "bank"]
#     features["keyword_flag"] = any(k in url.lower() for k in suspicious_keywords)

#     features["tld"] = ext.suffix
#     features["tld_len"] = len(ext.suffix)

#     return features







# @app.post("/predict_url")
# def predict_url(item: URLRequest):
#     feature_df = pd.DataFrame([extract_features(item.url)])
#     feature_df = pd.get_dummies(feature_df, columns=["tld"], drop_first=True)

#     # Align training columns
#     for col in url_model.feature_names_in_:
#         if col not in feature_df.columns:
#             feature_df[col] = 0

#     feature_df = feature_df[url_model.feature_names_in_]
#     prediction = url_model.predict(feature_df)[0]

#     return {
#         "url": item.url,
#         "prediction": "Phishing 🚨" if prediction == 1 else "Legitimate ✅",
#     }






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

    # Extract features
    df = pd.DataFrame([extract_features(req.url)])
    df = pd.get_dummies(df, columns=["tld"], drop_first=False)

    # Ensure all missing columns exist:
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0

    # Ensure no extra columns
    df = df[model.feature_names_in_]

    # Predict
    try:
        pred = model.predict(df)[0]
    except Exception as e:
        return {"error": "Model prediction failed", "details": str(e)}

    return {
        "url": req.url,
        "prediction": "Phishing 🚨" if pred == 1 else "Legitimate ✅"
    }

