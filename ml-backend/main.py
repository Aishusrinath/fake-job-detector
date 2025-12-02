from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import urlparse
import pandas as pd
import tldextract
import re


app = FastAPI()

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
    for col in model.feature_names_in_:
        if col not in feature_df.columns:
            feature_df[col] = 0
    feature_df = feature_df[url_model.feature_names_in_]
    
    prediction = url_model.predict(feature_df)[0]
    return {"url": item.url, "prediction": "Phishing 🚨" if prediction == 1 else "Legitimate ✅"}



    
# URL Phishing Prediction
# @app.post("/predict_url")
# def predict_url(data: URLRequest):
#     print("Received URL:", data.url)  # debug
#     try:
#         X = url_vectorizer.transform([data.url])
#         pred = url_model.predict(X)[0]
#         return {"prediction": "PHISHING" if pred == 0 else "LEGIT"}
#     except Exception as e:
#         print("Error:", e)
#         return {"error": str(e)}










