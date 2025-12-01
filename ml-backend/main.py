from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware

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
pipeline = joblib.load("domain_pipeline.pkl")

# url_model = joblib.load("url_model.pkl")
# url_vectorizer = joblib.load("url_vectorizer.pkl")


# Your trusted domains (same as notebook)
TRUSTED_DOMAINS = {
    "scotiabank.com",
    "bmo.com",
    "td.com",
    "rbc.com",
    "cibc.com",
    "desire2learn.com",
    "durhamcollege.ca",
}

# -------------------------------
# Helper functions
# -------------------------------

def get_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower().strip()
        if host.startswith("www."):
            host = host[4:]
        return host
    except:
        return ""

def is_trusted_domain(domain: str) -> bool:
    return domain in TRUSTED_DOMAINS




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
def predict_url(data: URLRequest):

    url = data.url.strip()
    domain = get_domain(url)

    if domain == "":
        return {"prediction": "UNKNOWN", "reason": "invalid_url"}

    # RULE: trusted domains always legit
    if is_trusted_domain(domain):
        return {"prediction": "LEGIT", "domain": domain, "reason": "trusted_domain"}

    # ML prediction using domain-only pipeline
    pred = pipeline.predict([domain])[0]
    prediction = "PHISHING" if int(pred) == 0 else "LEGIT"

    return {
        "prediction": prediction,
        "domain": domain,
        "reason": "model",
    }



    
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




