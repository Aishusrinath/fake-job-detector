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
url_model = joblib.load("url_model.pkl")
url_vectorizer = joblib.load("url_vectorizer.pkl")


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
    
# URL Phishing Prediction
# -----------------------------
@app.post("/predict_url")
def predict_url(data: URLRequest):
    X = url_vectorizer.transform([data.url])
    pred = url_model.predict(X)[0]   # 0 = phishing, 1 = legitimate

    return {
        "prediction": "PHISHING" if pred == 0 else "LEGIT"
    }


