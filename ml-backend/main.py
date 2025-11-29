from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

# Load model + vectorizer
model = joblib.load("fake_job_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
# ----------------------------
# Enable CORS
# ----------------------------
origins = [
    "https://fake-job-detect-front.onrender.com",  # Your deployed frontend
    "http://localhost:3000"                        # Local testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # allow these origins
    allow_credentials=True,
    allow_methods=["*"],          # allow GET, POST, etc.
    allow_headers=["*"],          # allow headers like Content-Type
)


class JobRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Fake Job Detector API is running"}

@app.post("/predict")
def predict(data: JobRequest):
    text_vec = vectorizer.transform([data.text])
    pred = model.predict(text_vec)[0]
    result = "FAKE" if pred == 1 else "REAL"

    return {"prediction": result}

