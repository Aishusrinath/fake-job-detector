from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load model + vectorizer
model = joblib.load("fake_job_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

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
