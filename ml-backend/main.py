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
model = joblib.load("fake_job_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

class JobRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Fake Job Detector API Running"}

@app.post("/predict")
def predict(data: JobRequest):
    # Transform text
    X = vectorizer.transform([data.text])

    # Predict
    pred = model.predict(X)[0]

    return {
        "prediction": "FAKE" if pred == 1 else "REAL"
    }
