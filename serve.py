
# serve.py
# Minimal FastAPI service that loads the exported bundle and serves /predict.
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

app = FastAPI(title="SoMe Scoring Service", version="1.0")

BUNDLE = joblib.load("artifacts/some_model.joblib")
PIPE = BUNDLE["pipeline"]
FEATURES = BUNDLE["feature_order"]
THRESH = BUNDLE["thresholds"]
VERSION = BUNDLE.get("version", "unknown")

class Record(BaseModel):
    engagement_rate: float = Field(..., ge=0)
    follower_count_c: float = Field(..., ge=0)
    posting_frequency_c: float = Field(..., ge=0)

class BatchRequest(BaseModel):
    records: List[Record]

def categorize(p: float) -> str:
    if p >= THRESH["t_launch"]:
        return "Launch"
    if p >= THRESH["t_review"]:
        return "Review"
    return "Ignore"

@app.get("/health")
def health():
    return {"status": "ok", "features": FEATURES, "version": VERSION}

@app.post("/predict")
def predict(req: BatchRequest):
    try:
        X = [[r.engagement_rate, r.follower_count_c, r.posting_frequency_c] for r in req.records]
        proba = PIPE.predict_proba(X)[:, 1].tolist()
        cats = [categorize(p) for p in proba]
        return {"prob_success": proba, "category": cats, "n": len(proba), "version": VERSION}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
