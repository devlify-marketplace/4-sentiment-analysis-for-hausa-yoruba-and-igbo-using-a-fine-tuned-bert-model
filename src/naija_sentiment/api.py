"""Local REST API for the fine-tuned NaijaSenti classifier."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .data import LANGUAGES, LABELS, validate_language
from .predict import SentimentPredictor

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.environ.get("NAIJA_MODEL_PATH", ROOT / "artifacts" / "model"))
_predictor: SentimentPredictor | None = None

app = FastAPI(title="NaijaSenti Hausa–Yoruba–Igbo Sentiment API", version="1.0.0",
              description="Coursework prototype. Outputs are model estimates, not facts or high-impact decisions.")


class PredictRequest(BaseModel):
    text: Annotated[str, Field(min_length=1, max_length=5000)]
    language: str = Field(description="hau (Hausa), yor (Yoruba), or ibo (Igbo)")


class BatchRequest(BaseModel):
    texts: Annotated[list[str], Field(min_length=1, max_length=100)]
    language: str


def get_predictor() -> SentimentPredictor:
    global _predictor
    if _predictor is None:
        if not (MODEL_DIR / "config.json").exists():
            raise HTTPException(status_code=503, detail="Model is not trained yet. Run the training command first.")
        try:
            _predictor = SentimentPredictor(MODEL_DIR)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Could not load the local model: {type(exc).__name__}") from exc
    return _predictor


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ready" if (MODEL_DIR / "config.json").exists() else "model_not_trained",
            "model": "Davlan/afro-xlmr-base", "model_path": str(MODEL_DIR), "ready": (MODEL_DIR / "config.json").exists()}


@app.get("/api/languages")
def languages():
    return {"languages": [{"code": code, "name": name} for code, name in LANGUAGES.items()],
            "labels": list(LABELS)}


@app.post("/api/predict")
def predict(request: PredictRequest):
    try:
        language = validate_language(request.language)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        return get_predictor().predict(request.text, language)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/predict-batch")
def predict_batch(request: BatchRequest):
    try:
        language = validate_language(request.language)
        if any(not text.strip() for text in request.texts):
            raise ValueError("Text entries must not be empty")
        if any(len(text) > 5000 for text in request.texts):
            raise ValueError("Each text must be 5,000 characters or fewer")
        return {"results": get_predictor().predict_batch(request.texts, language)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
