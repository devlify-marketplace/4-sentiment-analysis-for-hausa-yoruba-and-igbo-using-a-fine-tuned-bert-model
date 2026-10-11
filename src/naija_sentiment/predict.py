"""Inference wrapper for a locally saved Topic 4 classifier."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import LABELS, validate_language


class SentimentPredictor:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        if not (self.model_dir / "config.json").exists():
            raise FileNotFoundError(f"Fine-tuned model not found under {self.model_dir}; run `python -m naija_sentiment.train --root .` first.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        meta_path = self.model_dir.parent / "model_metadata.json"
        self.metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.max_length = int(self.metadata.get("max_length", self.model.config.max_position_embeddings - 2))
        self.labels = [self.model.config.id2label.get(i, LABELS[i]) for i in range(len(LABELS))]

    def predict(self, text: str, language: str) -> dict:
        return self.predict_batch([text], language)[0]

    def predict_batch(self, texts: list[str], language: str) -> list[dict]:
        language = validate_language(language)
        if not texts or len(texts) > 100:
            raise ValueError("Provide between 1 and 100 texts")
        cleaned = [str(text).strip() for text in texts]
        if any(not text for text in cleaned):
            raise ValueError("Text entries must not be empty")
        if any(len(text) > 5000 for text in cleaned):
            raise ValueError("Each text must be 5,000 characters or fewer")
        inputs = self.tokenizer(cleaned, truncation=True, max_length=self.max_length,
                                padding=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(**inputs).logits, dim=-1).cpu().tolist()
        output = []
        for scores in probabilities:
            index = max(range(len(scores)), key=scores.__getitem__)
            output.append({
                "language": language,
                "sentiment": self.labels[index],
                "confidence": float(scores[index]),
                "probabilities": {label: float(scores[i]) for i, label in enumerate(self.labels)},
            })
        return output
