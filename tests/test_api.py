from fastapi.testclient import TestClient

import naija_sentiment.api as api

client = TestClient(api.app)


def test_languages_endpoint():
    response = client.get("/api/languages")
    assert response.status_code == 200
    assert {x["code"] for x in response.json()["languages"]} == {"hau", "yor", "ibo"}
    assert response.json()["labels"] == ["negative", "neutral", "positive"]


def test_health_untrained(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODEL_DIR", tmp_path / "absent-model")
    monkeypatch.setattr(api, "_predictor", None)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["ready"] is False


def test_predict_untrained_returns_503(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODEL_DIR", tmp_path / "absent-model")
    monkeypatch.setattr(api, "_predictor", None)
    response = client.post("/api/predict", json={"text": "test", "language": "yor"})
    assert response.status_code == 503


def test_invalid_language_returns_422():
    response = client.post("/api/predict", json={"text": "hello", "language": "pcm"})
    assert response.status_code == 422
