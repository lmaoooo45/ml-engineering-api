import pytest
from fastapi.testclient import TestClient

from api import create_app


class FakeAnalyzer:
    def __init__(self):
        self.received = []

    def predict(self, text):
        self.received.append(text)
        return {
            "text": text,
            "label": "positive",
            "score": 0.9,
            "scores": {"negative": 0.1, "neutral": 0.2, "positive": 0.9},
            "processing_seconds": 0.01,
        }


@pytest.fixture
def client():
    analyzer = FakeAnalyzer()
    with TestClient(create_app(lambda: analyzer)) as test_client:
        yield test_client, analyzer


def test_health(client):
    test_client, analyzer = client
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert analyzer.received == []


def test_predict_passes_text_and_returns_result(client):
    test_client, analyzer = client
    text = "Мне понравился этот фильм"
    response = test_client.post("/predict", json={"text": text})
    assert response.status_code == 200
    assert response.json() == {
        "text": text,
        "label": "positive",
        "score": 0.9,
        "scores": {"negative": 0.1, "neutral": 0.2, "positive": 0.9},
        "processing_seconds": 0.01,
    }
    assert analyzer.received == [text]


@pytest.mark.parametrize("body", [
    {}, {"text": ""}, {"text": " \t\n"}, {"text": None},
    {"text": 123}, {"text": True}, {"text": ["фильм"]}, {"text": "а" * 5001},
])
def test_invalid_input_rejected_without_model_call(client, body):
    test_client, analyzer = client
    response = test_client.post("/predict", json=body)
    assert response.status_code == 422
    assert "detail" in response.json()
    assert analyzer.received == []


def test_malformed_json(client):
    test_client, analyzer = client
    response = test_client.post("/predict", content='{"text":', headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert analyzer.received == []


def test_predict_requires_post(client):
    test_client, _ = client
    assert test_client.get("/predict").status_code == 405


def test_model_loaded_once_for_multiple_requests():
    calls = []
    analyzer = FakeAnalyzer()

    def factory():
        calls.append("load")
        return analyzer

    with TestClient(create_app(factory)) as test_client:
        for text in ["Первый текст", "Второй текст"]:
            assert test_client.post("/predict", json={"text": text}).status_code == 200
        assert calls == ["load"]
        assert analyzer.received == ["Первый текст", "Второй текст"]
