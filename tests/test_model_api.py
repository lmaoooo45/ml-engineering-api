import pytest
from fastapi.testclient import TestClient

from api import create_app

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def real_client():
    with TestClient(create_app()) as client:
        yield client


@pytest.mark.parametrize("text, expected", [
    ("Мне понравился этот фильм", "positive"),
    ("Ужасное качество больше не буду покупать", "negative"),
    ("Посылка прибыла во вторник", "neutral"),
])
def test_real_model_prediction(real_client, text, expected):
    response = real_client.post("/predict", json={"text": text})
    assert response.status_code == 200
    result = response.json()
    assert result["text"] == text
    assert result["label"] == expected
    assert set(result["scores"]) == {"negative", "neutral", "positive"}
    assert all(0 <= value <= 1 for value in result["scores"].values())
    assert result["label"] == max(result["scores"], key=result["scores"].get)
    assert result["score"] == result["scores"][expected]
    assert result["processing_seconds"] >= 0
