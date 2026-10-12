from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator


class TextRequest(BaseModel):
    text: str = Field(strict=True, min_length=1, max_length=5000)

    @field_validator("text")
    @classmethod
    def check_text(cls, text):
        if not text.strip():
            raise ValueError("Введите непустой текст")
        return text


class Scores(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    negative: float = Field(ge=0, le=1)
    neutral: float = Field(ge=0, le=1)
    positive: float = Field(ge=0, le=1)


class Prediction(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    text: str
    label: Literal["negative", "neutral", "positive"]
    score: float = Field(ge=0, le=1)
    scores: Scores
    processing_seconds: float = Field(ge=0)


def load_model():
    from model import SentimentAnalyzer

    return SentimentAnalyzer()


def create_app(model_factory=load_model):
    @asynccontextmanager
    async def lifespan(app):
        app.state.analyzer = model_factory()
        yield
        del app.state.analyzer

    app = FastAPI(title="Анализ тональности текста", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict", response_model=Prediction)
    def predict(body: TextRequest, request: Request):
        return request.app.state.analyzer.predict(body.text)

    return app


app = create_app()
