from time import perf_counter

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "cointegrated/rubert-tiny-sentiment-balanced"


class SentimentAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.eval()

    def predict(self, text):
        if not text.strip():
            raise ValueError("Введите непустой текст")
        started = perf_counter()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0]
            scores = torch.sigmoid(logits)
        index = int(scores.argmax())
        return {
            "text": text,
            "label": self.model.config.id2label[index],
            "score": float(scores[index]),
            "scores": {self.model.config.id2label[i]: float(s) for i, s in enumerate(scores)},
            "processing_seconds": perf_counter() - started,
        }
