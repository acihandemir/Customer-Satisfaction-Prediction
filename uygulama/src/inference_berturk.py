"""BERTurk (fine-tuned) ile çıkarım — baseline ile AYNI arayüz.

predict(texts) -> list[dict]: predicted_rating, expected_rating, confidence, probabilities
Böylece app.py her iki backend'i de aynı şekilde kullanır.

Gereksinim:  pip install torch transformers
Model:       models/berturk_model/  (Colab'da trainer.save_model ile kaydedilen
             klasörü Drive'dan indirip buraya koy) — BERTURK_PATH env ile değişir.

Not: model label'ları 0..4, gerçek yıldız = index+1.
"""
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BERTURK_PATH = Path(os.environ.get("BERTURK_PATH", ROOT / "models" / "berturk_model"))
MAX_LEN = 160
STARS = np.array([1, 2, 3, 4, 5])


class BertRatingModel:
    """Tek seferde yükle, çok kez tahmin et. torch/transformers tembel import."""

    def __init__(self, path: Path = BERTURK_PATH, batch_size: int = 32):
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
        if not Path(path).exists():
            raise FileNotFoundError(
                f"BERTurk modeli bulunamadı: {path}\n"
                "Colab'da trainer.save_model ile kaydettiğin klasörü buraya indir."
            )
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path))
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.batch_size = batch_size
        self.classes_ = STARS

    def predict(self, texts):
        torch = self.torch
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i:i + self.batch_size])
            enc = self.tokenizer(batch, truncation=True, max_length=MAX_LEN,
                                 padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for p in probs:
                star = int(STARS[int(p.argmax())])
                out.append({
                    "predicted_rating": star,
                    "expected_rating": round(float(np.dot(p, STARS)), 3),
                    "confidence": float(p.max()),
                    "probabilities": {int(s): float(pr) for s, pr in zip(STARS, p)},
                })
        return out


_MODEL = None


def get_model():
    """Lazy singleton."""
    global _MODEL
    if _MODEL is None:
        _MODEL = BertRatingModel()
    return _MODEL
