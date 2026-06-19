"""Baseline model ile çıkarım — API ve testler bunu kullanır.

Eğitimdeki AYNI light_clean + metadata mantığını kullanır ki train/serve
tutarlı olsun.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import load
from scipy.sparse import hstack

from text_utils import light_clean

ROOT = Path(__file__).resolve().parent.parent
# MODEL_NAME env ile değiştirilebilir: baseline | baseline_balanced
# Varsayılan balanced -> sahte-puan (olumsuz metin) tespitini öne çıkarır.
MODEL_NAME = os.environ.get("MODEL_NAME", "baseline_balanced")
MODEL_PATH = ROOT / "models" / f"{MODEL_NAME}.joblib"


def compute_meta(raw_text: str):
    """Ham metinden 4 metadata özelliği (eğitimdeki tanımla aynı)."""
    t = raw_text if isinstance(raw_text, str) else ""
    n = max(len(t), 1)
    return [
        len(t),                              # review_length
        t.count("!"),                        # exclamation_count
        t.count("?"),                        # question_count
        sum(c.isupper() for c in t) / n,     # upper_ratio
    ]


class RatingModel:
    """Tek seferde yükle, çok kez tahmin et."""

    def __init__(self, model_path: Path = MODEL_PATH):
        bundle = load(model_path)
        self.word_vec = bundle["word_vec"]
        self.char_vec = bundle["char_vec"]
        self.scaler = bundle["scaler"]
        self.clf = bundle["clf"]
        self.meta_cols = bundle["meta_cols"]
        self.classes_ = self.clf.classes_  # [1,2,3,4,5]

    def _features(self, texts):
        clean = [light_clean(t) for t in texts]
        Xw = self.word_vec.transform(clean)
        Xc = self.char_vec.transform(clean)
        meta_df = pd.DataFrame([compute_meta(t) for t in texts], columns=self.meta_cols)
        meta = self.scaler.transform(meta_df)
        return hstack([Xw, Xc, sp.csr_matrix(meta)]).tocsr()

    def predict(self, texts):
        """texts: list[str] -> list[dict] (yıldız + olasılıklar + güven)."""
        X = self._features(texts)
        probs = self.clf.predict_proba(X)
        stars = self.classes_.astype(float)  # [1,2,3,4,5]
        out = []
        for p in probs:
            star = int(self.classes_[int(np.argmax(p))])
            # expected_rating = Σ olasılık×yıldız -> ortalama için argmax'tan
            # daha az yanlı (5★'ı sistematik şişirmez)
            expected = float(np.dot(p, stars))
            out.append({
                "predicted_rating": star,
                "expected_rating": round(expected, 3),
                "confidence": float(np.max(p)),
                "probabilities": {int(c): float(pr) for c, pr in zip(self.classes_, p)},
            })
        return out


_MODEL = None


def get_model():
    """Lazy singleton."""
    global _MODEL
    if _MODEL is None:
        _MODEL = RatingModel()
    return _MODEL
