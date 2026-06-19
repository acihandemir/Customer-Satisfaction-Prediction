"""FastAPI servisi — 2 endpoint + basit web arayüzü.

  POST /predict    -> tek yorumun tahmini yıldızı + olasılıklar
  POST /aggregate  -> yorum listesi -> metin-tabanlı gerçek ortalama puan
                      (+ verilen puanlarla uyuşmazlık / sahte-puan bayrağı)

Çalıştır:  uvicorn app:app --reload   (src/ klasöründen)
"""
import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Backend seçimi: BACKEND=baseline (hafif/hızlı, sklearn) | berturk (en iyi, torch)
BACKEND = os.environ.get("BACKEND", "baseline").lower()
if BACKEND == "berturk":
    from inference_berturk import get_model
else:
    from inference import get_model

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "web"

# Verilen puan ile metin-tahmini arası bu kadar (yıldız) fark -> şüpheli
FRAUD_THRESHOLD = 2

app = FastAPI(title="Türkçe Yorum Puan Sınıflandırıcı", version="0.1.0")


# ── Şemalar ────────────────────────────────────────────────────
class PredictIn(BaseModel):
    text: str = Field(..., min_length=1, description="Yorum metni")


class ReviewIn(BaseModel):
    text: str = Field(..., min_length=1)
    given_rating: Optional[int] = Field(
        None, ge=1, le=5, description="Kullanıcının verdiği yıldız (opsiyonel)"
    )


class AggregateIn(BaseModel):
    reviews: List[ReviewIn]
    include_details: bool = Field(
        False, description="Her yorumun tahminini de döndür"
    )


# ── Endpointler ────────────────────────────────────────────────
@app.post("/predict")
def predict(inp: PredictIn):
    return get_model().predict([inp.text])[0]


@app.post("/aggregate")
def aggregate(inp: AggregateIn):
    model = get_model()
    texts = [r.text for r in inp.reviews]
    preds = model.predict(texts)

    pred_stars = [p["predicted_rating"] for p in preds]
    # Ortalama: beklenen-değer (expected_rating) -> argmax'tan daha az yanlı,
    # gerçek ortalamayı daha iyi yakalar. argmax ortalaması da bilgi için döner.
    predicted_avg = sum(p["expected_rating"] for p in preds) / len(preds)
    predicted_avg_argmax = sum(pred_stars) / len(pred_stars)

    given = [r.given_rating for r in inp.reviews if r.given_rating is not None]
    given_avg = (sum(given) / len(given)) if given else None

    # Sahte-puan tespiti: verilen puan varsa, tahminle karşılaştır
    suspicious = []
    details = []
    for i, (r, p) in enumerate(zip(inp.reviews, preds)):
        flag = None
        if r.given_rating is not None:
            diff = r.given_rating - p["predicted_rating"]
            if abs(diff) >= FRAUD_THRESHOLD:
                # diff>0: verilen puan şişirilmiş (olumsuz metne yüksek puan)
                flag = "inflated" if diff > 0 else "deflated"
                suspicious.append({
                    "index": i,
                    "given_rating": r.given_rating,
                    "predicted_rating": p["predicted_rating"],
                    "type": flag,
                    "text": r.text[:120],
                })
        if inp.include_details:
            details.append({
                "index": i,
                "text": r.text[:120],
                "given_rating": r.given_rating,
                **p,
                "mismatch": flag,
            })

    resp = {
        "count": len(texts),
        "predicted_avg": round(predicted_avg, 3),
        "predicted_avg_argmax": round(predicted_avg_argmax, 3),
        "given_avg": round(given_avg, 3) if given_avg is not None else None,
        "rating_distribution": {s: pred_stars.count(s) for s in [1, 2, 3, 4, 5]},
        "suspicious_count": len(suspicious),
        "suspicious": suspicious,
    }
    if inp.include_details:
        resp["details"] = details
    return resp


@app.get("/health")
def health():
    return {"status": "ok", "backend": BACKEND}


# ── Web arayüzü (statik) ───────────────────────────────────────
if STATIC.exists():
    app.mount("/web", StaticFiles(directory=str(STATIC)), name="web")

    @app.get("/")
    def index():
        return FileResponse(str(STATIC / "index.html"))
