"""Baseline iyileştirme deneyleri — özellikleri BİR KEZ çıkar, çok model dene.

Karşılaştırılan eksenler:
  - Sınıf ağırlığı: yok / balanced / sqrt-yumuşatılmış
  - Loss: log_loss vs modified_huber, alpha taraması
  - Regresyon kafası (sürekli puan) -> ortalama için MAE
  - Beklenen-değer (E[yıldız]=Σ p·yıldız) tahmini -> daha hassas ortalama

Tüm metrikler aynı leakage-safe, dedup'lı test setinde.
"""
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from text_utils import light_clean

ROOT = Path(__file__).resolve().parent.parent          # uygulama/
CSV = ROOT.parent / "turkish_ecommerce_reviews.csv"     # proje kökündeki veri
META = ["review_length", "exclamation_count", "question_count", "upper_ratio"]
STARS = np.array([1, 2, 3, 4, 5])


def metrics(name, y_true, y_pred, ev=None):
    acc = accuracy_score(y_true, y_pred)
    off1 = np.mean(np.abs(y_true - y_pred) <= 1)
    mae = mean_absolute_error(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    ev_mae = mean_absolute_error(y_true, ev) if ev is not None else mae
    print(f"{name:<34} acc={acc*100:5.2f}  +-1={off1*100:5.2f}  "
          f"MAE={mae:.3f}  evMAE={ev_mae:.3f}  macroF1={mf1:.3f}", flush=True)
    return acc, off1, mae, mf1


def main():
    t0 = time.time()
    print("Veri + özellikler hazırlanıyor (bir kez)...", flush=True)
    df = pd.read_csv(CSV).rename(columns={"Rating (Star)": "rating"})
    df = df.dropna(subset=["Review", "rating"])
    df["review_clean"] = df["Review"].astype(str).map(light_clean)
    df = df[df["review_clean"].str.len() >= 2]
    df = df.drop_duplicates(subset=["review_clean"]).reset_index(drop=True)
    y = df["rating"].astype(int).values

    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    txt = df["review_clean"].values

    wv = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=120_000, sublinear_tf=True)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5,
                         max_features=200_000, sublinear_tf=True)
    Xw_tr, Xw_te = wv.fit_transform(txt[tr]), None
    Xc_tr = cv.fit_transform(txt[tr])
    Xw_te = wv.transform(txt[te]); Xc_te = cv.transform(txt[te])
    sc = StandardScaler()
    m_tr = sp.csr_matrix(sc.fit_transform(df.iloc[tr][META]))
    m_te = sp.csr_matrix(sc.transform(df.iloc[te][META]))
    X_tr = hstack([Xw_tr, Xc_tr, m_tr]).tocsr()
    X_te = hstack([Xw_te, Xc_te, m_te]).tocsr()
    y_tr, y_te = y[tr], y[te]
    print(f"  Hazır ({time.time()-t0:.0f} sn)  X_tr={X_tr.shape}\n", flush=True)
    print(f"  Test gerçek ortalama puan: {y_te.mean():.3f}\n", flush=True)

    # Yumuşatılmış sınıf ağırlıkları
    bal = compute_class_weight("balanced", classes=STARS, y=y_tr)
    w_sqrt = {s: float(np.sqrt(w)) for s, w in zip(STARS, bal)}

    def sgd(**kw):
        return SGDClassifier(max_iter=50, early_stopping=True, n_iter_no_change=3,
                             validation_fraction=0.05, random_state=42, n_jobs=-1, **kw)

    configs = [
        ("logloss a1e-5 (mevcut)", sgd(loss="log_loss", alpha=1e-5)),
        ("logloss a1e-5 balanced", sgd(loss="log_loss", alpha=1e-5, class_weight="balanced")),
        ("logloss a1e-5 sqrt-w",   sgd(loss="log_loss", alpha=1e-5, class_weight=w_sqrt)),
        ("logloss a3e-6",          sgd(loss="log_loss", alpha=3e-6)),
        ("logloss a3e-6 sqrt-w",   sgd(loss="log_loss", alpha=3e-6, class_weight=w_sqrt)),
        ("modhuber a1e-5 sqrt-w",  sgd(loss="modified_huber", alpha=1e-5, class_weight=w_sqrt)),
    ]

    print("=== SINIFLANDIRICILAR (proba'lı) ===", flush=True)
    for name, clf in configs:
        ts = time.time()
        clf.fit(X_tr, y_tr)
        p = clf.predict_proba(X_te)
        y_pred = STARS[p.argmax(1)]
        ev = p @ STARS                      # beklenen değer (sürekli puan)
        metrics(f"{name}", y_te, y_pred, ev=ev)
        print(f"{'':34}  tahmin-ort(argmax)={y_pred.mean():.3f}  "
              f"tahmin-ort(EV)={ev.mean():.3f}  ({time.time()-ts:.0f}sn)", flush=True)

    print("\n=== REGRESYON (ortalama puan için MAE-optimize) ===", flush=True)
    reg = SGDRegressor(alpha=1e-5, max_iter=50, early_stopping=True,
                       n_iter_no_change=3, random_state=42)
    reg.fit(X_tr, y_tr)
    r = np.clip(reg.predict(X_te), 1, 5)
    r_round = np.clip(np.rint(r), 1, 5).astype(int)
    metrics("SGDRegressor (round->sınıf)", y_te, r_round, ev=r)
    print(f"{'':34}  tahmin-ort(reg)={r.mean():.3f}", flush=True)

    print(f"\nToplam {time.time()-t0:.0f} sn", flush=True)


if __name__ == "__main__":
    main()
