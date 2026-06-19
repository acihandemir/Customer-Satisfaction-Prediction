"""Faz 0 + Faz 1: Veri hijyeni + geliştirilmiş klasik baseline.

Mevcut notebook'a göre 3 temel iyileştirme:
  1) Olumsuzluk korunur (stopword temizliği yok) -> "kötü değil" anlamı kalır.
  2) char_wb (2-5) n-gram -> Türkçe eklemeli morfolojiyi yakalar.
  3) Dedup + leakage-safe split -> şişmemiş, dürüst metrikler.

Metrikler: accuracy + MAE + ±1 accuracy + macro-F1 + confusion matrix.
±1 accuracy = komşu yıldızı doğru sayan tolerant metrik (sahte-puan tespiti
ve ortalama puan hesabı için exact accuracy'den daha anlamlı).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import dump
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from text_utils import light_clean

ROOT = Path(__file__).resolve().parent.parent          # uygulama/
CSV = ROOT.parent / "turkish_ecommerce_reviews.csv"     # proje kökündeki veri
MODELS = ROOT / "models"
META_COLS = ["review_length", "exclamation_count", "question_count", "upper_ratio"]


def log(msg):
    print(msg, flush=True)


def main(args):
    MODELS.mkdir(exist_ok=True)
    t0 = time.time()

    # ── Faz 0: Yükle + hijyen ──────────────────────────────────
    log("Veri yükleniyor...")
    df = pd.read_csv(CSV)
    df.rename(columns={"Rating (Star)": "rating"}, inplace=True)
    log(f"  Ham: {len(df):,} satır")

    df = df.dropna(subset=["Review", "rating"])
    df["review_clean"] = df["Review"].astype(str).map(light_clean)
    # Çok kısa / boş temizlenmiş yorumları at
    df = df[df["review_clean"].str.len() >= 2]
    # Dedup: aynı temizlenmiş metin + puan -> leakage'ı kapat
    before = len(df)
    df = df.drop_duplicates(subset=["review_clean"]).reset_index(drop=True)
    log(f"  Dedup: {before:,} -> {len(df):,} ({before - len(df):,} tekrar atıldı)")

    if args.sample:
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
        log(f"  ÖRNEKLEM modu: {len(df):,} satır")

    y = df["rating"].astype(int).values
    log("Sınıf dağılımı:")
    for r in [1, 2, 3, 4, 5]:
        c = int((y == r).sum())
        log(f"  {r}* -> {c:>7,} ({c/len(y)*100:5.2f}%)")

    # ── Faz 1: Özellikler ──────────────────────────────────────
    log("\nTF-IDF (word 1-2 gram) kuruluyor...")
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=5,
        max_features=120_000,
        sublinear_tf=True,
    )
    log("TF-IDF (char_wb 2-5 gram) kuruluyor...")
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=5,
        max_features=200_000,
        sublinear_tf=True,
    )

    # Split önce yapılır; vectorizer SADECE train'e fit edilir (leakage yok)
    idx = np.arange(len(df))
    idx_tr, idx_te = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y
    )
    txt = df["review_clean"].values

    log("Vektörler train'e fit ediliyor...")
    Xw_tr = word_vec.fit_transform(txt[idx_tr])
    Xc_tr = char_vec.fit_transform(txt[idx_tr])
    Xw_te = word_vec.transform(txt[idx_te])
    Xc_te = char_vec.transform(txt[idx_te])

    # Metadata
    scaler = StandardScaler()
    meta_tr = sp.csr_matrix(scaler.fit_transform(df.iloc[idx_tr][META_COLS]))
    meta_te = sp.csr_matrix(scaler.transform(df.iloc[idx_te][META_COLS]))

    X_tr = hstack([Xw_tr, Xc_tr, meta_tr]).tocsr()
    X_te = hstack([Xw_te, Xc_te, meta_te]).tocsr()
    y_tr, y_te = y[idx_tr], y[idx_te]
    log(f"  Train X: {X_tr.shape}  |  Test X: {X_te.shape}")

    # ── Model ──────────────────────────────────────────────────
    # SGDClassifier(log_loss): büyük seyrek veride çok hızlı + predict_proba var.
    # early_stopping ile yakınsamayı kendi kontrol eder.
    cw = "balanced" if args.balanced else None
    log(f"\nSGDClassifier(log_loss) eğitiliyor (class_weight={cw}, alpha={args.alpha})...")
    clf = SGDClassifier(
        loss="log_loss",
        alpha=args.alpha,
        class_weight=cw,
        max_iter=50,
        early_stopping=True,
        n_iter_no_change=3,
        validation_fraction=0.05,
        random_state=42,
        n_jobs=-1,
    )
    ts = time.time()
    clf.fit(X_tr, y_tr)
    log(f"  Eğitim {time.time()-ts:.1f} sn  (n_iter={clf.n_iter_})")

    # ── Değerlendirme ──────────────────────────────────────────
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
    off1 = np.mean(np.abs(y_te - y_pred) <= 1)
    mf1 = f1_score(y_te, y_pred, average="macro")

    log("\n" + "=" * 50)
    log("SONUÇLAR (test seti, dedup'lı, leakage-safe)")
    log("=" * 50)
    log(f"  Accuracy (exact) : {acc:.4f}  ({acc*100:.2f}%)")
    log(f"  ±1 Accuracy      : {off1:.4f}  ({off1*100:.2f}%)  <- komşu yıldız tolere")
    log(f"  MAE              : {mae:.4f}  yıldız")
    log(f"  Macro-F1         : {mf1:.4f}")
    log("\nClassification report:")
    log(classification_report(y_te, y_pred, target_names=["1*", "2*", "3*", "4*", "5*"]))
    log("Confusion matrix (satır=gerçek, sütun=tahmin):")
    log(str(confusion_matrix(y_te, y_pred)))

    # ── Kaydet ─────────────────────────────────────────────────
    out_path = MODELS / f"{args.name}.joblib"
    dump(
        {"word_vec": word_vec, "char_vec": char_vec, "scaler": scaler,
         "clf": clf, "meta_cols": META_COLS},
        out_path,
    )
    log(f"\nModel kaydedildi -> {out_path}")
    log(f"Toplam süre: {time.time()-t0:.1f} sn")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=1e-5,
                   help="SGD regularization (küçük=daha esnek)")
    p.add_argument("--balanced", action="store_true",
                   help="class_weight=balanced (macro-F1'i artırır, accuracy'yi düşürür)")
    p.add_argument("--name", type=str, default="baseline",
                   help="models/<name>.joblib olarak kaydet")
    p.add_argument("--sample", type=int, default=None,
                   help="Hızlı deneme için N satır örneklem")
    main(p.parse_args())
