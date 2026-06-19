"""Kayıtlı baseline modelini test setinde değerlendirir ve raporu DOSYAYA yazar.

train_baseline.py ile AYNI temizlik + dedup + split (random_state=42) kullanır,
böylece modelin hiç görmediği test seti birebir aynıdır.

Kullanım:
    python evaluate.py                  # models/baseline.joblib
    python evaluate.py baseline_balanced
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import load
from scipy.sparse import hstack
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, mean_absolute_error)
from sklearn.model_selection import train_test_split

from text_utils import light_clean

ROOT = Path(__file__).resolve().parent.parent          # uygulama/
CSV = ROOT.parent / "turkish_ecommerce_reviews.csv"     # proje kökündeki veri
RESULTS = ROOT / "results"
STARS = np.array([1, 2, 3, 4, 5])


def main(name):
    RESULTS.mkdir(exist_ok=True)
    bundle = load(ROOT / "models" / f"{name}.joblib")

    # --- train_baseline ile birebir aynı veri hazırlığı ---
    df = pd.read_csv(CSV).rename(columns={"Rating (Star)": "rating"})
    df = df.dropna(subset=["Review", "rating"])
    df["review_clean"] = df["Review"].astype(str).map(light_clean)
    df = df[df["review_clean"].str.len() >= 2]
    df = df.drop_duplicates(subset=["review_clean"]).reset_index(drop=True)
    y = df["rating"].astype(int).values

    idx = np.arange(len(df))
    _, te = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    txt = df["review_clean"].values

    Xw = bundle["word_vec"].transform(txt[te])
    Xc = bundle["char_vec"].transform(txt[te])
    meta = bundle["scaler"].transform(df.iloc[te][bundle["meta_cols"]])
    X = hstack([Xw, Xc, sp.csr_matrix(meta)]).tocsr()
    y_te = y[te]

    clf = bundle["clf"]
    y_pred = clf.predict(X)
    probs = clf.predict_proba(X)
    ev = probs @ STARS

    acc = accuracy_score(y_te, y_pred)
    off1 = np.mean(np.abs(y_te - y_pred) <= 1)
    mae = mean_absolute_error(y_te, y_pred)
    mf1 = f1_score(y_te, y_pred, average="macro")

    lines = []
    lines.append(f"MODEL: models/{name}.joblib")
    lines.append(f"Test seti: {len(y_te):,} yorum (dedup'lı, leakage-safe, random_state=42)")
    lines.append("")
    lines.append("=== ÖZET METRİKLER ===")
    lines.append(f"  Accuracy (exact) : {acc:.4f}  ({acc*100:.2f}%)")
    lines.append(f"  +-1 Accuracy     : {off1:.4f}  ({off1*100:.2f}%)")
    lines.append(f"  MAE              : {mae:.4f} yıldız")
    lines.append(f"  Macro-F1         : {mf1:.4f}")
    lines.append("")
    lines.append(f"  Gerçek ortalama puan      : {y_te.mean():.3f}")
    lines.append(f"  Tahmin ortalama (argmax)  : {y_pred.mean():.3f}")
    lines.append(f"  Tahmin ortalama (E-değer) : {ev.mean():.3f}  <- gerçeğe en yakın")
    lines.append("")
    lines.append("=== CLASSIFICATION REPORT ===")
    lines.append(classification_report(y_te, y_pred,
                 target_names=["1*", "2*", "3*", "4*", "5*"], digits=3))
    lines.append("=== CONFUSION MATRIX (satır=gerçek, sütun=tahmin) ===")
    cm = confusion_matrix(y_te, y_pred)
    lines.append("        " + "".join(f"{s}*".rjust(8) for s in [1, 2, 3, 4, 5]))
    for s, row in zip([1, 2, 3, 4, 5], cm):
        lines.append(f"   {s}*  " + "".join(f"{v:,}".rjust(8) for v in row))

    report = "\n".join(lines)
    print(report)
    out = RESULTS / f"{name}_report.txt"
    out.write_text(report, encoding="utf-8")
    print(f"\n--> Rapor kaydedildi: {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "baseline")
