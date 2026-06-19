# ⭐ Türkçe E-Ticaret Yorumları — Yıldız Puanı Sınıflandırması & Sahte Puan Tespiti

Yorum **metninden** 1–5 yıldız tahmin eden bir NLP sistemi ve bunu sunan küçük bir uygulama.
Amaç: bazı kullanıcıların yorumu öne çıkarmak için **olumsuz metin yazıp yüksek puan vermesi**
(ör. *"berbat ürün"* → 5★) gibi durumları yakalamak ve bir ürünün **metne dayalı gerçek puan
ortalamasını** hesaplamak.

<p>
<img alt="Python" src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white">
<img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white">
<img alt="BERTurk" src="https://img.shields.io/badge/BERTurk-Transformers-FFD21E?logo=huggingface&logoColor=black">
<img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white">
</p>

---

## 🎯 Öne Çıkanlar

- **Üç aşamalı modelleme:** Klasik TF-IDF + Lojistik Regresyon → geliştirilmiş doğrusal
  baseline (karakter + kelime n-gram) → **BERTurk** Transformer ince-ayarı.
- **Sahte / tutarsız puan tespiti:** Verilen yıldız ile metin tahmini arasındaki uyuşmazlığı
  işaretler (*inflated / deflated*).
- **Metne dayalı gerçek ortalama:** Argmax yerine **beklenen değer** ile, gerçek ortalamayı
  daha az yanlı yakalar.
- **Dürüst değerlendirme:** Tekrar ayıklama (dedup) + yalnızca eğitime fit ile veri sızıntısı
  kapatılır; sıralı yapıya uygun metrikler (**±1 doğruluk, MAE, makro-F1**).
- **Takılabilir servis:** Hafif `baseline` (torch'suz) ya da en iyi `berturk` backend'i tek
  ortam değişkeniyle seçilir; FastAPI + basit web arayüzü.

## 🏆 Sonuçlar (test kümesi)

| Model | Doğruluk | ±1 Doğruluk | MAE | Makro-F1 |
|-------|:--------:|:-----------:|:---:|:--------:|
| Klasik (TF-IDF + Lojistik Reg.) | %54,2 | — | 0,66 | 0,42 |
| Geliştirilmiş baseline (char+word n-gram) | %64,5 | %90,2 | 0,48 | 0,38 |
| **BERTurk (Transformer)** | **%64,5** | **%93,5** | **0,43** | **0,50** |

> BERTurk her anlamlı metrikte en iyisi; özellikle olumsuzlama ("kötü **değil**") ve azınlık
> (olumsuz) sınıfları doğru yakalar — bu da sahte-puan tespitinin kalbidir.

## 📁 Proje Yapısı

```
turkish_ecommerce_reviews/
├── klasik_nlp/                     # Klasik metin madenciliği çalışması
│   ├── Turkish_e-commerce_reviews_text_mining.ipynb
│   └── stopwords*.txt, words*.txt, top_bigrams.txt
├── berturk_egitim/
│   └── train_berturk_colab.ipynb   # BERTurk ince-ayar defteri (Colab + GPU)
├── uygulama/                       # Uygulama (en iyi modelle servis)
│   ├── src/                        #   kod (aşağıda)
│   ├── web/index.html              #   basit test arayüzü
│   ├── models/                     #   eğitilmiş modeller (baseline*.joblib, berturk_model/)
│   └── results/                    #   değerlendirme raporları
├── turkish_ecommerce_reviews.csv   # veri kümesi (272.216 yorum)
├── Proje_Raporu_IEEE.docx          # IEEE formatında proje raporu
├── requirements.txt
└── README.md
```

**`uygulama/src/` dosyaları**

| Dosya | Görev |
|-------|-------|
| `text_utils.py` | Ortak metin temizleme (eğitim + servis tutarlılığı) |
| `train_baseline.py` | Geliştirilmiş baseline'ı eğitir (`.joblib` üretir) |
| `evaluate.py` | Modeli test setinde değerlendirir → `results/` raporu |
| `experiments.py` | Baseline deneyleri (sqrt-ağırlık, beklenen değer) |
| `inference.py` | Baseline modeli yükler ve tahmin eder |
| `inference_berturk.py` | BERTurk modeli yükler (aynı arayüz) |
| `app.py` | FastAPI servisi (`/predict`, `/aggregate`, `/health`) |

## 🗂️ Veri Kümesi

`turkish_ecommerce_reviews.csv` — **272.216** Hepsiburada yorumu.
Kolonlar: yıldız puanı (1–5), yorum metni ve metinden türetilmiş üst-veri
(uzunluk, ünlem/soru sayısı, büyük harf oranı) ile ürün bilgileri.

Sınıf dağılımı **aşırı dengesiz**: 5★ %61,5 · 4★ %22,9 · 3★ %10,6 · 2★ %2,4 · 1★ %2,7.

## ⚙️ Kurulum

```bash
pip install -r requirements.txt
# BERTurk backend'i için ek olarak:
pip install torch transformers
```

## 🚀 Kullanım

### 1) Geliştirilmiş baseline'ı eğit (lokal, CPU, ~1–2 dk)
```bash
cd uygulama/src
python train_baseline.py            # accuracy odaklı
python train_baseline.py --balanced # azınlık (olumsuz) recall odaklı
```
Model `uygulama/models/<ad>.joblib` olarak kaydedilir.

### 2) API + web arayüzünü çalıştır
```bash
cd uygulama/src
python -m uvicorn app:app --port 8000               # hafif baseline (varsayılan)
```
En iyi model (BERTurk) ile (PowerShell):
```powershell
cd uygulama/src
$env:BACKEND="berturk"; python -m uvicorn app:app --port 8000
```
Tarayıcı: **http://127.0.0.1:8000/** · Sağlık kontrolü: `GET /health`

### 3) BERTurk ince-ayarı (Colab)
`berturk_egitim/train_berturk_colab.ipynb` defterini Colab'da **GPU** ile çalıştır.
Eğitilen model Drive'a kaydedilir; indirip `uygulama/models/berturk_model/` içine koyarak
servise alınır.

## 🔌 API

**`POST /predict`** — tek yorum
```json
{ "text": "ürün berbat hiç beğenmedim" }
```
```json
{ "predicted_rating": 1, "expected_rating": 1.14, "confidence": 0.93,
  "probabilities": { "1": 0.93, "2": 0.02, "3": 0.03, "4": 0.01, "5": 0.01 } }
```

**`POST /aggregate`** — yorum listesi → gerçek ortalama + sahte-puan tespiti
```json
{ "reviews": [
    { "text": "harika ürün çok memnunum", "given_rating": 5 },
    { "text": "berbat kargo geç ürün bozuk rezalet", "given_rating": 5 }
  ], "include_details": true }
```
```json
{ "count": 2, "predicted_avg": 2.79, "given_avg": 5.0,
  "suspicious_count": 1,
  "suspicious": [ { "index": 1, "given_rating": 5, "predicted_rating": 1, "type": "inflated" } ] }
```

> Örnekte ikinci yorum 5★ verilmiş ama metin olumsuz → model 1★ tahmin eder ve yorumu
> **"inflated" (şişirilmiş)** olarak işaretler; gerçek ortalama (2,79), verilen ortalamanın
> (5,0) altında doğru biçimde hesaplanır.

## 🧪 Neden BERTurk? (somut örnek)

> *"Daha iyi seçenekler mevcut zorunda değilseniz almayın ama çok da kötü değil"*

| Model | Tahmin |
|-------|:------:|
| Klasik baseline (TF-IDF) | **1★** (olumsuzlamayı kaçırır) |
| BERTurk | **3★** (kararsız/ortalama tonu doğru anlar) |

## 📊 Değerlendirme Metrikleri

Komşu yıldızlar (4★↔5★) doğal olarak karıştığından **exact doğruluk tek başına yanıltıcıdır**.
Bu yüzden **±1 doğruluk** (komşu yıldızı tolere eden), **MAE** (ortalama yıldız hatası) ve
sınıf dengesizliğine karşı **makro-F1** birlikte raporlanır. Detaylı raporlar
`uygulama/results/` altında; `python uygulama/src/evaluate.py <model>` ile üretilir.

## 🛠️ Teknolojiler

Python · scikit-learn · pandas/numpy/scipy · Hugging Face Transformers (BERTurk) · PyTorch ·
FastAPI · Uvicorn

## 📄 Rapor

Projenin IEEE formatındaki tam raporu: **`Proje_Raporu_IEEE.docx`**.
