# Klasik NLP Çalışması — Türkçe E-Ticaret Yorumları

Yorum metninden 1–5 yıldız tahmini yapan klasik metin madenciliği çalışması.

## İçerik

| Dosya | Açıklama |
|------|----------|
| `Turkish_e-commerce_reviews_text_mining.ipynb` | Ana çalışma defteri. Üç bölüm: (1) Klasik pipeline — EDA, ön işleme, TF-IDF, Logistic Regression, yorumlanabilirlik; (2) Sızıntısız değerlendirme ve veri-odaklı özellik seçimi deneyi; (3) Stemming, Naive Bayes karşılaştırması ve kelime bulutu. |
| `stopwords.txt` | İlk stopword listesi. |
| `stopwords_final.txt` | Ön işlemede kullanılan nihai stopword listesi. |
| `words_all.txt` | Tüm yorumlardaki kelimelerin frekans listesi. |
| `words_200plus.txt` | En az 200 kez geçen kelimeler. |
| `top_bigrams.txt` | En yüksek TF-IDF skorlu ikili kelime grupları (bigram). |

## Çalıştırma

Defter Google Colab için hazırlanmıştır (veri Google Drive'dan okunur).
Hücreler yukarıdan aşağıya sırayla çalıştırılır: Bölüm 1 → Bölüm 2 → Bölüm 3.

## Veri

`turkish_ecommerce_reviews.csv` — 272.216 Hepsiburada yorumu
(kolonlar: yıldız puanı, yorum metni ve metin tabanlı metadata özellikleri).
