# Analiz planı v2: Aşama 2, P–QRS ayrıştırma artırması (DONDURULMUŞ)

Sürüm 2.0. Bu dosya, Aşama 2'nin **hiçbir eğitimi başlamadan önce** Git'e işlenerek dondurulmuştur (işleme özeti Git geçmişinde). Aşama 1'in planı (`docs/analiz_plani.md`, v1.0) ve sonuçları değiştirilmeden makalede raporlanır.

## 1. Gerekçe (Aşama 1'den)
Aşama 1'in dondurulmuş birincil analizinde LUDB'de eğitilen bütün derin modeller BUT PDB'de P tepelerini çok yüksek PPV (≥ %97) ama düşük duyarlılıkla (%73–77) buldu. Kayıplar en çok AV blokta yoğunlaştı (Şekil 2: III. derece blokta bağımsız P'lerin hiçbiri bulunmadı). 12 derivasyon füzyonu P'yi düzeltmedi.

**Mekanizma hipotezi:** LUDB'de P neredeyse her zaman QRS'den 120–200 ms önce yer alıyor. Modeller "P = QRS'den hemen önceki dalga" kısayolunu öğreniyor. Aşama 2 bu kısayolu eğitim verisinde kırmayı hedefliyor.

**Açık beyan:** Aşama 2 yöntemi, Aşama 1'in dış veri sonuçları görüldükten sonra tasarlandı. Dış setlerde hiçbir parametre ayarlanmayacak; artırmanın bütün parametreleri aşağıda önceden sabitlendi. Bu durum makalede sınırlılık olarak yazılacak.

## 2. Yöntem: P–QRS ayrıştırma artırması (`src/dissoc_aug.py`)
Her eğitim kesitine olasılık **p = 0,5** ile uygulanır. P şablonları aynı kişinin aynı derivasyonundaki etiketli P'lerden alınır; başka kişi veya dış veri kullanılmaz.
- **Silme:** kesitteki her P, **0,3** olasılıkla başlangıç ve bitişini birleştiren doğru ve küçük gürültüyle (σ = 0,02) değiştirilir ve "arka plan" etiketi alır. Bu, P'nin olmadığı vuruşları (nodal ritim, AF benzeri) taklit eder.
- **Ekleme:** 1–**3** (düzgün dağılımlı) ek P şablonu, genlik × U(0,7; 1,3) ile, QRS'ye ve yoksay bölgelerine **40 ms**'den yakın olmayan rastgele konumlara eklenir. Konum T dalgası ve ST/TP segmenti olabilir; bu, bağımsız P'yi, T üstündeki P'yi ve iletilmeyen P'yi taklit eder. Şablonun kapladığı örnekler "P" etiketi alır; T ile çakışmada P önceliklidir.
- Diğer bütün eğitim ayarları Aşama 1 ile aynıdır: bütçe, öğrenme hızları (`configs/chosen_lr.json`), olağan veri artırma, son işlem, değerlendirme kuralları (plan dışı düzeltmeler dahil) ve bölmeler.

## 3. Modeller
U-Net, CNN-BiLSTM (Aşama 1'in en iyi temel modeli), U-Net-KAN (B-spline, G = 8, k = 3) ve DEQ (örtük). Her biri 5 katla eğitilir (tohum 1, etiket `s2dissoc`); sonuçlar 5 modelli toplulukla alınır. ConvKAN, işlemci bütçesi ve Aşama 1'deki zayıf LUDB sonucu nedeniyle Aşama 2'ye alınmadı.

## 4. Hipotezler ve uç noktalar
- **H4 (birincil):** Ayrıştırma artırması, aynı mimarinin artırmasız modeline göre BUT PDB P_peak F1'ini (±150 ms, her derivasyon) artırır.
  - Test: 4 mimarinin her biri için F1(artırmalı) − F1(artırmasız). Kişi düzeyinde eşleştirilmiş bootstrap (10 000 tekrar), iki yönlü p; 4 test için Holm düzeltmesi.
  - Destekleyici: kişi başına F1 üzerinde Wilcoxon.
- **H5 (güvenlik):** Artırma LUDB iç testinde 6 başlangıç/bitiş noktasının ortalama F1'ini bozmaz (aşağı kalmama sınırı −1,0 puan, %95 GA alt sınırı).
- **H6 (keşifsel):** KAN ve DEQ artırmadan CNN-BiLSTM'den daha çok yararlanır. Ölçüt: farkların farkı, eşleştirilmiş bootstrap.
- **Anahtar ikincil uç noktalar:**
  - BUT PDB P_peak duyarlılığı ve PPV'si;
  - BUT PDB AV blok alt grubunda Se ve F1;
  - pwave P_peak F1; QTDB P_on/P_off F1; çakışmasız BUT PDB;
  - ±50 ms P_peak;
  - dalgacık yöntemiyle karşılaştırma.
- **Raporlama kuralı:** Aşama 1 sonuçları olduğu gibi raporlanır. H4 desteklenmezse bu da raporlanır.

## 5. Plan dışı değişiklik günlüğü
| Tarih | Değişiklik | Gerekçe |
|---|---|---|
| – | – | – |
