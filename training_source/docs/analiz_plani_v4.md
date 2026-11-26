# Analiz planı v4: Ek veri tabanları; yeni dış testler ve Plan D (DONDURULMUŞ)

Sürüm 4.0. Herhangi bir model yeni veri tabanlarında değerlendirilmeden ve Plan D eğitimi başlamadan önce Git'e işlenmiştir. Ek veriler kullanıcı onayıyla (2026-11-23) indirildi. Kaynak, lisans ve bütünlük kaydı `docs/ek_veri_manifest.md` dosyasındadır.

## 1. Veri rolleri
| Veri | Kayıt | Etiket | Rol |
|---|---|---|---|
| **Zhejiang PVC/VT** (Ningbo; Jiménez-Pérez etiketleri; ham sinyal, 2000 Hz) | 334 | Global P/QRS/T başlangıç-bitiş | **Dış test**. Hiçbir eğitim, ayar ya da seçimde kullanılmaz. |
| **ISP** (ISP RAS, 1000 Hz) | 475 | Global P/QRS/T başlangıç-bitiş | **Dış test** (aynı kurumdan Multi-Expert'in bir kısmı eğitimde, bkz. aşağısı) |
| **RDB** (Chapman/Shaoxing, 500 Hz) | 2399 | Derivasyon başına P/QRS/T başlangıç-bitiş | **Plan D eğitimi** |
| **Multi-Expert (ISP-MED)** | 1527 (derivasyon II) | P/QRS/T başlangıç-tepe-bitiş | **Plan D eğitimi.** Yalnızca ISP ile çakışmayan **1065** kayıt kullanılır. |
| PTB-XL+ | 21 799 | Otomatik (ECGDeli) | Bu planda kullanılmaz (ön eğitim için yedek) |

- **Sızıntı kontrolü:** ISP'nin 462 kaydı, Multi-Expert'te birebir aynı sinyal olarak bulundu (derivasyon II korelasyonu > 0,9; `configs/me_isp_overlap.json`). Bu kayıtlar eğitimden çıkarıldı.
- **ISP sonuçlarının niteliği:** ISP sonuçları "aynı kurum, ayrık kayıtlar" olarak raporlanır. Zhejiang tamamen bağımsızdır.
- **Diğer kaynaklarla kesişme:** RDB ve Multi-Expert, dış P setleri (BUT PDB, pwave) ve QTDB kaynaklarıyla kayıt paylaşmaz (farklı kurum ve cihaz).

## 2. Yeni dış testler (bütün mevcut modellere uygulanır)
- **Modeller:**
  - Aşama 1'in 8 derin modeli ve dalgacık yöntemi;
  - Aşama 2 modelleri;
  - Plan B, C ve D modelleri (bittiklerinde).
- **Değerlendirme:** 5 modelli topluluk; ±150 ms; LUDB'deki kurallar.
  - Etiketler global olduğu için her derivasyon aynı referansla değerlendirilir ve havuzlanır.
  - İkincil: yalnızca derivasyon II.
  - Değerlendirme bölgesi: ilk ve son etiket arası.
- **Noktalar:** P_on, P_off, QRS_on, QRS_off, T_on, T_off. Bu setlerde tepe etiketi yoktur.
- **Uç noktalar:**
  - P_on ve P_off F1;
  - 6 başlangıç/bitiş noktasının ortalama F1'i;
  - ortalama |hata|.
- **İstatistik:** kişi düzeyinde bootstrap ile %95 GA. Karşılaştırmalar (her mimari için artırmalı ve artırmasız; önerilen ve CNN-BiLSTM) Holm düzeltmesiyle yapılır ve keşifsel olarak raporlanır.

## 3. Plan D: çok kaynaklı eğitim (etiket `s4multi`)
- **Eğitim verisi:** LUDB eğitim katı + RDB (2399 kayıt × 12 derivasyon) + Multi-Expert (ISP ile çakışmayan 1065 kayıt, derivasyon II).
- **Diğer ayarlar:** Aşama 2 ayrıştırma artırması (p = 0,5). Bütçe, öğrenme hızı ve son işlem Aşama 1 ile aynı.
- **Kaynak dengesi:** Örnekleme, LUDB ile ek veri havuzu arasında 50/50 yapılır. Ek havuz içinde RDB ve Multi-Expert örnekleri düzgün dağılımlı çekilir.
- **Model seçimi:** Yalnızca LUDB doğrulama katıyla, Aşama 1 ile aynı ölçütle.
- **Modeller:** CNN-BiLSTM, U-Net-KAN, DEQ; 5 kat; tohum 1.
- **Karşılaştırmalar:**
  - Aynı mimarinin Aşama 1 ve Aşama 2 modellerine karşı.
  - Dış setler: BUT PDB (birincil v2/v3 uç noktası), çakışmasız BUT PDB, pwave, QTDB, Zhejiang ve ISP.
  - Aileler içinde Holm düzeltmesi.
- **Önerilen yapılandırmanın seçimi:** v3 kuralı geçerlidir. Bütün varyantlar arasında pwave P_peak F1'i en yüksek olan seçilir. Seçilenin nihai testi, çakışmasız BUT PDB ve Zhejiang üzerinde yapılır.

## 4. Plan dışı değişiklik günlüğü
| Tarih | Değişiklik | Gerekçe |
|---|---|---|
| 2026-11-23 20:10 | **Plan dışı (veri düzeltmesi).** (1) Zhejiang etiketlerinin ham sinyalle değil, gürültüsü azaltılmış sürümle (`PVCVTECGData`) hizalı olduğu görsel kontrolle bulundu; ham sürümde kayıtların bir kısmı farklı kırpılmış. Zhejiang bu sürümle yeniden işlendi; o ana kadar alınan Zhejiang sonuçları (dalgacık, CNN-BiLSTM, U-Net) silinip yeniden hesaplandı. (2) Bütün veri setlerine etiket-sinyal hizası denetimi eklendi (`src/label_alignment_audit.py`): ham Zhejiang sürümünde 84 kayıt işaretlendi (negatif kontrol), düzeltilmiş sürümde 0. Multi-Expert kaydı 960 sınırda bulundu (oran 1,26) ve Plan D eğitiminden çıkarıldı. | Değerlendirme ve eğitim verisinin doğruluğu. Hiçbir Plan D koşusu ve hiçbir Zhejiang sonucu bu düzeltmeden sonra eski haliyle kullanılmadı. |
| 2026-11-23 22:50 | **Plan dışı (kapsam daraltma).** Plan B, C ve D'nin DEQ varyantları 0. kattan (s3xdb'de 1. kattan) sonra durduruldu; kalan DEQ koşuları başlatılmadı. Plan B, C ve D bu nedenle yalnızca CNN-BiLSTM ve U-Net-KAN için 5 katla tamamlandı; DEQ'nin kısmi kat sonuçları raporlanır ama topluluk değerlendirmesine girmez. | Kullanıcının belirlediği 8 saatlik süre: DEQ koşu başına en uzun model (~55 dk, toplamın yarısı). Şeffaflık notu: DEQ Aşama 2'de artırmadan en az yararlanan mimariydi; bu bilgi kararda rol oynadı. 'Mimariden bağımsızlık' iddiası, dört mimarinin tamamlandığı Aşama 2'ye dayanır. Seçim kuralı (pwave) ve doğrulama testleri değişmedi; aday kümesi tamamlanmış topluluklardır. |
| 2026-11-23 23:00 | **Yukarıdaki kapsam daraltması geri alındı.** Kullanıcı süre sınırını kaldırdı; DEQ varyantları da 5 katla tamamlanacak. Hiçbir sonuç daraltma sırasında görülmedi veya kullanılmadı. | Dondurulmuş planın eksiksiz uygulanması. |
| 2026-11-23 23:15 | **Plan dışı (ek analiz).** BUT PDB'de P duyarlılığı, yalnızca referans P ve QRS etiketlerine göre "QRS'e bağlı" (60–450 ms içinde referans QRS gelen ve arada başka P olmayan) ve "QRS'ten ayrışmış" (diğer bütün P'ler) olarak ayrı raporlanır (`src/coupling_analysis.py`). Eşleştirilmiş karşılaştırma için kesin McNemar testi (yalnızca toplamlar saklandığından olası bütün uyumsuz çift dağılımları üzerinden üst sınır) eklendi. | Kısayol mekanizmasının doğrudan testi. Sınıflama kuralı hiçbir model için hesaplanmadan önce yazıldı; planlı hiçbir analiz değiştirilmedi. |
