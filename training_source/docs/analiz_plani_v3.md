# Analiz planı v3: Paralel planlar B ve C (DONDURULMUŞ)

Sürüm 3.0. Bu plan, B ve C planlarının **hiçbir koşusu başlamadan önce** Git'e işlenmiştir. Aşama 1 (v1) ve Aşama 2 (v2) planları ve sonuçları değiştirilmeden raporlanır. Bu planın bütün varyantları raporlanır; hiçbiri gizlenmez.

## 1. Gerekçe
Aşama 1'de dış P hatası düşük duyarlılık (kaçırılan P) biçimindeydi; PPV ise çok yüksekti. Bu örüntüye iki ek açıklama getiriyoruz ve her biri için ayrı bir müdahale planlıyoruz:
- **B, sınıf dengesizliği ve karar eşiği:** P sınıfı örneklerin küçük bir kısmını oluşturuyor. Çapraz entropi, belirsiz P bölgelerini "arka plan" tarafına itiyor.
- **C, tek kaynaklı eğitim verisi:** LUDB tek merkez ve tek cihazdan geliyor. Farklı cihaz, derivasyon ve popülasyonlardan gelen etiketli veri, dağılım kaymasını azaltabilir.

## 2. Varyantlar (hepsi Aşama 2 ayrıştırma artırmasıyla, p = 0,5)
| Etiket | Müdahale | Sabit parametreler |
|---|---|---|
| `s3pw` (Plan B) | P sınıfı CE ağırlığı + P için yumuşak Dice | `--p_weight 3 --p_dice 0.5` |
| `s3xdb` (Plan C) | QTDB'nin 77 kaydıyla ek eğitim (yalnızca eğitimde) | `--xdb_qtdb 4` (4 kat çoğaltma, kesitlerin ≈ %25'i) |

- **Plan C'nin QTDB kaynağı:** BUT PDB ve pwave ile kaynak veri tabanı paylaşan bütün QTDB kayıtları eğitim dışında bırakılır: sel1xx ve sel2xx (MIT-BIH Aritmi, 15 kayıt), sel8xx (MIT-BIH SVDB, 13 kayıt). `src/xdb.py` dosyasına bakın.
- **QTDB'de etiket maskesi:** yalnızca etiketli vuruş pencereleri kullanılır. T başlangıcı yoksa QRS bitişinden T bitişine kadar olan bölge yoksayılır.
- **Modeller:** CNN-BiLSTM, U-Net-KAN, DEQ. Her biri 5 katla eğitilir; tohum 1; Aşama 1 öğrenme hızları ve eğitim bütçesi.
- **Parametreler** önceden sabitlenmiştir; LUDB dışı veride ayar yapılmaz.

## 3. Uç noktalar ve istatistik
- **Ana karşılaştırmalar:** her varyant × mimari için iki karşılaştırma yapılır, toplam 6 + 6:
  - (a) aynı mimarinin Aşama 1 modeline karşı (artırmasız, ağırlıksız);
  - (b) aynı mimarinin Aşama 2 modeline karşı (yalnızca artırma).
- **Ölçü:** BUT PDB P_peak F1, ±150 ms, her derivasyon havuzlanmış.
- **Test:** kişi düzeyinde eşleştirilmiş bootstrap (10 000 tekrar). Holm düzeltmesi, (a) ve (b) ailelerinin her birinde ayrı ayrı uygulanır.
- **Önerilen yapılandırmanın seçimi** (test setine göre seçimi önlemek için):
  - Aşama 1, 2 ve 3'ün bütün derin varyantları arasından, **pwave P_peak F1'i en yüksek olan** yapılandırma seçilir. pwave burada geliştirme seti işlevi görür.
  - Seçilen yapılandırmanın nihai iddiası **çakışmasız BUT PDB** (pwave ile ortak kaynak kesitleri çıkarılmış 41 kayıt) üzerinde, en iyi Aşama 1 temel modeline (CNN-BiLSTM) karşı, tek bir test olarak yapılır.
- **Plan C'nin QTDB değerlendirmesi:** yalnızca eğitimde kullanılmayan 28 kayıtta yapılır.
- **İkincil uç noktalar:**
  - LUDB iç testi aşağı kalmama (−1 puan);
  - AV blok alt grubunda Se ve F1;
  - ±50 ms P_peak;
  - QTDB P_on ve P_off.

## 4. Plan dışı değişiklik günlüğü
| Tarih | Değişiklik | Gerekçe |
|---|---|---|
| 2026-11-23 22:50 | **Plan dışı (kapsam daraltma).** Plan B, C ve D'nin DEQ varyantları 0. kattan (s3xdb'de 1. kattan) sonra durduruldu; kalan DEQ koşuları başlatılmadı. Plan B, C ve D bu nedenle yalnızca CNN-BiLSTM ve U-Net-KAN için 5 katla tamamlandı; DEQ'nin kısmi kat sonuçları raporlanır ama topluluk değerlendirmesine girmez. | Kullanıcının belirlediği 8 saatlik süre: DEQ koşu başına en uzun model (~55 dk, toplamın yarısı). Şeffaflık notu: DEQ Aşama 2'de artırmadan en az yararlanan mimariydi; bu bilgi kararda rol oynadı. 'Mimariden bağımsızlık' iddiası, dört mimarinin tamamlandığı Aşama 2'ye dayanır. Seçim kuralı (pwave) ve doğrulama testleri değişmedi; aday kümesi tamamlanmış topluluklardır. |
| 2026-11-23 23:00 | **Yukarıdaki kapsam daraltması geri alındı.** Kullanıcı süre sınırını kaldırdı; DEQ varyantları da 5 katla tamamlanacak. Hiçbir sonuç daraltma sırasında görülmedi veya kullanılmadı. | Dondurulmuş planın eksiksiz uygulanması. |
