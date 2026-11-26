# Analiz planı v5: Hakem değerlendirmesi sonrası ek analizler (DONDURULMUŞ)

Sürüm 5.0 (2026-11-24). Bu plan, iki bağımsız iç hakem raporundan sonra ve **hiçbir koşusu başlamadan önce** Git'e işlenmiştir.
- Buradaki bütün analizler önceden belirlenmiş v1–v4 analizlerine eklenir; hiçbirini değiştirmez.
- Makalelerde "keşifsel / revizyon analizi" olarak etiketlenir.
- Önceki doğrulama testleri ve seçim kuralı aynen geçerlidir.

## 1. Yeni eğitimler (`src/run_stage5.py`; 5 kat, Aşama 1 öğrenme hızları ve bütçesi)
| Etiket | Mimari | Müdahale | Amaç |
|---|---|---|---|
| `s5cd` | U-Net-KAN, CNN-BiLSTM | Artırma + QTDB (Plan C) + RDB/Multi-Expert (Plan D) birlikte | "İki çözüm birlikte" sorusunu doğrudan test etmek |
| `s5ins` | U-Net-KAN | Yalnızca P ekleme (`--dissoc_mode insert`) | Artırmanın bileşen ablasyonu |
| `s5del` | U-Net-KAN | Yalnızca P silme (`--dissoc_mode delete`) | Artırmanın bileşen ablasyonu |
| tohum 2 | U-Net, U-Net-KAN, DEQ, DEQ-unrolled (Aşama 1); U-Net-KAN Aşama 2, Plan C, Plan D | Aynı yapılandırma, `--seed 2` | Tohum değişkenliği (Paper A: artırma ve Plan C/D etkisinin kararlılığı; Paper B: H1/H2b) |

## 2. Değerlendirme
Bütün yeni topluluklar `final_eval` ile değerlendirilir: LUDB, QTDB, BUT PDB ve pwave. `s5cd`, `s5ins`, `s5del` ile tohum-2 U-Net-KAN Plan D için ayrıca şunlar yapılır:
- Zhejiang/ISP değerlendirmesi;
- ilk derivasyon değerlendirmesi;
- QRS bağlantı analizi.

**Birincil keşifsel karşılaştırmalar:**
- `s5cd` ile CNN-BiLSTM Aşama 1 arasında, v3/v4 doğrulama testlerinin aynısı: çakışmasız BUT PDB P_peak F1 ve Zhejiang P ortalama F1. Holm düzeltmesi bu iki test üzerinden uygulanır.
- `s5ins` ve `s5del` ile `s2dissoc` (U-Net-KAN) arasında: ayrışmış P duyarlılığı ve BUT PDB F1.
- Tohum: her yapılandırma için tohum 1 ve tohum 2 farkı ile iki tohumun ortalaması raporlanır. Ana karşılaştırmaların (artırma etkisi, Plan C/D etkisi, H1, H2b) iki tohumda da aynı yönde olup olmadığına bakılır.

## 3. Mevcut çıktılardan yeni analizler (yeni eğitim gerektirmez)
1. **QRS bağlantı analizi, kişi düzeyinde:**
   - kayıt 231 ayrı;
   - 231 hariç kalan ayrışmış P'ler;
   - yalnızca ilk derivasyon;
   - 100 ms içinde komşu P etiketi olan referanslar hariç (olası yinelenen etiketler).
2. **Plan C kazancının kaynağa göre ayrıştırılması:** BUT PDB kayıtları MIT-BIH (360 Hz) ve LTAFDB/SVDB (128 Hz) kökenli olarak ayrılır.
3. **Seçim kuralının duyarlılığı:** alternatif kurallar (pwave ±50 ms, LUDB doğrulama) hangi modeli seçerdi?
4. **DEQ:**
   - gecikme dağılımı (iç testin tamamı);
   - iterasyon sinyali için kişi düzeyinde test;
   - entropiyi kontrol eden kısmi korelasyon;
   - yüksek hatalı derivasyonları işaretlemede AUROC (iterasyon, entropi ve en büyük softmax karşılaştırması).
5. **KAN:**
   - parametre eşitlenmiş "yalnız başlık" modelinin iç testte aşağı kalmama analizi;
   - başlık kuralının 5 katta kararlılığı;
   - P'ye seçici kanal sıfırlandığında P F1'deki düşüş.
6. **Bağımsız setlerde mimari karşılaştırmaları:** ConvKAN ile CNN-BiLSTM (Zhejiang P_on) ve DEQ ile açık yığın (ISP 6 nokta), kişi düzeyinde eşleştirilmiş bootstrap ve Holm düzeltmesiyle.
7. **Daha sıkı tolerans:** CSE uyum tablosu ve ±40 ms sonuçları Paper B'ye eklenir.

## 4. Teknik düzeltmeler (sonuçları değiştirmeyen tutarlılık düzeltmeleri)
- **DEQ toplu çıkarımı:** Anderson çözücüsünün durma ölçütü yığının tamamına uygulandığı için Zhejiang/ISP'de DEQ, diğer bütün değerlendirmelerdeki gibi derivasyon derivasyon işletilir. Toplu işletilmiş DEQ varyantları yeniden değerlendirilir.
- **`extra_tables` bootstrap tohumu:** model ve veri setine göre sabitlenir; tablolar artık model kümesinden bağımsızdır.

## 5. Plan dışı değişiklik günlüğü
| Tarih | Değişiklik | Gerekçe |
|---|---|---|
| 2026-11-24 17:40 | **Plan dışı (ek analiz).** Kişi düzeyindeki bağlantı analizi, BUT PDB'deki 188 ayrışmış P'nin 160'ının tek hastadan (231) geldiğini gösterdi. Bu nedenle aynı analiz, MIT-BIH vuruş etiketleri (`.atr`, yalnızca değerlendirme için indirildi) kullanılarak MIT-BIH P-wave setine (12 kayıt, 22 108 P; 520 ayrışmış P) genişletildi. Sınıflama kuralı BUT PDB'deki ile aynı. | Mekanizma kanıtını birden fazla hastaya dayandırmak. |
| 2026-11-24 20:40 | **Plan dışı (ek analizler, dış hakem listesi B-1/B-2).** (1) Hesap yükü eşleşmiş DEQ ikizi: aynı blok, 18 açık adım (DEQ'in LUDB medyan iterasyonu), 5 kat (`s5u18`). (2) DEQ iterasyon sinyalinin Zhejiang ve ISP'de dış doğrulaması; hata eşiği ve işaretleme eşikleri LUDB iç testinde önceden sabitlendi (`src/deq_external_difficulty.py`). | Dış hakem önerileri; önceden belirlenmiş analizler değişmedi. |
| 2026-11-24 21:10 | **Plan dışı (ek analizler, dış hakem listesi B-3; B-1 düzeltmesi).** (1) Aynı kodlayıcı üzerinde başlık karşılaştırması: doğrusal (U-Net), yalnız KAN başlığı (`abl-headonly`) ve parametre eşlemeli GELU MLP başlığı (gizli 29, 397 parametre; `abl-mlphead`); ikişer tohum (`src/run_b3.py`). (2) KAN P kanalı her katın yalnızca doğrulama kişilerinde seçilip kilitlendi; test kümesinde rastgele ve enerji eşlemeli kanal kontrolleri (`src/kan_channel_control.py`). (3) B-1'de ilk çalıştırmadan sonra iki değişiklik: bileşik hata tanımında eşik `≥ 0` yerine `> 0` oldu (ilk sürüm her derivasyonu yüksek hatalı sayıyordu); LUDB tanımının veritabanı içi tekrarı (en kötü çeyrek) betimsel olarak eklendi. | Dış hakem önerileri; önceden belirlenmiş analizler değişmedi. |
| 2026-11-24 21:15 | **Plan dışı (ek analiz, dış hakem listesi A-2).** QRS'e bağlı sahte artırma kontrolü (`--dissoc_mode sham`, `s5sham`, U-Net-KAN, 5 kat, tohum 1): ekleme modundaki aynı şablonlar ve aynı sayıda yapıştırma (1–3), ancak her şablon mevcut bir P'nin yerine, kendi merkezine konuyor; P–QRS bağlantısı korunuyor, P silinmiyor. Karşılaştırma: yalnız ekleme (`s5ins`) ile sahte kontrol; birincil sonlanım çakışmasız BUT PDB P-tepe F1, ikincil MIT-BIH P-wave F1 ve kayıt 231'deki bağlantısız P duyarlılığı. | Ayrışmanın kendisinin mi, yoksa şablon ve genlik çeşitliliğinin mi etkili olduğunu ayırmak. |
| 2026-11-25 12:30 | **Plan dışı (ek değerlendirme, dış hakem listesi A-3).** (1) BUT PDB kayıt 03 (MIT-BIH 107, pacemaker'lı tam AV blok) ikinci AV blok hastası olarak ayrıca analiz edildi; tam blokta tüm P'ler iletilmediğinden zamanlama kuralı yerine fizyolojik tanım kullanıldı (`src/record107.py`). (2) AV blok tanılı açık kayıtlarda (PTB-XL 25, Chapman–Ningbo 114; AF/flutter hariç) kör uzman P etiketlemesi için çevrimdışı araç ve protokol hazırlandı (`src/avb_annotation.py`, `annotation/`). Birincil sonlanım: bloklu P duyarlılığı (±150 ms, ortak referans: tüm okuyucuların işaretlediği P'ler, ±40 ms eşleme), model ve aşama karşılaştırması kayıt 231 ile aynı; ikincil: okuyucular arası uyum (κ). Değerlendirme betiği etiketler gelmeden yazıldı ve dondurulur (`src/avb_annotation_eval.py`). | Bloklu P bulgusunu tek hastanın ötesinde sınamak. |
