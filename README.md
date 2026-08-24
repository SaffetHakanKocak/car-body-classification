# Car Body Type Classification

EfficientNet-B2 tabanlı, görsellerden araç gövde tipini sınıflandıran bir derin öğrenme projesi. Proje; eğitilmiş modeli kullanarak tek görsel tahmini, ZIP dosyasıyla toplu test ve model performansının görsel olarak incelenmesi için bir Streamlit arayüzü sunar.

## Özellikler

- Tek bir araç görseli için sınıf ve güven skoru tahmini
- `testdata.zip` üzerinden toplu tahmin
- İsteğe bağlı `True.txt` ile doğruluk, precision, recall, F1 ve karışıklık matrisi hesaplama
- Eğitim ve test metrikleri, sınıf bazlı rapor ve eğitim grafikleri
- CUDA destekli cihazlarda GPU, diğer durumlarda CPU kullanımı
- Görselleri en-boy oranını koruyarak 224 x 224 boyutuna dönüştürme

## Sınıflar

Model aşağıdaki sekiz araç gövde tipini sınıflandırır:

`SUV` · `VAN` · `STATION_WAGON` · `MICRO` · `OPEN_WHEEL` · `SEDAN` · `HATCHBACK` · `PICK_UP`

## Kurulum

Python 3.10 veya üzeri önerilir.

```bash
git clone https://github.com/SaffetHakanKocak/Car-Body-Classification.git
cd Car-Body-Classification
python -m venv .venv
```

Sanal ortamı etkinleştirin:

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

Bağımlılıkları yükleyin:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Uygulamayı Çalıştırma

```bash
streamlit run app/app.py
```

Tarayıcıda açılan arayüzde üç çalışma alanı bulunur:

1. **Tek Görsel:** JPG, JPEG, PNG, WEBP veya BMP formatında bir görsel yükleyip tahmin alın.
2. **Toplu Test:** Görselleri içeren `testdata.zip` dosyasını yükleyin. Performans ölçümü için aynı dosya adlarını kullanan isteğe bağlı `True.txt` dosyasını da ekleyebilirsiniz.
3. **Model Değerlendirme:** Eğitim özeti, test metrikleri, sınıf bazlı sınıflandırma raporu ve grafiklerini inceleyin.

### `True.txt` Formatı

Her satırda dosya adı ve 1 tabanlı gerçek sınıf etiketi bulunmalıdır:

```text
image_001.jpg | True: 1
image_002.jpg | True: 6
```

ZIP dosyası doğrudan görselleri veya tek bir kök klasör içeren bir yapıyı barındırabilir. Uygulama ZIP içindeki alt klasörleri de tarar.

## Komut Satırı Tahmini

Tek bir görseli Streamlit arayüzü olmadan sınıflandırmak için:

```bash
cd src
python predict.py --image_path "../path/to/image.jpg"
```

Sonuç JSON formatında; tahmin edilen sınıf, 1 tabanlı etiket, güven skoru ve tüm sınıflara ait olasılıklarla yazdırılır.

## Proje Yapısı

```text
├── app/app.py                         # Streamlit uygulaması
├── models/
│   ├── best_model.pth                 # Eğitilmiş EfficientNet-B2 ağırlıkları
│   └── class_names.json               # Sınıf sırası
├── notebooks/                         # Eğitim notebook'u
├── outputs_b2_v4/                     # Metrikler, raporlar ve eğitim grafikleri
├── report/                            # LaTeX raporu ve görselleri
├── src/
│   ├── model.py                       # Model oluşturma
│   ├── predict.py                     # Tek görsel tahmini
│   ├── predict_batch.py               # Toplu tahmin
│   ├── train.py                       # Eğitim akışı
│   ├── evaluate.py                    # Model değerlendirme
│   └── prepare_*.py                   # Veri hazırlama araçları
├── requirements.txt
└── README.md
```

## Model Bilgileri

| Parametre | Değer |
| --- | --- |
| Mimari | EfficientNet-B2 |
| Sınıf sayısı | 8 |
| Girdi boyutu | 224 x 224 |
| Dropout | 0.4 |
| Maksimum epoch | 30 |
| Öğrenme oranı | 0.0001 |

Model ağırlıkları ve değerlendirme çıktıları depoya dahil edilmiştir. Yeni bir eğitim çalıştırmadan önce veri klasörlerinin ve ilgili yolların `src/config.py` içindeki ayarlarla uyumlu olduğundan emin olun.

## Model Sonuçları

`outputs_b2_v4` içindeki kayıtlı değerlendirme sonuçlarına göre model, 796 görselden oluşan test setinde aşağıdaki performansı göstermiştir:

| Metrik | Sonuç |
| --- | ---: |
| Test Accuracy | **96.98%** |
| Macro Precision | **97.10%** |
| Macro Recall | **96.99%** |
| Macro F1-Score | **97.00%** |
| En iyi Validation Macro F1 | **98.62%** |

### Sınıf Bazlı Test Sonuçları

| Sınıf | Precision | Recall | F1-Score | Görsel sayısı |
| --- | ---: | ---: | ---: | ---: |
| SUV | 0.96 | 0.94 | 0.95 | 100 |
| VAN | 0.99 | 1.00 | 1.00 | 100 |
| STATION_WAGON | 0.97 | 1.00 | 0.99 | 100 |
| MICRO | 0.98 | 0.98 | 0.98 | 96 |
| OPEN_WHEEL | 1.00 | 1.00 | 1.00 | 100 |
| SEDAN | 0.99 | 0.90 | 0.94 | 100 |
| HATCHBACK | 0.88 | 0.95 | 0.91 | 100 |
| PICK_UP | 1.00 | 0.99 | 0.99 | 100 |

Değerlendirme çıktılarının tamamı için [`outputs_b2_v4/test_metrics.json`](outputs_b2_v4/test_metrics.json) ve [`outputs_b2_v4/test_classification_report.txt`](outputs_b2_v4/test_classification_report.txt) dosyalarına bakabilirsiniz.

## Lisans ve Veri Kullanımı

Bu depo eğitim/demo amaçlı hazırlanmıştır. Veri setlerinin ve raporda kullanılan görsellerin lisans ve kullanım koşulları, ilgili kaynakların koşullarına tabidir. Bu proje için ayrıca bir lisans tanımlanmamıştır.
