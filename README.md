# Car Body Type Classification

Bu klasor, demo icin gerekli Streamlit arayuzu, kaynak kodlar ve final model dosyalarini icerir.

## Demo Calistirma

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Streamlit acildiktan sonra:

- `Tek Gorsel` sekmesinden tek arac gorseli ile tahmin yapilabilir.
- `Toplu Test` sekmesinden `testdata.zip` yuklenerek toplu tahmin alinabilir.
- `True.txt` yuklenirse arayuz metrikleri ve karisiklik matrisini de gosterir.
- `Model Degerlendirme` sekmesi final modelin egitim ve test ozetini gosterir.

## Onemli Dosyalar

- `app/app.py`: Streamlit demo arayuzu
- `models/best_model.pth`: EfficientNet-B2 final model agirliklari
- `models/class_names.json`: Sinif sirasi
- `src/`: egitim, degerlendirme, veri hazirlama ve tahmin scriptleri
- `report/`: LaTeX rapor dosyasi ve rapor gorselleri
