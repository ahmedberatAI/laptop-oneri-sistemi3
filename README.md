# laptop-oneri-sistemi3
Akıllı laptop öneri sistemi - Streamlit uygulaması
# 💻 Akıllı Laptop Öneri Sistemi

Bu proje, kullanıcıların ihtiyaçlarına göre kişiselleştirilmiş laptop önerileri sunan akıllı bir sistem içerir.

## 🚀 Özellikler

- **Kişiselleştirilmiş Öneriler**: Bütçe, kullanım amacı ve tercihlerinize göre laptop önerileri
- **Fırsat Ürün Tespiti**: Anomali tespiti algoritması ile avantajlı fiyatlı ürünleri bulma
- **Pazar Analizi**: Detaylı fiyat ve trend analizleri
- **İnteraktif Arayüz**: Streamlit tabanlı kullanıcı dostu web arayüzü
- **Gerçek Zamanlı Görselleştirme**: Plotly ile interaktif grafikler

## 📊 Kullanılan Teknolojiler

- **Streamlit**: Web arayüzü
- **Pandas & NumPy**: Veri işleme
- **Scikit-learn**: Makine öğrenmesi algoritmaları
- **Plotly**: İnteraktif görselleştirme

## 🛠️ Kurulum

### Yerel Çalıştırma

1. Projeyi klonlayın:
```bash
git clone https://github.com/kullaniciadi/laptop-oneri-sistemi.git
cd laptop-oneri-sistemi
```

2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

3. Veri dosyalarınızı `data/` klasörüne yerleştirin:
```
data/
├── vatan_laptop_data_cleaned.csv
├── amazon_final.csv
└── cleaned_incehesap_data.csv
```

4. Uygulamayı çalıştırın:
```bash
streamlit run app.py
```

### Streamlit Cloud Deployment

1. GitHub'da bir repository oluşturun
2. Dosyalarınızı repository'ye yükleyin
3. [share.streamlit.io](https://share.streamlit.io) adresine gidin
4. GitHub hesabınızla giriş yapın
5. Repository'nizi seçin ve deploy edin

## 📁 Proje Yapısı

```
laptop-oneri-sistemi/
│
├── app.py                 # Ana Streamlit uygulaması
├── requirements.txt       # Python paket gereksinimleri
├── README.md             # Bu dosya
├── data/                 # Veri dosyaları (isteğe bağlı)
│   ├── vatan_laptop_data_cleaned.csv
│   ├── amazon_final.csv
│   └── cleaned_incehesap_data.csv
└── .streamlit/           # Streamlit konfigürasyonu (isteğe bağlı)
    └── config.toml
```

## 🎯 Nasıl Kullanılır

1. **Kişisel Öneri Sekmesi**: 
   - Bütçenizi belirleyin
   - Kullanım amacınızı seçin
   - Önem derecelerini ayarlayın
   - Önerileri görüntüleyin

2. **Fırsat Ürünleri Sekmesi**:
   - Anomali tespiti ile bulunan avantajlı ürünleri görün
   - Fırsat skorlarını inceleyin

3. **Pazar Analizi Sekmesi**:
   - Marka dağılımı ve fiyat trendlerini analiz edin
   - İnteraktif grafiklerle piyasayı keşfedin

## 🔧 Konfigürasyon

Sistem parametrelerini `app.py` içindeki `Config` sınıfından değiştirebilirsiniz:

- **GPU/CPU Skorları**: Donanım performans puanları
- **Marka Güvenilirlik Skorları**: Marka bazlı güvenilirlik katsayıları
- **Puanlama Ağırlıkları**: Öneri algoritmasının ağırlık değerleri

## 📈 Puanlama Algoritması

Sistem, aşağıdaki faktörleri dikkate alarak puanlama yapar:

- **Fiyat Uygunluğu** (%15): Belirtilen bütçeye uygunluk
- **Kullanım Amacı** (%30): Oyun, üretkenlik, taşınabilirlik, tasarım
- **Performans** (%12): GPU ve CPU skorları
- **Donanım Özellikleri** (%10): RAM ve SSD kapasiteleri
- **Marka Güvenilirliği** (%8): Marka bazlı güvenilirlik skoru

## 🚨 Dikkat Edilmesi Gerekenler

- Veri dosyalarınız yoksa sistem demo verileri kullanır
- Streamlit Cloud'da dosya boyut sınırları vardır (25MB)
- Büyük veri setleri için Git LFS kullanmanız önerilir

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Push edin (`git push origin feature/yeni-ozellik`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 📞 İletişim

Herhangi bir sorunuz için issue açabilir veya doğrudan iletişime geçebilirsiniz.

---

⭐ **Projeyi beğendiyseniz yıldız vermeyi unutmayın!**
