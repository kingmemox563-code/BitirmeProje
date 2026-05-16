# 3D Stereo Hacim Ölçüm Sistemi

Bu proje, stereo görüş (stereo vision) teknolojisi kullanarak nesnelerin 3 boyutlu hacmini, fiziksel boyutlarını ve tahmini ağırlığını hesaplayan gelişmiş bir analiz platformudur.

## 🚀 Özellikler

- **Stereo Vision Derinlik Analizi:** İki farklı açıdan çekilen fotoğrafları kullanarak yüksek hassasiyetli derinlik haritaları oluşturur.
- **YOLOv8 Entegrasyonu:** Nesne tespiti ve sınıflandırma için en güncel yapay zeka modellerini kullanır.
- **Otomatik Kalibrasyon:** Referans bir nesne kullanarak milimetrik hassasiyette gerçek dünya ölçümleri yapabilme.
- **3D Görselleştirme:** Nokta bulutu (point cloud) ve topografik ısı haritaları ile görsel analiz.
- **Profesyonel Raporlama:** Tüm ölçüm sonuçlarını ve görselleri içeren PDF raporları oluşturma.
- **Kullanıcı Dostu Arayüz:** Modern ve dinamik bir GUI (Tkinter/CustomTkinter tabanlı).

## 🛠️ Kurulum

1. Depoyu klonlayın:
   ```bash
   git clone https://github.com/kingmemox563-code/BitirmeProje.git
   cd BitirmeProje
   ```

2. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

3. Uygulamayı başlatın:
   ```bash
   python gui_app.py
   ```

## 📋 Kullanım Kuralları

En iyi sonuçlar için:
- **Paralel Hareket:** İkinci fotoğrafı çekerken kamerayı sadece sağa kaydırın (döndürmeyin).
- **Baseline:** İki çekim arası mesafe 5-10 cm olmalıdır.
- **Yüzey:** Dokulu ve mat yüzeylerde en iyi sonuçlar alınır.

## 📄 Lisans

Bu proje eğitim ve araştırma amaçlı geliştirilmiştir.

---
*Geliştirici: Antigravity AI Analiz Sistemleri*
