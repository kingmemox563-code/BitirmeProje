# 3D Stereo Hacim Ölçüm Sistemi - Kullanım Kılavuzu

Bu yazılım, iki farklı açıdan çekilmiş fotoğrafları kullanarak nesnelerin 3 boyutlu hacmini, ağırlığını ve fiziksel boyutlarını hesaplayan gelişmiş bir analiz platformudur.

---

## 1. Hızlı Başlangıç
1.  `python gui_app.py` komutuyla uygulamayı başlatın.
2.  **Sol Görüntüyü Çek (L)** butonuna basın.
3.  Kamerayı 5-10 cm sağa kaydırın (Döndürmeden!) ve **Sağ Görüntüyü Çek (R)** butonuna basın.
4.  **Hizala ve Hesapla** butonuna tıklayın.

---

## 2. En İyi Sonuç İçin Altın Kurallar (Önemli!)
Stereo Vision teknolojisi hassas bir matematiksel modele dayanır. Doğru ölçüm için şu kurallara uyun:

*   **Paralel Hareket:** İkinci fotoğrafı çekerken telefonu bir tren rayı üzerindeymiş gibi sadece sağa kaydırın. Telefonu sağa yatırmak veya döndürmek (pan yapmak) ölçümün hatalı çıkmasına neden olur.
*   **Mesafe:** İki fotoğraf arasındaki mesafe (baseline) yaklaşık 5-10 cm olmalıdır.
*   **Işık ve Yüzey:** Mat ve dokulu nesneler (kutu, meyve, ayakkabı) en iyi sonucu verir. Ayna, cam veya saf beyaz pürüzsüz yüzeylerde sistem derinlik algılamakta zorlanabilir.
*   **Odaklama:** Her iki fotoğrafta da nesnenin odakta ve net olduğundan emin olun.

---

## 3. Gelişmiş Özellikler

### Yapay Zeka (AI) Tespiti
*   **Gelişmiş Ayarlar** sekmesinden "Yapay Zeka Tespiti"ni aktif hale getirirseniz, sistem sahnede tanıdığı nesneleri (elma, şişe, bardak vb.) otomatik olarak seçer ve materyal yoğunluğunu kendisi ayarlar.
*   İnsanlar (person) sistem tarafından otomatik olarak elenir.

### Akıllı Tıklama (Smart Click)
*   Hesaplama tamamlandıktan sonra, derinlik haritası üzerinde ölçmek istediğiniz nesneye **tek tık** yapmanız yeterlidir. Sistem nesnenin sınırlarını otomatik olarak belirleyecektir.

### Otomatik Kalibrasyon
Ölçümlerin milimetrik hassasiyette olması için:
1.  Sahneye boyutu bilinen bir referans (örn: kredi kartı 8.5cm) koyun.
2.  Kartın üzerini farenizle seçin.
3.  Ayarlar sekmesine kartın gerçek genişliğini (8.5) girip **"Kalibre Et"** deyin.
4.  Sistem artık bu referansa göre tüm sahneyi yeniden ölçeklendirecektir.

---

## 4. Analiz ve Raporlama
*   **3D Görünüm:** Nesnenin derinlik haritasını 3 boyutlu bir nokta bulutu olarak inceleyebilirsiniz.
*   **Topografik Isı Haritası:** Nesne derinliğini coğrafi harita görünümüyle analiz edebilirsiniz.
*   **PDF Raporu:** "Raporu Kaydet" butonu; tüm ölçümleri, 3D görselleri ve analiz maskelerini içeren profesyonel bir PDF dosyası oluşturur (`raporlar` klasörüne kaydedilir).

---

## 5. Hata Giderme
*   **"Ortak Nokta Bulunamadı":** Nesneyi daha aydınlık bir yere alın veya iki fotoğraf arasındaki hareketi daha küçük tutun.
*   **"Değerler Çok Yüksek":** Muhtemelen kalibrasyon yapılmamıştır veya fotoğraf çekerken telefon çok fazla döndürülmüştür. Kalibrasyon özelliğini kullanın.

---
*Geliştirici: Antigravity AI Analiz Sistemleri*
