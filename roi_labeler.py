import cv2
import os
import glob
from tkinter import filedialog, Tk

def select_roi_and_save(image_dir, output_dir, class_id=0):
    """
    Resimler üzerinde ROI seçimi yapar ve YOLO formatında (.txt) kaydeder.
    Format: <class_id> <x_center> <y_center> <width> <height> (normalize edilmiş)
    """
    
    # Desteklenen resim formatları
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    
    if not image_files:
        print(f"Hata: '{image_dir}' dizininde resim bulunamadı.")
        return

    # Çıktı dizinini oluştur
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Bilgi: '{output_dir}' dizini oluşturuldu.")

    print("\n--- ROI Etiketleme Aracı ---")
    print("Talimatlar:")
    print("1. Fare ile kutu çizin.")
    print("2. Seçimi onaylamak için 'SPACE' veya 'ENTER' tuşuna basın.")
    print("3. Seçimi iptal etmek için 'c' tuşuna basın.")
    print("4. Sonraki resme geçmek için 'n' tuşuna basın (ROI seçmeden).")
    print("5. Çıkmak için 'q' veya 'ESC' tuşuna basın.")
    print("---------------------------\n")

    for img_path in image_files:
        img_name = os.path.basename(img_path)
        img_base = os.path.splitext(img_name)[0]
        txt_path = os.path.join(output_dir, f"{img_base}.txt")

        # Resmi oku
        image = cv2.imread(img_path)
        if image is None:
            print(f"Hata: {img_name} okunamadı, atlanıyor.")
            continue

        h, w = image.shape[:2]
        
        # ROI Seçimi (OpenCV dahili fonksiyonu)
        # showCrosshair=True: Merkezde artı işareti gösterir
        # fromCenter=False: Seçimi köşeden başlatır
        roi = cv2.selectROI(f"ROI Secimi: {img_name}", image, showCrosshair=True, fromCenter=False)
        
        # roi: (x, y, width, height) - Sol üst köşe koordinatları ve boyut
        x, y, rw, rh = roi

        # Pencereyi kapat
        cv2.destroyWindow(f"ROI Secimi: {img_name}")

        # Eğer genişlik ve yükseklik 0 değilse (yani bir seçim yapıldıysa)
        if rw > 0 and rh > 0:
            # YOLO Formatına Dönüştürme (Normalize edilmiş değerler)
            # YOLO formatı: center_x, center_y, width, height
            x_center = (x + rw / 2.0) / w
            y_center = (y + rh / 2.0) / h
            norm_width = rw / w
            norm_height = rh / h

            # Dosyaya yaz
            with open(txt_path, 'w') as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
            
            print(f"Kaydedildi: {txt_path}")
            print(f"Seçim: x={x}, y={y}, w={rw}, h={rh}")
        else:
            print(f"Bilgi: {img_name} için seçim yapılmadı, atlanıyor.")

        # Kullanıcı çıkmak istiyor mu kontrolü (Opsiyonel: selectROI sonrası bekleme)
        # Not: selectROI kendi döngüsüne sahip, ancak biz her resimden sonra sormak isteyebiliriz.
        # selectROI sonrasında bir tuş beklemeye gerek yok, direkt döngüye devam eder.

    print("\nİşlem tamamlandı.")

if __name__ == "__main__":
    # Tkinter arayüzünü gizli olarak başlat
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True) # Pencereyi en üste getir

    print("Lütfen resimlerin olduğu klasörü seçin...")
    input_dir = filedialog.askdirectory(title="Resim Klasörünü Seçin")
    
    if not input_dir:
        print("İşlem iptal edildi (Giriş klasörü seçilmedi).")
        exit()

    print("Lütfen etiketlerin (txt) kaydedileceği klasörü seçin...")
    output_dir = filedialog.askdirectory(title="Kayıt Klasörünü Seçin")

    if not output_dir:
        print("İşlem iptal edildi (Çıkış klasörü seçilmedi).")
        exit()
    
    # Ana fonksiyonu çalıştır
    select_roi_and_save(input_dir, output_dir)
