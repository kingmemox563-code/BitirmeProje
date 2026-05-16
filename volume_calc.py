import cv2
import numpy as np

class VolumeCalculator:
    def __init__(self, pixel_cm_ratio=0.1, disparity_to_depth_scale=0.1):
        """
        Kamera kalibre edilmediği için bu değerler tahminseldir (Approximation).
        Gerçek bir projede bu değerler D = f*B / disparity formülünden gelir.
        
        pixel_cm_ratio: X ve Y ekseninde 1 pikselin kaç cm'ye denk geldiği
        disparity_to_depth_scale: Disparity (kayma) miktarını Z ekseni derinliğine (cm) çeviren ölçek
        """
        self.px_to_cm = pixel_cm_ratio
        self.disp_to_depth = disparity_to_depth_scale

    def auto_calibrate(self, reference_px_w, reference_px_h, real_cm_w, real_cm_h):
        """
        Referans bir nesnenin (Örn: Kart) piksel boyutu ve gerçek boyutu verilince
        hem pixel_cm_ratio hem de disp_to_depth değerlerini günceller.
        """
        old_ratio = self.px_to_cm
        
        ratio_w = real_cm_w / reference_px_w if reference_px_w > 0 else old_ratio
        ratio_h = real_cm_h / reference_px_h if reference_px_h > 0 else old_ratio
        
        new_ratio = (ratio_w + ratio_h) / 2
        
        # Derinlik ölçeğini de orantılı olarak güncelle (Z ekseni de XY ile ölçeklenmeli)
        scaling_factor = new_ratio / old_ratio
        self.disp_to_depth *= scaling_factor
        
        self.px_to_cm = new_ratio
        return self.px_to_cm

    def calculate(self, base_image, disparity_map, foreground_threshold=15.0, roi=None, density=1.0):
        # Arka plan ayıklama (Segmentation)
        # Sadece belirli bir disparity (derinlik) seviyesinden daha yüksek olan pikselleri "nesne" kabul et.
        
        # Disparity değerlerini pozitif yap (gürültü filtreleme)
        disparity_map = np.where(disparity_map < 0, 0, disparity_map)
        
        # ROI Uygulama (Eğer seçilmişse)
        if roi is not None:
            x1, y1, x2, y2 = roi
            roi_mask = np.zeros(disparity_map.shape, dtype=bool)
            roi_mask[y1:y2, x1:x2] = True
            disparity_map = np.where(roi_mask, disparity_map, 0)
        
        # EKSRA FİLTRELEME: Derinlik haritasındaki kumlanmayı (speckle noise) azaltmak için Median Blur uygula
        disp_smooth = cv2.medianBlur(disparity_map, 5)
        
        # Dinamik eşik değeri (Arayüzden gelen)
        mask = disp_smooth > foreground_threshold
        
        # Ekstra Filtre: Maske üzerindeki küçük delikleri kapatmak ve küçük pürüzleri yok etmek için morfolojik işlemler (Morphological Ops)
        kernel = np.ones((5, 5), np.uint8)
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel) # küçük gürültüleri sil
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel) # delikleri kapat
        mask = mask_uint8 > 0
        
        # Nesnenin Z yüksekliği (Hacim hesabı için)
        object_z_values = disp_smooth[mask] * self.disp_to_depth
        
        if len(object_z_values) == 0:
            return 0, base_image, {"area_pixels": 0, "mean_depth": 0, "max_depth": 0, "center_of_mass": (0, 0), "dims": (0,0,0), "mass": 0}
            
        # Alan hesaplaması: 1 pikselin XY düzlemindeki cm^2 alanı
        pixel_area_cm2 = self.px_to_cm ** 2
        
        # Toplam Hacim: Her bir nesne pikselinin alanı ile Z yüksekliğinin çarpımı (İntegral)
        total_volume_cm3 = np.sum(object_z_values * pixel_area_cm2)
        
        # Kütle Hesabı
        estimated_mass_g = total_volume_cm3 * density
        
        # Ekstra İstatistikler Seçme
        area_pixels = np.sum(mask)
        mean_depth = np.mean(object_z_values) if area_pixels > 0 else 0
        max_depth = np.max(object_z_values) if area_pixels > 0 else 0
        
        # Görselleştirme: Tespit edilen nesneyi boya
        viz_image = base_image.copy()
        
        # Saydam yeşil maske (Overlay) daha belirgin olması için
        mask_layer = np.zeros_like(viz_image)
        mask_layer[mask] = [0, 255, 0] # BGR (Green)
        
        cv2.addWeighted(mask_layer, 0.4, viz_image, 1 - 0.4, 0, viz_image)
        
        # Sınırları (Kontur) Kırmızı Çiz çizerek nesneyi net olarak ayır
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(viz_image, contours, -1, (0, 0, 255), 2)
        
        # Ağırlık Merkezi (Center of Mass) ve Nesne Tespiti Bounding Box
        cx, cy = 0, 0
        estimated_distance_cm = 0.0
        dims_cm = (0, 0, 0) # W, H, L
        
        if len(contours) > 0:
            # En büyük konturu al (Ana nesne)
            c = max(contours, key=cv2.contourArea)
            
            # 1. Bounding Box Çizimi (Yeşil Dikdörtgen)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(viz_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(viz_image, "Nesne", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Gerçek Boyutlar (cm)
            dim_w_cm = w * self.px_to_cm
            dim_h_cm = h * self.px_to_cm
            dim_l_cm = max_depth # Nesnenin en derin (yüksek) noktası
            dims_cm = (dim_w_cm, dim_h_cm, dim_l_cm)
            
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Merkeze bir nokta ve hedef işareti çiz
                cv2.circle(viz_image, (cx, cy), 5, (255, 0, 0), -1)
                
            # Kameraya Uzaklık Tahmini
            mean_disparity = np.mean(disp_smooth[mask])
            if mean_disparity > 0:
                estimated_distance_cm = 1500.0 / mean_disparity
                cv2.putText(viz_image, f"Mesafe: ~{estimated_distance_cm:.1f}cm", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            # Boyut Yazısı
            dim_text = f"{dim_w_cm:.1f}x{dim_h_cm:.1f}x{dim_l_cm:.1f} cm"
            cv2.putText(viz_image, dim_text, (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        stats = {
            "area_pixels": int(area_pixels),
            "mean_depth": float(mean_depth),
            "max_depth": float(max_depth),
            "center_of_mass": (cx, cy),
            "estimated_distance_cm": float(estimated_distance_cm),
            "dims_cm": dims_cm,
            "mass_g": float(estimated_mass_g)
        }
        
        return total_volume_cm3, viz_image, stats

    def get_topographic_heatmap(self, disparity_map, colormap=cv2.COLORMAP_JET):
        """
        Derinlik haritasını ısı haritasına çevirir ve üzerine kontur çizgileri ekler.
        """
        # Normalize et
        disp_norm = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Renklendir
        heatmap = cv2.applyColorMap(disp_norm, colormap)
        
        # Kontur çizgileri ekle (Topografik görünüm için)
        gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        # Belirli aralıklarla eşikleme yaparak çizgiler oluştur
        for level in range(0, 256, 32):
            _, thresh = cv2.threshold(disp_norm, level, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(heatmap, contours, -1, (255, 255, 255), 1)
            
        return heatmap
