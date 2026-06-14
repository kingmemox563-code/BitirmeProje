import cv2
import numpy as np

class VolumeCalculator:
    def __init__(self, pixel_cm_ratio=0.055, baseline_cm=10.0, focal_length_px=600.0):
        """
        pixel_cm_ratio: X ve Y ekseninde 1 pikselin kaç cm'ye denk geldiği (Kalibrasyonla güncellenir)
        baseline_cm: İki çekim arasındaki mesafe (Varsayılan 10cm)
        focal_length_px: Kameranın piksel cinsinden odak uzaklığı (Standart kameralar için ~600)
        """
        self.px_to_cm = pixel_cm_ratio
        self.baseline = baseline_cm
        self.focal_length = focal_length_px
        
        # Derinlik çarpanı: f * B 
        # Z = (f * B) / disparity
        self.depth_constant = self.focal_length * self.baseline

    def auto_calibrate(self, reference_px_w, reference_px_h, real_cm_w, real_cm_h, current_distance=None):
        """
        Referans bir nesnenin (Örn: Kart) piksel boyutu ve gerçek boyutu verilince
        hem pixel_cm_ratio hem de focal_length tahminini günceller.
        """
        ratio_w = real_cm_w / reference_px_w if reference_px_w > 0 else self.px_to_cm
        ratio_h = real_cm_h / reference_px_h if reference_px_h > 0 else self.px_to_cm
        
        self.px_to_cm = (ratio_w + ratio_h) / 2
        
        # X-Y ölçeği değiştiğinde, eğer nesnenin mevcut tahmini mesafesini biliyorsak
        # odak uzaklığını (focal length) çok daha doğru hesaplayabiliriz.
        # f = Z / px_to_cm
        if current_distance and current_distance > 0:
            # px_to_cm oranı Z / f'ye eşittir.
            # Ancak Z'nin kendisi eski focal_length üzerinden hesaplanmıştı (Z_old = f_old * B / d).
            # Gerçekte Z / f oranı px_to_cm'dir.
            # Disparity (d) sabit kalır. d = B / px_to_cm olmalıdır.
            # f = Z_real / px_to_cm. Ancak Z_real'i bilmiyoruz. 
            pass
            
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
        # ROI seçiliyse ve sabit eşik çok yüksekse, ROI içindeki değerlere göre otomatik ayarla
        effective_threshold = foreground_threshold
        if roi is not None:
            x1r, y1r, x2r, y2r = roi
            roi_region = disp_smooth[y1r:y2r, x1r:x2r]
            roi_max = np.max(roi_region) if roi_region.size > 0 else 0
            roi_mean = np.mean(roi_region[roi_region > 0]) if np.any(roi_region > 0) else 0
            
            # Eğer ROI içindeki en yüksek disparity bile eşiğin altındaysa,
            # eşiği ROI ortalamasının %30'una düşür (nesneyi kaybetme)
            if roi_max > 0 and roi_max <= foreground_threshold:
                effective_threshold = max(roi_mean * 0.3, 1.0)
            # Eğer eşik, ROI içindeki piksellerin çoğunu siliyorsa da düşür
            elif roi_max > 0:
                roi_above = np.sum(roi_region > foreground_threshold)
                roi_total = np.sum(roi_region > 0)
                if roi_total > 0 and (roi_above / roi_total) < 0.15:
                    effective_threshold = max(roi_mean * 0.3, 1.0)
        
        mask = disp_smooth > effective_threshold
        
        # Ekstra Filtre: Maske üzerindeki küçük delikleri kapatmak ve küçük pürüzleri yok etmek için morfolojik işlemler (Morphological Ops)
        kernel = np.ones((5, 5), np.uint8)
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel) # küçük gürültüleri sil
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel) # delikleri kapat
        mask = mask_uint8 > 0
        
        # Nesnenin Z yüksekliği (Hacim hesabı için)
        # Formül: Z = (f * B) / disparity
        # Nesnenin 'kalınlığını' bulmak için: Z_zemin - Z_nesne
        
        dist_map = np.zeros_like(disp_smooth, dtype=np.float32)
        valid_disp = disp_smooth > 0.5
        dist_map[valid_disp] = self.depth_constant / disp_smooth[valid_disp]
        
        # Nesnenin kameraya olan tahmini uzaklığı
        object_dist = np.median(dist_map[mask]) if np.any(mask) else 0.0

        # Zeminin ortalama uzaklığını bul
        background_mask = (disp_smooth > 2) & (~mask)
        if np.any(background_mask):
            ground_dist = np.median(dist_map[background_mask])
        else:
            ground_dist = np.max(dist_map[mask]) if np.any(mask) else 100.0
            
        # XY boyutlarını hesapla
        y_idx, x_idx = np.where(mask)
        if len(y_idx) > 0:
            w_px = np.max(x_idx) - np.min(x_idx)
            h_px = np.max(y_idx) - np.min(y_idx)
            dim_w_cm = w_px * self.px_to_cm
            dim_h_cm = h_px * self.px_to_cm
        else:
            dim_w_cm, dim_h_cm = 0, 0

        # --- FİZİKSEL KALINLIK SINIRI ---
        # Bir nesne kendi en kısa XY boyutundan daha kalın olamaz (fizik kuralı).
        # Örn: 10 cm genişliğindeki cüzdan max ~4 cm kalın olabilir.
        min_xy = min(dim_w_cm, dim_h_cm) if min(dim_w_cm, dim_h_cm) > 0 else 5.0
        max_xy = max(dim_w_cm, dim_h_cm) if max(dim_w_cm, dim_h_cm) > 0 else 10.0
        # Mutlak kalınlık üst sınırı: min boyutun %40'ı
        absolute_thickness_cap = min_xy * 0.40
        if absolute_thickness_cap < 0.3:
            absolute_thickness_cap = 0.3

        if ground_dist - object_dist > max_xy * 1.5:
            # Nesne havada tutuluyor — zemin referansı kullanılamaz
            # Gerçekçi kalınlık tahmini: min boyutun %12'si (cüzdan/telefon için uygundur)
            assumed_thickness = min_xy * 0.12
            if assumed_thickness < 0.3:
                assumed_thickness = 0.3
            
            # Yüzey pürüzü (derinlik varyasyonu) sınırı: min boyutun %8'i
            relative_heights = object_dist - dist_map[mask]
            max_variation = min_xy * 0.08
            relative_heights = np.clip(relative_heights, 0, max_variation)
            
            object_thickness_values = relative_heights + assumed_thickness
        else:
            # Nesne zemin üzerinde
            object_thickness_values = ground_dist - dist_map[mask]
            # Kalınlık sınırı: max boyutun %35'i (max 8 cm)
            max_allowed = min(max_xy * 0.35, 8.0)
            if max_allowed < 1.0:
                max_allowed = 1.0
            object_thickness_values = np.clip(object_thickness_values, 0, max_allowed)

        # Her iki durumda da mutlak fiziksel üst sınırı uygula
        object_thickness_values = np.clip(object_thickness_values, 0, absolute_thickness_cap)
        
        if len(object_thickness_values) == 0:
            # YEDEK HESAPLAMA: Stereo eşleştirme başarısız olduysa ama ROI seçiliyse,
            # ROI piksel boyutlarından hacim tahmini yap (düz ince nesne varsayımı)
            if roi is not None:
                x1, y1, x2, y2 = roi
                roi_w_px = abs(x2 - x1)
                roi_h_px = abs(y2 - y1)
                
                roi_w_cm = roi_w_px * self.px_to_cm
                roi_h_cm = roi_h_px * self.px_to_cm
                # İnce nesne varsayımı: kalınlık = en küçük boyutun %10'u (min 0.5 cm)
                roi_thickness_cm = max(min(roi_w_cm, roi_h_cm) * 0.1, 0.5)
                
                fallback_vol = roi_w_cm * roi_h_cm * roi_thickness_cm
                fallback_mass = fallback_vol * density
                
                # Yedek görselleştirme
                viz_fb = base_image.copy()
                cv2.rectangle(viz_fb, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(viz_fb, "ROI Tahmini (Stereo Yetersiz)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                cx_fb = (x1 + x2) // 2
                cy_fb = (y1 + y2) // 2
                cv2.circle(viz_fb, (cx_fb, cy_fb), 5, (255, 0, 0), -1)
                
                dim_text = f"{roi_w_cm:.1f}x{roi_h_cm:.1f}x{roi_thickness_cm:.1f} cm"
                cv2.putText(viz_fb, dim_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(viz_fb, "Yedek Mod: Duz nesne varsayimi", (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1)
                
                stats_fb = {
                    "area_pixels": int(roi_w_px * roi_h_px),
                    "mean_depth": float(roi_thickness_cm),
                    "max_depth": float(roi_thickness_cm),
                    "center_of_mass": (cx_fb, cy_fb),
                    "dims_cm": (roi_w_cm, roi_h_cm, roi_thickness_cm),
                    "mass_g": float(fallback_mass),
                    "estimated_distance_cm": 30.0,  # Varsayılan yakın mesafe
                    "fallback_mode": True
                }
                return fallback_vol, viz_fb, stats_fb
            
            return 0, base_image, {"area_pixels": 0, "mean_depth": 0, "max_depth": 0, "center_of_mass": (0, 0), "dims_cm": (0,0,0), "mass_g": 0, "estimated_distance_cm": 0}
            
        # Alan hesaplaması: 1 pikselin XY düzlemindeki cm^2 alanı
        pixel_area_cm2 = self.px_to_cm ** 2
        
        # Toplam Hacim: Her bir nesne pikselinin alanı ile kalınlığın çarpımı (İntegral)
        total_volume_cm3 = np.sum(object_thickness_values * pixel_area_cm2)
        
        # Kütle Hesabı
        estimated_mass_g = total_volume_cm3 * density
        
        # Ekstra İstatistikler Seçme
        area_pixels = np.sum(mask)
        mean_thickness = np.mean(object_thickness_values) if area_pixels > 0 else 0
        max_thickness = np.max(object_thickness_values) if area_pixels > 0 else 0
        
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
            
            # Gerçek Boyutlar (cm) (Zaten yukarıda mask üzerinden bulduk ama BB daha tutarlı olabilir)
            dim_w_cm = w * self.px_to_cm
            dim_h_cm = h * self.px_to_cm
            dim_l_cm = max_thickness # Nesnenin en kalın (yükseklik) noktası
            dims_cm = (dim_w_cm, dim_h_cm, dim_l_cm)
            
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Merkeze bir nokta ve hedef işareti çiz
                cv2.circle(viz_image, (cx, cy), 5, (255, 0, 0), -1)
                
            # Kameraya Uzaklık Tahmini
            display_dist = min(object_dist, 45.0)
            if display_dist > 0:
                cv2.putText(viz_image, f"Mesafe: ~{display_dist:.1f}cm", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            # Boyut Yazısı
            dim_text = f"{dim_w_cm:.1f}x{dim_h_cm:.1f}x{dim_l_cm:.1f} cm"
            cv2.putText(viz_image, dim_text, (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        stats = {
            "area_pixels": int(area_pixels),
            "mean_depth": float(mean_thickness),
            "max_depth": float(max_thickness),
            "center_of_mass": (cx, cy),
            "estimated_distance_cm": float(min(object_dist, 45.0)),
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
