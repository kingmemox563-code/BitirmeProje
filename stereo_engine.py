import cv2
import numpy as np

class StereoEngine:
    def __init__(self):
        # SIFT detector (scale-invariant feature transform)
        self.sift = cv2.SIFT_create()
        
        # Flann bazlı matcher (hızlı nokta eşleştirme)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def compute_rectification(self, img_left, img_right):
        """
        Görüntüleri alır, ortak noktaları bulur ve yatayda hizalar (Rectification).
        Bu işlem ağırdır (yavaştır), bu yüzden sadece resim çekildiğinde 1 kere çalıştırılmalıdır.
        """
        # Gri tonlamaya çevir (SIFT ve Disparity genelde gri görüntüyle çalışır)
        gray1 = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        h, w = gray1.shape
        
        # 1. SIFT ile köşe/nokta (feature) tespiti
        kp1, des1 = self.sift.detectAndCompute(gray1, None)
        kp2, des2 = self.sift.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None:
            return None, None, False
            
        # 2. Noktaları FLANN ile eşleştirme
        matches = self.flann.knnMatch(des1, des2, k=2)
        
        # Daha sıkı oran testi (0.8 -> 0.7) hatalı eşleşmeleri azaltır
        good_matches = []
        pts1 = []
        pts2 = []
        for dMatch in matches:
            if len(dMatch) == 2:
                m, n = dMatch
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
                    pts2.append(kp2[m.trainIdx].pt)
                    pts1.append(kp1[m.queryIdx].pt)
                
        pts1 = np.float32(pts1)
        pts2 = np.float32(pts2)
        
        if len(good_matches) < 20: # Sınırı tekrar 20'ye çektik (daha esnek)
            return None, None, False

        # 3. Fundamental Matrix (Temel Matris) bulma
        # Varsa daha yeni ve güçlü olan USAC_MAGSAC kullan, yoksa RANSAC
        algo = getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC)
        F, mask = cv2.findFundamentalMat(pts1, pts2, algo, 2.0, 0.99) # Eşik 1.0 -> 2.0 (daha esnek)
        
        if F is None or mask is None:
            return None, None, False
            
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]
        
        if len(pts1) < 10: # Minimum nokta sayısını 15 -> 10 yaptık
            return None, None, False
            
        # 4. Uncalibrated Stereo Rectification
        success, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F, imgSize=(w, h))

        if not success:
            return None, None, False
            
        # Homography matrislerini kontrol et (Aşırı bükülmeyi engellemek için)
        # Eğer matrislerin determinantı çok küçükse veya bükülme çok fazlaysa başarısız say
        if np.abs(np.linalg.det(H1)) < 1e-3 or np.abs(np.linalg.det(H2)) < 1e-3:
            return None, None, False

        # Görüntüleri bük (warp)
        rectified_L = cv2.warpPerspective(img_left, H1, (w, h))
        rectified_R = cv2.warpPerspective(img_right, H2, (w, h))
        
        return rectified_L, rectified_R, True

    def compute_disparity(self, rect_L, rect_R, block_size=5, num_disp_mult=6, uniqueness=10, speckle_win=100, speckle_range=32):
        """
        Hizalanmış görüntüleri alıp WLS Filtresi kullanarak pürüzsüz derinlik haritası hesaplar.
        """
        rect_gray_L = cv2.cvtColor(rect_L, cv2.COLOR_BGR2GRAY)
        rect_gray_R = cv2.cvtColor(rect_R, cv2.COLOR_BGR2GRAY)
        
        num_disp = 16 * num_disp_mult
        
        # Sol Eşleştirici
        left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=uniqueness,
            speckleWindowSize=speckle_win,
            speckleRange=speckle_range,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        # Sağ Eşleştirici (Filtreleme için gerekli)
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        
        # Disparity hesapla
        dispL = left_matcher.compute(rect_gray_L, rect_gray_R)
        dispR = right_matcher.compute(rect_gray_R, rect_gray_L)
        
        # WLS Filtresi ayarları
        lmbda = 80000
        sigma = 1.2
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)
        
        # WLS Filtresini uygula (Parametre ismi versiyona göre değişebildiği için pozisyonel kullanıyoruz)
        # filtered_disp = wls_filter.filter(dispL, rect_gray_L, None, disparity_right=dispR) # Eski hali
        filtered_disp = wls_filter.filter(dispL, rect_gray_L, None, dispR)
        
        # Normalizasyon
        disparity = filtered_disp.astype(np.float32) / 16.0
        disp_viz = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return disparity, disp_viz
