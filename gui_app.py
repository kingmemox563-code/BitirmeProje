import customtkinter as ctk
from PIL import Image, ImageOps
import cv2
import numpy as np
import os
from datetime import datetime
from stereo_engine import StereoEngine
from volume_calc import VolumeCalculator
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from ultralytics import YOLO
from collections import deque
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image as RLImage, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Türkçe Karakter Desteği İçin Font Kaydı
try:
    # Windows sistem fontunu kullan
    font_path = "C:/Windows/Fonts/arial.ttf"
    pdfmetrics.registerFont(TTFont('Arial', font_path))
    pdfmetrics.registerFont(TTFont('Arial-Bold', "C:/Windows/Fonts/arialbd.ttf"))
    FONT_NAME = 'Arial'
    FONT_NAME_BOLD = 'Arial-Bold'
except:
    # Eğer font bulunamazsa varsayılana dön (ama kutucuk sorunu devam edebilir)
    FONT_NAME = 'Helvetica'
    FONT_NAME_BOLD = 'Helvetica-Bold'

# Tema Ayarları
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("3D Hacim Ölçüm Sistemi - Stereo Vision (GUI)")
        self.geometry("1250x750")

        # İşlem Motorları
        self.engine = StereoEngine()
        self.calculator = VolumeCalculator()
        
        # YENI: AI Model (YOLO)
        try:
            self.model = YOLO("yolov8n.pt")
            self.ai_active = True
        except:
            self.model = None
            self.ai_active = False

        # Kamera
        self.cap = cv2.VideoCapture(0)
        self.camera_mode = True
        self.img_left = None
        self.img_right = None
        self.rect_L = None
        self.rect_R = None
        self.disp_viz_raw = None
        self.current_volume = 0.0
        self.current_viz_img = None
        self.current_stats = {}
        
        # YENI: ROI ve Materyal Durumları
        self.roi_start = None
        self.roi_end = None
        self.is_selecting_roi = False
        self.selected_roi = None # [x1, y1, x2, y2]
        self.current_density = 1.0
        
        # YENI: Smoothing Buffer
        self.volume_buffer = deque(maxlen=5)
        self.mass_buffer = deque(maxlen=5)
        
        # YENI: Multi-View (Çoklu Bakış) Oturumu
        self.multi_view_data = [] # List of {volume, mass, stats, img}
        self.ai_confidence = 0.45 # Varsayılan AI güven eşiği
        
        # Materyal Listesi (YOLO sınıflarıyla eşleşenler eklendi)
        self.materials = {
            "Su (1.0)": 1.0,
            "Plastik (0.9)": 0.9,
            "Cüzdan (Deri) (0.86)": 0.86,
            "Telefon (Elektronik) (1.3)": 1.3,
            "Araba Anahtarı (Metal/Plastik) (2.0)": 2.0,
            "Tahta (0.7)": 0.7,
            "Demir (7.8)": 7.8,
            "Alüminyum (2.7)": 2.7,
            "Elma/Meyve (0.8)": 0.8,
            "Ekmek (0.5)": 0.5,
            "Özel Giriş": 1.0
        }
        
        # YOLO Sınıf -> Materyal Eşleşmesi (günlük nesneler eklendi)
        self.ai_material_map = {
            # Elektronik
            "cell phone": "Telefon (Elektronik) (1.3)",
            "laptop": "Telefon (Elektronik) (1.3)",
            "keyboard": "Telefon (Elektronik) (1.3)",
            "mouse": "Telefon (Elektronik) (1.3)",
            "remote": "Plastik (0.9)",
            "tv": "Telefon (Elektronik) (1.3)",
            # Kişisel eşyalar
            "handbag": "Cüzdan (Deri) (0.86)",
            "backpack": "Cüzdan (Deri) (0.86)",
            "suitcase": "Plastik (0.9)",
            "tie": "Plastik (0.9)",
            "scissors": "Araba Anahtarı (Metal/Plastik) (2.0)",
            "book": "Tahta (0.7)",
            # Yiyecekler
            "apple": "Elma/Meyve (0.8)",
            "orange": "Elma/Meyve (0.8)",
            "banana": "Elma/Meyve (0.8)",
            "sandwich": "Ekmek (0.5)",
            "donut": "Ekmek (0.5)",
            "cake": "Ekmek (0.5)",
            # Mutfak / Günlük
            "bottle": "Plastik (0.9)",
            "cup": "Plastik (0.9)",
            "bowl": "Plastik (0.9)",
            "wine glass": "Plastik (0.9)",
            "vase": "Plastik (0.9)",
            "clock": "Elektronik (1.3)",
        }
        
        # Canlı kamerada atlanacak büyük/sabit nesne sınıfları
        self.skip_live_classes = {
            "person", "chair", "couch", "sofa", "bed", "dining table",
            "tv", "monitor", "refrigerator", "oven", "sink",
            "toilet", "door", "window", "wall", "floor", "ceiling"
        }

        # -- Genel Çerçeve (Layout) --
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ==========================================
        # SOL MENÜ (Kontroller)
        # ==========================================
        self.sidebar_frame = ctk.CTkScrollableFrame(self, width=280, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(20, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="3D VOLUME AI", font=ctk.CTkFont(size=24, weight="bold"), text_color="#00e6e6")
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # --- SEKME SİSTEMİ (SOL MENÜ) ---
        self.side_tabs = ctk.CTkTabview(self.sidebar_frame, width=250, height=550)
        self.side_tabs.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.tab_ops = self.side_tabs.add("İşlem Akışı")
        self.tab_adv = self.side_tabs.add("Ayarlar")
        
        # ==========================================
        # 1. SEKME: İŞLEM AKIŞI
        # ==========================================
        
        # Çekim / Yükleme Butonları
        self.btn_left_load = ctk.CTkButton(self.tab_ops, text="Sol Resim Seç", command=self.load_left, height=30, fg_color="#4d4d4d")
        self.btn_left_load.pack(fill="x", padx=10, pady=(10, 2))
        
        self.btn_left = ctk.CTkButton(self.tab_ops, text="1. SOL ÇEK (L)", command=self.capture_left, height=45, font=ctk.CTkFont(weight="bold"))
        self.btn_left.pack(fill="x", padx=10, pady=(2, 15))

        self.btn_right_load = ctk.CTkButton(self.tab_ops, text="Sağ Resim Seç", command=self.load_right, height=30, fg_color="#4d4d4d")
        self.btn_right_load.pack(fill="x", padx=10, pady=(5, 2))

        self.btn_right = ctk.CTkButton(self.tab_ops, text="2. SAĞ ÇEK (R)", command=self.capture_right, state="disabled", height=45, font=ctk.CTkFont(weight="bold"))
        self.btn_right.pack(fill="x", padx=10, pady=(2, 15))

        self.btn_calc = ctk.CTkButton(self.tab_ops, text="3. HESAPLA", command=self.calculate_stereo, state="disabled", fg_color="#2eb82e", hover_color="#248f24", height=50, font=ctk.CTkFont(size=16, weight="bold"))
        self.btn_calc.pack(fill="x", padx=10, pady=(15, 20))

        # Rapor Butonu
        self.btn_report = ctk.CTkButton(self.tab_ops, text="PDF Raporu Oluştur", command=self.save_report, state="disabled", height=35, fg_color="#3d3d5c")
        self.btn_report.pack(fill="x", padx=10, pady=5)

        # Çoklu Bakış Kontrolleri
        self.lbl_multi = ctk.CTkLabel(self.tab_ops, text="Çoklu Bakış Seansı", font=ctk.CTkFont(weight="bold"))
        self.lbl_multi.pack(pady=(20, 0))
        
        self.btn_add_view = ctk.CTkButton(self.tab_ops, text="Görünüm Ekle (+)", command=self.add_current_view, state="disabled", fg_color="#b8860b", hover_color="#996515")
        self.btn_add_view.pack(fill="x", padx=10, pady=5)
        
        self.btn_reset_session = ctk.CTkButton(self.tab_ops, text="Seansı Sıfırla", command=self.reset_session, fg_color="#444444")
        self.btn_reset_session.pack(fill="x", padx=10, pady=5)

        # ==========================================
        # 2. SEKME: TEKNİK AYARLAR
        # ==========================================
        
        # Slider Ayarları
        ctk.CTkLabel(self.tab_adv, text="Derinlik Detayı", font=ctk.CTkFont(weight="bold")).pack(pady=(5, 0), anchor="w", padx=10)
        self.slider_block = ctk.CTkSlider(self.tab_adv, from_=1, to=10, number_of_steps=9, command=self.on_slider_change)
        self.slider_block.set(3)
        self.slider_block.pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkLabel(self.tab_adv, text="Hassasiyet (Num Disp)", font=ctk.CTkFont(weight="bold")).pack(pady=(0, 0), anchor="w", padx=10)
        self.slider_disp = ctk.CTkSlider(self.tab_adv, from_=1, to=10, number_of_steps=9, command=self.on_slider_change)
        self.slider_disp.set(6)
        self.slider_disp.pack(fill="x", padx=10, pady=(0, 10))
        
        ctk.CTkLabel(self.tab_adv, text="Parazit Filtresi", font=ctk.CTkFont(weight="bold")).pack(pady=(0, 0), anchor="w", padx=10)
        self.slider_speckle = ctk.CTkSlider(self.tab_adv, from_=0, to=200, number_of_steps=20, command=self.on_slider_change)
        self.slider_speckle.set(100)
        self.slider_speckle.pack(fill="x", padx=10, pady=(0, 10))
        
        ctk.CTkLabel(self.tab_adv, text="Eşik Değeri", font=ctk.CTkFont(weight="bold")).pack(pady=(0, 0), anchor="w", padx=10)
        self.slider_thresh = ctk.CTkSlider(self.tab_adv, from_=0, to=50, number_of_steps=50, command=self.on_slider_change)
        self.slider_thresh.set(15)
        self.slider_thresh.pack(fill="x", padx=10, pady=(0, 10))
        
        ctk.CTkLabel(self.tab_adv, text="Renk Paleti", font=ctk.CTkFont(weight="bold")).pack(pady=(5, 0), anchor="w", padx=10)
        self.combo_cmap = ctk.CTkComboBox(self.tab_adv, values=["JET", "PLASMA", "MAGMA", "VIRIDIS", "INFERNO"], command=self.on_slider_change)
        self.combo_cmap.set("JET")
        self.combo_cmap.pack(fill="x", padx=10, pady=(0, 20))
        
        # Sabit Bilgi (Kullanıcıya bildirim)
        ctk.CTkLabel(self.tab_adv, text="Not: Kaydırma mesafesi\n10 cm olarak sabitlendi.", text_color="orange", font=ctk.CTkFont(size=11)).pack(pady=10)

        # ==========================================
        # SAĞ EKRAN (Sekmeli Görünüm)
        # ==========================================
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=1, padx=(10, 20), pady=20, sticky="nsew")
        self.tab_main = self.tabview.add("Ölçüm ve Analiz")
        self.tab_3d = self.tabview.add("3D Görünüm")
        self.tab_history = self.tabview.add("Seans Geçmişi")
        self.tab_settings = self.tabview.add("Gelişmiş Ayarlar")

        # --- TAB 1: ÖLÇÜM VE ANALİZ ---
        self.tab_main.grid_rowconfigure(1, weight=1)
        self.tab_main.grid_columnconfigure((0, 1, 2), weight=1)

        # Ana Kamera
        self.lbl_cam = ctk.CTkLabel(self.tab_main, text="Kamera Bekleniyor...")
        self.lbl_cam.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
        self.lbl_cam.bind("<Button-1>", self.on_mouse_down)
        self.lbl_cam.bind("<B1-Motion>", self.on_mouse_move)
        self.lbl_cam.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Sonuçlar
        self.lbl_res1 = ctk.CTkLabel(self.tab_main, text="Sol Görüntü")
        self.lbl_res1.grid(row=1, column=0, padx=5, pady=10)

        self.lbl_res2 = ctk.CTkLabel(self.tab_main, text="Derinlik Haritası")
        self.lbl_res2.grid(row=1, column=1, padx=5, pady=10)
        
        self.lbl_res3 = ctk.CTkLabel(self.tab_main, text="Analiz Maskesi")
        self.lbl_res3.grid(row=1, column=2, padx=5, pady=10)
        
        # İstatistik Kartı
        self.stats_frame = ctk.CTkFrame(self.tab_main, fg_color="#1a1a1a")
        self.stats_frame.grid(row=2, column=0, columnspan=3, pady=10, sticky="ew", padx=20)
        self.stats_frame.grid_columnconfigure((0,1,2,3), weight=1)

        self.lbl_volume = ctk.CTkLabel(self.stats_frame, text="Hacim: - cm³", font=ctk.CTkFont(size=28, weight="bold"), text_color="#00e6e6")
        self.lbl_volume.grid(row=0, column=0, pady=15)
        
        self.lbl_mass = ctk.CTkLabel(self.stats_frame, text="Ağırlık: - gr", font=ctk.CTkFont(size=28, weight="bold"), text_color="#ffcc00")
        self.lbl_mass.grid(row=0, column=1, pady=15)
        
        self.lbl_dims = ctk.CTkLabel(self.stats_frame, text="Boyut: - cm", font=ctk.CTkFont(size=18), text_color="#ffffff")
        self.lbl_dims.grid(row=0, column=2, pady=15)

        self.lbl_dist = ctk.CTkLabel(self.stats_frame, text="Mesafe: - cm", font=ctk.CTkFont(size=18), text_color="#aaaaaa")
        self.lbl_dist.grid(row=0, column=3, pady=15)

        # --- TAB 2: 3D GÖRÜNÜM ---
        self.fig = plt.figure(figsize=(5, 5), dpi=100)
        self.fig.patch.set_facecolor('#2b2b2b')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#2b2b2b')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_3d)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # --- TAB 3: SEANS GEÇMİŞİ ---
        self.tab_history.grid_rowconfigure(0, weight=1)
        self.tab_history.grid_columnconfigure(0, weight=1)
        
        self.history_frame = ctk.CTkScrollableFrame(self.tab_history, label_text="Eklenen Görünümler")
        self.history_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.lbl_no_history = ctk.CTkLabel(self.history_frame, text="Henüz bir görünüm eklenmedi.\n'Görünüm Ekle' butonunu kullanarak ölçümleri kaydedebilirsiniz.", text_color="gray")
        self.lbl_no_history.pack(pady=20)
        
        self.history_widgets = []

        # --- TAB 3: AYARLAR ---
        self.settings_scroll = ctk.CTkScrollableFrame(self.tab_settings)
        self.settings_scroll.pack(fill="both", expand=True)

        # AI Kontrolleri
        self.lbl_ai = ctk.CTkLabel(self.settings_scroll, text="Yapay Zeka (YOLO) Ayarları", font=ctk.CTkFont(weight="bold", size=16))
        self.lbl_ai.pack(pady=(10, 5))
        self.switch_ai = ctk.CTkSwitch(self.settings_scroll, text="Yapay Zeka Tespiti Aktif", command=self.on_ai_switch)
        self.switch_ai.select()
        self.switch_ai.pack(pady=5)
        
        self.lbl_conf = ctk.CTkLabel(self.settings_scroll, text=f"AI Güven Eşiği: {self.ai_confidence:.2f}")
        self.lbl_conf.pack(pady=0)
        self.slider_conf = ctk.CTkSlider(self.settings_scroll, from_=0.1, to=0.9, number_of_steps=16, command=self.on_conf_change)
        self.slider_conf.set(self.ai_confidence)
        self.slider_conf.pack(pady=5)

        # Kalibrasyon
        ctk.CTkLabel(self.settings_scroll, text="Otomatik Kalibrasyon", font=ctk.CTkFont(weight="bold", size=16)).pack(pady=(20, 5))
        self.lbl_cal_info = ctk.CTkLabel(self.settings_scroll, text="Seçili ROI'yi referans nesne (kart vb.) kabul et.", font=ctk.CTkFont(size=12))
        self.lbl_cal_info.pack(pady=0)
        
        self.entry_cal_cm = ctk.CTkEntry(self.settings_scroll, placeholder_text="Gerçek Genişlik (cm)...")
        self.entry_cal_cm.pack(pady=5)
        self.btn_calibrate = ctk.CTkButton(self.settings_scroll, text="ROI Boyutuna Göre Kalibre Et", command=self.do_calibration, fg_color="#2b8a8a")
        self.btn_calibrate.pack(pady=5)

        # Materyal & Smoothing
        ctk.CTkLabel(self.settings_scroll, text="Ölçüm ve Materyal", font=ctk.CTkFont(weight="bold", size=16)).pack(pady=(20, 5))
        self.switch_smoothing = ctk.CTkSwitch(self.settings_scroll, text="Hareketli Ortalama (Smoothing)")
        self.switch_smoothing.select()
        self.switch_smoothing.pack(pady=5)
        
        self.combo_material = ctk.CTkComboBox(self.settings_scroll, values=list(self.materials.keys()), command=self.on_material_change)
        self.combo_material.set("Su (1.0)")
        self.combo_material.pack(pady=5)
        
        self.entry_density = ctk.CTkEntry(self.settings_scroll, placeholder_text="Yoğunluk...")
        self.entry_density.pack(pady=5)
        
        # Görselleştirme
        ctk.CTkLabel(self.settings_scroll, text="Görselleştirme", font=ctk.CTkFont(weight="bold", size=16)).pack(pady=(20, 5))
        self.switch_heatmap = ctk.CTkSwitch(self.settings_scroll, text="Topografik Isı Haritası", command=self.on_slider_change)
        self.switch_heatmap.pack(pady=5)

        self.btn_reset_roi = ctk.CTkButton(self.settings_scroll, text="Seçili Alanı (ROI) Sıfırla", command=self.reset_roi, fg_color="#a63d3d")
        self.btn_reset_roi.pack(pady=20)

        self.is_calculating = False
        
        # Kısayol Tuşları Destekleri
        self.bind('<l>', lambda e: self.capture_left())
        self.bind('<r>', lambda e: self.capture_right() if self.img_left is not None else None)

        self.update_camera()

    def update_camera(self):
        ret, frame = self.cap.read()
        
        # Eğer kamera yoksa veya okunamadıysa boş bir frame oluştur (Dosyadan yükleme yapabilmek için)
        if not ret:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Kamera Bulunamadi veya Kapali.", (140, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Dosyadan resim yukleyebilirsiniz.", (150, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            ret = True

        if ret:
            display_frame = frame.copy()
            
            # --- Canlı Nesne Tespiti (2D Önizleme) ---
            if self.img_left is None and not self.is_calculating:
                detected_by_ai = False
                if self.ai_active and self.model is not None:
                    # YOLO ile akıllı tespit (İnsanları yoksayarak)
                    results = self.model(display_frame, verbose=False, conf=self.ai_confidence)[0]
                    for box in results.boxes:
                        cls_id = int(box.cls[0])
                        label = results.names[cls_id]
                        # Büyük sabit nesneleri ve insanı atla
                        if label in self.skip_live_classes:
                            continue
                            
                        x, y, x2, y2 = map(int, box.xyxy[0].tolist())
                        w, h = x2 - x, y2 - y
                        
                        height, width = frame.shape[:2]
                        if w < width * 0.9 and h < height * 0.9:
                            detected_by_ai = True
                            # Nesne adını Türkçeleştir (kısaca)
                            tr_label = self._translate_label(label)
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(display_frame, f"Hedef: {tr_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.putText(display_frame, "Uzaklik hesaplanacak", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            # Ekranda çok fazla nesne işaretlememesi için ilk uygun olanı bulup çık
                            break
                
                # Eğer AI aktif değilse veya hedef bulamadıysa geleneksel yönteme dön
                if not detected_by_ai:
                    # Geleneksel yöntem
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
                    edges = cv2.Canny(blurred, 30, 100)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
                    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
                    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        valid_contours = [c for c in contours if cv2.contourArea(c) > 3000]
                        if valid_contours:
                            c = max(valid_contours, key=cv2.contourArea)
                            x, y, w, h = cv2.boundingRect(c)
                            
                            height, width = frame.shape[:2]
                            if w < width * 0.9 and h < height * 0.9:
                                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                cv2.putText(display_frame, "Hedef Nesne (Geleneksel)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                cv2.putText(display_frame, "Uzaklik & Hacim hesaplanacak", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # ----------------------------------------
            
            # --- ROI Görselleştirme ---
            # 1. Mevcut Seçili ROI (Varsa)
            if self.selected_roi:
                # Orijinal resim (ref_img) boyutlarından display_frame boyutlarına ölçekle
                ref_img_for_roi = self.rect_L if self.rect_L is not None else (self.img_left if self.img_left is not None else None)
                if ref_img_for_roi is not None:
                    orig_h, orig_w = ref_img_for_roi.shape[:2]
                else:
                    orig_h, orig_w = frame.shape[:2]
                    
                disp_h, disp_w = display_frame.shape[:2]
                scale_x = disp_w / orig_w
                scale_y = disp_h / orig_h
                
                x1, y1, x2, y2 = self.selected_roi
                d_x1, d_y1 = int(x1 * scale_x), int(y1 * scale_y)
                d_x2, d_y2 = int(x2 * scale_x), int(y2 * scale_y)
                
                cv2.rectangle(display_frame, (d_x1, d_y1), (d_x2, d_y2), (255, 255, 0), 2)
                cv2.putText(display_frame, "ROI AKTIF", (d_x1, d_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # 2. Canlı Çizim Önizlemesi (Fare sürüklenirken)
            if self.is_selecting_roi and self.roi_start and self.roi_end:
                lbl_w = self.lbl_cam.winfo_width()
                lbl_h = self.lbl_cam.winfo_height()
                
                # display_frame boyutuna doğrudan ölçekle (offset yok, NW anchor)
                img_h, img_w = display_frame.shape[:2]
                sx, sy = img_w / lbl_w, img_h / lbl_h
                
                p1_x = int(self.roi_start[0] * sx)
                p1_y = int(self.roi_start[1] * sy)
                p2_x = int(self.roi_end[0] * sx)
                p2_y = int(self.roi_end[1] * sy)
                
                p1_x = max(0, min(p1_x, img_w))
                p1_y = max(0, min(p1_y, img_h))
                p2_x = max(0, min(p2_x, img_w))
                p2_y = max(0, min(p2_y, img_h))
                
                cv2.rectangle(display_frame, (p1_x, p1_y), (p2_x, p2_y), (0, 255, 255), 2)

            # EĞER DOSYADAN YÜKLENMİŞSE ÖNİZLEME OLARAK ONU GÖSTER
            if not self.camera_mode:
                if self.img_left is not None and self.img_right is None:
                    # Sadece sol yüklü
                    preview_img = self.img_left.copy()
                    preview_img = cv2.resize(preview_img, (640, 480))
                    cv2.putText(preview_img, "SOL YUKLENDI. Sagi yukleyin.", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    display_frame = preview_img
                elif self.img_left is not None and self.img_right is not None:
                    # İkisi de yüklü
                    preview_img = self.img_left.copy()
                    preview_img = cv2.resize(preview_img, (640, 480))
                    cv2.putText(preview_img, "IKI RESIM HAZIR. HESAPLAYIN.", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    display_frame = preview_img
            else:
                # CANLI KAMERA MODU - Kamerayı Dondurma
                if self.img_left is not None and self.img_right is None:
                    cv2.putText(display_frame, "SOL CEKILDI. Sagi cekin.", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                elif self.img_left is not None and self.img_right is not None:
                    cv2.putText(display_frame, "IKI RESIM DE CEKILDI. HESAPLAYIN.", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            d_img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(d_img)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 400))
            self.lbl_cam.configure(image=ctk_img, text="")
            self.lbl_cam.image = ctk_img

        if not self.is_calculating:
            self.after(30, self.update_camera)

    def on_mouse_down(self, event):
        self.roi_start = (event.x, event.y)
        self.is_selecting_roi = True

    def on_mouse_move(self, event):
        if self.is_selecting_roi:
            self.roi_end = (event.x, event.y)

    def on_mouse_up(self, event):
        if self.is_selecting_roi:
            self.roi_end = (event.x, event.y)
            self.is_selecting_roi = False
            
            # Koordinatları sırala (label koordinatlarında)
            x1 = min(self.roi_start[0], self.roi_end[0])
            y1 = min(self.roi_start[1], self.roi_end[1])
            x2 = max(self.roi_start[0], self.roi_end[0])
            y2 = max(self.roi_start[1], self.roi_end[1])
            
            # Label boyutları
            lbl_w = self.lbl_cam.winfo_width()
            lbl_h = self.lbl_cam.winfo_height()

            # Seçim yapılacak referans görüntüyü belirle
            ref_img = self.rect_L if self.rect_L is not None else (self.img_left if self.img_left is not None else None)
            
            if ref_img is not None:
                img_h, img_w = ref_img.shape[:2]
            else:
                ret, tmp_f = self.cap.read()
                if ret:
                    img_h, img_w = tmp_f.shape[:2]
                else:
                    img_h, img_w = 480, 640

            # CTkLabel görüntüyü sol-üst (NW) köşeden başlatır, offset yok
            sx = img_w / lbl_w
            sy = img_h / lbl_h
            
            # EĞER TIKLAMA İSE (Sürükleme değilse, < 8px)
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if dist < 8 and self.disp_viz_raw is not None and self.rect_L is not None:
                # Akıllı Tıklama (Smart Click)
                click_x, click_y = int(event.x * sx), int(event.y * sy)
                
                if 0 <= click_y < img_h and 0 <= click_x < img_w:
                    depth_val = float(self.disp_viz_raw[click_y, click_x])
                    new_thresh = max(5, depth_val - 10)
                    self.slider_thresh.set(new_thresh)
                    
                    mask = (self.disp_viz_raw > new_thresh).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    found = False
                    for c in contours:
                        if cv2.pointPolygonTest(c, (click_x, click_y), False) >= 0:
                            bx, by, bw, bh = cv2.boundingRect(c)
                            self.selected_roi = [bx, by, bx+bw, by+bh]
                            found = True
                            break
                    
                    if not found:
                        self.selected_roi = [max(0, click_x-40), max(0, click_y-40), 
                                             min(img_w, click_x+40), min(img_h, click_y+40)]
            else:
                # Normal Dikdörtgen ROI Seçimi
                norm_x1 = x1 * sx
                norm_y1 = y1 * sy
                norm_x2 = x2 * sx
                norm_y2 = y2 * sy
                self.selected_roi = [int(max(0, norm_x1)), int(max(0, norm_y1)), 
                                     int(min(img_w, norm_x2)), int(min(img_h, norm_y2))]
            
            if self.rect_L is not None:
                self.on_slider_change(None)

    def reset_roi(self):
        self.selected_roi = None
        if self.rect_L is not None:
            self.on_slider_change(None)

    def on_material_change(self, value):
        dens = self.materials.get(value, 1.0)
        self.entry_density.delete(0, "end")
        self.entry_density.insert(0, str(dens))
        self.on_slider_change(None)

    def on_ai_switch(self):
        self.ai_active = self.switch_ai.get()
        if self.ai_active and self.rect_L is not None:
            self.run_ai_detection()

    def on_conf_change(self, value):
        self.ai_confidence = float(value)
        self.lbl_conf.configure(text=f"AI Güven Eşiği: {self.ai_confidence:.2f}")

    def _translate_label(self, label: str) -> str:
        """YOLO sınıf adını kısa Türkçeye çevirir."""
        _MAP = {
            "cell phone": "Telefon",
            "laptop": "Dizüstü",
            "keyboard": "Klavye",
            "mouse": "Mouse",
            "remote": "Kumanda",
            "handbag": "Çanta/Cüzdan",
            "backpack": "Sırt Çantası",
            "suitcase": "Bavul",
            "book": "Kitap",
            "scissors": "Makas",
            "bottle": "Şişe",
            "cup": "Bardak",
            "bowl": "Kase",
            "apple": "Elma",
            "orange": "Portakal",
            "banana": "Muz",
            "sandwich": "Sandviç",
            "clock": "Saat",
            "vase": "Vazo",
            "knife": "Bıçak",
            "fork": "Çatal",
            "spoon": "Kaşık",
            "pen": "Kalem",
            "pencil": "Kalem",
        }
        return _MAP.get(label, label)

    def run_ai_detection(self):
        if self.model is None or self.img_left is None: return
        
        # YOLO Çıkarımı
        results = self.model(self.img_left, verbose=False, conf=self.ai_confidence)[0]
        
        # Görüntü merkezi
        img_h, img_w = self.img_left.shape[:2]
        center_x, center_y = img_w / 2, img_h / 2
        
        detections = []
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = results.names[cls_id]
            
            # Büyük sabit nesneleri ve insanı atla
            if label in self.skip_live_classes:
                continue
                
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bx, by = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Merkeze olan uzaklık
            dist_to_center = np.sqrt((bx - center_x)**2 + (by - center_y)**2)
            
            detections.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "label": label,
                "dist": dist_to_center,
                "area": (x2-x1)*(y2-y1)
            })
        
        if detections:
            # Merkeze en yakın olanı seç
            detections.sort(key=lambda d: d["dist"])
            best = detections[0]
            
            self.selected_roi = best["box"]
            # Materyal otomatik seçimi
            if best["label"] in self.ai_material_map:
                mat_name = self.ai_material_map[best["label"]]
                self.combo_material.set(mat_name)
                self.on_material_change(mat_name)
            
            tr_label = self._translate_label(best['label'])
            self.lbl_volume.configure(text=f"AI: {tr_label} Tespit Edildi!", text_color="#2eb82e")
            self.on_slider_change(None)
        else:
            # Geleneksel yöntemle en büyük nesneyi bul (Cüzdan, Anahtar vb. YOLO tarafından tanınmayanlar için)
            gray = cv2.cvtColor(self.img_left, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            edges = cv2.Canny(blurred, 30, 100)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                valid_contours = [c for c in contours if cv2.contourArea(c) > 3000]
                if valid_contours:
                    c = max(valid_contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(c)
                    self.selected_roi = [x, y, x+w, y+h]
                    self.lbl_volume.configure(text="Geleneksel Kontur Tespiti Aktif", text_color="#2eb82e")
                    self.on_slider_change(None)

    def add_current_view(self):
        """Mevcut ölçümü seansa ekler."""
        if self.current_volume <= 0: return
        
        # UI Geribildirimi (Buton animasyonu gibi)
        self.btn_add_view.configure(text="EKLENDI!", fg_color="#2eb82e")
        self.after(1000, lambda: self.btn_add_view.configure(text="Görünüm Ekle (+)", fg_color="#b8860b"))

        view_data = {
            "volume": self.current_volume,
            "mass": self.mass_buffer[-1] if self.mass_buffer else self.current_stats.get('mass_g', 0),
            "stats": self.current_stats.copy(),
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        self.multi_view_data.append(view_data)
        
        # Geçmiş tablosunu güncelle
        if self.lbl_no_history:
            self.lbl_no_history.pack_forget()
            self.lbl_no_history = None
            
        count = len(self.multi_view_data)
        
        # Yeni bir satır (row) ekle geçmişe
        row_frame = ctk.CTkFrame(self.history_frame, fg_color="#262626")
        row_frame.pack(fill="x", padx=5, pady=2)
        
        lbl_info = ctk.CTkLabel(row_frame, text=f"#{count} | {view_data['timestamp']} | {view_data['volume']:.1f} cm³ | {view_data['mass']:.1f} gr", anchor="w")
        lbl_info.pack(side="left", padx=10, pady=5)
        
        self.history_widgets.append(row_frame)
        
        self.lbl_multi.configure(text=f"Çoklu Bakış: {count} Görünüm", text_color="#ffcc00")
        
        # Ortalama hesapla ve göster
        avg_vol = sum(d['volume'] for d in self.multi_view_data) / count
        avg_mass = sum(d['mass'] for d in self.multi_view_data) / count
        
        self.lbl_volume.configure(text=f"Ort. Hacim: {avg_vol:.2f} cm³", text_color="#00e6e6")
        self.lbl_mass.configure(text=f"Ort. Ağırlık: {avg_mass:.1f} gr")
        
        # Otomatik olarak geçmiş sekmesine odaklan (isteğe bağlı)
        # self.tabview.set("Seans Geçmişi")

    def reset_session(self):
        """Tüm çoklu bakış verilerini sıfırlar."""
        self.multi_view_data = []
        self.camera_mode = True # Seans sıfırlanınca kameraya dön
        
        # Geçmiş arayüzünü temizle
        for widget in self.history_widgets:
            widget.destroy()
        self.history_widgets = []
        
        if self.lbl_no_history is None:
            self.lbl_no_history = ctk.CTkLabel(self.history_frame, text="Henüz bir görünüm eklenmedi.\n'Görünüm Ekle' butonunu kullanarak ölçümleri kaydedebilirsiniz.", text_color="gray")
            self.lbl_no_history.pack(pady=20)
            
        self.lbl_multi.configure(text="Çoklu Bakış Seansı", text_color="white")
        self.lbl_volume.configure(text="Seans sıfırlandı.", text_color="white")
        self.on_slider_change(None)

    def do_calibration(self):
        if self.selected_roi is None:
            self.lbl_volume.configure(text="Lütfen önce farenizle referans nesneyi seçin!", text_color="orange")
            return
            
        try:
            # Girişi temizle (cm ekini sil, virgülü noktaya çevir)
            val_str = self.entry_cal_cm.get().lower().replace("cm", "").replace(",", ".").strip()
            real_cm = float(val_str)
            
            x1, y1, x2, y2 = self.selected_roi
            px_w = x2 - x1
            px_h = y2 - y1
            
            new_ratio = self.calculator.auto_calibrate(px_w, px_h, real_cm, real_cm) 
            self.lbl_volume.configure(text=f"✓ Kalibrasyon Başarılı! (Oran: {new_ratio:.4f})", text_color="#2eb82e")
            self.on_slider_change(None)
        except Exception as e:
            self.lbl_volume.configure(text="Geçersiz değer! Sadece sayı girin (Örn: 8.5)", text_color="red")

    def load_left(self):
        self.selected_roi = None  # Yeni resim yüklendiği için eski ROI'yi sıfırla
        filepath = ctk.filedialog.askopenfilename(title="Sol Görüntüyü Seç", filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if filepath:
            self.camera_mode = False
            try:
                # PIL kullanarak yükle (EXIF rotasyonunu düzeltmek için)
                pil_img = Image.open(filepath).convert("RGB")
                pil_img = ImageOps.exif_transpose(pil_img)
                numpy_img = np.array(pil_img)
                self.img_left = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)
                
                if self.img_left is not None:
                    self.btn_right.configure(state="normal")
                    self.lbl_volume.configure(text="Sol resim başarıyla yüklendi.", text_color="#00e6e6")
                else:
                    self.lbl_volume.configure(text="Hata: Resim dosyası okunamadı!", text_color="red")
            except Exception as e:
                self.lbl_volume.configure(text=f"Yükleme Hatası: {str(e)}", text_color="red")
            
    def load_right(self):
        filepath = ctk.filedialog.askopenfilename(title="Sağ Görüntüyü Seç", filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if filepath:
            self.camera_mode = False
            try:
                pil_img = Image.open(filepath).convert("RGB")
                pil_img = ImageOps.exif_transpose(pil_img)
                numpy_img = np.array(pil_img)
                self.img_right = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)
                
                if self.img_right is not None:
                    self.btn_calc.configure(state="normal")
                    self.lbl_volume.configure(text="Sağ resim başarıyla yüklendi. Hesaplayabilirsiniz.", text_color="#00e6e6")
                else:
                    self.lbl_volume.configure(text="Hata: Resim dosyası okunamadı!", text_color="red")
            except Exception as e:
                self.lbl_volume.configure(text=f"Yükleme Hatası: {str(e)}", text_color="red")

    def capture_left(self):
        self.selected_roi = None  # Yeni çekim yapıldığı için eski ROI'yi sıfırla
        ret, frame = self.cap.read()
        if ret:
            self.camera_mode = True
            self.img_left = frame.copy()
            self.btn_right.configure(state="normal")
            
    def capture_right(self):
        ret, frame = self.cap.read()
        if ret:
            self.camera_mode = True
            self.img_right = frame.copy()
            self.btn_calc.configure(state="normal")

    def calculate_stereo(self):
        if self.img_left is None or self.img_right is None: return
        
        self.is_calculating = True
        self.lbl_volume.configure(text="AI ve Derinlik İşleniyor...", text_color="#ffcc00")
        self.update()

        # Faz 2: AI Tespiti (Önce yapıyoruz ki ROI hazır olsun)
        if self.switch_ai.get():
            self.run_ai_detection()

        # Aşama 1: Hizalama (Rectification)
        self.rect_L, self.rect_R, success = self.engine.compute_rectification(self.img_left, self.img_right)
        
        if not success:
            # EĞER HİZALAMA BAŞARISIZSA (Resimler zaten hizalı olabilir - Middlebury gibi)
            # Orijinal resimlerle devam etmeyi dene
            self.lbl_volume.configure(text="Hizalama yapılamadı, orijinal resimlerle deneniyor...", text_color="#ffcc00")
            self.update()
            self.rect_L = self.img_left.copy()
            self.rect_R = self.img_right.copy()
            # Resim boyutlarını eşitle
            h, w = self.rect_L.shape[:2]
            self.rect_R = cv2.resize(self.rect_R, (w, h))
            
        # Orijinal görüntüyü arayüzdeki pencereye yansıt
        rl_img = cv2.cvtColor(self.rect_L, cv2.COLOR_BGR2RGB)
        imL = Image.fromarray(rl_img)
        cL = ctk.CTkImage(light_image=imL, dark_image=imL, size=(300, 220))
        self.lbl_res1.configure(image=cL, text="")
        
        self.btn_report.configure(state="normal")
        self.btn_add_view.configure(state="normal")
        # Aşama 2: Slider değerlerini çekip Derinlik Haritası çıkarma fonksiyonunu tetikle
        try:
            self.on_slider_change(None) 
        except Exception as e:
            self.lbl_volume.configure(text=f"Hata: {str(e)}", text_color="red")
            self.is_calculating = False
            self.update_camera()

    def on_slider_change(self, value):
        if self.rect_L is None or self.rect_R is None: return
            
        try:
            # Slider değerlerini SGBM Block bilgilerine çevir
            block_val = int(self.slider_block.get()) # 1-10 arası
            block_s = (block_val * 2) + 1 # Tek sayı olmak zorunda (3, 5, 7 ... 21)
            
            disp_m = int(self.slider_disp.get()) # 1-10 arası
            speckle_w = int(self.slider_speckle.get())
            thresh_v = float(self.slider_thresh.get())
            cmap_name = self.combo_cmap.get()
            
            # Colormap Haritalama
            cmap_dict = {
                "JET": cv2.COLORMAP_JET,
                "PLASMA": cv2.COLORMAP_PLASMA,
                "MAGMA": cv2.COLORMAP_MAGMA,
                "VIRIDIS": cv2.COLORMAP_VIRIDIS,
                "INFERNO": cv2.COLORMAP_INFERNO
            }
            cmap_code = cmap_dict.get(cmap_name, cv2.COLORMAP_JET)
            
            # Anlık hızlı hesaplama yap
            try:
                disparity, disp_viz = self.engine.compute_disparity(self.rect_L, self.rect_R, block_size=block_s, num_disp_mult=disp_m, speckle_win=speckle_w)
            except Exception as e:
                self.lbl_volume.configure(text=f"Derinlik Hatası: {str(e)}", text_color="red")
                return

            self.disp_viz_raw = disp_viz.copy()
            self.current_cmap = cmap_code # Raporda kullanmak için kaydet
            
            # Derinliği bir Renk Haritasına dökelim ki (Isı haritası gibi) güzel görünsün
            if self.switch_heatmap.get():
                disp_color = self.calculator.get_topographic_heatmap(disp_viz, cmap_code)
            else:
                disp_color = cv2.applyColorMap(disp_viz, cmap_code)
                
            disp_rgb = cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB)
            imD = Image.fromarray(disp_rgb)
            cD = ctk.CTkImage(light_image=imD, dark_image=imD, size=(300, 220))
            self.lbl_res2.configure(image=cD, text="")
            
            # Hacmi hesapla
            try:
                dens_str = self.entry_density.get()
                density_val = float(dens_str) if dens_str else 1.0
            except:
                density_val = 1.0

            vol, viz_img, stats = self.calculator.calculate(
                self.rect_L, 
                disparity, 
                foreground_threshold=thresh_v,
                roi=self.selected_roi,
                density=density_val
            )
            
            # YENI: Smoothing (Hareketli Ortalama)
            if self.switch_smoothing.get():
                self.volume_buffer.append(vol)
                self.mass_buffer.append(stats.get('mass_g', 0.0))
                vol = sum(self.volume_buffer) / len(self.volume_buffer)
                mass = sum(self.mass_buffer) / len(self.mass_buffer)
            else:
                mass = stats.get('mass_g', 0.0)

            self.current_volume = vol
            self.current_viz_img = viz_img
            self.current_stats = stats
            
            # UI Güncelle
            dist = stats.get('estimated_distance_cm', 0.0)
            dims = stats.get('dims_cm', (0,0,0))

            self.lbl_volume.configure(text=f"Hacim: {vol:.2f} cm³", text_color="#00e6e6")
            self.lbl_mass.configure(text=f"Ağırlık: {mass:.1f} gr")
            self.lbl_dims.configure(text=f"Boyut: {dims[0]:.1f}x{dims[1]:.1f}x{dims[2]:.1f} cm")
            self.lbl_dist.configure(text=f"Mesafe: ~{dist:.1f} cm")
            
            # Maskeyi göster
            viz_rgb = cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB)
            imV = Image.fromarray(viz_rgb)
            cV = ctk.CTkImage(light_image=imV, dark_image=imV, size=(300, 220))
            self.lbl_res3.configure(image=cV, text="")
            
            # 3D GÖRSELLEŞTİRME GÜNCELLE
            try:
                self.update_3d_view(disparity, thresh_v)
            except:
                pass # 3D çizim hatası kritik değil
        except Exception as e:
            self.lbl_volume.configure(text=f"Hesaplama Hatası: {str(e)}", text_color="red")

    def update_3d_view(self, disparity, threshold):
        # 3D Grafik ağır olabileceği için veriyi seyreltiyoruz (Downsampling)
        # Sadece threshold üzerindeki noktaları al
        mask = disparity > threshold
        if self.selected_roi:
            x1, y1, x2, y2 = self.selected_roi
            roi_mask = np.zeros(disparity.shape, dtype=bool)
            roi_mask[y1:y2, x1:x2] = True
            mask = mask & roi_mask

        points = np.argwhere(mask)
        if len(points) > 5000: # Max 5000 nokta göster
            idx = np.random.choice(len(points), 5000, replace=False)
            points = points[idx]
        
        if len(points) == 0: return

        # X, Y, Z (Depth)
        y_pts = points[:, 0]
        x_pts = points[:, 1]
        z_pts = disparity[mask]
        if len(z_pts) > 5000: z_pts = z_pts[idx]

        self.ax.clear()
        self.ax.set_facecolor('#2b2b2b')
        # Derinliği ters çeviriyoruz ki yakın noktalar yukarıda görünsün (Z-axis inversion for intuitive view)
        p3d = self.ax.scatter(x_pts, y_pts, z_pts, c=z_pts, cmap='jet', s=1)
        
        self.ax.set_title("Nesne 3D Nokta Bulutu", color='white')
        self.ax.set_xlabel("X (px)", color='white')
        self.ax.set_ylabel("Y (px)", color='white')
        self.ax.set_zlabel("Derinlik", color='white')
        self.ax.tick_params(colors='white')
        
        self.canvas.draw()

    def save_report(self):
        save_dir = "raporlar"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = os.path.join(save_dir, f"rapor_{timestamp}.pdf")
        
        # Geçici resimleri kaydet (PDF'e gömmek için)
        tmp_img_l = os.path.join(save_dir, "tmp_l.jpg")
        tmp_img_d = os.path.join(save_dir, "tmp_d.jpg")
        tmp_img_v = os.path.join(save_dir, "tmp_v.jpg")
        tmp_img_3d = os.path.join(save_dir, "tmp_3d.png")
        
        cv2.imwrite(tmp_img_l, self.rect_L)
        disp_color = cv2.applyColorMap(self.disp_viz_raw, getattr(self, "current_cmap", cv2.COLORMAP_JET))
        cv2.imwrite(tmp_img_d, disp_color)
        if self.current_viz_img is not None:
            cv2.imwrite(tmp_img_v, self.current_viz_img)
            
        # 3D Grafik görüntüsünü al
        self.fig.savefig(tmp_img_3d, facecolor='#2b2b2b')

        # PDF Oluşturma (ReportLab)
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Türkçe Karakter Destekli Yeni Stiller
        styles.add(ParagraphStyle(name='TR_Normal', fontName=FONT_NAME, fontSize=10, leading=12))
        styles.add(ParagraphStyle(name='TR_Title', fontName=FONT_NAME_BOLD, fontSize=18, leading=22, alignment=1, spaceAfter=20))
        styles.add(ParagraphStyle(name='TR_Bold', fontName=FONT_NAME_BOLD, fontSize=10, leading=12))

        elements = []

        # Başlık
        title = Paragraph(f"3D STEREO HACİM ANALİZ RAPORU", styles['TR_Title'])
        elements.append(title)
        elements.append(Paragraph(f"Tarih: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", styles['TR_Normal']))
        elements.append(Spacer(1, 12))

        # Analiz Özet Tablosu
        summary_vol = self.current_volume
        summary_mass = self.mass_buffer[-1] if self.mass_buffer else self.current_stats.get('mass_g', 0)
        
        # Eğer çoklu bakış varsa ortalamaları kullan
        if self.multi_view_data:
            summary_vol = sum(d['volume'] for d in self.multi_view_data) / len(self.multi_view_data)
            summary_mass = sum(d['mass'] for d in self.multi_view_data) / len(self.multi_view_data)
            elements.append(Paragraph(f"<b>NOT:</b> Bu rapor {len(self.multi_view_data)} farklı görünümün ortalamasını içermekmaktadır.", styles['TR_Normal']))
            elements.append(Spacer(1, 10))

        data = [
            ["Parametre", "Değer"],
            ["Hesaplanan Hacim (Ort)", f"{summary_vol:.2f} cm3"],
            ["Tahmini Ağırlık (Ort)", f"{summary_mass:.1f} gr"],
            ["Boyutlar (ExBxY)", f"{self.current_stats.get('dims_cm', (0,0,0))[0]:.1f} x {self.current_stats.get('dims_cm', (0,0,0))[1]:.1f} x {self.current_stats.get('dims_cm', (0,0,0))[2]:.1f} cm"],
            ["Kameraya Uzaklık", f"{self.current_stats.get('estimated_distance_cm', 0):.1f} cm"],
            ["Son Nesne Alanı", f"{self.current_stats.get('area_pixels', 0)} piksel"]
        ]
        table = Table(data, colWidths=[200, 200])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), FONT_NAME), # Tüm tabloya font uygula
            ('FONTNAME', (0, 0), (-1, 0), FONT_NAME_BOLD), # Başlığa kalın font
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
        elements.append(Spacer(1, 20))

        # Görüntüler (Yan yana)
        img_w, img_h = 240, 180
        img_l_rl = RLImage(tmp_img_l, width=img_w, height=img_h)
        img_d_rl = RLImage(tmp_img_d, width=img_w, height=img_h)
        
        img_table_data = [
            [img_l_rl, img_d_rl],
            [Paragraph("Orijinal Görüntü (Sol)", styles['TR_Normal']), Paragraph("Derinlik Haritası (WLS Filter)", styles['TR_Normal'])]
        ]
        img_table = Table(img_table_data)
        elements.append(img_table)
        elements.append(Spacer(1, 10))

        img_v_rl = RLImage(tmp_img_v, width=img_w, height=img_h)
        img_3d_rl = RLImage(tmp_img_3d, width=img_w, height=img_h)
        
        img_table_data2 = [
            [img_v_rl, img_3d_rl],
            [Paragraph("Analiz & Segmentasyon", styles['TR_Normal']), Paragraph("3D Nokta Bulutu Projeksiyonu", styles['TR_Normal'])]
        ]
        img_table2 = Table(img_table_data2)
        elements.append(img_table2)

        # Ayarlar Bilgisi
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("<b>Kullanılan Ayarlar:</b>", styles['TR_Normal']))
        params = f"Block Size: {(int(self.slider_block.get())*2)+1} | Num Disp: {int(self.slider_disp.get())*16} | Threshold: {self.slider_thresh.get()} | Materyal: {self.combo_material.get()}"
        elements.append(Paragraph(params, styles['TR_Normal']))

        # PDF'i Kaydet
        doc.build(elements)
        
        # Geçici dosyaları sil (Opsiyonel, temizlik için)
        for f in [tmp_img_l, tmp_img_d, tmp_img_v, tmp_img_3d]:
            if os.path.exists(f): os.remove(f)

        self.lbl_volume.configure(text=f"✓ PDF Raporu Hazırlandı: {pdf_path}", text_color="#2eb82e")

    def on_closing(self):
        self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
