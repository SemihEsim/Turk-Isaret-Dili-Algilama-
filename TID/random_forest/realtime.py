# realtime.py

import cv2
import numpy as np
import joblib
import mediapipe as mp
from collections import deque

from config import (
    TEK_EL_HARFLER,
    MODEL_TEK_EL, MODEL_IKI_EL,
    LE_TEK_EL, LE_IKI_EL
)
from feature_extractor import extract_from_frame, get_hand_landmarks_for_drawing

# ──────────────────────────────────────────────
# Model Yükle
# ──────────────────────────────────────────────

print("Modeller yükleniyor...")
model_tek = joblib.load(MODEL_TEK_EL)
model_iki = joblib.load(MODEL_IKI_EL)
le_tek    = joblib.load(LE_TEK_EL)
le_iki    = joblib.load(LE_IKI_EL)
print("Modeller hazır!\n")

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

# ──────────────────────────────────────────────
# Tahmin Stabilizasyonu
# ──────────────────────────────────────────────

# Son 15 tahmini tutarak titreme önlenir
# Örneğin 15 frame'den 11'i 'A' diyorsa → 'A' göster
tahmin_buffer = deque(maxlen=15)

# ──────────────────────────────────────────────
# Yardımcı Fonksiyonlar
# ──────────────────────────────────────────────

def tahmin_yap(features, el_sayisi):
    """
    El sayısına bakarak doğru modeli seçer ve tahmin yapar.
    
    Mantık:
    - 1 el görünüyorsa: önce tek el modelini dene (C,I,L,O,P,U,V)
      Ama iki el harfi de 1 elle yapılmış olabilir (diğer el kameraya girmediyse)
      Bu yüzden her iki modelden güven skoru alınır, yüksek olanı seçilir.
    - 2 el görünüyorsa: direkt iki el modeli kullanılır.
    """
    if el_sayisi == 2:
        # Kesinlikle iki el modeli
        pred_idx = model_iki.predict([features])[0]
        olasilik = model_iki.predict_proba([features])[0]
        guven    = olasilik[pred_idx]
        harf     = le_iki.inverse_transform([pred_idx])[0]
        return harf, guven, "iki_el"

    else:
        # Tek el görünüyor → her iki modelden skor al
        # Tek el modeli (63 özellik)
        pred_tek  = model_tek.predict([features])[0]
        prob_tek  = model_tek.predict_proba([features])[0]
        guven_tek = prob_tek[pred_tek]
        harf_tek  = le_tek.inverse_transform([pred_tek])[0]

        # İki el modeli de dene ama özelliği pad'le (126 özellik, ikinci el sıfır)
        features_padded = features + [0.0] * 63
        pred_iki  = model_iki.predict([features_padded])[0]
        prob_iki  = model_iki.predict_proba([features_padded])[0]
        guven_iki = prob_iki[pred_iki]
        harf_iki  = le_iki.inverse_transform([pred_iki])[0]

        # Daha yüksek güven skorlu modeli seç
        if guven_tek >= guven_iki:
            return harf_tek, guven_tek, "tek_el"
        else:
            return harf_iki, guven_iki, "iki_el"


def stabil_tahmin(yeni_harf):
    """Buffer'daki en sık tahmini döner (titreme önleme)."""
    tahmin_buffer.append(yeni_harf)
    return max(set(tahmin_buffer), key=tahmin_buffer.count)


def ekrana_yaz(frame, harf, guven, model_tipi, el_sayisi, stabil):
    """Ekranın sol üstüne bilgileri çizer."""
    h, w = frame.shape[:2]

    # Arka plan kutusu
    cv2.rectangle(frame, (0, 0), (320, 110), (20, 20, 20), -1)

    # Güvene göre renk: yeşil > %75, turuncu > %50, kırmızı < %50
    if guven > 0.75:
        renk = (0, 230, 0)
        durum = "Güçlü"
    elif guven > 0.50:
        renk = (0, 165, 255)
        durum = "Orta"
    else:
        renk = (0, 0, 220)
        durum = "Zayıf"

    # Tahmin harfi (büyük)
    cv2.putText(frame, f"Harf: {harf}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, renk, 3)

    # Detay bilgiler
    cv2.putText(frame, f"Guven: {guven:.0%}  [{durum}]", (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, renk, 2)
    cv2.putText(frame, f"Model: {model_tipi}  |  El: {el_sayisi}", (10, 102),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # Stabil değilse uyarı
    if not stabil:
        cv2.putText(frame, "Sabit tut...", (w - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)


# ──────────────────────────────────────────────
# Ana Döngü
# ──────────────────────────────────────────────

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("HATA: Kamera açılamadı!")
    exit()

print("Kamera açıldı.")
print("Çıkmak için 'q' | Bufferı sıfırlamak için 'r'\n")

guncel_harf  = "?"
guncel_guven = 0.0
guncel_model = "-"
guncel_el    = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame    = cv2.flip(frame, 1)  # Ayna görüntüsü (daha doğal)
    img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Özellik çıkar
    features, el_sayisi = extract_from_frame(img_rgb)

    if features is not None:
        harf, guven, model_tipi = tahmin_yap(features, el_sayisi)
        stabil_harf = stabil_tahmin(harf)

        guncel_harf  = stabil_harf
        guncel_guven = guven
        guncel_model = model_tipi
        guncel_el    = el_sayisi
        stabil       = (list(tahmin_buffer).count(stabil_harf) >= 10)
    else:
        # El görünmüyor
        tahmin_buffer.clear()
        stabil = False

    # El iskeletini çiz
    landmark_result = get_hand_landmarks_for_drawing(img_rgb)
    if landmark_result.multi_hand_landmarks:
        for hand_lm in landmark_result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_lm,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

    # Bilgileri ekrana yaz
    if features is not None:
        ekrana_yaz(frame, guncel_harf, guncel_guven,
                   guncel_model, guncel_el, stabil)
    else:
        cv2.rectangle(frame, (0, 0), (260, 45), (20, 20, 20), -1)
        cv2.putText(frame, "El bulunamadı", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)

    cv2.imshow("TID Alfabe Tahmini", frame)

    tus = cv2.waitKey(1) & 0xFF
    if tus == ord('q'):
        break
    elif tus == ord('r'):
        tahmin_buffer.clear()
        print("Buffer sıfırlandı.")

cap.release()
cv2.destroyAllWindows()
print("Kapatıldı.")