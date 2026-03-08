# config.py
import os
# Tek el kullanan harfler
TEK_EL_HARFLER = {'C', 'I', 'L', 'O', 'P', 'U', 'V'}

# İki el kullanan harfler
IKI_EL_HARFLER = {
    'A', 'B', 'D', 'E', 'F', 'G', 'H',
    'J', 'K', 'M', 'N', 'R', 'S',
    'T', 'Y', 'Z'
}

# Model kayıt yolları
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_TEK_EL = os.path.join(BASE_DIR, "models", "model_tek_el.pkl")
MODEL_IKI_EL = os.path.join(BASE_DIR, "models", "model_iki_el.pkl")
LE_TEK_EL    = os.path.join(BASE_DIR, "models", "le_tek_el.pkl")
LE_IKI_EL    = os.path.join(BASE_DIR, "models", "le_iki_el.pkl")

# Dataset yolları
TRAIN_PATH = "dataset/train"
TEST_PATH  = "dataset/test"

# MediaPipe güven eşikleri
DETECTION_CONFIDENCE  = 0.3   # Eğitim için düşük tut (daha fazla foto işlenir)
TRACKING_CONFIDENCE   = 0.5   # Gerçek zamanlı için
REALTIME_CONFIDENCE   = 0.7   # Gerçek zamanlı detection