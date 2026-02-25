# feature_extractor.py

import cv2
import numpy as np
import mediapipe as mp
from config import TEK_EL_HARFLER, DETECTION_CONFIDENCE

mp_hands = mp.solutions.hands

# Eğitim için iki ayrı hands nesnesi (static_image_mode=True)
_hands_tek = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=DETECTION_CONFIDENCE
)

_hands_iki = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=DETECTION_CONFIDENCE
)

# Gerçek zamanlı için (static_image_mode=False → daha hızlı tracking)
_hands_rt_tek = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

_hands_rt_iki = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)


def _normalize(landmarks):
    """
    21 el noktasını bileğe (nokta 0) göre normalize eder.
    Böylece elin ekrandaki pozisyonu ve boyutu fark etmez.
    Çıktı: 63 float (21 nokta × x,y,z)
    """
    wrist_x = landmarks[0].x
    wrist_y = landmarks[0].y
    wrist_z = landmarks[0].z
    features = []
    for lm in landmarks:
        features.append(lm.x - wrist_x)
        features.append(lm.y - wrist_y)
        features.append(lm.z - wrist_z)
    return features


def extract_tek_el(img_rgb, realtime=False):
    """
    Tek el harfler için özellik çıkarır.
    Çıktı: 63 float veya None (el bulunamazsa)
    """
    hands = _hands_rt_tek if realtime else _hands_tek
    result = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        return None

    return _normalize(result.multi_hand_landmarks[0].landmark)


def extract_iki_el(img_rgb, realtime=False):
    """
    İki el harfler için özellik çıkarır.
    Sol el + sağ el koordinatları birleştirilir.
    Bir el bulunamazsa o el sıfırlanır (padding).
    Çıktı: 126 float veya None (hiç el bulunamazsa)
    """
    hands = _hands_rt_iki if realtime else _hands_iki
    result = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        return None

    sol_el = [0.0] * 63
    sag_el = [0.0] * 63

    for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
        handedness = result.multi_handedness[i].classification[0].label
        features = _normalize(hand_landmarks.landmark)
        if handedness == "Left":
            sol_el = features
        else:
            sag_el = features

    # İki el de sıfırsa (handedness alınamadıysa) ilk eli sağ say
    if sol_el == [0.0] * 63 and sag_el == [0.0] * 63:
        return None

    return sol_el + sag_el  # 126 özellik


def extract_from_path(image_path, harf):
    """
    Dosya yolundan harfe göre doğru özellik çıkarımını yapar.
    Eğitim sırasında kullanılır.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if harf.upper() in TEK_EL_HARFLER:
        return extract_tek_el(img_rgb, realtime=False)
    else:
        return extract_iki_el(img_rgb, realtime=False)


def extract_from_frame(img_rgb):
    """
    Gerçek zamanlı kameradan gelen frame'i işler.
    Kaç el göründüğüne bakarak doğru özelliği döner.
    Döndürür: (features, el_sayisi) veya (None, 0)
    """
    # Önce iki eli dene
    result_iki = _hands_rt_iki.process(img_rgb)

    if not result_iki.multi_hand_landmarks:
        return None, 0

    el_sayisi = len(result_iki.multi_hand_landmarks)

    if el_sayisi == 1:
        # Tek el görünüyor
        features = _normalize(result_iki.multi_hand_landmarks[0].landmark)
        return features, 1
    else:
        # İki el görünüyor
        sol_el = [0.0] * 63
        sag_el = [0.0] * 63
        for i, hand_landmarks in enumerate(result_iki.multi_hand_landmarks):
            handedness = result_iki.multi_handedness[i].classification[0].label
            features = _normalize(hand_landmarks.landmark)
            if handedness == "Left":
                sol_el = features
            else:
                sag_el = features
        return sol_el + sag_el, 2


def get_hand_landmarks_for_drawing(img_rgb):
    """Sadece çizim için landmark ve bağlantıları döner."""
    result = _hands_rt_iki.process(img_rgb)
    return result