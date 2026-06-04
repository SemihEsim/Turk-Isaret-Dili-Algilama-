import cv2
import numpy as np
import mediapipe as mp
import time
import os
import io
from gtts import gTTS
import pygame
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image  # Türkçe karakter çizimi için eklendi

# ==================================================
# 1. TÜRKÇE SES MOTORU AYARLARI
# ==================================================
def play_turkish_audio(text):
    try:
        pygame.mixer.init()
        tts = gTTS(text=text.lower(), lang='tr', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        pygame.mixer.music.load(fp)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            cv2.waitKey(10)
        pygame.mixer.quit()
    except Exception as e:
        print(f"Seslendirme Hatası: {e}")

# ==================================================
# 2. GÜNCEL TÜRKÇE ETİKET LİSTESİ VE MODEL (70 Kelime)
# ==================================================
# Gönderdiğin görseldeki güncellenmiş Türkçe karakterli liste birebir eklendi
actions = np.array([
    'abla', 'acıkmak', 'afiyet_olsun', 'ağabey', 'ağlamak', 'aile', 'alışveriş', 'anne', 'arkadaş', 'baba',
    'bahçe', 'bakmak', 'bayrak', 'bebek', 'ben', 'biz', 'çalışmak', 'çarşamba', 'çocuk', 'çorba',
    'cuma', 'cumartesi', 'dede', 'doktor', 'dün', 'ev', 'evet', 'futbol', 'gülmek', 'hasta',
    'hastane', 'hayır', 'hoşçakal', 'içmek', 'ilaç', 'iyi', 'kardeş', 'kim', 'kötü', 'nasıl',
    'neden', 'nerede', 'öğretmen', 'okul', 'olmaz', 'olur', 'onlar', 'özür_dilemek', 'pazar', 'pazartesi',
    'perşembe', 'rica_etmek', 'salı', 'sen', 'sevmek', 'siz', 'tamam', 'tatlı', 'teşekkür', 'tuvalet',
    'var', 'yanlış', 'yapmak', 'yardım', 'yarın', 'yemek', 'yemek_pişirmek', 'yok', 'zaman', 'zor'
])

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, 'tid_70_kelime_cnn1d_best.h5')

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH, compile=False)
    print(">> Model ve güncel Türkçe etiketler başarıyla yüklendi.")
else:
    print(f"!! HATA: {MODEL_PATH} bulunamadı!")
    exit()

SEQUENCE_LENGTH = 60 

# Windows'un standart Arial yazı tipini yüklüyoruz (Türkçe karakterleri destekler)
try:
    font_path = "arial.ttf"
    font_large = ImageFont.truetype(font_path, 24) # Tahminler ve Durum için
    font_small = ImageFont.truetype(font_path, 18) # Alt bar cümle alanı için
except:
    # Eğer sistemde font bulunamazsa varsayılanı kullanır
    font_large = ImageFont.load_default()
    font_small = ImageFont.load_default()

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils 

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    return np.concatenate([lh, rh, pose])

# OYNAK TÜRKÇE KARAKTER YAZMA FONKSİYONU
def draw_turkish_text(img, text, position, font, color):
    """OpenCV görüntüsüne Pillow kullanarak kusursuz Türkçe karakter basar"""
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

# ==================================================
# ISI HARİTASI FONKSİYONU
# ==================================================
def create_blended_heatmap(shape, sequences, bg_img, prediction_text):
    heatmap_canvas = np.zeros((shape[0], shape[1]), dtype=np.float32)
    for frame_points in sequences:
        for i in range(0, 126, 3):  
            x = int(frame_points[i] * shape[1])
            y = int(frame_points[i+1] * shape[0])
            if 0 < x < shape[1] and 0 < y < shape[0]:
                cv2.circle(heatmap_canvas, (x, y), 12, 1, -1)
    
    heatmap_canvas = cv2.GaussianBlur(heatmap_canvas, (65, 65), 0)
    max_val = np.max(heatmap_canvas)
    if max_val > 0:
        heatmap_canvas = (heatmap_canvas / max_val) * 255
        
    heatmap_colored = cv2.applyColorMap(heatmap_canvas.astype(np.uint8), cv2.COLORMAP_JET)
    gray = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
    bg_unheated = cv2.bitwise_and(bg_img, bg_img, mask=mask_inv)
    blended_heated = cv2.addWeighted(heatmap_colored, 0.6, bg_img, 0.4, 0)
    bg_heated = cv2.bitwise_and(blended_heated, blended_heated, mask=mask)
    final_blended = cv2.add(bg_unheated, bg_heated)
  
    h_shape, w_shape = shape
    cv2.rectangle(final_blended, (0, h_shape-70), (w_shape, h_shape), (20, 20, 20), -1)
    
    text_color = (255, 0, 0) if "EL ALGILANMADI" in prediction_text or "ANLASILMADI" in prediction_text else (0, 255, 0)
    # Isı haritası alt barındaki Türkçe yazıyı güncelledik (Pillow ile RGB formatında)
    final_blended = draw_turkish_text(final_blended, f"ANALİZ SONUCU: {prediction_text}", (20, h_shape-50), font_large, text_color[::-1])
    
    return final_blended

# ==================================================
# 3. ANA DÖNGÜ VE DURUM KONTROLÜ
# ==================================================
cap = cv2.VideoCapture(0)

KAYIT_SURESI = 3.0   
SONUC_SURESI = 4.0   
CUMLE_BITIRME_SURESI = 3.0 

state = "KAYIT"  
state_start_time = time.time()

recorded_sequence = []
hand_detected_frames = 0
last_prediction = "BEKLENİYOR..."

current_sentence = []      
final_sentence_output = "" 
no_hand_start_time = None  

with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1) 
        h, w, _ = frame.shape
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        elapsed = time.time() - state_start_time
        ekranda_el_var_mi = bool(results.left_hand_landmarks or results.right_hand_landmarks)

        # --- OTOMATİK CÜMLE BİTİRME ---
        if state == "DURDURULDU":
            if ekranda_el_var_mi:
                state = "KAYIT"
                state_start_time = time.time()
                recorded_sequence = []
                hand_detected_frames = 0
                current_sentence = [] 
                final_sentence_output = ""
                last_prediction = "SİSTEM AKTİF"
                no_hand_start_time = None
                try: cv2.destroyWindow('Isi Haritasi ') 
                except: pass
                continue
        else:
            if not ekranda_el_var_mi:
                if no_hand_start_time == None:
                    no_hand_start_time = time.time()
                else:
                    suresiz_gecen = time.time() - no_hand_start_time
                    if suresiz_gecen >= CUMLE_BITIRME_SURESI and len(current_sentence) > 0:
                        state = "DURDURULDU"
                        final_sentence_output = " ".join(current_sentence).upper()
                        last_prediction = "CÜMLE BİTTİ!"
                        
                        try: cv2.destroyWindow('Isi Haritasi ') 
                        except: pass
                        
                        cv2.rectangle(image, (30, h//2 - 40), (w - 30, h//2 + 40), (20, 20, 20), -1)
                        image = draw_turkish_text(image, "SESLİ OKUNUYOR...", (w//2-120, h//2-15), font_large, (0, 255, 0))
                        cv2.imshow('TID Algilama', image)
                        cv2.waitKey(100)
                        
                        play_turkish_audio(final_sentence_output)
                        continue
            else:
                no_hand_start_time = None

        # --- DURUM MAKİNESİ AKIŞI ---
        if state == "KAYIT":
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            recorded_sequence.append(extract_keypoints(results))
            if ekranda_el_var_mi:
                hand_detected_frames += 1

            cv2.rectangle(image, (0, 0), (w, 60), (150, 0, 0), -1) # BGR Formatı
            image = draw_turkish_text(image, "DURUM: KAYIT YAPILIYOR... (HAREKETİ YAPIN)", (15, 15), font_large, (255, 255, 255))
            
            kalan_oran = min(1.0, elapsed / KAYIT_SURESI)
            cv2.rectangle(image, (0, 60), (int(w * kalan_oran), 70), (0, 0, 255), -1)

            if elapsed >= KAYIT_SURESI:
                if hand_detected_frames > 10:
                    seq = np.array(recorded_sequence)
                    if len(seq) >= SEQUENCE_LENGTH:
                        input_data = seq[np.linspace(0, len(seq)-1, SEQUENCE_LENGTH, dtype=int)]
                    else:
                        input_data = np.concatenate([seq, [seq[-1]] * (SEQUENCE_LENGTH - len(seq))])
                    
                    res = model.predict(np.expand_dims(input_data, axis=0), verbose=0)[0]
                    confidence = np.max(res)
                    
                    if confidence > 0.50:
                        detected_word = actions[np.argmax(res)].upper()
                        last_prediction = f"{detected_word} (%{confidence*100:.0f})"
                        
                        if len(current_sentence) == 0 or current_sentence[-1] != detected_word:
                            current_sentence.append(detected_word)
                    else:
                        last_prediction = "ANLAŞILMADI"
                    
                    heatmap_img = create_blended_heatmap((h, w), recorded_sequence, image.copy(), last_prediction)
                    cv2.imshow('Isi Haritasi ', heatmap_img)
                else:
                    last_prediction = "EL ALGILANMADI"
                    try: cv2.destroyWindow('Isi Haritasi ') 
                    except: pass

                state = "SONUC"
                state_start_time = time.time()

        elif state == "SONUC":
            cv2.rectangle(image, (0, 0), (w, 60), (0, 150, 0), -1)
            image = draw_turkish_text(image, "DURUM: ANALİZ TAMAMLANDI (BEKLEYİN 4 SN)", (15, 15), font_large, (255, 255, 255))
            
            kalan_oran = max(0, (SONUC_SURESI - elapsed) / SONUC_SURESI)
            cv2.rectangle(image, (0, 60), (int(w * kalan_oran), 70), (0, 255, 0), -1)

            if elapsed >= SONUC_SURESI:
                state = "KAYIT"
                state_start_time = time.time()
                recorded_sequence = []
                hand_detected_frames = 0
                last_prediction = "BEKLENİYOR..."
                try: cv2.destroyWindow('Isi Haritasi ') 
                except: pass

        elif state == "DURDURULDU":
            cv2.rectangle(image, (0, 0), (w, 60), (0, 0, 100), -1)
            image = draw_turkish_text(image, "SİSTEM KİLİTLİ - CÜMLE BİTTİ", (15, 15), font_large, (255, 255, 255))
            
            cv2.rectangle(image, (30, h//2 - 50), (w - 30, h//2 + 40), (20, 20, 20), -1)
            image = draw_turkish_text(image, f"Okunan: '{final_sentence_output}'", (40, h//2 - 25), font_small, (0, 255, 255))
            image = draw_turkish_text(image, "Yeni cümle için kameraya ELİNİZİ GÖSTERİN.", (40, h//2 + 10), font_small, (0, 255, 0))

        # ==================================================
        # ORTAK ALT ARAYÜZ (PILLOW ENTEGRASYONLU TÜRKÇE ALAN)
        # ==================================================
        cv2.rectangle(image, (0, h-90), (w, h), (30, 30, 30), -1)
        
        text_color = (255, 0, 0) if last_prediction in ["EL ALGILANMADI", "ANLAŞILMADI", "CÜMLE BİTTİ!"] else (255, 255, 0)
        image = draw_turkish_text(image, f"TAHMİN: {last_prediction}", (20, h-80), font_large, text_color)

        if len(current_sentence) > 0:
            active_sentence_str = " -> ".join(current_sentence)
            image = draw_turkish_text(image, f"Cümle: {active_sentence_str}", (20, h-35), font_small, (0, 255, 0))
        elif final_sentence_output:
            image = draw_turkish_text(image, f"Nihai Cümle: {final_sentence_output}.", (20, h-35), font_small, (255, 170, 0))

        cv2.imshow('TID Algilama', image)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()