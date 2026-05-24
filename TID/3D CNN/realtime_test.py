import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# --- AYARLAR ---
SEQUENCE_LENGTH = 32 
IMAGE_SIZE = 64
CLASSES = ["abla", "bayrak", "ev", "fil", "hastane", "kitap", "misafir", "para", "saat", "yarin"]

print("Hareket tabanli model yukleniyor...")
model = load_model('motion_3dcnn_model.h5')

cap = cv2.VideoCapture(0)
# Fark almak için SEQUENCE_LENGTH + 1 kare tutmaliyiz
raw_frames_queue = deque(maxlen=SEQUENCE_LENGTH + 1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    # Gri tonlama ve boyutlandırma
    resized = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    raw_frames_queue.append(gray)

    # Elimizde fark alacak kadar (17 kare) veri birikince:
    if len(raw_frames_queue) == SEQUENCE_LENGTH + 1:
        diff_sequence = []
        
        # Ardışık karelerin farkını hesapla
        for i in range(1, len(raw_frames_queue)):
            diff = cv2.absdiff(raw_frames_queue[i], raw_frames_queue[i-1])
            norm_diff = diff / 255.0
            rgb_diff = np.stack((norm_diff,)*3, axis=-1)
            diff_sequence.append(rgb_diff)
            
        # Tahmin yap
        input_data = np.expand_dims(diff_sequence, axis=0)
        predictions = model.predict(input_data, verbose=0)[0]
        idx = np.argmax(predictions)
        conf = predictions[idx]

        if conf > 0.65:
            label = f"{CLASSES[idx]} %{int(conf*100)}"
            cv2.putText(frame, label, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Ekranda ne olduğunu gör (İsteğe bağlı: fark görüntüsünü göster)
    cv2.imshow("Hareket Odakli Tanimlama", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()