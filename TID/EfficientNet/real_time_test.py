"""
3_realtime_efficientnet.py - EfficientNetB0 Gercek Zamanli Kamera Testi
Tuslar: Q=Cik  S=Ekran goruntusu  R=Temizle  +/-=ROI boyutu
"""

import cv2
import numpy as np
import tensorflow as tf
import json
import time
from collections import deque

IMG_SIZE         = 224
MODEL_PATH       = "EfficientNet/results/EfficientNetB0_TID_final"
CLASS_NAMES_PATH = "EfficientNet/results/class_names_eff.json"
SMOOTH_WINDOW    = 12
CONF_THRESHOLD   = 0.55
ROI_RATIO        = 0.45
COLOR            = (0, 200, 100)

def preprocess(roi):
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return tf.expand_dims(img, axis=0)

def get_roi(frame, ratio):
    h, w = frame.shape[:2]
    size = int(min(h, w) * ratio)
    cx, cy = w // 2, h // 2
    x1, y1 = cx - size // 2, cy - size // 2
    x2, y2 = cx + size // 2, cy + size // 2
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

def draw_roi_box(frame, bbox):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, 2)
    L = 22
    for px, py, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (px, py), (px+dx*L, py), COLOR, 3)
        cv2.line(frame, (px, py), (px, py+dy*L), COLOR, 3)
    cv2.putText(frame, "Elinizi buraya getirin", (x1+5, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

def draw_badge(frame, label, conf):
    h, w = frame.shape[:2]
    bx, by, size = w-130, 70, 90
    cv2.rectangle(frame, (bx, by), (bx+size, by+size), COLOR, -1)
    cv2.rectangle(frame, (bx, by), (bx+size, by+size), (255,255,255), 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 2.5, 3)
    cv2.putText(frame, label,
                (bx+(size-tw)//2, by+(size+th)//2-5),
                cv2.FONT_HERSHEY_DUPLEX, 2.5, (0,0,0), 3)
    cv2.putText(frame, f"Guven: {conf:.0%}", (bx, by+size+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR, 2)

def draw_conf_bar(frame, x, y, conf, width=200):
    cv2.rectangle(frame, (x, y), (x+width, y+16), (40,40,40), -1)
    cv2.rectangle(frame, (x, y), (x+int(width*conf), y+16), COLOR, -1)
    cv2.rectangle(frame, (x, y), (x+width, y+16), (100,100,100), 1)

def draw_top5(frame, avg_pred, class_names):
    top5 = np.argsort(avg_pred)[::-1][:5]
    cv2.putText(frame, "Top-5:", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)
    for i, idx in enumerate(top5):
        y = 100 + i*30
        cv2.putText(frame, class_names[idx], (10, y+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR, 1)
        draw_conf_bar(frame, 40, y, avg_pred[idx], width=150)
        cv2.putText(frame, f"{avg_pred[idx]:.0%}", (195, y+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

def draw_header(frame, fps, roi_ratio):
    w = frame.shape[1]
    cv2.rectangle(frame, (0,0), (w,55), (15,15,15), -1)
    cv2.putText(frame, "EfficientNetB0 - TID El Alfabesi",
                (w//2-220, 38), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 2)
    cv2.putText(frame, f"FPS:{fps:.0f}", (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,100), 2)
    cv2.putText(frame, f"ROI:{roi_ratio:.0%}", (w-80, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150,150,150), 1)

def draw_footer(frame, history):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0,h-50), (w,h), (15,15,15), -1)
    cv2.putText(frame, "Gecmis: " + " ".join(history[-18:]),
                (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.putText(frame, "Q:Cik  S:Ekran goruntusu  R:Temizle  +/-:ROI boyutu",
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120,120,120), 1)

def run():
    with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
        class_names = json.load(f)

    print("Model yukleniyor...")
    loaded = tf.saved_model.load(MODEL_PATH)
    infer  = loaded.signatures["serving_default"]
    
    # Çıktı tensor adını bul
    output_key = list(infer.structured_outputs.keys())[0]
    print(f"Model hazir! ({len(class_names)} sinif)\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    buffer     = deque(maxlen=SMOOTH_WINDOW)
    history    = []
    fps_buf    = deque(maxlen=30)
    prev_time  = time.time()
    roi_ratio  = ROI_RATIO
    prev_label = ""
    shot_n     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        now = time.time()
        fps_buf.append(1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        fps = np.mean(fps_buf)

        roi, bbox = get_roi(frame, roi_ratio)
        if roi.size == 0:
            continue

        inp  = preprocess(roi)
        out  = infer(inp)
        pred = out[output_key].numpy()[0]

        buffer.append(pred)
        avg   = np.mean(buffer, axis=0)
        idx   = np.argmax(avg)
        conf  = float(avg[idx])
        label = class_names[idx] if conf >= CONF_THRESHOLD else "?"

        if label != "?" and label != prev_label:
            history.append(label)
            prev_label = label
            if len(history) > 40:
                history.pop(0)

        draw_header(frame, fps, roi_ratio)
        draw_roi_box(frame, bbox)
        draw_badge(frame, label, conf)
        draw_top5(frame, avg, class_names)

        x1, y1, x2, y2 = bbox
        draw_conf_bar(frame, x1, y2+8, conf, width=x2-x1)

        draw_footer(frame, history)
        cv2.imshow("TID - EfficientNetB0", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fn = f"screenshot_eff_{shot_n}.png"
            cv2.imwrite(fn, frame)
            print(f"Kaydedildi: {fn}")
            shot_n += 1
        elif key == ord('r'):
            history.clear()
            buffer.clear()
            prev_label = ""
            print("Temizlendi")
        elif key == ord('+'):
            roi_ratio = min(roi_ratio + 0.02, 0.75)
        elif key == ord('-'):
            roi_ratio = max(roi_ratio - 0.02, 0.15)

    cap.release()
    cv2.destroyAllWindows()
    print("Kapatildi.")

if __name__ == "__main__":
    run()