print("--- KAMERA KODU BAŞLATILIYOR ---")
import sys
import time

print("1. cv2 ve diğer kütüphaneler yükleniyor...")
import cv2
import numpy as np
import torch

print("2. Model import ediliyor...")
from Model import TIDTCN

print("3. mediapipe yükleniyor...")
try:
    import mediapipe as mp
    import mediapipe.python.solutions as mp_solutions
    mp.solutions = mp_solutions
    HAS_MP = True
    print(">> mediapipe başarıyla yüklendi!")
except ImportError as e:
    print(f"HATA: mediapipe yüklenemedi: {e}")
    HAS_MP = False

import json
import collections
import threading
import queue
import argparse

NUM_FRAMES  = 32
FEATURE_DIM = 138
POSE_IDX    = [11, 12, 15, 16]

GREEN  = (0, 210, 80)
BLUE   = (200, 120, 0)
RED    = (0, 50, 220)
WHITE  = (255, 255, 255)
GRAY   = (150, 150, 150)
DARK   = (20, 20, 20)
YELLOW = (0, 200, 255)

class KeypointExtractor:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )

    def extract(self, frame_bgr):
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        feat = np.zeros(FEATURE_DIM, dtype=np.float32)

        h_res = self.hands.process(rgb)
        if h_res.multi_hand_landmarks and h_res.multi_handedness:
            for lm, hd in zip(h_res.multi_hand_landmarks, h_res.multi_handedness):
                label = hd.classification[0].label
                pts   = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32).flatten()
                if label == "Left":
                    feat[0:63]   = pts
                else:
                    feat[63:126] = pts

        p_res = self.pose.process(rgb)
        if p_res.pose_landmarks:
            for i, pi in enumerate(POSE_IDX):
                lm = p_res.pose_landmarks.landmark[pi]
                feat[126 + i*3: 126 + i*3 + 3] = [lm.x, lm.y, lm.z]
        return feat

    def close(self):
        self.hands.close()
        self.pose.close()

class PredictorThread(threading.Thread):
    def __init__(self, model, classes, device):
        super().__init__(daemon=True)
        self.model   = model
        self.classes = classes
        self.device  = device
        self.in_q    = queue.Queue(maxsize=1)
        self.out_q   = queue.Queue(maxsize=3)
        self.running = True

    def submit(self, keypoints):
        try: self.in_q.get_nowait()
        except queue.Empty: pass
        self.in_q.put(keypoints)

    def get_result(self):
        try: return self.out_q.get_nowait()
        except queue.Empty: return None

    @torch.no_grad()
    def run(self):
        while self.running:
            try:
                kp = self.in_q.get(timeout=0.5)
            except queue.Empty:
                continue

            arr = np.stack(kp, axis=0).astype(np.float32)

            nonzero = arr[arr != 0]
            if len(nonzero) > 0:
                mn, mx = nonzero.min(), nonzero.max()
                if mx > mn:
                    mask = (arr != 0)
                    arr[mask] = 2.0 * (arr[mask] - mn) / (mx - mn) - 1.0

            x = torch.from_numpy(arr).unsqueeze(0).to(self.device)
            self.model.eval()
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
            top5_i = probs.argsort()[::-1][:5]
            top5   = [(self.classes[i], float(probs[i])) for i in top5_i]

            try: self.out_q.put_nowait(top5)
            except queue.Full: pass

    def stop(self): self.running = False

def overlay_rect(img, x, y, w, h, color, alpha=0.65):
    sub = img[y:y+h, x:x+w]
    rect = np.full_like(sub, color)
    cv2.addWeighted(rect, alpha, sub, 1 - alpha, 0, sub)
    img[y:y+h, x:x+w] = sub

def txt(img, text, pos, scale, color, thick=1):
    x, y = pos
    cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+1)
    cv2.putText(img, text, (x, y),     cv2.FONT_HERSHEY_SIMPLEX, scale, color,   thick)

def draw_hud(frame, pred, conf, top5, progress, history, fps, recording):
    H, W = frame.shape[:2]
    overlay_rect(frame, 8, 8, 290, 210, DARK)
    txt(frame, "Tahmin:", (18, 35), 0.52, GRAY)
    disp = (pred[:24] + "..") if len(pred) > 24 else pred
    txt(frame, disp, (18, 65), 0.9, GREEN, thick=2)
    txt(frame, f"Guven: %{conf*100:.1f}", (18, 88), 0.5, WHITE)

    bw = 255
    cv2.rectangle(frame, (18, 96), (18+bw, 106), (60,60,60), -1)
    filled = int(bw * conf)
    clr = GREEN if conf > 0.6 else YELLOW if conf > 0.35 else RED
    if filled > 0:
        cv2.rectangle(frame, (18, 96), (18+filled, 106), clr, -1)

    txt(frame, "Top-5:", (18, 126), 0.43, GRAY)
    for i, (c, p) in enumerate(top5[:5]):
        yy  = 142 + i * 17
        bw2 = int(250 * p)
        if bw2 > 0:
            cv2.rectangle(frame, (18, yy-10), (18+bw2, yy-2), (0, int(60+120*p), 30), -1)
        txt(frame, f"{c[:20]:<20} {p*100:4.1f}%", (18, yy), 0.37, WHITE)

    px, py = W - 215, 8
    overlay_rect(frame, px, py, 207, 55, DARK)
    txt(frame, "Kayit penceresi:", (px+6, py+18), 0.42, GRAY)
    bclr = RED if recording else BLUE
    cv2.rectangle(frame, (px+6, py+24), (px+6+190, py+40), (60,60,60), -1)
    filled2 = int(190 * progress)
    if filled2 > 0:
        cv2.rectangle(frame, (px+6, py+24), (px+6+filled2, py+40), bclr, -1)
    txt(frame, f"{int(progress*100)}%", (px+200, py+40), 0.38, WHITE)

    if recording:
        cv2.circle(frame, (W-18, 18), 8, RED, -1)
        txt(frame, "REC", (W-52, 24), 0.44, RED)

    txt(frame, f"FPS:{fps:.0f}", (W-70, H-10), 0.4, GRAY)

    if history:
        overlay_rect(frame, W//2-195, H-52, 390, 44, DARK)
        txt(frame, "Son:", (W//2-185, H-32), 0.4, GRAY)
        txt(frame, "  ->  ".join(history[-4:]), (W//2-185, H-14), 0.46, GREEN)

    overlay_rect(frame, 8, H-52, 255, 44, DARK)
    txt(frame, "r: kayit  a: otomatik  s: anlik  q: cikis", (14, H-14), 0.36, GRAY)

def run(args):
    print("\n>> ANA KOD ÇALIŞTIRILIYOR...")
    if not HAS_MP:
        print("HATA: mediapipe bulunamadı, çıkılıyor.")
        return

    print(">> Cihaz ayarlanıyor...")
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f">> Model Yükleniyor: {args.checkpoint}")
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        classes = ckpt["classes"]
        model   = TIDTCN(num_classes=len(classes), input_dim=FEATURE_DIM)
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()
        print(f">> Model Başarıyla Yüklendi! ({len(classes)} Sınıf)")
    except Exception as e:
        print(f"HATA: Model yüklenirken çöktü! Hata Detayı: {e}")
        return

    print(">> İş Parçacıkları (Thread) başlatılıyor...")
    extractor = KeypointExtractor()
    predictor = PredictorThread(model, classes, device)
    predictor.start()

    print(f">> Kameraya {args.camera} numarası üzerinden bağlanılıyor (DirectShow)...")
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"\nHATA: {args.camera} Numaralı Kamera Açılamadı!")
        print("Çözüm 1: Kameranızın kapağı kapalı olabilir.")
        print("Çözüm 2: Komutun sonuna --camera 1 veya --camera 2 eklemeyi deneyin.")
        return

    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    win_size   = int(args.window * 30)
    buf        = collections.deque(maxlen=win_size)
    mode         = "auto"
    manual_buf   = []
    manual_rec   = False
    last_auto_t  = 0
    pred    = "—"
    conf    = 0.0
    top5    = []
    history = []
    fps_dq  = collections.deque(maxlen=30)

    print(">> KAMERA PENCERESİ AÇILIYOR... (Ekrana yansıması 2-3 saniye sürebilir)")

    while True:
        ret, frame = cap.read()
        if not ret: 
            print("HATA: Kameradan görüntü alınamıyor. Bağlantı koptu!")
            break

        fps_dq.append(time.time())
        fps = (len(fps_dq)-1) / max(fps_dq[-1]-fps_dq[0], 1e-6) if len(fps_dq) > 1 else 0

        kp = extractor.extract(frame)
        buf.append(kp)
        if manual_rec: manual_buf.append(kp)

        progress  = 0.0
        recording = manual_rec
        if mode == "auto":
            progress = len(buf) / win_size
            now = time.time()
            if len(buf) >= win_size and now - last_auto_t >= args.window:
                kps = list(buf)
                idx = np.linspace(0, len(kps)-1, NUM_FRAMES, dtype=int)
                predictor.submit([kps[i] for i in idx])
                last_auto_t = now
        else:
            if manual_rec:
                progress = min(len(manual_buf) / win_size, 1.0)
                if len(manual_buf) >= win_size:
                    idx = np.linspace(0, len(manual_buf)-1, NUM_FRAMES, dtype=int)
                    predictor.submit([manual_buf[i] for i in idx])
                    manual_buf.clear()
                    manual_rec = False

        res = predictor.get_result()
        if res:
            pred, conf = res[0]
            top5       = res
            if not history or history[-1] != pred:
                history.append(pred)
                if len(history) > 10: history.pop(0)

        draw_hud(frame, pred, conf, top5, progress, history, fps, recording)
        cv2.imshow("TID TCN - Isaret Dili Tahmini", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'):
            if mode == "auto": mode = "manual"
            else:
                if not manual_rec:
                    manual_rec = True
                    manual_buf.clear()
                else:
                    if len(manual_buf) >= NUM_FRAMES:
                        idx = np.linspace(0, len(manual_buf)-1, NUM_FRAMES, dtype=int)
                        predictor.submit([manual_buf[i] for i in idx])
                    manual_buf.clear()
                    manual_rec = False
        elif key == ord('a'):
            mode = "auto"
            manual_rec = False
            manual_buf.clear()
        elif key == ord('s'):
            if len(buf) >= NUM_FRAMES:
                kps = list(buf)
                idx = np.linspace(0, len(kps)-1, NUM_FRAMES, dtype=int)
                predictor.submit([kps[i] for i in idx])

    predictor.stop()
    extractor.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print(">> ARGÜMANLAR AYRIŞTIRILIYOR...")
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--camera",     type=int,   default=0)
    p.add_argument("--window",     type=float, default=3.0)
    args = p.parse_args()
    run(args)