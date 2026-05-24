"""
TID SlowFast — Gerçek Zamanlı Kamera Tahmini (Keypoint tabanlı)
================================================================

Kamera açıldıktan sonra:
  • MediaPipe ile el + pose landmark'ları çıkarılır
  • 3 saniye keypoint biriktirir
  • Otomatik tahmin üretir → ekranda gösterir
  • 'r' tuşu: Manuel kayıt başlat/bitir
  • 'q' tuşu: Çıkış

Keypoint formatı (eğitim verisiyle uyumlu):
  [0:63]   = Sol el   (21 nokta × 3: x,y,z) — mp.solutions.hands
  [63:126]  = Sağ el   (21 nokta × 3: x,y,z) — mp.solutions.hands
  [126:138] = 4 Pose noktası (11,12,15,16) × 3 — mp.solutions.pose

Kullanım:
    python camera.py --checkpoint outputs/best_model.pt
    python camera.py --checkpoint outputs/best_model.pt --camera 1
    python camera.py --checkpoint outputs/best_model.pt --window 3
"""

import os
import sys
import json
import time
import argparse
import threading
import collections
import queue

import cv2
import torch
import numpy as np

try:
    import mediapipe as mp
    import mediapipe.python.solutions as mp_solutions
    mp.solutions = mp_solutions
    HAS_MP = True
except ImportError:
    HAS_MP = False

from model import SlowFastTID


# ─────────────────────────────────────────
# Sabitler
# ─────────────────────────────────────────
NUM_FRAMES  = 32
FEATURE_DIM = 138   # 63 + 63 + 12
POSE_IDX    = [11, 12, 15, 16]  # sol omuz, sağ omuz, sol bilek, sağ bilek

# Renk sabitleri (BGR)
CLR_GREEN  = (0, 220, 80)
CLR_BLUE   = (220, 140, 0)
CLR_RED    = (0, 60, 220)
CLR_WHITE  = (255, 255, 255)
CLR_BLACK  = (0, 0, 0)
CLR_GRAY   = (160, 160, 160)
CLR_YELLOW = (0, 210, 255)
CLR_DARK   = (20, 20, 20)


# ─────────────────────────────────────────
# KeypointExtractor — Eğitim verisiyle AYNI format
# ─────────────────────────────────────────

class KeypointExtractor:
    """
    MediaPipe Hands + Pose ile frame'den keypoint çıkarır.
    EĞİTİM VERİSİYLE AYNI FORMAT:
      [0:63]   = Sol el   (21 × 3)
      [63:126]  = Sağ el   (21 × 3)
      [126:138] = 4 Pose   (11,12,15,16) × 3
    Toplam: 138 feature
    """

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
        # Çizim yardımcıları
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands   = mp.solutions.hands
        self.mp_pose_mod = mp.solutions.pose

    def extract(self, frame_bgr):
        """
        BGR frame → (138,) keypoint vektörü.
        Eğitim verisindeki extract_keypoints.py ile AYNI sıra ve format.
        """
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        feat = np.zeros(FEATURE_DIM, dtype=np.float32)

        # ── El noktaları ──
        h_res = self.hands.process(rgb)
        if h_res.multi_hand_landmarks and h_res.multi_handedness:
            for lm, hd in zip(h_res.multi_hand_landmarks, h_res.multi_handedness):
                label = hd.classification[0].label   # "Left" veya "Right"
                pts = np.array(
                    [[p.x, p.y, p.z] for p in lm.landmark],
                    dtype=np.float32
                ).flatten()  # 63

                if label == "Left":
                    feat[0:63] = pts
                elif label == "Right":
                    feat[63:126] = pts

        # ── Pose noktaları ──
        p_res = self.pose.process(rgb)
        if p_res.pose_landmarks:
            for i, pi in enumerate(POSE_IDX):
                lm = p_res.pose_landmarks.landmark[pi]
                feat[126 + i*3 : 126 + i*3 + 3] = [lm.x, lm.y, lm.z]

        return feat

    def draw_landmarks(self, frame_bgr):
        """Frame üzerine landmark'ları çiz (görselleştirme için)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Eller
        h_res = self.hands.process(rgb)
        if h_res.multi_hand_landmarks:
            for hand_lm in h_res.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame_bgr, hand_lm,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=1, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=1, circle_radius=1),
                )

        # Pose (sadece üst gövde)
        p_res = self.pose.process(rgb)
        if p_res.pose_landmarks:
            # Sadece omuz-bilek noktalarını çiz
            for pi in POSE_IDX:
                lm = p_res.pose_landmarks.landmark[pi]
                h, w = frame_bgr.shape[:2]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame_bgr, (cx, cy), 5, (245, 117, 66), -1)
                cv2.circle(frame_bgr, (cx, cy), 7, (245, 66, 230), 1)

        return frame_bgr

    def close(self):
        self.hands.close()
        self.pose.close()


# ─────────────────────────────────────────
# Keypoint → SlowFast Tensor dönüşümü
# ─────────────────────────────────────────

def preprocess_keypoints(keypoint_list, num_fast=32, alpha=4):
    """
    Keypoint listesi (herhangi uzunlukta) → (slow_tensor, fast_tensor)
    Uniform sampling ile num_fast frame'e örnekler.
    """
    n = len(keypoint_list)

    # Uniform sampling → num_fast frame seç
    fast_idx = np.linspace(0, n - 1, num_fast, dtype=int)
    slow_idx = fast_idx[::alpha]  # her alpha'da bir

    # Stack: (T, F)
    fast_data = np.stack([keypoint_list[i] for i in fast_idx], axis=0)  # (32, 138)
    slow_data = np.stack([keypoint_list[i] for i in slow_idx], axis=0)  # (8, 138)

    # Tensor: (1, F, T)
    fast_tensor = torch.from_numpy(fast_data.T.copy()).unsqueeze(0)  # (1, 138, 32)
    slow_tensor = torch.from_numpy(slow_data.T.copy()).unsqueeze(0)  # (1, 138, 8)

    return slow_tensor, fast_tensor


# ─────────────────────────────────────────
# Tahmin iş parçacığı
# ─────────────────────────────────────────

class PredictorThread(threading.Thread):
    def __init__(self, model, classes, device):
        super().__init__(daemon=True)
        self.model   = model
        self.classes = classes
        self.device  = device
        self.in_q    = queue.Queue(maxsize=1)
        self.out_q   = queue.Queue(maxsize=5)
        self.running = True

    def submit(self, keypoints):
        try:
            self.in_q.get_nowait()
        except queue.Empty:
            pass
        self.in_q.put(keypoints)

    def get_result(self):
        try:
            return self.out_q.get_nowait()
        except queue.Empty:
            return None

    @torch.no_grad()
    def run(self):
        while self.running:
            try:
                kp_list = self.in_q.get(timeout=0.5)
            except queue.Empty:
                continue

            slow_t, fast_t = preprocess_keypoints(kp_list)
            slow_t = slow_t.to(self.device)
            fast_t = fast_t.to(self.device)

            self.model.eval()
            with torch.amp.autocast(device_type=self.device.type):
                logits = self.model([slow_t, fast_t])

            probs = torch.softmax(logits, dim=1)[0].cpu()
            top_k = min(5, len(self.classes))
            top5_v, top5_i = probs.topk(top_k)

            top5 = [(self.classes[i], float(v)) for i, v in zip(top5_i, top5_v)]
            pred, conf = top5[0]

            try:
                self.out_q.put_nowait((pred, conf, top5))
            except queue.Full:
                pass

    def stop(self):
        self.running = False


# ─────────────────────────────────────────
# HUD çizim fonksiyonları
# ─────────────────────────────────────────

def overlay_rect(img, x, y, w, h, color, alpha=0.65):
    """Yarı saydam dikdörtgen."""
    y2, x2 = min(y + h, img.shape[0]), min(x + w, img.shape[1])
    sub = img[y:y2, x:x2]
    rect = np.full_like(sub, color)
    cv2.addWeighted(rect, alpha, sub, 1 - alpha, 0, sub)
    img[y:y2, x:x2] = sub


def put_text_shadow(img, text, pos, scale, color, thickness=1):
    x, y = pos
    cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, scale, CLR_BLACK, thickness+1)
    cv2.putText(img, text, (x, y),     cv2.FONT_HERSHEY_SIMPLEX, scale, color,     thickness)


def draw_progress_bar(img, x, y, w, h, progress, color_bg, color_fg):
    cv2.rectangle(img, (x, y), (x + w, y + h), color_bg, -1)
    filled = int(w * min(progress, 1.0))
    if filled > 0:
        cv2.rectangle(img, (x, y), (x + filled, y + h), color_fg, -1)


def draw_hud(frame, state):
    H, W = frame.shape[:2]

    pred      = state.get("pred", "—")
    conf      = state.get("conf", 0.0)
    top5      = state.get("top5", [])
    progress  = state.get("progress", 0.0)
    recording = state.get("recording", False)
    fps       = state.get("fps", 0.0)
    history   = state.get("history", [])
    mode      = state.get("mode", "auto")
    lm_found  = state.get("landmarks_found", False)

    # ── Sol panel ──
    panel_w = 300
    overlay_rect(frame, 10, 10, panel_w, 220, CLR_DARK, 0.7)

    put_text_shadow(frame, "Tahmin:", (22, 38), 0.55, CLR_GRAY)
    pred_disp = pred[:22] if len(pred) > 22 else pred
    put_text_shadow(frame, pred_disp, (22, 68), 0.9, CLR_GREEN, thickness=2)

    conf_pct = f"Guven: %{conf*100:.1f}"
    put_text_shadow(frame, conf_pct, (22, 92), 0.52, CLR_WHITE)

    draw_progress_bar(frame, 22, 100, panel_w - 30, 8, conf,
                      (60, 60, 60), CLR_GREEN if conf > 0.6 else CLR_YELLOW if conf > 0.3 else CLR_RED)

    lm_color = CLR_GREEN if lm_found else CLR_RED
    lm_text = "El: ALGILANDI" if lm_found else "El: YOK"
    put_text_shadow(frame, lm_text, (22, 118), 0.42, lm_color)

    put_text_shadow(frame, "Top-5:", (22, 140), 0.45, CLR_GRAY)
    for i, (cls, prob) in enumerate(top5[:5]):
        bar_w = int((panel_w - 35) * prob)
        yy = 156 + i * 17
        if bar_w > 0:
            cv2.rectangle(frame, (22, yy - 10), (22 + bar_w, yy - 2),
                          (0, int(80 + 100 * prob), int(40 * (1 - prob))), -1)
        label = f"{cls[:18]:<18} {prob*100:4.1f}%"
        put_text_shadow(frame, label, (22, yy), 0.38, CLR_WHITE)

    # ── Sağ üst: ilerleme çubuğu ──
    bar_x, bar_y = W - 220, 10
    overlay_rect(frame, bar_x - 5, bar_y - 5, 225, 50, CLR_DARK, 0.65)
    put_text_shadow(frame, "Kayit penceresi:", (bar_x, bar_y + 12), 0.42, CLR_GRAY)
    rec_clr = CLR_RED if recording else CLR_BLUE
    draw_progress_bar(frame, bar_x, bar_y + 18, 210, 18, progress, (60, 60, 60), rec_clr)
    pct_txt = f"{int(progress*100)}%"
    cv2.putText(frame, pct_txt, (bar_x + 213, bar_y + 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, CLR_WHITE, 1)

    if recording:
        cv2.circle(frame, (W - 20, 20), 8, CLR_RED, -1)
        put_text_shadow(frame, "REC", (W - 50, 26), 0.45, CLR_RED)

    # ── FPS ──
    put_text_shadow(frame, f"FPS: {fps:.0f}", (W - 90, H - 12), 0.42, CLR_GRAY)

    # ── Geçmiş ──
    if history:
        hist_y = H - 45
        overlay_rect(frame, W//2 - 200, hist_y - 18, 400, 56, CLR_DARK, 0.6)
        put_text_shadow(frame, "Son tahminler:", (W//2 - 190, hist_y), 0.4, CLR_GRAY)
        hist_str = "  >  ".join(history[-4:])
        put_text_shadow(frame, hist_str, (W//2 - 190, hist_y + 22), 0.48, CLR_GREEN)

    # ── Kontroller ──
    overlay_rect(frame, 10, H - 55, 310, 48, CLR_DARK, 0.6)
    mode_str = "Mod: OTOMATIK" if mode == "auto" else "Mod: MANUEL (r=kayit)"
    put_text_shadow(frame, mode_str, (18, H - 36), 0.42, CLR_YELLOW)
    put_text_shadow(frame, "r: mod  a: otomatik  s: anlik  q: cikis", (18, H - 16), 0.36, CLR_GRAY)

    return frame


# ─────────────────────────────────────────
# Ana kamera döngüsü
# ─────────────────────────────────────────

def run_camera(args):
    if not HAS_MP:
        print("HATA: mediapipe bulunamadı! pip install mediapipe")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    # Model yükle
    ckpt    = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    in_features = ckpt.get("in_features", 138)

    model = SlowFastTID(num_classes=len(classes), dropout=0.0, in_features=in_features)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"Model yüklendi: {len(classes)} sınıf | {in_features} feature")

    # Keypoint çıkarıcı (eğitim verisiyle aynı format)
    extractor = KeypointExtractor()
    print("MediaPipe Hands + Pose başlatıldı.")

    # Tahmin thread'i
    predictor = PredictorThread(model, classes, device)
    predictor.start()

    # Kamera
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Hata: Kamera {args.camera} açılamadı!")
        return

    target_fps = 30
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_frames = int(args.window * target_fps)  # 3sn × 30fps = 90

    # Keypoint buffer
    buffer = collections.deque(maxlen=window_frames)

    state = {
        "mode": "auto",
        "pred": "—",
        "conf": 0.0,
        "top5": [],
        "progress": 0.0,
        "recording": False,
        "fps": 0.0,
        "history": [],
        "landmarks_found": False,
    }

    last_predict_time = 0
    predict_interval  = 1.0   # her 1 saniyede bir tahmin
    manual_recording  = False
    manual_buffer     = []

    fps_times = collections.deque(maxlen=30)

    print(f"\nKamera açık. Pencere: {args.window}sn ({window_frames} frame)")
    print("'r' = mod değiştir, 'a' = otomatik, 's' = anlık, 'q' = çıkış\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kamera okunamadı.")
            break

        # Keypoint çıkar
        keypoints = extractor.extract(frame)
        landmarks_found = np.any(keypoints != 0)
        state["landmarks_found"] = landmarks_found

        # Landmark çiz
        if args.show_landmarks:
            extractor.draw_landmarks(frame)

        # Buffer'a ekle
        buffer.append(keypoints)

        # ── Otomatik mod ──
        if state["mode"] == "auto":
            progress = len(buffer) / window_frames
            state["progress"] = progress

            now = time.time()
            if (len(buffer) >= window_frames and
                    now - last_predict_time >= predict_interval):
                # Uniform sampling ile NUM_FRAMES keypoint seç
                kps = list(buffer)
                idx = np.linspace(0, len(kps) - 1, NUM_FRAMES, dtype=int)
                predictor.submit([kps[i] for i in idx])
                last_predict_time = now

        # ── Manuel mod ──
        else:
            if manual_recording:
                manual_buffer.append(keypoints)
                progress = min(len(manual_buffer) / window_frames, 1.0)
                state["progress"] = progress
                state["recording"] = True

                if len(manual_buffer) >= window_frames:
                    idx = np.linspace(0, len(manual_buffer) - 1, NUM_FRAMES, dtype=int)
                    predictor.submit([manual_buffer[i] for i in idx])
                    manual_buffer.clear()
                    manual_recording = False
                    state["recording"] = False
            else:
                state["progress"] = 0.0
                state["recording"] = False

        # ── Tahmin sonucu ──
        result = predictor.get_result()
        if result:
            pred, conf, top5 = result
            state["pred"] = pred
            state["conf"] = conf
            state["top5"] = top5
            if not state["history"] or state["history"][-1] != pred:
                state["history"].append(pred)
                if len(state["history"]) > 10:
                    state["history"].pop(0)

        # ── FPS ──
        fps_times.append(time.time())
        if len(fps_times) > 1:
            state["fps"] = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0])

        # ── HUD ──
        display = frame.copy()
        draw_hud(display, state)
        cv2.imshow("TID SlowFast - Turk Isaret Dili Tanima", display)

        # ── Klavye ──
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('r'):
            if state["mode"] == "auto":
                state["mode"] = "manual"
                manual_recording = False
                manual_buffer.clear()
                print("Mod: MANUEL — 'r' ile kayıt başlat/bitir")
            else:
                if not manual_recording:
                    manual_recording = True
                    manual_buffer.clear()
                    print("Kayıt başladı...")
                else:
                    if len(manual_buffer) >= NUM_FRAMES:
                        idx = np.linspace(0, len(manual_buffer) - 1, NUM_FRAMES, dtype=int)
                        predictor.submit([manual_buffer[i] for i in idx])
                        print(f"Tahmin gönderildi ({len(manual_buffer)} keypoint)")
                    else:
                        print("Çok kısa! En az 1 saniye kayıt yapın.")
                    manual_buffer.clear()
                    manual_recording = False
                    state["recording"] = False

        elif key == ord('a'):
            state["mode"] = "auto"
            manual_recording = False
            manual_buffer.clear()
            print("Mod: OTOMATİK")

        elif key == ord('s'):
            if len(buffer) >= NUM_FRAMES:
                kps = list(buffer)
                idx = np.linspace(0, len(kps) - 1, NUM_FRAMES, dtype=int)
                predictor.submit([kps[i] for i in idx])
                print("Anlık tahmin gönderildi.")

    # Temizlik
    predictor.stop()
    extractor.close()
    cap.release()
    cv2.destroyAllWindows()
    print("\nSon tahmin geçmişi:", " → ".join(state["history"]))


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="TID SlowFast Gerçek Zamanlı Kamera Tahmini")
    p.add_argument("--checkpoint", required=True,
                   help="Eğitilmiş model: outputs/best_model.pt")
    p.add_argument("--camera",  type=int, default=0,
                   help="Kamera indeksi (varsayılan: 0)")
    p.add_argument("--window",  type=float, default=2.0,
                   help="Kayıt penceresi saniye (varsayılan: 2)")
    p.add_argument("--show_landmarks", action="store_true", default=True,
                   help="MediaPipe landmark'larını göster")
    p.add_argument("--no_landmarks", dest="show_landmarks", action="store_false",
                   help="Landmark çizimini kapat")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_camera(args)