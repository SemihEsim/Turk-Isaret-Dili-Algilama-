"""
ADIM 1 — Keypoint Çıkarma
==========================
Tüm videoları bir kez işleyip .npy olarak kaydeder.
Eğitim sırasında video açılmaz, doğrudan .npy okunur → çok hızlı.

Kullanım:
    python extract_keypoints.py --data_dir archive --output_dir keypoints
    python extract_keypoints.py --data_dir archive --output_dir keypoints --splits train val test

Çıktı yapısı:
    keypoints/
        train/ kelime/ video.npy
        val/   kelime/ video.npy
        test/  kelime/ video.npy

Her .npy dosyası: shape (32, 138)
  - Sol el:  21 nokta × 3 = 63
  - Sağ el:  21 nokta × 3 = 63
  - Poz:      4 nokta × 3 = 12
  Toplam: 138 özellik/frame
"""
print("MERHABA, KOD ÇALIŞMAYA BAŞLADI")
import sys
import os
print("1. os kütüphanesi yüklendi.")
user_site = os.path.join(os.environ['APPDATA'], 'Python', 'Python39', 'site-packages')
if user_site not in sys.path:
    sys.path.append(user_site)

import numpy as np
print("2. numpy kütüphanesi yüklendi.")

import argparse
import json
print("3. argparse ve json yüklendi.")

print("4. cv2 (OpenCV) yükleniyor, lütfen bekleyin...")
import cv2
print("5. cv2 başarıyla yüklendi.")

print("6. mediapipe yükleniyor, lütfen bekleyin...")
try:
    import mediapipe as mp
    import mediapipe.python.solutions as mp_solutions  # Alt modülü zorla yüklüyoruz
    mp.solutions = mp_solutions  # Bağlantıyı manuel kuruyoruz
    print("7. mediapipe başarıyla yüklendi.")
    HAS_MP = True
except ImportError:
    print("7. HATA: mediapipe yüklü değil!")
    HAS_MP = False

NUM_FRAMES  = 32
FEATURE_DIM = 138   # 63 + 63 + 12

# Poz noktaları: sol omuz(11), sağ omuz(12), sol bilek(15), sağ bilek(16)
POSE_INDICES = [11, 12, 15, 16]


def extract_one_video(video_path, num_frames=NUM_FRAMES):
    """Video → (num_frames, FEATURE_DIM) numpy array"""
    cap   = cv2.VideoCapture(video_path)
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)

    # Uniform sampling
    indices = np.linspace(0, total - 1, num_frames, dtype=int)

    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.3
    )
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.3
    )

    features = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            features.append(np.zeros(FEATURE_DIM, dtype=np.float32))
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_feat = np.zeros(FEATURE_DIM, dtype=np.float32)

        # El noktaları
        hand_res = mp_hands.process(rgb)
        left_done = right_done = False
        if hand_res.multi_hand_landmarks and hand_res.multi_handedness:
            for lm, hd in zip(hand_res.multi_hand_landmarks, hand_res.multi_handedness):
                label = hd.classification[0].label  # "Left" veya "Right"
                pts   = np.array([[p.x, p.y, p.z] for p in lm.landmark],
                                 dtype=np.float32).flatten()  # 63
                if label == "Left" and not left_done:
                    frame_feat[0:63] = pts
                    left_done = True
                elif label == "Right" and not right_done:
                    frame_feat[63:126] = pts
                    right_done = True

        # Poz noktaları
        pose_res = mp_pose.process(rgb)
        if pose_res.pose_landmarks:
            for i, pi in enumerate(POSE_INDICES):
                lm = pose_res.pose_landmarks.landmark[pi]
                frame_feat[126 + i*3 : 126 + i*3 + 3] = [lm.x, lm.y, lm.z]

        features.append(frame_feat)

    cap.release()
    mp_hands.close()
    mp_pose.close()

    return np.stack(features, axis=0)  # (32, 138)


def process_split(data_dir, out_dir, split, classes=None):
    split_in  = os.path.join(data_dir, split)
    split_out = os.path.join(out_dir,  split)

    if not os.path.isdir(split_in):
        print(f"  Atlandı (bulunamadı): {split_in}")
        return

    kelimeler = sorted([
        d for d in os.listdir(split_in)
        if os.path.isdir(os.path.join(split_in, d)) and not d.startswith(".")
    ])
    if classes:
        kelimeler = [k for k in kelimeler if k in classes]

    print(f"\n[{split}] {len(kelimeler)} kelime işlenecek")

    ok = fail = 0
    for kelime in kelimeler:
        in_dir  = os.path.join(split_in,  kelime)
        out_k   = os.path.join(split_out, kelime)
        os.makedirs(out_k, exist_ok=True)

        videolar = [f for f in os.listdir(in_dir)
                    if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

        for vid in videolar:
            npy_name = os.path.splitext(vid)[0] + ".npy"
            npy_path = os.path.join(out_k, npy_name)

            if os.path.exists(npy_path):
                ok += 1
                continue  # zaten çıkarılmış

            try:
                arr = extract_one_video(os.path.join(in_dir, vid))
                np.save(npy_path, arr)
                ok += 1
            except Exception as e:
                print(f"    HATA {vid}: {e}")
                fail += 1

        print(f"  {kelime}: {len(videolar)} video → {ok} OK, {fail} hata", end="\r")

    print(f"\n[{split}] Tamamlandı: {ok} OK, {fail} hata")


def main(args):
    if not HAS_MP:
        print("HATA: mediapipe yüklü değil!\npip install mediapipe")
        return

    splits  = args.splits.split(",")
    classes = args.classes.split(",") if args.classes else None

    os.makedirs(args.output_dir, exist_ok=True)

    # Sınıf listesini kaydet (train'den al)
    train_dir = os.path.join(args.data_dir, "train")
    if os.path.isdir(train_dir):
        all_classes = sorted([
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        ])
        if classes:
            all_classes = [c for c in all_classes if c in classes]
        with open(os.path.join(args.output_dir, "classes.json"), "w", encoding="utf-8") as f:
            json.dump(all_classes, f, ensure_ascii=False, indent=2)
        print(f"Sınıflar kaydedildi: {len(all_classes)} kelime")

    for split in splits:
        process_split(args.data_dir, args.output_dir, split.strip(), classes)

    print("\nKeypoint çıkarma tamamlandı!")
    print(f"Çıktı klasörü: {args.output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default="archive",    help="Ana veri klasörü")
    p.add_argument("--output_dir",  default="keypoints",  help="Keypoint çıktı klasörü")
    p.add_argument("--splits",      default="train,val,test")
    p.add_argument("--classes",     default=None,
                   help="Sadece belirli kelimeler: 'elma,araba,anne' (boş=hepsi)")
    args = p.parse_args()
    main(args)