"""
model_manager.py — Tüm TID modellerinin merkezi yükleme ve tahmin yöneticisi
==============================================================================
Desteklenen modeller:
  - EfficientNetB0  (TensorFlow SavedModel)
  - MobileNetV2     (Keras .keras)
  - Random Forest   (scikit-learn .pkl — tek el + iki el)
  - TCN             (PyTorch .pt — kelime bazlı)
"""

import os
import sys
import json
import numpy as np
import logging
import threading
from collections import deque

logger = logging.getLogger(__name__)

# ───── Proje kök dizini ─────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ───── Model yolları ─────
PATHS = {
    "efficientnet": {
        "model": os.path.join(BASE_DIR, "EfficientNet", "results", "EfficientNetB0_TID_final"),
        "classes": os.path.join(BASE_DIR, "EfficientNet", "results", "class_names_eff.json"),
    },
    "mobilenet": {
        "model": os.path.join(BASE_DIR, "MobileNetV2", "results", "MobileNetV2_TID_final.keras"),
        "classes": os.path.join(BASE_DIR, "MobileNetV2", "results", "class_names.json"),
    },
    "random_forest": {
        "model_tek": os.path.join(BASE_DIR, "random_forest", "models", "model_tek_el.pkl"),
        "model_iki": os.path.join(BASE_DIR, "random_forest", "models", "model_iki_el.pkl"),
        "le_tek": os.path.join(BASE_DIR, "random_forest", "models", "le_tek_el.pkl"),
        "le_iki": os.path.join(BASE_DIR, "random_forest", "models", "le_iki_el.pkl"),
    },
    "tcn": {
        "model": os.path.join(BASE_DIR, "TCN", "outputs", "best_model.pt"),
        "classes": os.path.join(BASE_DIR, "TCN", "outputs", "classes.json"),
    },
}

# Model bilgileri (UI için)
MODEL_INFO = {
    "efficientnet": {
        "name": "EfficientNetB0",
        "type": "harf",
        "accuracy": 99.92,
        "description": "Yüksek doğruluklu görsel harf tanıma modeli",
        "input": "frame",
        "framework": "tensorflow",
    },
    "mobilenet": {
        "name": "MobileNetV2",
        "type": "harf",
        "accuracy": 99.91,
        "description": "Hızlı ve hafif görsel harf tanıma modeli",
        "input": "frame",
        "framework": "tensorflow",
    },
    "random_forest": {
        "name": "Random Forest",
        "type": "harf",
        "accuracy": 95.0,
        "description": "MediaPipe el noktaları ile harf tanıma",
        "input": "landmarks",
        "framework": "sklearn",
    },
    "tcn": {
        "name": "TCN (Temporal)",
        "type": "kelime",
        "accuracy": 68.2,
        "description": "Video sekansından kelime tanıma (32 frame)",
        "input": "sequence",
        "framework": "pytorch",
    },
}

IMG_SIZE = 224
SMOOTH_WINDOW = 10
TCN_NUM_FRAMES = 32
TCN_FEATURE_DIM = 138


class ModelManager:
    """Tüm modelleri lazy-load eder ve tahmin yapar."""

    def __init__(self):
        self._models = {}
        self._classes = {}
        self._smooth_buffers = {}
        self._tcn_frame_buffer = deque(maxlen=96)  # ~3 saniye (30fps)
        self._load_locks = {key: threading.Lock() for key in MODEL_INFO}
        self._loading = set()  # Şu anda yüklenmekte olan modeller
        logger.info("ModelManager başlatıldı")

    # ──────────────────────────────────────
    # Yardımcı
    # ──────────────────────────────────────

    def get_available_models(self):
        """Kullanılabilir modellerin listesini döner."""
        available = {}
        for key, info in MODEL_INFO.items():
            paths = PATHS[key]
            # İlk dosya yolunu kontrol et
            first_path = list(paths.values())[0]
            exists = os.path.exists(first_path)
            available[key] = {**info, "available": exists}
        return available

    def _get_buffer(self, model_key):
        if model_key not in self._smooth_buffers:
            self._smooth_buffers[model_key] = deque(maxlen=SMOOTH_WINDOW)
        return self._smooth_buffers[model_key]

    # ──────────────────────────────────────
    # Model Yükleme (Lazy)
    # ──────────────────────────────────────

    def _load_efficientnet(self):
        if "efficientnet" in self._models:
            return
        with self._load_locks["efficientnet"]:
            if "efficientnet" in self._models:  # Double-check lock
                return
            import tensorflow as tf
            logger.info("EfficientNetB0 yükleniyor...")
            loaded = tf.saved_model.load(PATHS["efficientnet"]["model"])
            infer = loaded.signatures["serving_default"]
            output_key = list(infer.structured_outputs.keys())[0]
            with open(PATHS["efficientnet"]["classes"], "r", encoding="utf-8") as f:
                classes = json.load(f)
            self._models["efficientnet"] = {"infer": infer, "output_key": output_key}
            self._classes["efficientnet"] = classes
            logger.info(f"EfficientNetB0 hazır ({len(classes)} sınıf)")

    def _load_mobilenet(self):
        if "mobilenet" in self._models:
            return
        with self._load_locks["mobilenet"]:
            if "mobilenet" in self._models:  # Double-check lock
                return
            import tensorflow as tf
            logger.info("MobileNetV2 yükleniyor...")
            model = tf.keras.models.load_model(PATHS["mobilenet"]["model"])
            with open(PATHS["mobilenet"]["classes"], "r", encoding="utf-8") as f:
                classes = json.load(f)
            self._models["mobilenet"] = {"model": model}
            self._classes["mobilenet"] = classes
            logger.info(f"MobileNetV2 hazır ({len(classes)} sınıf)")

    def _load_random_forest(self):
        if "random_forest" in self._models:
            return
        with self._load_locks["random_forest"]:
            if "random_forest" in self._models:  # Double-check lock
                return
            import joblib
            logger.info("Random Forest yükleniyor...")
            p = PATHS["random_forest"]
            self._models["random_forest"] = {
                "model_tek": joblib.load(p["model_tek"]),
                "model_iki": joblib.load(p["model_iki"]),
                "le_tek": joblib.load(p["le_tek"]),
                "le_iki": joblib.load(p["le_iki"]),
            }
            self._classes["random_forest"] = list("ABCDEFGHIJKLMNOPRSTUVYZ")
            logger.info("Random Forest hazır")

    def _load_tcn(self):
        if "tcn" in self._models:
            return
        with self._load_locks["tcn"]:
            if "tcn" in self._models:  # Double-check lock
                return
            import torch

            # TCN model tanımını import et
            sys.path.insert(0, os.path.join(BASE_DIR, "TCN"))
            from Model import TIDTCN

            logger.info("TCN yükleniyor...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(PATHS["tcn"]["model"], map_location=device, weights_only=False)
            classes = ckpt["classes"]
            model = TIDTCN(num_classes=len(classes), input_dim=TCN_FEATURE_DIM)
            model.load_state_dict(ckpt["model_state"])
            model.to(device).eval()
            self._models["tcn"] = {"model": model, "device": device}
            self._classes["tcn"] = classes
            logger.info(f"TCN hazır ({len(classes)} sınıf)")

    def load_model(self, model_key):
        """Model yükler (lazy loading)."""
        loaders = {
            "efficientnet": self._load_efficientnet,
            "mobilenet": self._load_mobilenet,
            "random_forest": self._load_random_forest,
            "tcn": self._load_tcn,
        }
        if model_key in loaders:
            loaders[model_key]()
            return True
        return False

    # ──────────────────────────────────────
    # Tahmin
    # ──────────────────────────────────────

    def predict_efficientnet(self, frame_rgb):
        """frame_rgb: numpy (H, W, 3) uint8 RGB"""
        import tensorflow as tf
        self._load_efficientnet()
        m = self._models["efficientnet"]

        img = self._preprocess_frame_tf(frame_rgb)
        out = m["infer"](img)
        pred = out[m["output_key"]].numpy()[0]

        buf = self._get_buffer("efficientnet")
        buf.append(pred)
        avg = np.mean(buf, axis=0)
        idx = int(np.argmax(avg))
        conf = float(avg[idx])
        label = self._classes["efficientnet"][idx]

        top5 = self._get_top5(avg, self._classes["efficientnet"])
        return {"label": label, "confidence": conf, "top5": top5}

    def predict_mobilenet(self, frame_rgb):
        """frame_rgb: numpy (H, W, 3) uint8 RGB"""
        self._load_mobilenet()
        m = self._models["mobilenet"]

        img = self._preprocess_frame_tf(frame_rgb, normalize=True)
        pred = m["model"].predict(img, verbose=0)[0]

        buf = self._get_buffer("mobilenet")
        buf.append(pred)
        avg = np.mean(buf, axis=0)
        idx = int(np.argmax(avg))
        conf = float(avg[idx])
        label = self._classes["mobilenet"][idx]

        top5 = self._get_top5(avg, self._classes["mobilenet"])
        return {"label": label, "confidence": conf, "top5": top5}

    def predict_random_forest(self, landmarks, hand_count):
        """
        landmarks: liste (63 veya 126 float — normalize edilmiş el noktaları)
        hand_count: 1 veya 2
        """
        self._load_random_forest()
        m = self._models["random_forest"]

        if hand_count == 2:
            pred_idx = m["model_iki"].predict([landmarks])[0]
            prob = m["model_iki"].predict_proba([landmarks])[0]
            conf = float(prob[pred_idx])
            label = m["le_iki"].inverse_transform([pred_idx])[0]
        else:
            # Tek el → her iki modelden güven al
            pred_tek = m["model_tek"].predict([landmarks[:63]])[0]
            prob_tek = m["model_tek"].predict_proba([landmarks[:63]])[0]
            guven_tek = float(prob_tek[pred_tek])
            harf_tek = m["le_tek"].inverse_transform([pred_tek])[0]

            features_padded = list(landmarks[:63]) + [0.0] * 63
            pred_iki = m["model_iki"].predict([features_padded])[0]
            prob_iki = m["model_iki"].predict_proba([features_padded])[0]
            guven_iki = float(prob_iki[pred_iki])
            harf_iki = m["le_iki"].inverse_transform([pred_iki])[0]

            if guven_tek >= guven_iki:
                label, conf = harf_tek, guven_tek
            else:
                label, conf = harf_iki, guven_iki

        return {"label": label, "confidence": conf, "top5": [{"label": label, "confidence": conf}]}

    def predict_tcn(self, keypoints_frame):
        """
        keypoints_frame: 138-boyutlu özellik vektörü (tek frame)
        Her frame biriktirilir, yeterli olunca tahmin yapılır.
        """
        import torch
        self._load_tcn()
        m = self._models["tcn"]

        self._tcn_frame_buffer.append(np.array(keypoints_frame, dtype=np.float32))

        if len(self._tcn_frame_buffer) < TCN_NUM_FRAMES:
            return {
                "label": "Bekleniyor...",
                "confidence": 0.0,
                "top5": [],
                "buffer_progress": len(self._tcn_frame_buffer) / TCN_NUM_FRAMES,
            }

        # Son frame'lerden uniform sampling
        kps = list(self._tcn_frame_buffer)
        indices = np.linspace(0, len(kps) - 1, TCN_NUM_FRAMES, dtype=int)
        selected = [kps[i] for i in indices]
        arr = np.stack(selected, axis=0).astype(np.float32)

        # Normalize (sıfır olmayanlar)
        nonzero = arr[arr != 0]
        if len(nonzero) > 0:
            mn, mx = nonzero.min(), nonzero.max()
            if mx > mn:
                mask = arr != 0
                arr[mask] = 2.0 * (arr[mask] - mn) / (mx - mn) - 1.0

        x = torch.from_numpy(arr).unsqueeze(0).to(m["device"])
        with torch.no_grad():
            m["model"].eval()
            logits = m["model"](x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = self._classes["tcn"][idx]

        top5 = self._get_top5(probs, self._classes["tcn"])
        return {"label": label, "confidence": conf, "top5": top5, "buffer_progress": 1.0}

    # ──────────────────────────────────────
    # Genel tahmin dispatcher
    # ──────────────────────────────────────

    def predict(self, model_key, data):
        """
        model_key: 'efficientnet' | 'mobilenet' | 'random_forest' | 'tcn'
        data: model_key'e göre değişen veri
        """
        if model_key == "efficientnet":
            return self.predict_efficientnet(data["frame"])
        elif model_key == "mobilenet":
            return self.predict_mobilenet(data["frame"])
        elif model_key == "random_forest":
            return self.predict_random_forest(data["landmarks"], data["hand_count"])
        elif model_key == "tcn":
            return self.predict_tcn(data["keypoints"])
        else:
            return {"label": "?", "confidence": 0.0, "top5": []}

    # ──────────────────────────────────────
    # Yardımcı fonksiyonlar
    # ──────────────────────────────────────

    def _preprocess_frame_tf(self, frame_rgb, normalize=False):
        """Frame'i TF modeli için ön işlemden geçirir."""
        import tensorflow as tf
        import cv2
        img = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32)
        if normalize:
            img = img / 255.0
        return tf.expand_dims(img, axis=0)

    def _get_top5(self, predictions, class_names):
        """Top 5 tahmin listesi döner."""
        top5_idx = np.argsort(predictions)[::-1][:5]
        return [
            {"label": class_names[i], "confidence": float(predictions[i])}
            for i in top5_idx
        ]

    def clear_buffers(self):
        """Tüm smoothing ve TCN buffer'larını temizler."""
        self._smooth_buffers.clear()
        self._tcn_frame_buffer.clear()
