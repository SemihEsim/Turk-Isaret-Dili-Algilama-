"""
app.py — TID İşaret Dili Web Uygulaması
========================================
Flask + Socket.IO ile gerçek zamanlı işaret dili tanıma.

Çalıştırma:
    cd TID/webapp
    python app.py
"""

import os
import sys
import json
import base64
import logging
import time

import cv2
import numpy as np
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from model_manager import ModelManager

# ───── Logging ─────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ───── Flask ─────
app = Flask(__name__)
app.config["SECRET_KEY"] = "tid-isaret-dili-2026"
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                    max_http_buffer_size=10 * 1024 * 1024)

# ───── Model Manager ─────
manager = ModelManager()

# ───── İstatistikler ─────
stats = {
    "total_predictions": 0,
    "predictions_per_model": {},
    "start_time": time.time(),
    "letter_counts": {},
}


# ──────────────────────────────────────────
# Route'lar
# ──────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/models")
def api_models():
    """Kullanılabilir model listesi."""
    return jsonify(manager.get_available_models())


@app.route("/api/stats")
def api_stats():
    """İstatistikler."""
    uptime = time.time() - stats["start_time"]
    return jsonify({
        **stats,
        "uptime_seconds": uptime,
    })


# ──────────────────────────────────────────
# WebSocket event'leri
# ──────────────────────────────────────────

@socketio.on("connect")
def handle_connect():
    logger.info("İstemci bağlandı")
    emit("connection_status", {"status": "connected"})


@socketio.on("disconnect")
def handle_disconnect():
    logger.info("İstemci ayrıldı")


@socketio.on("load_model")
def handle_load_model(data):
    """Model ön yükleme."""
    model_key = data.get("model", "efficientnet")
    try:
        manager.load_model(model_key)
        emit("model_loaded", {"model": model_key, "status": "ok"})
    except Exception as e:
        logger.error(f"Model yükleme hatası: {e}")
        emit("model_loaded", {"model": model_key, "status": "error", "message": str(e)})


@socketio.on("predict_frame")
def handle_predict_frame(data):
    """
    Frame tabanlı tahmin (EfficientNet, MobileNetV2).
    data: { model, frame_base64 }
    """
    try:
        model_key = data.get("model", "efficientnet")
        frame_b64 = data.get("frame")

        if not frame_b64:
            logger.warning("Boş frame geldi")
            return

        # Base64 → numpy
        img_bytes = base64.b64decode(frame_b64)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            logger.warning("Frame decode edilemedi")
            return

        logger.info(f"Frame alındı: {frame.shape} → {model_key} ile tahmin yapılıyor...")

        # BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = manager.predict(model_key, {"frame": frame_rgb})

        logger.info(f"Tahmin: {result['label']} (güven: {result['confidence']:.2f})")

        # İstatistik güncelle
        stats["total_predictions"] += 1
        stats["predictions_per_model"][model_key] = stats["predictions_per_model"].get(model_key, 0) + 1
        if result["label"] != "?":
            stats["letter_counts"][result["label"]] = stats["letter_counts"].get(result["label"], 0) + 1

        emit("prediction", {
            "model": model_key,
            "label": result["label"],
            "confidence": result["confidence"],
            "top5": result["top5"],
        })

    except Exception as e:
        logger.error(f"Tahmin hatası: {e}", exc_info=True)
        emit("prediction_error", {"error": str(e)})


@socketio.on("predict_landmarks")
def handle_predict_landmarks(data):
    """
    Landmark tabanlı tahmin (Random Forest).
    data: { landmarks: [...], hand_count: 1|2 }
    """
    try:
        landmarks = data.get("landmarks", [])
        hand_count = data.get("hand_count", 1)

        if not landmarks:
            return

        result = manager.predict("random_forest", {
            "landmarks": landmarks,
            "hand_count": hand_count,
        })

        stats["total_predictions"] += 1
        stats["predictions_per_model"]["random_forest"] = stats["predictions_per_model"].get("random_forest", 0) + 1

        emit("prediction", {
            "model": "random_forest",
            "label": result["label"],
            "confidence": result["confidence"],
            "top5": result["top5"],
        })

    except Exception as e:
        logger.error(f"RF tahmin hatası: {e}")
        emit("prediction_error", {"error": str(e)})


@socketio.on("predict_tcn")
def handle_predict_tcn(data):
    """
    TCN sekans tabanlı tahmin.
    data: { keypoints: [138 float] }  — her frame tek tek gönderilir
    """
    try:
        keypoints = data.get("keypoints", [])

        if not keypoints or len(keypoints) != 138:
            return

        result = manager.predict("tcn", {"keypoints": keypoints})

        stats["total_predictions"] += 1
        stats["predictions_per_model"]["tcn"] = stats["predictions_per_model"].get("tcn", 0) + 1

        emit("prediction", {
            "model": "tcn",
            "label": result["label"],
            "confidence": result["confidence"],
            "top5": result.get("top5", []),
            "buffer_progress": result.get("buffer_progress", 0),
        })

    except Exception as e:
        logger.error(f"TCN tahmin hatası: {e}")
        emit("prediction_error", {"error": str(e)})


@socketio.on("clear_buffers")
def handle_clear_buffers():
    """Tüm tampon belleği temizle."""
    manager.clear_buffers()
    emit("buffers_cleared", {"status": "ok"})


# ──────────────────────────────────────────
# Başlatma
# ──────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  TID İşaret Dili Web Uygulaması")
    print("=" * 55)
    print(f"\n🌐 http://localhost:5000\n")

    available = manager.get_available_models()
    for key, info in available.items():
        status = "✅" if info["available"] else "❌"
        print(f"  {status} {info['name']:<20} ({info['type']}) — %{info['accuracy']:.1f}")

    print(f"\n{'=' * 55}\n")

    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
