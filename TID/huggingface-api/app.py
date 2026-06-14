"""
TID İşaret Dili API — Hugging Face Spaces Backend
===================================================
Random Forest (harf tanıma) + 1D CNN (kelime tanıma)
"""
import os
import json
import numpy as np
import joblib
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TID İşaret Dili API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════ Model Yolları ═══════
BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, "models")

# ═══════ 1D CNN Etiketleri (70 kelime) ═══════
CNN1D_ACTIONS = [
    'abla', 'acıkmak', 'afiyet_olsun', 'ağabey', 'ağlamak', 'aile', 'alışveriş', 'anne', 'arkadaş', 'baba',
    'bahçe', 'bakmak', 'bayrak', 'bebek', 'ben', 'biz', 'çalışmak', 'çarşamba', 'çocuk', 'çorba',
    'cuma', 'cumartesi', 'dede', 'doktor', 'dün', 'ev', 'evet', 'futbol', 'gülmek', 'hasta',
    'hastane', 'hayır', 'hoşçakal', 'içmek', 'ilaç', 'iyi', 'kardeş', 'kim', 'kötü', 'nasıl',
    'neden', 'nerede', 'öğretmen', 'okul', 'olmaz', 'olur', 'onlar', 'özür_dilemek', 'pazar', 'pazartesi',
    'perşembe', 'rica_etmek', 'salı', 'sen', 'sevmek', 'siz', 'tamam', 'tatlı', 'teşekkür', 'tuvalet',
    'var', 'yanlış', 'yapmak', 'yardım', 'yarın', 'yemek', 'yemek_pişirmek', 'yok', 'zaman', 'zor'
]

SEQUENCE_LENGTH = 60

# ═══════ Global Model Referansları ═══════
rf_models = {}
cnn1d_model = None


def load_models():
    """Modelleri başlangıçta yükle."""
    global rf_models, cnn1d_model

    # Random Forest
    try:
        rf_models["model_tek"] = joblib.load(os.path.join(MODELS_DIR, "model_tek_el.pkl"))
        rf_models["model_iki"] = joblib.load(os.path.join(MODELS_DIR, "model_iki_el.pkl"))
        rf_models["le_tek"] = joblib.load(os.path.join(MODELS_DIR, "le_tek_el.pkl"))
        rf_models["le_iki"] = joblib.load(os.path.join(MODELS_DIR, "le_iki_el.pkl"))
        logger.info("✅ Random Forest modelleri yüklendi")
    except Exception as e:
        logger.error(f"❌ RF yükleme hatası: {e}")

    # 1D CNN
    try:
        from tensorflow.keras.models import load_model
        cnn1d_model = load_model(os.path.join(MODELS_DIR, "tid_70_kelime_cnn1d_best.h5"), compile=False)
        logger.info("✅ 1D CNN modeli yüklendi")
    except Exception as e:
        logger.error(f"❌ 1D CNN yükleme hatası: {e}")


# ═══════ Pydantic Modelleri ═══════
class RFRequest(BaseModel):
    landmarks: List[float]
    hand_count: int = 1

class CNN1DRequest(BaseModel):
    sequence: List[List[float]]  # 60×225

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    top5: List[dict]


# ═══════ Endpoint'ler ═══════
@app.get("/")
def root():
    return {
        "status": "ok",
        "models": {
            "random_forest": bool(rf_models),
            "cnn1d": cnn1d_model is not None,
        }
    }


@app.post("/predict/rf", response_model=PredictionResponse)
def predict_rf(req: RFRequest):
    if not rf_models:
        raise HTTPException(503, "RF modeli yüklenmedi")

    landmarks = np.array(req.landmarks, dtype=np.float32)

    if req.hand_count == 2 and len(landmarks) >= 126:
        model = rf_models["model_iki"]
        le = rf_models["le_iki"]
        features = landmarks[:126].reshape(1, -1)
    else:
        # Tek el — iki modelden de güven karşılaştır
        tek_feat = landmarks[:63].reshape(1, -1)
        pred_tek = rf_models["model_tek"].predict(tek_feat)[0]
        prob_tek = rf_models["model_tek"].predict_proba(tek_feat)[0]
        guven_tek = float(prob_tek[pred_tek])
        harf_tek = rf_models["le_tek"].inverse_transform([pred_tek])[0]

        padded = np.concatenate([landmarks[:63], np.zeros(63)]).reshape(1, -1)
        pred_iki = rf_models["model_iki"].predict(padded)[0]
        prob_iki = rf_models["model_iki"].predict_proba(padded)[0]
        guven_iki = float(prob_iki[pred_iki])
        harf_iki = rf_models["le_iki"].inverse_transform([pred_iki])[0]

        if guven_tek >= guven_iki:
            return PredictionResponse(
                label=harf_tek, confidence=guven_tek,
                top5=[{"label": harf_tek, "confidence": guven_tek}]
            )
        else:
            return PredictionResponse(
                label=harf_iki, confidence=guven_iki,
                top5=[{"label": harf_iki, "confidence": guven_iki}]
            )

    pred_idx = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    conf = float(prob[pred_idx])
    label = le.inverse_transform([pred_idx])[0]

    top5_idx = np.argsort(prob)[::-1][:5]
    top5 = [{"label": le.inverse_transform([i])[0], "confidence": float(prob[i])} for i in top5_idx]

    return PredictionResponse(label=label, confidence=conf, top5=top5)


@app.post("/predict/cnn1d", response_model=PredictionResponse)
def predict_cnn1d(req: CNN1DRequest):
    if cnn1d_model is None:
        raise HTTPException(503, "1D CNN modeli yüklenmedi")

    seq = np.array(req.sequence, dtype=np.float32)

    # Sekansı 60 frame'e normalize et
    if len(seq) >= SEQUENCE_LENGTH:
        indices = np.linspace(0, len(seq) - 1, SEQUENCE_LENGTH, dtype=int)
        input_data = seq[indices]
    else:
        padding = np.tile(seq[-1:], (SEQUENCE_LENGTH - len(seq), 1))
        input_data = np.concatenate([seq, padding])

    pred = cnn1d_model.predict(np.expand_dims(input_data, axis=0), verbose=0)[0]
    idx = int(np.argmax(pred))
    conf = float(pred[idx])
    label = CNN1D_ACTIONS[idx]

    top5_idx = np.argsort(pred)[::-1][:5]
    top5 = [{"label": CNN1D_ACTIONS[i], "confidence": float(pred[i])} for i in top5_idx]

    return PredictionResponse(label=label, confidence=conf, top5=top5)


# ═══════ Başlatma ═══════
@app.on_event("startup")
def startup():
    load_models()
