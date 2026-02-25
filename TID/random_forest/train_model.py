# train_model.py

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

from config import (
    TEK_EL_HARFLER, TRAIN_PATH, TEST_PATH,
    MODEL_TEK_EL, MODEL_IKI_EL, LE_TEK_EL, LE_IKI_EL
)
from feature_extractor import extract_from_path

os.makedirs("models", exist_ok=True)


# ──────────────────────────────────────────────
# 1. Dataset Yükleme
# ──────────────────────────────────────────────

def load_dataset(base_path):
    """
    Dataset klasöründeki tüm harfleri okur.
    Harfe göre tek_el ve iki_el olarak ayırır.
    """
    X_tek, y_tek = [], []
    X_iki, y_iki = [], []

    atlanan_tek = 0
    atlanan_iki = 0
    toplam = 0

    siniflar = sorted(os.listdir(base_path))
    print(f"Bulunan sınıflar: {siniflar}\n")

    for sinif in siniflar:
        sinif_path = os.path.join(base_path, sinif)
        if not os.path.isdir(sinif_path):
            continue

        dosyalar = [
            f for f in os.listdir(sinif_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        basarili = 0
        for dosya in dosyalar:
            toplam += 1
            img_path = os.path.join(sinif_path, dosya)
            features = extract_from_path(img_path, sinif)

            if features is not None:
                if sinif.upper() in TEK_EL_HARFLER:
                    X_tek.append(features)
                    y_tek.append(sinif.upper())
                else:
                    X_iki.append(features)
                    y_iki.append(sinif.upper())
                basarili += 1
            else:
                if sinif.upper() in TEK_EL_HARFLER:
                    atlanan_tek += 1
                else:
                    atlanan_iki += 1

        print(f"  [{sinif}] {'TEK EL' if sinif.upper() in TEK_EL_HARFLER else 'IKI EL':7s} "
              f"| {basarili}/{len(dosyalar)} foto işlendi")

    print(f"\nToplam: {toplam} fotoğraf")
    print(f"Tek el: {len(X_tek)} başarılı, {atlanan_tek} atlandı")
    print(f"İki el: {len(X_iki)} başarılı, {atlanan_iki} atlandı\n")

    return (np.array(X_tek), np.array(y_tek),
            np.array(X_iki), np.array(y_iki))


# ──────────────────────────────────────────────
# 2. Model Eğitimi
# ──────────────────────────────────────────────

def egit_model(X_train, y_train, X_test, y_test, model_path, le_path, isim):
    """
    Verilen veriyle Random Forest modeli eğitir,
    test seti üzerinde değerlendirir ve kaydeder.
    """
    print(f"\n{'='*50}")
    print(f"  {isim} MODELİ EĞİTİLİYOR")
    print(f"{'='*50}")
    print(f"Train: {len(X_train)} örnek | Test: {len(X_test)} örnek")
    print(f"Özellik boyutu: {X_train.shape[1]}")

    # Label encoding
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    print(f"Sınıflar: {list(le.classes_)}")

    # Random Forest
    # n_estimators=200 → 200 karar ağacı, daha fazlası daha iyi ama yavaş
    # class_weight='balanced' → harf başına foto sayısı farklıysa dengeleme yapar
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1          # Tüm CPU çekirdeklerini kullan
    )

    print("Eğitim başlıyor...")
    model.fit(X_train, y_train_enc)
    print("Eğitim tamamlandı!")

    # Değerlendirme
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test_enc).mean()
    print(f"\nTest Doğruluğu: %{accuracy*100:.2f}")
    print("\nDetaylı Rapor:")
    print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

    # Confusion Matrix kaydet
    cm = confusion_matrix(y_test_enc, y_pred)
    fig_w = max(10, len(le.classes_))
    fig, ax = plt.subplots(figsize=(fig_w, fig_w - 2))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
    plt.title(f"Confusion Matrix - {isim}")
    plt.tight_layout()
    cm_path = f"models/confusion_{isim.lower().replace(' ', '_')}.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix kaydedildi: {cm_path}")

    # Kaydet
    joblib.dump(model, model_path)
    joblib.dump(le, le_path)
    print(f"Model kaydedildi: {model_path}")
    print(f"Label encoder kaydedildi: {le_path}")

    return model, le, accuracy


# ──────────────────────────────────────────────
# 3. Ana Akış
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== TRAIN SETİ YÜKLENİYOR ===")
    X_train_tek, y_train_tek, X_train_iki, y_train_iki = load_dataset(TRAIN_PATH)

    print("\n=== TEST SETİ YÜKLENİYOR ===")
    X_test_tek,  y_test_tek,  X_test_iki,  y_test_iki  = load_dataset(TEST_PATH)

    # Tek el modeli (63 özellik: C, I, L, O, P, U, V)
    model_tek, le_tek, acc_tek = egit_model(
        X_train_tek, y_train_tek,
        X_test_tek,  y_test_tek,
        MODEL_TEK_EL, LE_TEK_EL,
        "Tek El"
    )

    # İki el modeli (126 özellik: diğer tüm harfler)
    model_iki, le_iki, acc_iki = egit_model(
        X_train_iki, y_train_iki,
        X_test_iki,  y_test_iki,
        MODEL_IKI_EL, LE_IKI_EL,
        "Iki El"
    )

    print("\n" + "="*50)
    print("  EĞİTİM TAMAMLANDI")
    print("="*50)
    print(f"  Tek El Model Doğruluğu : %{acc_tek*100:.2f}")
    print(f"  İki El Model Doğruluğu : %{acc_iki*100:.2f}")
    print("="*50)
    print("\nSıradaki adım: python realtime.py")