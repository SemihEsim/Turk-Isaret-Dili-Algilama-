import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, TensorBoard)
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime

# ─────────────────────────────────────────────
DATASET_DIR     = "random_forest/dataset"
IMG_SIZE        = 224
BATCH_SIZE      = 32
EPOCHS_FROZEN   = 20
EPOCHS_FINETUNE = 20

os.makedirs("results",     exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs",        exist_ok=True)

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(f"✅ GPU: {[g.name for g in gpus] if gpus else 'YOK (CPU)'}")
print(f"✅ TensorFlow: {tf.__version__}\n")

# ─────────────────────────────────────────────
def get_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.75, 1.25],
        channel_shift_range=20.0,
        fill_mode='nearest'
    )
    base_datagen = ImageDataGenerator(rescale=1./255)

    def flow(datagen, split, shuffle=True):
        return datagen.flow_from_directory(
            os.path.join(DATASET_DIR, split),
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=shuffle
        )

    train_gen = flow(train_datagen, "train", shuffle=True)
    val_gen   = flow(base_datagen,  "val",   shuffle=False)
    test_gen  = flow(base_datagen,  "test",  shuffle=False)

    print(f"📊 Train: {train_gen.samples} | Val: {val_gen.samples} | Test: {test_gen.samples}")
    print(f"📊 Sınıflar: {list(train_gen.class_indices.keys())}\n")

    class_names = list(train_gen.class_indices.keys())
    with open("results/class_names.json", 'w', encoding='utf-8') as f:
        json.dump(class_names, f, ensure_ascii=False)

    return train_gen, val_gen, test_gen, class_names

# ─────────────────────────────────────────────
def build_model(num_classes):
    base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                       include_top=False, weights='imagenet')
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs, name="MobileNetV2_TID"), base

# ─────────────────────────────────────────────
def train(model, base_model, train_gen, val_gen):
    print(f"\n{'='*60}")
    print(f"  MobileNetV2 EĞİTİMİ")
    print(f"{'='*60}")

    log_dir = os.path.join("logs", "MobileNetV2_" + datetime.now().strftime("%H%M%S"))

    def callbacks(suffix):
        return [
            EarlyStopping(monitor='val_accuracy', patience=6,
                          restore_best_weights=True, verbose=1),
            ModelCheckpoint(f"checkpoints/MobileNetV2{suffix}.keras",
                            monitor='val_accuracy', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, min_lr=1e-7, verbose=1),
            TensorBoard(log_dir=log_dir)
        ]

    # Aşama 1: Frozen
    print(f"\n[AŞAMA 1] Frozen — max {EPOCHS_FROZEN} epoch")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    h1 = model.fit(train_gen, epochs=EPOCHS_FROZEN,
                   validation_data=val_gen, callbacks=callbacks("_frozen"), verbose=1)

    # Aşama 2: Fine-tune
    print(f"\n[AŞAMA 2] Fine-tune — son 40 katman — max {EPOCHS_FINETUNE} epoch")
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    h2 = model.fit(train_gen, epochs=EPOCHS_FINETUNE,
                   validation_data=val_gen, callbacks=callbacks("_finetuned"), verbose=1)

    history = {
        'accuracy':      h1.history['accuracy']     + h2.history['accuracy'],
        'val_accuracy':  h1.history['val_accuracy'] + h2.history['val_accuracy'],
        'loss':          h1.history['loss']          + h2.history['loss'],
        'val_loss':      h1.history['val_loss']      + h2.history['val_loss'],
        'frozen_epochs': len(h1.history['accuracy'])
    }

    model.save("results/MobileNetV2_TID_final.keras")
    with open("results/MobileNetV2_history.json", 'w') as f:
        json.dump(history, f)

    print(f"\n✅ Model kaydedildi → results/MobileNetV2_TID_final.keras")
    return model, history

# ─────────────────────────────────────────────
def evaluate(model, test_gen, class_names):
    print(f"\n[DEĞERLENDİRME]")
    test_gen.reset()
    loss, acc = model.evaluate(test_gen, verbose=1)
    print(f"  Test Accuracy: {acc*100:.2f}%")

    test_gen.reset()
    preds  = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes

    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    print(report)
    with open("results/MobileNetV2_report.txt", 'w', encoding='utf-8') as f:
        f.write(f"Test Accuracy: {acc*100:.2f}%\nTest Loss: {loss:.4f}\n\n{report}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 13))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, linewidths=0.5)
    plt.title('Confusion Matrix — MobileNetV2', fontsize=14, fontweight='bold')
    plt.ylabel('Gerçek'); plt.xlabel('Tahmin')
    plt.tight_layout()
    plt.savefig('results/MobileNetV2_confusion_matrix.png', dpi=130)
    plt.close()

    # Eğitim grafiği
    with open("results/MobileNetV2_history.json") as f:
        history = json.load(f)

    n = history['frozen_epochs']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('MobileNetV2 — Eğitim', fontsize=14, fontweight='bold')
    axes[0].plot(history['accuracy'],     label='Train', lw=2)
    axes[0].plot(history['val_accuracy'], label='Val',   lw=2, ls='--')
    axes[0].axvline(x=n, color='gray', lw=1, ls=':', label='Fine-tune')
    axes[0].set_title('Accuracy'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history['loss'],     label='Train', lw=2)
    axes[1].plot(history['val_loss'], label='Val',   lw=2, ls='--')
    axes[1].axvline(x=n, color='gray', lw=1, ls=':')
    axes[1].set_title('Loss'); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/MobileNetV2_training.png', dpi=130)
    plt.show()

    print(f"\n Sonuçlar 'results/' klasörüne kaydedildi.")
    return acc

# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(" MobileNetV2 TİD Eğitimi Başlıyor\n")

    train_gen, val_gen, test_gen, class_names = get_generators()
    model, base = build_model(len(class_names))
    model.summary()
    model, history = train(model, base, train_gen, val_gen)
    acc = evaluate(model, test_gen, class_names)

    print(f"\n{'='*50}")
    print(f"  SONUÇ: %{acc*100:.2f} Test Accuracy")
    print(f"{'='*50}")
    print("  Sıradaki adım: python 3_realtime_test.py")