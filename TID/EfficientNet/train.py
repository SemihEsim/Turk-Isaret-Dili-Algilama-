import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

DATASET_DIR     = "random_forest/dataset"
IMG_SIZE        = 224
BATCH_SIZE      = 64
EPOCHS_FROZEN   = 20
EPOCHS_FINETUNE = 20

os.makedirs("results",     exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(f"GPU: {[g.name for g in gpus] if gpus else 'YOK (CPU)'}")
print(f"TensorFlow: {tf.__version__}\n")

def get_generators():
    train_datagen = ImageDataGenerator(
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
    base_datagen = ImageDataGenerator()

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

    print(f"Train: {train_gen.samples} | Val: {val_gen.samples} | Test: {test_gen.samples}")
    print(f"Siniflar: {list(train_gen.class_indices.keys())}\n")

    class_names = list(train_gen.class_indices.keys())
    with open("results/class_names_eff.json", 'w', encoding='utf-8') as f:
        json.dump(class_names, f, ensure_ascii=False)

    return train_gen, val_gen, test_gen, class_names

def build_model(num_classes):
    base = EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
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

    return Model(inputs, outputs, name="EfficientNetB0_TID"), base

def get_callbacks():
    return [
        EarlyStopping(monitor='val_accuracy', patience=6,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1),
    ]

def save_model(model, path):
    tf.saved_model.save(model, path)
    print(f"Model kaydedildi: {path}")

def train(model, base_model, train_gen, val_gen):
    print("\n" + "="*60)
    print("  EfficientNetB0 EGITIMI")
    print("="*60)

    # Asama 1: Frozen
    print(f"\n[ASAMA 1] Frozen - max {EPOCHS_FROZEN} epoch")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    h1 = model.fit(train_gen, epochs=EPOCHS_FROZEN,
                   validation_data=val_gen,
                   callbacks=get_callbacks(), verbose=1)

    print(f"\nAsama 1 en iyi val_accuracy: {max(h1.history['val_accuracy']):.4f}")
    save_model(model, "checkpoints/EfficientNetB0_frozen")

    # Asama 2: Fine-tune
    print(f"\n[ASAMA 2] Fine-tune - son 40 katman - max {EPOCHS_FINETUNE} epoch")
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    h2 = model.fit(train_gen, epochs=EPOCHS_FINETUNE,
                   validation_data=val_gen,
                   callbacks=get_callbacks(), verbose=1)

    save_model(model, "results/EfficientNetB0_TID_final")

    history = {
        'accuracy':      [float(x) for x in h1.history['accuracy']     + h2.history['accuracy']],
        'val_accuracy':  [float(x) for x in h1.history['val_accuracy'] + h2.history['val_accuracy']],
        'loss':          [float(x) for x in h1.history['loss']         + h2.history['loss']],
        'val_loss':      [float(x) for x in h1.history['val_loss']     + h2.history['val_loss']],
        'frozen_epochs': len(h1.history['accuracy'])
    }
    with open("results/EfficientNetB0_history.json", 'w') as f:
        json.dump(history, f)

    return model, history

def evaluate(model, test_gen, class_names):
    print("\n[DEGERLENDIRME]")
    test_gen.reset()
    loss, acc = model.evaluate(test_gen, verbose=1)
    print(f"  Test Accuracy: {acc*100:.2f}%")

    test_gen.reset()
    preds  = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes

    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    print(report)
    with open("results/EfficientNetB0_report.txt", 'w', encoding='utf-8') as f:
        f.write(f"Test Accuracy: {acc*100:.2f}%\nTest Loss: {loss:.4f}\n\n{report}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 13))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names, linewidths=0.5)
    plt.title('Confusion Matrix - EfficientNetB0', fontsize=14, fontweight='bold')
    plt.ylabel('Gercek'); plt.xlabel('Tahmin')
    plt.tight_layout()
    plt.savefig('results/EfficientNetB0_confusion_matrix.png', dpi=130)
    plt.close()

    with open("results/EfficientNetB0_history.json") as f:
        history = json.load(f)

    n = history['frozen_epochs']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('EfficientNetB0 - Egitim', fontsize=14, fontweight='bold')
    axes[0].plot(history['accuracy'],     label='Train', color='#2E7D32', lw=2)
    axes[0].plot(history['val_accuracy'], label='Val',   color='#2E7D32', lw=2, ls='--')
    axes[0].axvline(x=n, color='gray', lw=1, ls=':', label='Fine-tune')
    axes[0].set_title('Accuracy'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history['loss'],     label='Train', color='#FF6F00', lw=2)
    axes[1].plot(history['val_loss'], label='Val',   color='#FF6F00', lw=2, ls='--')
    axes[1].axvline(x=n, color='gray', lw=1, ls=':')
    axes[1].set_title('Loss'); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/EfficientNetB0_training.png', dpi=130)
    plt.show()

    print("\nSonuclar 'results/' klasorune kaydedildi.")
    return acc

if __name__ == "__main__":
    print("EfficientNetB0 TID Egitimi Basliyor\n")

    train_gen, val_gen, test_gen, class_names = get_generators()
    model, base = build_model(len(class_names))
    model.summary()
    model, history = train(model, base, train_gen, val_gen)
    acc = evaluate(model, test_gen, class_names)

    print(f"\n{'='*50}")
    print(f"  SONUC: %{acc*100:.2f} Test Accuracy")
    print(f"{'='*50}")