"""
1_prepare_data.py — Veri Hazırlama
===================================
- train klasöründen %15 val seti ayırır
- Veri setini kontrol eder ve istatistik gösterir
- Örnek görüntüleri gösterir

Çalıştır: python 1_prepare_data.py

Beklenen yapı:
  dataset/
    train/
      A/  B/  C/  ...
    test/
      A/  B/  C/  ...
"""

import os
import shutil
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────
DATASET_DIR  = "random_forest/dataset"
TRAIN_DIR    = os.path.join(DATASET_DIR, "train")
VAL_DIR      = os.path.join(DATASET_DIR, "val")
TEST_DIR     = os.path.join(DATASET_DIR, "test")
VAL_SPLIT    = 0.15   # train'in %15'i val olur
SEED         = 42
# ─────────────────────────────────────────────

random.seed(SEED)


def create_val_split():
    """Train'den val ayır (eğer val yoksa)."""
    if os.path.exists(VAL_DIR) and len(os.listdir(VAL_DIR)) > 0:
        print("✅ Val klasörü zaten var, atlanıyor.")
        return

    print(f"📂 Val seti oluşturuluyor (%{int(VAL_SPLIT*100)} train'den)...\n")
    classes = sorted(os.listdir(TRAIN_DIR))

    for cls in classes:
        src_dir = os.path.join(TRAIN_DIR, cls)
        val_cls_dir = os.path.join(VAL_DIR, cls)
        os.makedirs(val_cls_dir, exist_ok=True)

        images = [f for f in os.listdir(src_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        random.shuffle(images)

        n_val = int(len(images) * VAL_SPLIT)
        val_images = images[:n_val]

        for img in val_images:
            src  = os.path.join(src_dir, img)
            dst  = os.path.join(val_cls_dir, img)
            shutil.move(src, dst)

        print(f"  {cls}: {len(val_images)} görüntü val'e taşındı "
              f"({len(images)-len(val_images)} train'de kaldı)")

    print(f"\n Val seti oluşturuldu: {VAL_DIR}")


def count_images(directory):
    """Klasördeki görüntü sayısını sınıf bazında say."""
    stats = {}
    if not os.path.exists(directory):
        return stats
    for cls in sorted(os.listdir(directory)):
        cls_path = os.path.join(directory, cls)
        if not os.path.isdir(cls_path):
            continue
        imgs = [f for f in os.listdir(cls_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        stats[cls] = len(imgs)
    return stats


def print_stats():
    """Veri seti istatistiklerini yazdır."""
    print("\n" + "="*60)
    print("  VERİ SETİ İSTATİSTİKLERİ")
    print("="*60)

    train_stats = count_images(TRAIN_DIR)
    val_stats   = count_images(VAL_DIR)
    test_stats  = count_images(TEST_DIR)

    all_classes = sorted(set(list(train_stats.keys()) +
                             list(val_stats.keys()) +
                             list(test_stats.keys())))

    print(f"\n{'Sınıf':<8} {'Train':>8} {'Val':>8} {'Test':>8} {'Toplam':>8}")
    print("-"*40)

    total_train = total_val = total_test = 0
    for cls in all_classes:
        t = train_stats.get(cls, 0)
        v = val_stats.get(cls, 0)
        te = test_stats.get(cls, 0)
        total_train += t
        total_val   += v
        total_test  += te
        print(f"  {cls:<6} {t:>8} {v:>8} {te:>8} {t+v+te:>8}")

    print("-"*40)
    print(f"  {'TOPLAM':<6} {total_train:>8} {total_val:>8} {total_test:>8} "
          f"{total_train+total_val+total_test:>8}")

    print(f"\n  Sınıf sayısı : {len(all_classes)}")
    print(f"  Sınıflar     : {all_classes}")

    # Uyarı: dengesiz sınıflar
    if train_stats:
        min_cls = min(train_stats, key=train_stats.get)
        max_cls = max(train_stats, key=train_stats.get)
        ratio   = train_stats[max_cls] / (train_stats[min_cls] + 1e-6)
        if ratio > 2:
            print(f"\n  ⚠️  Dengesizlik uyarısı: {max_cls}({train_stats[max_cls]}) "
                  f"vs {min_cls}({train_stats[min_cls]}) — oran: {ratio:.1f}x")

    # Class names kaydet
    os.makedirs("results", exist_ok=True)
    with open("results/class_names.json", 'w', encoding='utf-8') as f:
        json.dump(all_classes, f, ensure_ascii=False)
    print(f"\n  ✅ Sınıf isimleri kaydedildi: results/class_names.json")

    return all_classes, train_stats


def plot_distribution(train_stats):
    """Sınıf dağılımı grafiği."""
    classes = list(train_stats.keys())
    counts  = list(train_stats.values())

    plt.figure(figsize=(14, 5))
    bars = plt.bar(classes, counts, color='steelblue', edgecolor='white', linewidth=0.5)
    plt.title('Train Seti — Sınıf Dağılımı', fontsize=14, fontweight='bold')
    plt.xlabel('Harf')
    plt.ylabel('Görüntü Sayısı')
    plt.axhline(y=np.mean(counts), color='red', linestyle='--',
                label=f'Ortalama: {np.mean(counts):.0f}')
    plt.legend()
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 str(count), ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig('results/data_distribution.png', dpi=120)
    plt.show()
    print("  ✅ Dağılım grafiği: results/data_distribution.png")


def show_samples(n_per_class=3):
    """Her sınıftan örnek görüntüler göster."""
    classes = sorted(os.listdir(TRAIN_DIR))
    n_cols  = n_per_class
    n_rows  = len(classes)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
    fig.suptitle('Örnek Görüntüler (Train)', fontsize=14, fontweight='bold')

    for r, cls in enumerate(classes):
        cls_path = os.path.join(TRAIN_DIR, cls)
        images   = [f for f in os.listdir(cls_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        samples  = random.sample(images, min(n_per_class, len(images)))

        for c, img_name in enumerate(samples):
            ax  = axes[r, c] if n_rows > 1 else axes[c]
            img = mpimg.imread(os.path.join(cls_path, img_name))
            ax.imshow(img)
            ax.axis('off')
            if c == 0:
                ax.set_ylabel(cls, fontsize=10, fontweight='bold', rotation=0,
                              labelpad=20, va='center')

    plt.tight_layout()
    plt.savefig('results/sample_images.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("  ✅ Örnek görüntüler: results/sample_images.png")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Veri Hazırlama Başlıyor\n")

    # Klasörleri kontrol et
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ HATA: '{TRAIN_DIR}' bulunamadı!")
        print("   dataset/train/A/, dataset/train/B/ ... şeklinde olmalı")
        exit(1)

    # Val split oluştur
    create_val_split()

    # İstatistikler
    all_classes, train_stats = print_stats()

    # Grafikler
    print("\n📊 Grafikler oluşturuluyor...")
    plot_distribution(train_stats)
    show_samples(n_per_class=3)

    print("\n🎉 Veri hazırlama tamamlandı!")
    print("   Sıradaki adım: python 2_train.py")