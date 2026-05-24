"""
Dataset — .npy keypoint dosyalarını okur
=========================================
extract_keypoints.py çalıştırıldıktan sonra kullanılır.
Video açılmaz, sadece .npy okunur → çok hızlı veri yükleme.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


NUM_FRAMES  = 32
FEATURE_DIM = 138


class TIDKeypointDataset(Dataset):

    def __init__(self, keypoint_dir, split="train", classes=None, augment=False):
        """
        keypoint_dir : extract_keypoints.py çıktısı (keypoints/)
        split        : "train" | "val" | "test"
        classes      : sadece bu kelimeleri yükle (None = hepsi)
        augment      : train için veri artırma
        """
        self.augment = augment
        split_dir    = os.path.join(keypoint_dir, split)

        # Sınıfları belirle
        if classes is not None:
            self.classes = sorted(classes)
        else:
            # classes.json varsa oradan al
            cls_file = os.path.join(keypoint_dir, "classes.json")
            if os.path.exists(cls_file):
                with open(cls_file, encoding="utf-8") as f:
                    self.classes = json.load(f)
            else:
                self.classes = sorted([
                    d for d in os.listdir(split_dir)
                    if os.path.isdir(os.path.join(split_dir, d))
                ])

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.num_classes  = len(self.classes)

        # Tüm .npy dosyalarını listele
        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.endswith(".npy"):
                    self.samples.append(
                        (os.path.join(cls_dir, fname), self.class_to_idx[cls])
                    )

        print(f"[{split}] {self.num_classes} sınıf, {len(self.samples)} örnek")

    def __len__(self):
        return len(self.samples)

    def _augment(self, x):
        """
        x: (T, F) numpy array
        Basit augmentasyonlar — keypoint verisine uygun
        """
        # 1) Zaman gürültüsü: frame'leri hafif kaydır
        if np.random.random() < 0.5:
            shift = np.random.randint(-3, 4)
            x = np.roll(x, shift, axis=0)

        # 2) Ölçek gürültüsü: koordinatları biraz büyüt/küçült
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            x = x * scale

        # 3) Yatay çevirme (sağ/sol el koordinatlarını değiştir)
        if np.random.random() < 0.5:
            flipped = x.copy()
            # Sol el (0:63) ↔ Sağ el (63:126)
            flipped[:, 0:63]  = x[:, 63:126]
            flipped[:, 63:126] = x[:, 0:63]
            # X koordinatlarını ters çevir (1 - x)
            for col in range(0, 126, 3):
                flipped[:, col] = 1.0 - flipped[:, col]
            x = flipped

        # 4) Gaussian gürültü
        if np.random.random() < 0.3:
            x = x + np.random.randn(*x.shape).astype(np.float32) * 0.01

        return x

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        try:
            x = np.load(path).astype(np.float32)  # (32, 138)
        except Exception:
            x = np.zeros((NUM_FRAMES, FEATURE_DIM), dtype=np.float32)

        # Shape garantisi
        if x.shape[0] != NUM_FRAMES:
            # Yeniden örnekle
            indices = np.linspace(0, len(x) - 1, NUM_FRAMES, dtype=int)
            x = x[indices]
        if x.shape[1] != FEATURE_DIM:
            tmp = np.zeros((NUM_FRAMES, FEATURE_DIM), dtype=np.float32)
            tmp[:, :x.shape[1]] = x[:, :FEATURE_DIM]
            x = tmp

        # Normalizasyon: sıfır olmayan değerleri [-1, 1]'e ölçekle
        nonzero = x[x != 0]
        if len(nonzero) > 0:
            x_min, x_max = nonzero.min(), nonzero.max()
            if x_max > x_min:
                mask  = (x != 0)
                x[mask] = 2.0 * (x[mask] - x_min) / (x_max - x_min) - 1.0

        if self.augment:
            x = self._augment(x)

        return torch.from_numpy(x), torch.tensor(label, dtype=torch.long)


def get_dataloaders(keypoint_dir, batch_size=32, classes=None):
    """
    Train/val/test DataLoader'larını döndür.
    Windows'ta num_workers=0 olmalı.
    """
    train_ds = TIDKeypointDataset(keypoint_dir, "train", classes, augment=True)
    val_ds   = TIDKeypointDataset(keypoint_dir, "val",   classes, augment=False)
    test_ds  = TIDKeypointDataset(keypoint_dir, "test",  classes, augment=False)

    kw = dict(num_workers=0, pin_memory=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw)

    return train_loader, val_loader, test_loader, train_ds.num_classes, train_ds.classes