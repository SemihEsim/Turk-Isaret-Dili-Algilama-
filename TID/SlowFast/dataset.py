import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random


class TIDKeypointDataset(Dataset):
    """
    TID (Türk İşaret Dili) Keypoint Dataset for SlowFast

    Klasör yapısı:
        keypoints/
            train/
                kelime1/
                    signer0_sample123_color.npy   → (32, 138)
                    ...
                kelime2/
                    ...
            val/
                ...
            test/
                ...

    Her .npy dosyası (32, 138) boyutunda:
        32 frame, 138 feature (46 landmark × 3 koordinat: x, y, z)
    """

    def __init__(
        self,
        root_dir,
        split="train",              # "train" | "val" | "test" | None
        num_frames_slow=8,
        num_frames_fast=32,
        alpha=4,
        in_features=138,
    ):
        self.root_dir = root_dir
        self.split = split
        self.num_frames_slow = num_frames_slow
        self.num_frames_fast = num_frames_fast
        self.alpha = alpha
        self.in_features = in_features

        assert num_frames_fast == num_frames_slow * alpha, \
            f"num_frames_fast ({num_frames_fast}) == num_frames_slow ({num_frames_slow}) * alpha ({alpha}) olmalı"

        # split=None → root_dir'in kendisi kelime klasörlerini içeriyor
        self.split_dir = os.path.join(root_dir, split) if split else root_dir

        # Sadece klasörleri al (gizli dosyaları atla)
        self.classes = sorted([
            d for d in os.listdir(self.split_dir)
            if os.path.isdir(os.path.join(self.split_dir, d)) and not d.startswith(".")
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(self.split_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.endswith(".npy"):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

        # Augmentation sadece train için
        self.augment = (split == "train")

        print(f"[{split}] {len(self.classes)} sınıf, {len(self.samples)} keypoint dosyası yüklendi.")

    def __len__(self):
        return len(self.samples)

    def _load_keypoints(self, npy_path):
        """
        .npy dosyasından keypoint'leri yükle.
        Beklenen shape: (T, F) → (32, 138)
        Eğer T != num_frames_fast ise yeniden örnekle.
        """
        data = np.load(npy_path).astype(np.float32)  # (T, F)

        T, F = data.shape

        # Feature boyutu kontrol
        if F != self.in_features:
            # Eksik feature'ları sıfırla veya fazlalıkları kes
            if F < self.in_features:
                pad = np.zeros((T, self.in_features - F), dtype=np.float32)
                data = np.concatenate([data, pad], axis=1)
            else:
                data = data[:, :self.in_features]

        # Temporal boyut kontrol
        if T != self.num_frames_fast:
            indices = np.linspace(0, T - 1, self.num_frames_fast, dtype=int)
            data = data[indices]

        return data  # (num_frames_fast, in_features)

    def _augment(self, data):
        """
        Train augmentation:
        1. Gaussian noise ekleme (koordinat pertürbasyonu)
        2. Temporal jitter (frame sırasını hafifçe karıştırma)
        3. Horizontal flip (x koordinatlarını aynalandırma)
        4. Temporal scaling (hız değiştirme)
        """
        T, F = data.shape

        # 1. Gaussian noise (%50 olasılık)
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.01, data.shape).astype(np.float32)
            data = data + noise

        # 2. Temporal jitter (%30 olasılık)
        if random.random() < 0.3:
            for i in range(T):
                j = min(max(i + random.randint(-1, 1), 0), T - 1)
                if i != j:
                    data[i], data[j] = data[j].copy(), data[i].copy()

        # 3. Horizontal flip (%50 olasılık)
        #    x koordinatları her 3. eleman (idx 0, 3, 6, ...)
        if random.random() < 0.5:
            for c in range(0, F, 3):
                data[:, c] = 1.0 - data[:, c]

        # 4. Temporal scaling (%30 olasılık)
        #    Random subset seçip interpole et
        if random.random() < 0.3:
            scale = random.uniform(0.8, 1.2)
            new_T = int(T * scale)
            new_T = max(new_T, 4)  # minimum 4 frame
            indices = np.linspace(0, T - 1, new_T, dtype=int)
            stretched = data[indices]
            # Geri T frame'e örnekle
            final_idx = np.linspace(0, len(stretched) - 1, T, dtype=int)
            data = stretched[final_idx]

        return data

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]

        # Keypoint verisini yükle
        data = self._load_keypoints(npy_path)  # (32, 138)

        # Augmentation
        if self.augment:
            data = self._augment(data)

        # Fast pathway: tüm frame'ler → (in_features, T_fast)
        fast_tensor = torch.from_numpy(data.T.copy())  # (138, 32)

        # Slow pathway: her alpha'da bir frame → (in_features, T_slow)
        slow_data = data[::self.alpha]  # (8, 138)
        slow_tensor = torch.from_numpy(slow_data.T.copy())  # (138, 8)

        return [slow_tensor, fast_tensor], torch.tensor(label, dtype=torch.long)


# Windows'ta multiprocessing pickle sorunu yaşamamak için
# collate_fn modül düzeyinde tanımlanmalı (local fonksiyon olamaz)
def collate_fn(batch):
    slow_list = [b[0][0] for b in batch]
    fast_list = [b[0][1] for b in batch]
    labels    = [b[1] for b in batch]
    return [torch.stack(slow_list), torch.stack(fast_list)], torch.stack(labels)


def get_dataloaders(root_dir, batch_size=32, num_workers=0, num_frames_slow=8, alpha=4):
    """
    Train/val/test DataLoader'larını döndür.
    num_workers=0: Windows'ta spawn multiprocessing sorununu önler.
                   GPU'lu Linux sistemlerde 4 yapabilirsiniz.
    """
    kwargs = dict(
        root_dir=root_dir,
        num_frames_slow=num_frames_slow,
        num_frames_fast=num_frames_slow * alpha,
        alpha=alpha,
    )

    train_ds = TIDKeypointDataset(split="train", **kwargs)
    val_ds   = TIDKeypointDataset(split="val",   **kwargs)
    test_ds  = TIDKeypointDataset(split="test",  **kwargs)

    # pin_memory sadece CUDA varsa açık olmalı
    pin = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn,
                              pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=collate_fn,
                              pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=collate_fn,
                              pin_memory=pin)

    num_classes = train_ds.num_classes
    classes = train_ds.classes

    return train_loader, val_loader, test_loader, num_classes, classes