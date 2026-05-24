import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────
# Temel bloklar (1D — Keypoint tabanlı)
# ─────────────────────────────────────────

class ResBlock1D(nn.Module):
    """1D Temporal Residual Block"""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)

        self.relu      = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


def make_layer(in_channels, out_channels, num_blocks, stride=1):
    """1D residual katmanlar oluştur."""
    layers = []
    downsample = None

    if in_channels != out_channels or stride != 1:
        downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels)
        )

    layers.append(ResBlock1D(in_channels, out_channels, stride=stride, downsample=downsample))
    for _ in range(1, num_blocks):
        layers.append(ResBlock1D(out_channels, out_channels))

    return nn.Sequential(*layers)


# ─────────────────────────────────────────
# Slow Pathway (1D Temporal)
# ─────────────────────────────────────────

class SlowPathway(nn.Module):
    """
    Slow pathway: az frame, yüksek kanal sayısı.
    El pozisyonu ve parmak açılarına odaklanır.
    Giriş: (B, num_features, T_slow)  → (B, 138, 8)
    """

    def __init__(self, in_features=138):
        super().__init__()

        # Stem: feature boyutunu kanal sayısına dönüştür
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # ResNet benzeri yapı
        self.layer1 = make_layer(64,  128, 2, stride=1)   # T: 8
        self.layer2 = make_layer(128, 256, 2, stride=2)   # T: 4
        self.layer3 = make_layer(256, 512, 2, stride=2)   # T: 2

        self.out_channels = [128, 256, 512]

    def forward(self, x):
        x  = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        return c1, c2, c3


# ─────────────────────────────────────────
# Fast Pathway (1D Temporal)
# ─────────────────────────────────────────

class FastPathway(nn.Module):
    """
    Fast pathway: çok frame, düşük kanal sayısı (beta=4).
    Jest hareketinin dinamiğine odaklanır.
    Giriş: (B, num_features, T_fast)  → (B, 138, 32)
    """

    BETA = 4  # kanal azaltma oranı

    def __init__(self, in_features=138):
        super().__init__()
        B = self.BETA

        # Stem (temporal conv ile hareket yakalama)
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, 64 // B, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64 // B),
            nn.ReLU(inplace=True),
        )

        self.layer1 = make_layer(64//B,  128//B, 2, stride=1)   # T: 32
        self.layer2 = make_layer(128//B, 256//B, 2, stride=2)   # T: 16
        self.layer3 = make_layer(256//B, 512//B, 2, stride=2)   # T: 8

        self.out_channels = [128//B, 256//B, 512//B]

    def forward(self, x):
        x  = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        return c1, c2, c3


# ─────────────────────────────────────────
# Lateral Connection (Fast → Slow)
# ─────────────────────────────────────────

class LateralConnection(nn.Module):
    """
    Fast pathway'den Slow pathway'e bilgi aktarımı.
    Fast'ın temporal boyutunu slow'a eşitlemek için
    temporal stride veya interpolation uygulanır.
    """

    def __init__(self, fast_channels, target_temporal_ratio):
        super().__init__()
        self.target_ratio = target_temporal_ratio

        # Fast kanallarını 2x'e çıkar (lateral bilgi)
        self.conv = nn.Conv1d(
            fast_channels,
            fast_channels * 2,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn   = nn.BatchNorm1d(fast_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fast_feat, slow_feat):
        """fast_feat → slow temporal boyutuna uyarla ve concat et."""
        lat = self.relu(self.bn(self.conv(fast_feat)))

        # Temporal boyutu slow'a eşitle
        T_slow = slow_feat.shape[2]
        if lat.shape[2] != T_slow:
            lat = F.adaptive_avg_pool1d(lat, T_slow)

        return torch.cat([slow_feat, lat], dim=1)


# ─────────────────────────────────────────
# SlowFast Ana Model (Keypoint tabanlı)
# ─────────────────────────────────────────

class SlowFastTID(nn.Module):
    """
    SlowFast Türk İşaret Dili Tanıma Modeli (Keypoint tabanlı).

    Giriş:
        inputs[0] = slow_keypoints: (B, num_features, T_slow)  → (B, 138, 8)
        inputs[1] = fast_keypoints: (B, num_features, T_fast)  → (B, 138, 32)

    Çıkış:
        logits: (B, num_classes)
    """

    ALPHA = 4   # fast_frames / slow_frames

    def __init__(self, num_classes=20, dropout=0.5, in_features=138):
        super().__init__()

        self.slow = SlowPathway(in_features)
        self.fast = FastPathway(in_features)

        # Lateral connections (her aşama için)
        fast_ch = self.fast.out_channels   # [32, 64, 128]
        slow_ch = self.slow.out_channels   # [128, 256, 512]

        # target_temporal_ratio: fast/slow temporal oranı (her katmanda)
        self.lat1 = LateralConnection(fast_ch[0], self.ALPHA)
        self.lat2 = LateralConnection(fast_ch[1], self.ALPHA)

        # Son katman: doğrudan global pooling + concat
        total_dim = slow_ch[2] + fast_ch[2]

        # Lateral'den gelen ekstra kanalları hesaba kat
        # layer1 sonrası slow: 128 + (fast_ch[0]*2) = 128 + 64 = 192 → layer2'ye giriş
        # Ama basitlik için lateral'i sadece son fusion'da kullanıyoruz

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(total_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, inputs):
        slow_input, fast_input = inputs  # unpack list

        # Her pathway'den ara feature'lar
        s1, s2, s3 = self.slow(slow_input)
        f1, f2, f3 = self.fast(fast_input)

        # Son katman: global pooling + concat
        s3_pooled = F.adaptive_avg_pool1d(s3, 1).flatten(1)  # (B, 512)
        f3_pooled = F.adaptive_avg_pool1d(f3, 1).flatten(1)  # (B, 128)

        fused = torch.cat([s3_pooled, f3_pooled], dim=1)  # (B, 640)
        fused = self.dropout(fused)
        logits = self.fc(fused)

        return logits


# ─────────────────────────────────────────
# Pretrained yükleyici (opsiyonel)
# ─────────────────────────────────────────

def load_pretrained_slowfast(model, checkpoint_path):
    """
    Önceden eğitilmiş checkpoint'i yükle.
    Uyumsuz katmanları (fc gibi) atla.
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Farklı checkpoint formatları
    if "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model_state = model.state_dict()
    filtered = {
        k: v for k, v in state_dict.items()
        if k in model_state and v.shape == model_state[k].shape
    }
    missing = [k for k in model_state if k not in filtered]
    unexpected = [k for k in state_dict if k not in model_state]

    model.load_state_dict(filtered, strict=False)
    print(f"Pretrained yüklendi: {len(filtered)} katman eşleşti.")
    print(f"Eksik (random init): {len(missing)}")
    print(f"Beklenmeyen (atlandı): {len(unexpected)}")
    return model


if __name__ == "__main__":
    model = SlowFastTID(num_classes=20, dropout=0.5, in_features=138)

    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Toplam parametre: {total:.2f}M")
    print(f"Eğitilebilir: {trainable:.2f}M")

    # Dummy test
    B = 4
    slow = torch.randn(B, 138, 8)
    fast = torch.randn(B, 138, 32)

    with torch.no_grad():
        out = model([slow, fast])
    print(f"Çıkış shape: {out.shape}")  # (4, 20)