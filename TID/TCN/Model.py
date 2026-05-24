"""
TCN — Temporal Convolutional Network
=====================================
İşaret dili için tasarlanmış hafif ve hızlı mimari.

Giriş: (B, T, C) = (batch, 32 frame, 138 özellik)
Çıkış: (B, num_classes)

Neden TCN?
- 1D dilated conv → geniş temporal alıcı alan, az parametre
- Paralel hesaplama (RNN gibi sıralı değil) → CPU'da hızlı
- Residual bağlantılar → derin ağlarda stabil eğitim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    """
    TCN temel bloğu:
    Dilated kausal konvolüsyon + residual bağlantı
    """

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        # Kausal padding: gelecek frame'lere bakma
        pad = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=pad, dilation=dilation)

        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        self.relu  = nn.ReLU()
        self.pad   = pad

        # Residual projeksiyon (kanal sayısı değişiyorsa)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        # x: (B, C, T)
        identity = x

        out = self.conv1(x)
        out = out[:, :, :-self.pad]   # kausal: fazladan padding'i kırp
        out = self.relu(self.bn1(out))
        out = self.drop1(out)

        out = self.conv2(out)
        out = out[:, :, :-self.pad]
        out = self.relu(self.bn2(out))
        out = self.drop2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


class TIDTCN(nn.Module):
    """
    TID TCN Ana Model

    Giriş: (B, T, input_dim)   — T=32, input_dim=138
    Çıkış: (B, num_classes)
    """

    def __init__(
        self,
        num_classes,
        input_dim   = 138,    # 63(sol el) + 63(sağ el) + 12(poz)
        num_channels = None,  # Her katmandaki kanal sayıları listesi
        kernel_size  = 3,
        dropout      = 0.3,
    ):
        super().__init__()

        if num_channels is None:
            # 4 katman, her seferinde 2x büyür
            num_channels = [128, 128, 256, 256]

        # Giriş projeksiyonu
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, num_channels[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # TCN katmanları — dilation 1, 2, 4, 8 (üstel büyüme)
        layers = []
        in_ch  = num_channels[0]
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch

        self.tcn = nn.Sequential(*layers)

        # Attention pooling: tüm zaman adımlarına ağırlık ver
        self.attn = nn.Sequential(
            nn.Linear(num_channels[-1], 1),
            nn.Softmax(dim=1),
        )

        # Sınıflandırıcı
        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1], 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (B, T, input_dim)
        """
        B, T, _ = x.shape

        # Giriş projeksiyonu: (B, T, C0)
        x = self.input_proj(x)

        # TCN için (B, C, T) formatına geç
        x = x.permute(0, 2, 1)
        x = self.tcn(x)

        # Attention pooling: (B, T, C) → (B, C)
        x = x.permute(0, 2, 1)           # (B, T, C)
        attn_w = self.attn(x)             # (B, T, 1)
        x = (x * attn_w).sum(dim=1)       # (B, C)

        return self.classifier(x)         # (B, num_classes)


if __name__ == "__main__":
    model = TIDTCN(num_classes=30, input_dim=138)

    total     = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Toplam parametre:    {total:.2f}M")
    print(f"Eğitilebilir:        {trainable:.2f}M")

    # Dummy test
    x   = torch.randn(4, 32, 138)
    out = model(x)
    print(f"Giriş shape:  {x.shape}")
    print(f"Çıkış shape:  {out.shape}")   # (4, 30)