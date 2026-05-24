"""
ADIM 2 — Eğitim
================
extract_keypoints.py bittikten sonra çalıştırın.

Kullanım:
    python train.py --keypoint_dir keypoints --output_dir outputs
    python train.py --keypoint_dir keypoints --output_dir outputs --classes elma,araba,anne
"""

import os
import time
import json
import math
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from Dataset import get_dataloaders
from Model import TIDTCN


# ─────────────────────────────────────────
# Yardımcılar
# ─────────────────────────────────────────

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum = self.count = 0
    def update(self, v, n=1): self.sum += v * n; self.count += n
    @property
    def avg(self): return self.sum / max(self.count, 1)


def accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        B    = target.size(0)
        _, pred = output.topk(maxk, dim=1)
        correct  = pred.t().eq(target.view(1, -1).expand_as(pred.t()))
        return [(correct[:k].reshape(-1).float().sum() / B * 100).item() for k in topk]


def cosine_lr(optimizer, epoch, warmup, total, base_lr, min_lr=1e-6):
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / max(warmup, 1)
    else:
        p  = (epoch - warmup) / max(total - warmup, 1)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * p))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ─────────────────────────────────────────
# Train / Eval
# ─────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, training):
    model.train() if training else model.eval()
    loss_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss   = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            k = min(5, logits.size(1))
            a1, a5 = accuracy(logits, y, topk=(1, k))
            B = y.size(0)
            loss_m.update(loss.item(), B)
            top1_m.update(a1, B)
            top5_m.update(a5, B)

    return loss_m.avg, top1_m.avg, top5_m.avg


# ─────────────────────────────────────────
# Ana fonksiyon
# ─────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    # Sınıf filtresi
    classes = args.classes.split(",") if args.classes else None

    train_loader, val_loader, test_loader, num_classes, class_list = get_dataloaders(
        keypoint_dir = args.keypoint_dir,
        batch_size   = args.batch_size,
        classes      = classes,
    )
    print(f"Sınıf sayısı: {num_classes}")

    # Model
    model = TIDTCN(
        num_classes  = num_classes,
        input_dim    = 138,
        num_channels = [128, 128, 256, 256],
        kernel_size  = 3,
        dropout      = args.dropout,
    ).to(device)

    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parametresi: {total:.2f}M")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(class_list, f, ensure_ascii=False, indent=2)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val  = 0.0
    patience  = 0
    history   = []

    print(f"\n{'='*55}")
    print(f"Eğitim başlıyor: {args.epochs} epoch | batch={args.batch_size}")
    print(f"{'='*55}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        lr = cosine_lr(optimizer, epoch - 1, args.warmup, args.epochs, args.lr)

        tr_loss, tr1, tr5 = run_epoch(model, train_loader, criterion, optimizer, device, True)
        vl_loss, vl1, vl5 = run_epoch(model, val_loader,   criterion, optimizer, device, False)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs}  LR={lr:.5f}  {elapsed:.0f}s")
        print(f"  Train  loss={tr_loss:.4f}  top1={tr1:.1f}%  top5={tr5:.1f}%")
        print(f"  Val    loss={vl_loss:.4f}  top1={vl1:.1f}%  top5={vl5:.1f}%")

        history.append(dict(epoch=epoch, lr=lr,
                            tr_loss=tr_loss, tr1=tr1, tr5=tr5,
                            vl_loss=vl_loss, vl1=vl1, vl5=vl5))

        if vl1 > best_val:
            best_val = vl1
            patience = 0
            torch.save({"model_state": model.state_dict(),
                        "classes": class_list,
                        "val_top1": vl1,
                        "epoch": epoch},
                       os.path.join(args.output_dir, "best_model.pt"))
            print(f"  ✓ Kaydedildi  (val top1={best_val:.1f}%)")
        else:
            patience += 1
            print(f"  Patience {patience}/{args.patience}")
            if patience >= args.patience:
                print("Early stopping.")
                break

        with open(os.path.join(args.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        print()

    # Test
    print(f"\n{'='*55}")
    ckpt = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    ts_loss, ts1, ts5 = run_epoch(model, test_loader, criterion, optimizer, device, False)
    print(f"TEST  loss={ts_loss:.4f}  top1={ts1:.1f}%  top5={ts5:.1f}%")

    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(dict(test_loss=ts_loss, test_top1=ts1, test_top5=ts5,
                       best_val_top1=best_val), f, indent=2)


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--keypoint_dir", default="keypoints",  help="extract_keypoints.py çıktısı")
    p.add_argument("--output_dir",   default="outputs",    help="Checkpoint klasörü")
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=32,  help="CPU'da 32-64 iyi çalışır")
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--warmup",       type=int,   default=5)
    p.add_argument("--patience",     type=int,   default=20)
    p.add_argument("--classes",      default=None,
                   help="Sadece belirli kelimeler: 'elma,araba,anne'")
    args = p.parse_args()
    train(args)