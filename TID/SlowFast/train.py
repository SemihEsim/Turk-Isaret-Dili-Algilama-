import os
import time
import json
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import SlowFastTID


# ─────────────────────────────────────────
# Metrik yardımcıları
# ─────────────────────────────────────────

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def top_k_accuracy(output, target, k=(1, 5)):
    with torch.no_grad():
        maxk = min(max(k), output.size(1))  # sınıf sayısından büyük olamaz
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        results = []
        for ki in k:
            ki = min(ki, maxk)
            correct_k = correct[:ki].reshape(-1).float().sum()
            results.append((correct_k / batch_size * 100).item())
        return results


# ─────────────────────────────────────────
# Warmup + Cosine LR Scheduler
# ─────────────────────────────────────────

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.base_lr       = base_lr
        self.min_lr        = min_lr

    def step(self, epoch):
        import math
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / max(self.warmup_epochs, 1)
        else:
            progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


# ─────────────────────────────────────────
# Train / Evaluate  (CPU + GPU uyumlu)
# ─────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, use_amp):
    model.train()
    loss_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    
    # Döngü başlamadan hemen önce zamanı kaydediyoruz
    start_time = time.time()

    for step, (inputs, labels) in enumerate(loader):
        slow   = inputs[0].to(device)
        fast   = inputs[1].to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model([slow, fast])
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        acc1, acc5 = top_k_accuracy(logits, labels, k=(1, 5))
        B = labels.size(0)
        loss_m.update(loss.item(), B)
        top1_m.update(acc1, B)
        top5_m.update(acc5, B)

        if (step + 1) % 10 == 0 or (step + 1) == len(loader):
            # --- ZAMAN (ETA) HESAPLAMASI ---
            elapsed_time = time.time() - start_time
            steps_done = step + 1
            time_per_step = elapsed_time / steps_done
            steps_left = len(loader) - steps_done
            eta_seconds = steps_left * time_per_step

            # Saniyeyi Saat:Dakika:Saniye formatına çevirme
            m, s = divmod(eta_seconds, 60)
            h, m = divmod(m, 60)
            eta_str = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
            # -------------------------------

            print(f"  Epoch {epoch:3d} | Step {step+1:4d}/{len(loader)} | "
                  f"Loss {loss_m.avg:.4f} | Top1 {top1_m.avg:.2f}% | Top5 {top5_m.avg:.2f}% | ETA: {eta_str}")

    return loss_m.avg, top1_m.avg, top5_m.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    loss_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    for inputs, labels in loader:
        slow   = inputs[0].to(device)
        fast   = inputs[1].to(device)
        labels = labels.to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model([slow, fast])
            loss   = criterion(logits, labels)

        acc1, acc5 = top_k_accuracy(logits, labels, k=(1, 5))
        B = labels.size(0)
        loss_m.update(loss.item(), B)
        top1_m.update(acc1, B)
        top5_m.update(acc5, B)

    return loss_m.avg, top1_m.avg, top5_m.avg


# ─────────────────────────────────────────
# Ana eğitim fonksiyonu
# ─────────────────────────────────────────

def train(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"   # AMP sadece GPU'da açık
    print(f"Cihaz: {device} | Mixed Precision: {use_amp}")

    # ── Veri ──────────────────────────────
    train_loader, val_loader, test_loader, num_classes, classes = get_dataloaders(
        root_dir        = args.data_dir,
        batch_size      = args.batch_size,
        num_workers     = 0,           # Windows'ta 0 olmalı
        num_frames_slow = args.num_frames_slow,
        alpha           = args.alpha,
    )
    print(f"Sınıf sayısı: {num_classes}")

    # ── Model ─────────────────────────────
    model = SlowFastTID(
        num_classes=num_classes,
        dropout=args.dropout,
        in_features=args.in_features,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Toplam parametre: {total_params:.2f}M")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    # ── Optimizer ─────────────────────────
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if "fc" in name or "lat" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=args.weight_decay)

    scheduler  = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs, args.lr)
    criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Eğitim döngüsü ────────────────────
    best_val_top1    = 0.0
    patience_counter = 0
    history          = []

    print(f"\n{'='*60}")
    print(f"Eğitim başlıyor: {args.epochs} epoch")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        lr = scheduler.step(epoch - 1)

        tr_loss, tr_top1, tr_top5 = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, use_amp)

        vl_loss, vl_top1, vl_top5 = evaluate(
            model, val_loader, criterion, device, use_amp)

        elapsed = time.time() - t0
        print(f"\nEpoch {epoch:3d}/{args.epochs} | LR {lr:.6f} | {elapsed:.1f}s")
        print(f"  Train → Loss {tr_loss:.4f} | Top1 {tr_top1:.2f}% | Top5 {tr_top5:.2f}%")
        print(f"  Val   → Loss {vl_loss:.4f} | Top1 {vl_top1:.2f}% | Top5 {vl_top5:.2f}%")

        record = dict(epoch=epoch, lr=lr,
                      tr_loss=tr_loss, tr_top1=tr_top1, tr_top5=tr_top5,
                      vl_loss=vl_loss, vl_top1=vl_top1, vl_top5=vl_top5)
        history.append(record)

        if vl_top1 > best_val_top1:
            best_val_top1    = vl_top1
            patience_counter = 0
            ckpt_path        = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_top1":    vl_top1,
                "classes":     classes,
                "in_features": args.in_features,
            }, ckpt_path)
            print(f"  ✓ En iyi model kaydedildi → Val Top1: {best_val_top1:.2f}%")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{args.patience}")

        with open(os.path.join(args.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if patience_counter >= args.patience:
            print(f"\nEarly stopping: {args.patience} epoch iyileşme olmadı.")
            break
        print()

    # ── Test ──────────────────────────────
    print(f"\n{'='*60}")
    print("Test değerlendirmesi (en iyi model)")
    print(f"{'='*60}")

    ckpt = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state"])

    ts_loss, ts_top1, ts_top5 = evaluate(model, test_loader, criterion, device, use_amp)
    print(f"Test → Loss {ts_loss:.4f} | Top1 {ts_top1:.2f}% | Top5 {ts_top5:.2f}%")

    results = dict(test_loss=ts_loss, test_top1=ts_top1, test_top5=ts_top5,
                   best_val_top1=best_val_top1)
    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="TID SlowFast Eğitimi (Keypoint)")
    p.add_argument("--data_dir",        default="../keypoints", help="Keypoint veri klasörü (train/val/test içermeli)")
    p.add_argument("--output_dir",      default="outputs",     help="Checkpoint klasörü")
    p.add_argument("--epochs",          type=int,   default=80)
    p.add_argument("--batch_size",      type=int,   default=32,   help="Keypoint verisi hafif, 32-64 kullanılabilir")
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--dropout",         type=float, default=0.5)
    p.add_argument("--warmup_epochs",   type=int,   default=5)
    p.add_argument("--patience",        type=int,   default=15)
    p.add_argument("--num_frames_slow", type=int,   default=8)
    p.add_argument("--alpha",           type=int,   default=4)
    p.add_argument("--in_features",     type=int,   default=138,  help="Keypoint feature sayısı (46 nokta × 3)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)