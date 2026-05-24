import os
import json
import argparse
import torch
import numpy as np

from model import SlowFastTID


class TIDPredictor:
    """
    Tek .npy keypoint dosyası veya klasör üzerinde TID tahmini yapar.

    Kullanım:
        predictor = TIDPredictor("outputs/best_model.pt")
        pred, conf = predictor.predict("keypoint.npy")
        top5 = predictor.predict_top5("keypoint.npy")
    """

    def __init__(self, checkpoint_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.classes = ckpt["classes"]
        self.in_features = ckpt.get("in_features", 138)
        num_classes  = len(self.classes)

        self.model = SlowFastTID(
            num_classes=num_classes, dropout=0.0, in_features=self.in_features)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()

        print(f"Model yüklendi: {num_classes} sınıf | {self.in_features} feature | Cihaz: {self.device}")
        print(f"Val Top1 (eğitim): {ckpt.get('val_top1', 'N/A'):.2f}%")

        # Parametreler
        self.num_frames_slow = 8
        self.num_frames_fast = 32
        self.alpha = 4

    def _preprocess(self, npy_path):
        """
        .npy dosyası → [slow_tensor, fast_tensor]
        """
        data = np.load(npy_path).astype(np.float32)  # (T, F)
        T, F = data.shape

        # Feature boyutu kontrol
        if F != self.in_features:
            if F < self.in_features:
                pad = np.zeros((T, self.in_features - F), dtype=np.float32)
                data = np.concatenate([data, pad], axis=1)
            else:
                data = data[:, :self.in_features]

        # Temporal boyut kontrol
        if T != self.num_frames_fast:
            indices = np.linspace(0, T - 1, self.num_frames_fast, dtype=int)
            data = data[indices]

        # Fast: (1, F, T_fast)
        fast_tensor = torch.from_numpy(data.T.copy()).unsqueeze(0).to(self.device)

        # Slow: her alpha'da bir → (1, F, T_slow)
        slow_data = data[::self.alpha]
        slow_tensor = torch.from_numpy(slow_data.T.copy()).unsqueeze(0).to(self.device)

        return slow_tensor, fast_tensor

    @torch.no_grad()
    def predict(self, npy_path):
        """
        En olası sınıfı döndür.
        Returns:
            predicted_class (str), confidence (float, 0-1)
        """
        slow, fast = self._preprocess(npy_path)
        with torch.amp.autocast(device_type=self.device.type):
            logits = self.model([slow, fast])
        probs = torch.softmax(logits, dim=1)[0]
        idx   = probs.argmax().item()
        return self.classes[idx], probs[idx].item()

    @torch.no_grad()
    def predict_top5(self, npy_path):
        """
        En olası 5 sınıfı döndür.
        Returns:
            list of (class_name, confidence) tuples
        """
        slow, fast = self._preprocess(npy_path)
        with torch.amp.autocast(device_type=self.device.type):
            logits = self.model([slow, fast])
        probs   = torch.softmax(logits, dim=1)[0]
        top_k   = min(5, len(self.classes))
        top5    = probs.topk(top_k)
        results = [(self.classes[i], p.item()) for i, p in zip(top5.indices, top5.values)]
        return results

    def predict_folder(self, folder_path):
        """
        Bir klasördeki tüm .npy dosyalarını tahmin et.
        Returns:
            dict: {filename: {"pred": str, "conf": float}}
        """
        files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]
        results = {}
        for i, fname in enumerate(files, 1):
            path = os.path.join(folder_path, fname)
            pred, conf = self.predict(path)
            results[fname] = {"pred": pred, "conf": round(conf, 4)}
            print(f"[{i}/{len(files)}] {fname} → {pred} ({conf*100:.1f}%)")
        return results

    def evaluate_folder(self, folder_path, true_label=None):
        """
        Klasördeki .npy dosyalarını değerlendir.
        true_label: klasörün gerçek sınıfı (biliniyorsa doğruluk hesapla)
        """
        results = self.predict_folder(folder_path)
        if true_label:
            correct = sum(1 for r in results.values() if r["pred"] == true_label)
            total   = len(results)
            print(f"\nDoğruluk: {correct}/{total} = {correct/total*100:.1f}%")
        return results


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="TID SlowFast Inference (Keypoint)")
    p.add_argument("--checkpoint", required=True,      help="best_model.pt yolu")
    p.add_argument("--npy",        default=None,        help="Tek .npy keypoint dosyası yolu")
    p.add_argument("--folder",     default=None,        help="Klasör yolu")
    p.add_argument("--top5",       action="store_true", help="Top-5 göster")
    p.add_argument("--true_label", default=None,        help="Değerlendirme için gerçek etiket")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predictor = TIDPredictor(args.checkpoint)

    if args.npy:
        if args.top5:
            results = predictor.predict_top5(args.npy)
            print(f"\nTahminler — {os.path.basename(args.npy)}:")
            for rank, (cls, conf) in enumerate(results, 1):
                bar = "█" * int(conf * 30)
                print(f"  {rank}. {cls:<30} {bar} {conf*100:.1f}%")
        else:
            pred, conf = predictor.predict(args.npy)
            print(f"\nTahmin: {pred}  ({conf*100:.1f}% güven)")

    elif args.folder:
        results = predictor.evaluate_folder(args.folder, true_label=args.true_label)
        out_path = "predictions.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSonuçlar kaydedildi: {out_path}")

    else:
        print("--npy veya --folder belirtin.")