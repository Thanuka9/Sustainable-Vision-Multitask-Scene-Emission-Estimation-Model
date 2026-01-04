import os
import argparse
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from dataset import get_default_transforms
from model import MultiTaskResNet50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMISSION_LABELS = ["very_low", "low", "medium", "high", "very_high"]

# Map Intel folders -> emission class index (same mapping you used)
INTEL_FOLDER_TO_EMISSION = {
    "forest": 0,
    "glacier": 0,
    "mountain": 0,
    "sea": 1,
    "buildings": 3,
    "street": 4,
}

DEFAULT_CHECKPOINTS_DIR = r"D:\sustainable_vision\checkpoints"
DEFAULT_INTEL_TEST_DIR = r"D:\datasets\Intel Image Classification Dataset\seg_test"


def strip_module_prefix(state_dict):
    """Utility to handle DataParallel state dicts by removing 'module.' prefix."""
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


class IntelEmissionTestDataset(Dataset):
    """
    Dataset class for evaluating on the Intel Image Classification dataset's test split.
    Uses folder names to derive ground-truth emission labels.
    """
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for cls_name in os.listdir(root_dir):
            cls_path = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_path):
                continue

            if cls_name not in INTEL_FOLDER_TO_EMISSION:
                continue

            y = INTEL_FOLDER_TO_EMISSION[cls_name]
            for fname in os.listdir(cls_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    self.samples.append((os.path.join(cls_path, fname), y))

        if not self.samples:
            raise RuntimeError(f"No images found under: {root_dir}")

        print(f"[Dataset] Loaded {len(self.samples)} test images from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(y, dtype=torch.long)


@torch.no_grad()
def eval_checkpoint(ckpt_path: str, num_scenes: int, loader: DataLoader):
    """
    Evaluates a single checkpoint on the Intel test dataset.
    Returns accuracy stats or None if the model is incompatible.
    """
    model = MultiTaskResNet50(num_scenes=num_scenes).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state_dict = ckpt["model_state_dict"]
    state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    total = 0
    correct = 0
    pred_counter = Counter()

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        out = model(x)

        if not isinstance(out, tuple) or len(out) < 3:
            # model without emission head can't be compared here
            return None

        _, _, emission_logits = out
        preds = emission_logits.argmax(dim=1)

        total += y.size(0)
        correct += (preds == y).sum().item()

        pred_counter.update(preds.detach().cpu().tolist())

    acc = correct / max(1, total)
    return {
        "acc": acc,
        "total": total,
        "pred_dist": pred_counter,
        "epoch": ckpt.get("epoch", None),
        "val_loss": ckpt.get("val_loss", None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intel-test", type=str, default=DEFAULT_INTEL_TEST_DIR)
    ap.add_argument("--checkpoints-dir", type=str, default=DEFAULT_CHECKPOINTS_DIR)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="Limit test images (0 = no limit)")
    args = ap.parse_args()

    # Build dataset/loader
    tfm = get_default_transforms(train=False)
    ds = IntelEmissionTestDataset(args.intel_test, transform=tfm)

    if args.limit and args.limit > 0:
        ds.samples = ds.samples[: args.limit]
        print(f"[Dataset] Limited to {len(ds.samples)} images")

    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)

    # Determine num_scenes from your training setting
    # Your model expects Places365 num classes = 365 (as you loaded)
    num_scenes = 365

    ckpts = []
    for f in os.listdir(args.checkpoints_dir):
        if f.lower().endswith(".pt"):
            ckpts.append(os.path.join(args.checkpoints_dir, f))

    if not ckpts:
        raise RuntimeError(f"No .pt checkpoints found in {args.checkpoints_dir}")

    results = []
    print("\n=== Evaluating checkpoints on Intel seg_test (emission accuracy) ===\n")

    for ckpt_path in sorted(ckpts):
        name = os.path.basename(ckpt_path)
        print(f"-> {name}")
        r = eval_checkpoint(ckpt_path, num_scenes, loader)

        if r is None:
            print("   (Skipped: checkpoint has no emission head output)\n")
            continue

        results.append((name, r["acc"], r["total"], r["epoch"], r["val_loss"]))
        print(f"   emission_acc = {r['acc']*100:.2f}%  (n={r['total']})\n")

    if not results:
        print("No comparable checkpoints found (none produced emission logits).")
        return

    # Sort results by accuracy (highest first)
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n=== Leaderboard (best -> worst) ===")
    for i, (name, acc, total, epoch, vloss) in enumerate(results, start=1):
        print(f"{i:02d}. {name:40s}  acc={acc*100:6.2f}%  n={total}  epoch={epoch}  val_loss={vloss}")

    best = results[0]
    print("\n✅ BEST CHECKPOINT:")
    print(f"   {best[0]}  (emission_acc={best[1]*100:.2f}%)")


if __name__ == "__main__":
    main()
