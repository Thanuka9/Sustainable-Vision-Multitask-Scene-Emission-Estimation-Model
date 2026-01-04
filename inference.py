# inference.py

import os
import argparse
from io import BytesIO
import random
from urllib.parse import urlparse

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
import requests

from dataset import Places365WithAttributes, get_default_transforms
from model import MultiTaskResNet50

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset root (used only for loading class names initially)
DATA_ROOT = r"D:\datasets\torchvision_places365"

# Model Checkpoint Paths
BASE_CKPT_PATH = os.path.join("checkpoints", "best_multitask_resnet50_emission.pt")
INTEL_CKPT_PATH = os.path.join("checkpoints", "best_multitask_resnet50_emission_intel.pt")

USE_SMALL_PLACES = True
TOP_K = 5 # Number of top predictions to display for scene classification

# Output Labels for Emission Head
EMISSION_LABELS = ["very_low", "low", "medium", "high", "very_high"]

# Default path for Intel dataset (for evaluation/finetuning)
DEFAULT_INTEL_ROOT = r"D:\datasets\Intel Image Classification Dataset"

# Intel Dataset Class Mapping (folder name -> emission level)
# This mapping allows us to use the Intel dataset for ground-truth emission evaluation.
INTEL_FOLDER_TO_EMISSION = {
    "forest": 0,
    "glacier": 0,
    "mountain": 0,
    "sea": 1,
    "buildings": 3,
    "street": 4,
}

# HTTP Headers for image downloading
# A friendly UA helps avoid 403 blocks from many sites
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}

# ----------------------------------------


def load_classes():
    """
    Loads the list of scene class names from the Places365 dataset.
    This is needed to map model output indices to human-readable names.
    """
    ds = Places365WithAttributes(
        root=DATA_ROOT,
        split="train-standard",
        small=USE_SMALL_PLACES,
        download=False,
        transform=get_default_transforms(train=False),
    )
    classes = ds.classes
    print(f"Loaded {len(classes)} Places365 classes.")
    return classes


def load_model(num_scenes: int, ckpt_path: str):
    """
    Loads the MultiTaskResNet50 model from a checkpoint.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading model from: {ckpt_path}")
    model = MultiTaskResNet50(num_scenes=num_scenes)

    # Load state dict
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state_dict = ckpt["model_state_dict"]

    # Fix for DataParallel models (remove 'module.' prefix if present)
    if any(k.startswith("module.") for k in state_dict.keys()):
        from collections import OrderedDict
        new_sd = OrderedDict()
        for k, v in state_dict.items():
            new_sd[k.replace("module.", "", 1)] = v
        state_dict = new_sd

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval() # Set to evaluation mode

    print(
        f"  Loaded checkpoint at epoch {ckpt.get('epoch', 'N/A')}, "
        f"val_loss={ckpt.get('val_loss', 'N/A')}"
    )
    return model


def _is_url(s: str) -> bool:
    """Checks if a string looks like a URL."""
    return s.startswith("http://") or s.startswith("https://")


def _download_image_bytes(url: str, timeout: int = 20) -> bytes:
    """
    Robust image downloader:
      - Adds User-Agent
      - Follows redirects
      - Validates Content-Type is image/*
      - Gives clear errors when URL is HTML, blocked, etc.
    """
    session = requests.Session()
    try:
        resp = session.get(
            url,
            headers=DEFAULT_HEADERS,
            timeout=timeout,
            allow_redirects=True,
            stream=True,
        )
        resp.raise_for_status()

        ctype = (resp.headers.get("Content-Type") or "").lower()
        if not ctype.startswith("image/"):
            # Many “image” links return HTML pages. Show a helpful snippet.
            try:
                text_snippet = resp.text[:250].replace("\n", " ")
            except Exception:
                text_snippet = "<unable to read response text>"
            raise RuntimeError(
                f"URL did not return an image.\n"
                f"  URL: {url}\n"
                f"  Content-Type: {ctype}\n"
                f"  Response starts with: {text_snippet}"
            )

        return resp.content

    except requests.exceptions.HTTPError as e:
        # Common: 403 for missing UA (we added UA), or hotlink protection.
        raise RuntimeError(f"Failed to download image (HTTP error): {e}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to download image (request error): {e}")
    finally:
        session.close()


def preprocess_image(img_path: str) -> torch.Tensor:
    """
    Load and transform an image into a tensor ready for the model.
    Supports:
      - Local filesystem paths
      - HTTP/HTTPS direct image URLs
    """
    transform = get_default_transforms(train=False)

    if _is_url(img_path):
        print(f"Downloading image from URL: {img_path}")
        img_bytes = _download_image_bytes(img_path)

        try:
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
        except UnidentifiedImageError:
            raise RuntimeError(
                "Downloaded content could not be decoded as an image. "
                "This usually means the URL returned HTML or blocked content. "
                "Use a direct .jpg/.png/.webp link or download the file locally."
            )

    else:
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert("RGB")

    tensor = transform(img).unsqueeze(0) # Add batch dimension
    return tensor


@torch.no_grad()
def _run_on_single_image(img_path: str, model: torch.nn.Module, classes, top_k: int = TOP_K):
    """
    Runs inference on a single image and prints the results.
    """
    x = preprocess_image(img_path).to(DEVICE)
    out = model(x)

    # Unpack outputs based on available heads
    if isinstance(out, tuple):
        if len(out) == 3:
            scene_logits, attr_logits, emission_logits = out
        elif len(out) == 2:
            scene_logits, attr_logits = out
            emission_logits = None
        else:
            raise RuntimeError(f"Unexpected model output tuple length: {len(out)}")
    else:
        raise RuntimeError(f"Unexpected model output type: {type(out)}")

    # 1. Process Scene Predictions
    scene_probs = torch.softmax(scene_logits[0], dim=0)
    top_probs, top_idxs = torch.topk(scene_probs, k=top_k)

    # 2. Process Attribute Predictions
    attr_probs = torch.sigmoid(attr_logits[0]).cpu().tolist()

    # Print Results
    print("\n=== Scene prediction (top-{}) ===".format(top_k))
    for rank, (p, idx) in enumerate(zip(top_probs.cpu().tolist(), top_idxs.cpu().tolist()), start=1):
        print(f"{rank}. {classes[idx]}  ({p*100:.2f}%)")

    print("\n=== Attribute probabilities ===")
    for i, p in enumerate(attr_probs):
        print(f"attr_{i}: {p:.3f}")

    # 3. Process Emission Predictions
    if emission_logits is not None:
        emission_probs = torch.softmax(emission_logits[0], dim=0)
        best_idx = int(torch.argmax(emission_probs).item())
        best_p = float(emission_probs[best_idx].item())

        label = EMISSION_LABELS[best_idx] if 0 <= best_idx < len(EMISSION_LABELS) else f"class_{best_idx}"

        print("\n=== Estimated carbon emission level ===")
        print(f"Predicted: {label}  ({best_p*100:.2f}%)")
        print("Full distribution:")
        for i, p in enumerate(emission_probs.cpu().tolist()):
            name = EMISSION_LABELS[i] if i < len(EMISSION_LABELS) else f"class_{i}"
            print(f"  {i}: {name:>10s}  {p*100:.2f}%")
    else:
        print("\n(No emission head found in this checkpoint — only scenes + attributes.)")


# ---------------------------------------------------------------------
# Generic folder evaluation (works for ANY dataset folder)
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate_on_folder(folder: str, model: torch.nn.Module, classes, top_k: int = TOP_K, num_images: int = 20):
    """
    Runs qualitative evaluation on a folder of images.
    Randomly selects 'num_images' and runs inference on them.
    """
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    all_paths = []
    for dirpath, _, filenames in os.walk(folder):
        for fname in filenames:
            if fname.lower().endswith(exts):
                all_paths.append(os.path.join(dirpath, fname))

    if not all_paths:
        print(f"No images found in folder: {folder}")
        return

    random.shuffle(all_paths)
    sample_paths = all_paths[: min(num_images, len(all_paths))]

    print(f"\nFound {len(all_paths)} images in '{folder}'. Running on {len(sample_paths)} sample(s)...")

    for idx, img_path in enumerate(sample_paths, start=1):
        print("\n" + "-" * 80)
        print(f"[{idx}/{len(sample_paths)}] {img_path}")
        print("-" * 80)
        _run_on_single_image(img_path, model, classes, top_k=top_k)


# ---------------------------------------------------------------------
# Intel dataset helpers (unchanged)
# ---------------------------------------------------------------------
def resolve_intel_dirs(root: str):
    """
    Locates the 'seg_train' and 'seg_test' folders within the Intel dataset root.
    Handles potential nested directory structures.
    """
    train_base = os.path.join(root, "seg_train")
    alt_train = os.path.join(train_base, "seg_train")
    if os.path.isdir(alt_train):
        train_dir = alt_train
    elif os.path.isdir(train_base):
        train_dir = train_base
    else:
        raise FileNotFoundError(
            f"Could not find Intel train folder under: {root} "
            "(expected 'seg_train' or 'seg_train/seg_train')."
        )

    test_base = os.path.join(root, "seg_test")
    alt_test = os.path.join(test_base, "seg_test")
    if os.path.isdir(alt_test):
        test_dir = alt_test
    elif os.path.isdir(test_base):
        test_dir = test_base
    else:
        test_dir = None

    return train_dir, test_dir


class IntelEmissionDataset(Dataset):
    """
    Custom Dataset for the Intel Image Classification dataset.
    Used for fine-tuning the emission head.
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
                print(f"[IntelEmissionDataset] Skipping folder '{cls_name}' (no emission mapping defined).")
                continue

            emm_label = INTEL_FOLDER_TO_EMISSION[cls_name]
            for fname in os.listdir(cls_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    self.samples.append((os.path.join(cls_path, fname), emm_label))

        if not self.samples:
            raise RuntimeError(f"No usable images found in Intel train folder: {root_dir}")

        print(f"[IntelEmissionDataset] Collected {len(self.samples)} images from: {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, emm_label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(emm_label, dtype=torch.long)


def finetune_emission_on_intel(train_dir: str, model: torch.nn.Module, epochs: int = 1, batch_size: int = 32, lr: float = 1e-4):
    """
    Fine-tunes ONLY the carbon emission head on the Intel dataset.
    The backbone and other heads are frozen.
    """
    print(
        f"\n🔧 Fine-tuning emission head on Intel data:\n  train_dir = {train_dir}\n"
        f"  epochs={epochs}, batch_size={batch_size}, lr={lr:.1e}"
    )

    # Freeze all parameters first
    for p in model.parameters():
        p.requires_grad = False

    if not hasattr(model, "emission_head"):
        print("⚠ Model has no 'emission_head' attribute; skipping fine-tune.")
        return

    # Unfreeze only the emission head
    for p in model.emission_head.parameters():
        p.requires_grad = True

    transform = get_default_transforms(train=True)
    ds = IntelEmissionDataset(train_dir, transform=transform)

    # Windows-safe default: allow override with env var if you want
    num_workers = int(os.getenv("SV_NUM_WORKERS", "0"))

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        for images, emm_labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            emm_labels = emm_labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            out = model(images)
            if not isinstance(out, tuple) or len(out) < 3:
                raise RuntimeError("Model output does not contain emission head while fine-tuning.")

            _, _, emission_logits = out
            loss = criterion(emission_logits, emm_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total += images.size(0)

            preds = emission_logits.argmax(dim=1)
            correct += (preds == emm_labels).sum().item()

        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)
        print(f"  [Intel FT] Epoch {ep:02d} | loss={avg_loss:.4f}, emission_acc={acc:.3f}")

    model.eval()

    # Save the fine-tuned model
    os.makedirs(os.path.dirname(INTEL_CKPT_PATH), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epochs,
            "val_loss": None,
            "note": "Fine-tuned emission head on Intel dataset",
        },
        INTEL_CKPT_PATH,
    )
    print(f"\n✅ Saved fine-tuned model to: {INTEL_CKPT_PATH}")


# ---------------------------------------------------------------------
# Arg parsing & entrypoint
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run inference with multitask Places365 model.\n"
            "Mode 1: Single image/URL (-i/--image)\n"
            "Mode 2: Qualitative test on Intel dataset (--eval-intel)\n"
            "Mode 3: Fine-tune emission head on Intel then test (--finetune-intel)\n"
            "Mode 4: Qualitative test on ANY folder (--eval-folder)"
        )
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", "-i", type=str, help="Path to input image OR a direct HTTP/HTTPS image URL.")
    group.add_argument("--eval-intel", action="store_true", help="Run qualitative evaluation on Intel dataset.")
    group.add_argument("--finetune-intel", action="store_true", help="Fine-tune emission head on Intel then evaluate.")
    group.add_argument("--eval-folder", type=str, help="Evaluate any folder of images (recursive).")

    parser.add_argument("--use-intel-ckpt", action="store_true", help="Use Intel-finetuned checkpoint instead of base.")
    parser.add_argument("--topk", type=int, default=TOP_K, help="How many top scene predictions to show.")
    parser.add_argument("--intel-root", type=str, default=DEFAULT_INTEL_ROOT, help="Root folder of Intel dataset.")
    parser.add_argument("--num-images", type=int, default=20, help="Number of images to sample for eval.")
    parser.add_argument("--ft-epochs", type=int, default=1, help="Epochs for --finetune-intel.")
    parser.add_argument("--ft-batch", type=int, default=32, help="Batch size for --finetune-intel.")
    parser.add_argument("--ft-lr", type=float, default=1e-4, help="Learning rate for --finetune-intel.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 1. Load class names
    classes = load_classes()
    num_scenes = len(classes)

    # 2. Determine which checkpoint to load
    if args.finetune_intel:
        ckpt_path = BASE_CKPT_PATH
        print(f"📘 Using BASE Places365 checkpoint for fine-tuning: {ckpt_path}")
    else:
        if args.use_intel_ckpt:
            ckpt_path = INTEL_CKPT_PATH
            print(f"🔄 Using INTEL fine-tuned checkpoint: {ckpt_path}")
        else:
            ckpt_path = BASE_CKPT_PATH
            print(f"📘 Using BASE Places365 checkpoint: {ckpt_path}")

    model = load_model(num_scenes=num_scenes, ckpt_path=ckpt_path)

    # 3. Execute requested mode
    if args.image:
        _run_on_single_image(args.image, model, classes, top_k=args.topk)

    elif args.eval_folder:
        evaluate_on_folder(
            folder=args.eval_folder,
            model=model,
            classes=classes,
            top_k=args.topk,
            num_images=args.num_images,
        )

    else:
        # Intel Dataset operations
        train_dir, test_dir = resolve_intel_dirs(args.intel_root)

        if args.finetune_intel:
            finetune_emission_on_intel(
                train_dir=train_dir,
                model=model,
                epochs=args.ft_epochs,
                batch_size=args.ft_batch,
                lr=args.ft_lr,
            )

        eval_dir = test_dir if test_dir is not None else train_dir
        evaluate_on_folder(
            folder=eval_dir,
            model=model,
            classes=classes,
            top_k=args.topk,
            num_images=args.num_images,
        )
