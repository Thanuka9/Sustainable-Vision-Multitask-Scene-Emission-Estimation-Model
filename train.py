# train.py

import os
import time
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import optuna  # kept so you can re-enable tuning later if needed

from dataset import Places365WithAttributes, get_default_transforms
from model import MultiTaskResNet50

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True  # speed up convs for fixed image size

# Where TorchVision will download Places365 data:
DATA_ROOT = r"D:\datasets\torchvision_places365"

# Directory to save checkpoints & plots
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)

# ❌ We do NOT resume from old checkpoints here because the model
# architecture has changed (extra emission head).
RESUME_FROM_CHECKPOINT: Optional[str] = None

# Training config (FINAL MODEL)
BASE_BATCH_SIZE = 32
BASE_NUM_WORKERS = 4          # use multiple workers to speed up loading (lower to 0/2 if issues)
BASE_NUM_EPOCHS = 15
PATIENCE = 5                  # early stopping patience

# Loss weights
LAMBDA_ATTR = 1.3691009043636835   # from your best Optuna trial
LAMBDA_EMISSION = 1.0              # weight for carbon-emission head

# Optuna config (turned OFF for final training)
USE_OPTUNA = False
N_TRIALS = 10
TUNE_EPOCHS = 8

# Limit samples
MAX_TRAIN_SAMPLES = 300_000
MAX_VAL_SAMPLES = 30_000

# Use smaller 256x256 version of Places365
USE_SMALL_PLACES = True
# ------------------------------------------------------------------


def compute_attribute_stats(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    preds: probabilities [N, 4]
    targets: 0/1 tensor [N, 4]
    returns macro precision, recall, f1
    """
    eps = 1e-8
    preds_bin = (preds > 0.5).float()

    tp = (preds_bin * targets).sum(dim=0)
    fp = (preds_bin * (1 - targets)).sum(dim=0)
    fn = ((1 - preds_bin) * targets).sum(dim=0)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    scene_loss_fn: nn.Module,
    attr_loss_fn: nn.Module,
    emission_loss_fn: nn.Module,
    lambda_attr: float,
    lambda_emission: float,
    optimizer: torch.optim.Optimizer,
):
    model.train()
    total_loss = 0.0
    correct_scene = 0
    correct_emission = 0
    total_samples = 0

    all_attr_preds = []
    all_attr_targets = []

    for batch in loader:
        images = batch["image"].to(DEVICE, non_blocking=True)
        scene_labels = batch["scene_label"].to(DEVICE, non_blocking=True)
        attr_labels = batch["attribute_label"].to(DEVICE, non_blocking=True)
        emission_labels = batch["emission_label"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        # 3-head model
        scene_logits, attr_logits, emission_logits = model(images)

        loss_scene = scene_loss_fn(scene_logits, scene_labels)
        loss_attr = attr_loss_fn(attr_logits, attr_labels)
        loss_emission = emission_loss_fn(emission_logits, emission_labels)

        loss = loss_scene + lambda_attr * loss_attr + lambda_emission * loss_emission

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # scene accuracy
        _, pred_scene = torch.max(scene_logits, dim=1)
        correct_scene += (pred_scene == scene_labels).sum().item()

        # emission accuracy
        _, pred_emission = torch.max(emission_logits, dim=1)
        correct_emission += (pred_emission == emission_labels).sum().item()

        # attributes
        attr_probs = torch.sigmoid(attr_logits).detach().cpu()
        all_attr_preds.append(attr_probs)
        all_attr_targets.append(attr_labels.cpu())

    avg_loss = total_loss / max(1, total_samples)
    scene_acc = correct_scene / max(1, total_samples)
    emission_acc = correct_emission / max(1, total_samples)

    all_attr_preds = torch.cat(all_attr_preds, dim=0)
    all_attr_targets = torch.cat(all_attr_targets, dim=0)
    attr_stats = compute_attribute_stats(all_attr_preds, all_attr_targets)

    return avg_loss, scene_acc, emission_acc, attr_stats


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    scene_loss_fn: nn.Module,
    attr_loss_fn: nn.Module,
    emission_loss_fn: nn.Module,
    lambda_attr: float,
    lambda_emission: float,
):
    model.eval()
    total_loss = 0.0
    correct_scene = 0
    correct_emission = 0
    total_samples = 0

    all_attr_preds = []
    all_attr_targets = []

    for batch in loader:
        images = batch["image"].to(DEVICE, non_blocking=True)
        scene_labels = batch["scene_label"].to(DEVICE, non_blocking=True)
        attr_labels = batch["attribute_label"].to(DEVICE, non_blocking=True)
        emission_labels = batch["emission_label"].to(DEVICE, non_blocking=True)

        scene_logits, attr_logits, emission_logits = model(images)

        loss_scene = scene_loss_fn(scene_logits, scene_labels)
        loss_attr = attr_loss_fn(attr_logits, attr_labels)
        loss_emission = emission_loss_fn(emission_logits, emission_labels)

        loss = loss_scene + lambda_attr * loss_attr + lambda_emission * loss_emission

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        _, pred_scene = torch.max(scene_logits, dim=1)
        correct_scene += (pred_scene == scene_labels).sum().item()

        _, pred_emission = torch.max(emission_logits, dim=1)
        correct_emission += (pred_emission == emission_labels).sum().item()

        attr_probs = torch.sigmoid(attr_logits).cpu()
        all_attr_preds.append(attr_probs)
        all_attr_targets.append(attr_labels.cpu())

    avg_loss = total_loss / max(1, total_samples)
    scene_acc = correct_scene / max(1, total_samples)
    emission_acc = correct_emission / max(1, total_samples)

    all_attr_preds = torch.cat(all_attr_preds, dim=0)
    all_attr_targets = torch.cat(all_attr_targets, dim=0)
    attr_stats = compute_attribute_stats(all_attr_preds, all_attr_targets)

    return avg_loss, scene_acc, emission_acc, attr_stats


def _maybe_limit_dataset(ds, max_samples: Optional[int]):
    """
    If max_samples is set and smaller than len(ds),
    return a Subset with randomly sampled indices.
    """
    if max_samples is None or max_samples <= 0 or max_samples >= len(ds):
        return ds

    indices = torch.randperm(len(ds))[:max_samples]
    return Subset(ds, indices.tolist())


def build_dataloaders(batch_size: int, num_workers: int):
    """
    Create train/val datasets + loaders for Places365 and return them + num_scenes.
    Automatically disables downloading if data already exists.
    """
    transform_train = get_default_transforms(train=True)
    transform_val = get_default_transforms(train=False)

    def dataset_exists(root, small: bool):
        folder = "data_256_standard" if small else "data_large_standard"
        expected_dir = os.path.join(root, folder)
        return os.path.exists(expected_dir)

    skip_download = dataset_exists(DATA_ROOT, USE_SMALL_PLACES)

    train_ds = Places365WithAttributes(
        root=DATA_ROOT,
        split="train-standard",
        small=USE_SMALL_PLACES,
        download=not skip_download,
        transform=transform_train,
    )

    val_ds = Places365WithAttributes(
        root=DATA_ROOT,
        split="val",
        small=USE_SMALL_PLACES,
        download=False,
        transform=transform_val,
    )

    train_ds = _maybe_limit_dataset(train_ds, MAX_TRAIN_SAMPLES)
    val_ds = _maybe_limit_dataset(val_ds, MAX_VAL_SAMPLES)

    if MAX_TRAIN_SAMPLES:
        print(f"\nTrain size: {len(train_ds)} images (limited)")
    else:
        print(f"\nTrain size: {len(train_ds)} images")

    if MAX_VAL_SAMPLES:
        print(f"Val   size: {len(val_ds)} images (limited)")
    else:
        print(f"Val   size: {len(val_ds)} images")

    base_ds = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
    num_scenes = len(base_ds.classes)

    print(f"Number of scene classes: {num_scenes}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda"),
    )

    return train_loader, val_loader, num_scenes


def plot_curves(history: Dict[str, list], save_dir: str = SAVE_DIR, suffix: str = "") -> None:
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Multi-task Loss (scene + attributes + emissions)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"loss_curves{suffix}.png"))
    plt.close()

    # Scene accuracy
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train scene acc")
    plt.plot(epochs, history["val_acc"], label="Val scene acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Scene Classification Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"scene_accuracy{suffix}.png"))
    plt.close()

    # Emission accuracy
    if "train_emission_acc" in history:
        plt.figure()
        plt.plot(epochs, history["train_emission_acc"], label="Train emission acc")
        plt.plot(epochs, history["val_emission_acc"], label="Val emission acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Carbon Emission Level Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"emission_accuracy{suffix}.png"))
        plt.close()

    # Attribute F1
    plt.figure()
    plt.plot(epochs, history["train_attr_f1"], label="Train attr F1")
    plt.plot(epochs, history["val_attr_f1"], label="Val attr F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 score")
    plt.title("Attribute Detection F1 (macro)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"attr_f1{suffix}.png"))
    plt.close()

    print(f"\n📊 Saved training curves in: {save_dir}")


def train_model(
    lr: float,
    weight_decay: float,
    lambda_attr: float,
    lambda_emission: float,
    num_epochs: int,
    batch_size: int,
    num_workers: int,
    save_best_path: Optional[str] = None,
    record_history: bool = False,
    resume_from: Optional[str] = None,
):
    """
    Train a model with given hyperparameters.
    Returns (best_val_loss, history or None).
    """
    print(
        f"\n🔧 Training with lr={lr:.2e}, weight_decay={weight_decay:.2e}, "
        f"lambda_attr={lambda_attr:.3f}, lambda_emission={lambda_emission:.3f}, "
        f"epochs={num_epochs}, batch={batch_size}"
    )

    train_loader, val_loader, num_scenes = build_dataloaders(
        batch_size=batch_size, num_workers=num_workers
    )

    model = MultiTaskResNet50(num_scenes=num_scenes)

    if DEVICE.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"🌐 Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(DEVICE)

    scene_loss_fn = nn.CrossEntropyLoss()
    attr_loss_fn = nn.BCEWithLogitsLoss()
    emission_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Resume only if you explicitly set resume_from (currently None)
    if resume_from is not None and os.path.isfile(resume_from):
        ckpt = torch.load(resume_from, map_location=DEVICE)
        state_dict = ckpt["model_state_dict"]

        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)

        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e:
            print(f"⚠ Couldn't load optimizer state: {e}")

        print(f"  🔁 Loaded weights from {resume_from} (epoch {ckpt.get('epoch', '?')})")

    best_val_loss = float("inf")
    best_epoch = -1
    no_improve = 0

    history: Optional[Dict[str, list]] = None
    if record_history:
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_emission_acc": [],
            "val_emission_acc": [],
            "train_attr_f1": [],
            "val_attr_f1": [],
        }

    for epoch in range(1, num_epochs + 1):
        start = time.time()

        train_loss, train_acc, train_emission_acc, train_attr = train_epoch(
            model,
            train_loader,
            scene_loss_fn,
            attr_loss_fn,
            emission_loss_fn,
            lambda_attr,
            lambda_emission,
            optimizer,
        )
        val_loss, val_acc, val_emission_acc, val_attr = eval_epoch(
            model,
            val_loader,
            scene_loss_fn,
            attr_loss_fn,
            emission_loss_fn,
            lambda_attr,
            lambda_emission,
        )

        dt = time.time() - start

        if history is not None:
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["train_emission_acc"].append(train_emission_acc)
            history["val_emission_acc"].append(val_emission_acc)
            history["train_attr_f1"].append(train_attr["f1"])
            history["val_attr_f1"].append(val_attr["f1"])

        print(f"\nEpoch {epoch:02d} | {dt:.1f}s")
        print(
            f"  Train: loss={train_loss:.4f}, "
            f"scene_acc={train_acc:.3f}, "
            f"emission_acc={train_emission_acc:.3f}, "
            f"attr_f1={train_attr['f1']:.3f}, "
            f"P={train_attr['precision']:.3f}, "
            f"R={train_attr['recall']:.3f}"
        )
        print(
            f"  Val  : loss={val_loss:.4f}, "
            f"scene_acc={val_acc:.3f}, "
            f"emission_acc={val_emission_acc:.3f}, "
            f"attr_f1={val_attr['f1']:.3f}, "
            f"P={val_attr['precision']:.3f}, "
            f"R={val_attr['recall']:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            if save_best_path is not None:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "hyperparams": {
                            "lr": lr,
                            "weight_decay": weight_decay,
                            "lambda_attr": lambda_attr,
                            "lambda_emission": lambda_emission,
                            "batch_size": batch_size,
                        },
                    },
                    save_best_path,
                )
                print(f"  ✅ Saved best model to {save_best_path}")
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve} epoch(s).")
            if no_improve >= PATIENCE:
                print("  ⏹ Early stopping (no improvement).")
                break

    print(
        f"\n🏁 Finished training. Best epoch={best_epoch}, "
        f"best_val_loss={best_val_loss:.4f}"
    )

    return best_val_loss, history


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective: tune lr, weight_decay, lambda_attr.
    Emission loss weight kept fixed.
    """
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
    lambda_attr = trial.suggest_float("lambda_attr", 0.5, 2.0)

    batch_size = BASE_BATCH_SIZE
    num_workers = BASE_NUM_WORKERS

    best_val_loss, _ = train_model(
        lr=lr,
        weight_decay=weight_decay,
        lambda_attr=lambda_attr,
        lambda_emission=LAMBDA_EMISSION,
        num_epochs=TUNE_EPOCHS,
        batch_size=batch_size,
        num_workers=num_workers,
        save_best_path=None,
        record_history=False,
        resume_from=None,
    )

    return best_val_loss


def main():
    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"Data root: {DATA_ROOT}")
    print(f"Checkpoints: {SAVE_DIR}")

    if USE_OPTUNA:
        print("\n🚀 Running Optuna hyperparameter search on Places365...")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=N_TRIALS)

        print("\n🔎 Optuna results:")
        print(f"  Best value (val_loss): {study.best_value:.4f}")
        print(f"  Best params: {study.best_params}")

        best_lr = study.best_params["lr"]
        best_wd = study.best_params["weight_decay"]
        best_lambda_attr = study.best_params["lambda_attr"]

        print("\n🎯 Retraining final model with best hyperparameters...")
        save_path = os.path.join(SAVE_DIR, "best_multitask_resnet50_optuna_emission.pt")

        best_val_loss, history = train_model(
            lr=best_lr,
            weight_decay=best_wd,
            lambda_attr=best_lambda_attr,
            lambda_emission=LAMBDA_EMISSION,
            num_epochs=BASE_NUM_EPOCHS,
            batch_size=BASE_BATCH_SIZE,
            num_workers=BASE_NUM_WORKERS,
            save_best_path=save_path,
            record_history=True,
            resume_from=None,
        )

        if history is not None:
            plot_curves(history, save_dir=SAVE_DIR, suffix="_optuna_emission")

        print(f"\n✅ Final best val loss (optuna model): {best_val_loss:.4f}")
    else:
        print("\n▶ Running FINAL training with best hyperparameters (from previous Optuna run)...")

        lr = 3.177436480338081e-05
        weight_decay = 2.1989387889816935e-05
        lambda_attr = LAMBDA_ATTR

        save_path = os.path.join(SAVE_DIR, "best_multitask_resnet50_emission.pt")

        best_val_loss, history = train_model(
            lr=lr,
            weight_decay=weight_decay,
            lambda_attr=lambda_attr,
            lambda_emission=LAMBDA_EMISSION,
            num_epochs=BASE_NUM_EPOCHS,
            batch_size=BASE_BATCH_SIZE,
            num_workers=BASE_NUM_WORKERS,
            save_best_path=save_path,
            record_history=True,
            resume_from=RESUME_FROM_CHECKPOINT,  # currently None → fresh training
        )

        if history is not None:
            plot_curves(history, save_dir=SAVE_DIR, suffix="_final_emission")

        print(f"\n✅ Final best val loss (final model): {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
