# dual_inference_portable.py
# Portable dual-model inference (BASE + INTEL) with AUTO selection
# ✅ No Places365 dataset root required
# ✅ Runs on a single image OR a folder of images
# ✅ Auto-downloads Places365 class list (365) if missing/wrong
# ✅ Outputs: both models (optional) + chosen model + emission + scene-type + CO2 estimate

import os
import argparse
import random
from io import BytesIO
from typing import List, Dict, Tuple

import torch
from PIL import Image, UnidentifiedImageError

try:
    import requests
except ImportError:
    requests = None

from dataset import get_default_transforms
from model import MultiTaskResNet50


# -----------------------------
# CONFIG
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_CKPT = r"checkpoints\best_multitask_resnet50_emission.pt"
INTEL_CKPT = r"checkpoints\best_multitask_resnet50_emission_intel.pt"

ASSETS_CLASSES_FILE = r"assets\categories_places365.txt"
PLACES365_CLASSES_URL = "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt"

EXPECTED_NUM_SCENES = 365
EMISSION_LABELS = ["very_low", "low", "medium", "high", "very_high"]
TOPK_DEFAULT = 5


# -----------------------------
# STANDARD-BASED REFERENCES
# -----------------------------
# EPA passenger car tank-to-wheel CO2 factor: 0.313 kg / vehicle-mile (Sep 2023)
EPA_PASSENGER_CAR_G_PER_KM = (0.313 * 1000.0) / 1.609344  # ~194.5 g/km

DEFAULT_GRID_G_PER_KWH = 445.0
DEFAULT_BUILDING_EUI_KWH_PER_M2_YR = 120.0

LABEL_MULTIPLIER = {
    "very_low": 0.3,
    "low": 0.6,
    "medium": 1.0,
    "high": 1.5,
    "very_high": 2.1,
}

TRANSPORT_KW = ["street", "highway", "downtown", "crosswalk", "parking", "intersection", "road", "alley", "bridge", "traffic"]
BUILT_KW = ["building", "apartment", "office", "house", "hotel", "skyscraper", "mall", "library", "construction", "tower", "city", "residential"]
INDUSTRIAL_KW = ["industrial", "factory", "power_plant", "refinery", "warehouse", "plant", "oilrig", "smokestack", "chimney"]
NATURE_KW = ["forest", "rainforest", "mountain", "glacier", "ice", "beach", "coast", "ocean", "lake", "river", "valley", "field", "desert", "park"]


# -----------------------------
# UTIL
# -----------------------------
def ensure_requests():
    if requests is None:
        raise RuntimeError("requests not installed. Run: pip install requests")


def ensure_places365_classes_file(path: str) -> None:
    """
    Ensures the official Places365 categories file exists locally and has 365 labels.
    Auto-downloads if missing or wrong length.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def count_nonempty_lines(p: str) -> int:
        if not os.path.isfile(p):
            return 0
        with open(p, "r", encoding="utf-8") as f:
            return len([ln for ln in f.read().splitlines() if ln.strip()])

    n = count_nonempty_lines(path)
    if n == EXPECTED_NUM_SCENES:
        return

    ensure_requests()
    print(f"⚠️ Places365 class list missing/wrong ({n} lines). Downloading official list...")
    resp = requests.get(PLACES365_CLASSES_URL, timeout=30)
    resp.raise_for_status()

    with open(path, "w", encoding="utf-8") as f:
        f.write(resp.text)

    n2 = count_nonempty_lines(path)
    if n2 != EXPECTED_NUM_SCENES:
        raise RuntimeError(
            f"Downloaded classes file but got {n2} lines (expected {EXPECTED_NUM_SCENES}).\n"
            f"Check file: {path}"
        )

    print(f"✅ Saved Places365 classes to: {path} ({n2} labels)")


def load_classes_from_file(path: str) -> List[str]:
    """
    Loads Places365 class labels from file.
    Supports:
      - "/a/abbey"
      - "0 /a/abbey"
    Returns exactly 365 labels (pads/truncates to be safe).
    """
    classes: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                classes.append(parts[1])
            else:
                classes.append(line)

    if len(classes) < EXPECTED_NUM_SCENES:
        classes += [f"class_{i}" for i in range(len(classes), EXPECTED_NUM_SCENES)]
    elif len(classes) > EXPECTED_NUM_SCENES:
        classes = classes[:EXPECTED_NUM_SCENES]

    return classes


def strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def load_model(num_scenes: int, ckpt_path: str) -> torch.nn.Module:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = MultiTaskResNet50(num_scenes=num_scenes).to(DEVICE)

    # safer load if supported
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)

    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    state_dict = strip_module_prefix(state_dict)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _download_image_bytes(url: str) -> bytes:
    ensure_requests()
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "image/*,*/*;q=0.8"}
    resp = requests.get(url, timeout=30, headers=headers, stream=True)
    resp.raise_for_status()

    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "image" not in ctype:
        raise RuntimeError(f"URL is not an image (Content-Type={ctype}). Use a direct image URL.")
    return resp.content


def preprocess_image(img_input: str) -> torch.Tensor:
    transform = get_default_transforms(train=False)

    if img_input.startswith("http://") or img_input.startswith("https://"):
        raw = _download_image_bytes(img_input)
        try:
            img = Image.open(BytesIO(raw)).convert("RGB")
        except UnidentifiedImageError:
            raise RuntimeError("Downloaded content is not a valid image.")
    else:
        if not os.path.isfile(img_input):
            raise FileNotFoundError(f"Image not found: {img_input}")
        img = Image.open(img_input).convert("RGB")

    return transform(img).unsqueeze(0)


def collect_images(folder: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    out: List[str] = []
    for dp, _, fns in os.walk(folder):
        for fn in fns:
            if fn.lower().endswith(exts):
                out.append(os.path.join(dp, fn))
    return out


def is_intel_domain(path: str) -> bool:
    s = path.lower()
    if "intel image classification dataset" in s:
        return True
    if "\\seg_test\\" in s or "\\seg_train\\" in s:
        return True
    intel_folders = ["\\forest\\", "\\glacier\\", "\\mountain\\", "\\sea\\", "\\street\\", "\\buildings\\"]
    return any(tok in s for tok in intel_folders)


@torch.no_grad()
def predict(model: torch.nn.Module, x: torch.Tensor, classes: List[str], topk: int) -> Dict:
    x = x.to(DEVICE)
    scene_logits, attr_logits, emission_logits = model(x)

    eprob = torch.softmax(emission_logits[0], dim=0)
    top2 = torch.topk(eprob, k=2)
    best_idx = int(top2.indices[0].item())
    best_prob = float(top2.values[0].item())
    second_prob = float(top2.values[1].item())
    gap = best_prob - second_prob

    label = EMISSION_LABELS[best_idx] if best_idx < len(EMISSION_LABELS) else f"class_{best_idx}"

    sprob = torch.softmax(scene_logits[0], dim=0)
    sp, si = torch.topk(sprob, k=topk)
    scenes = [(classes[int(i)], float(p)) for p, i in zip(sp, si)]

    return {"label": label, "prob": best_prob, "gap": gap, "scenes": scenes}


def weighted_scene_type(scenes: List[Tuple[str, float]]) -> str:
    score = {"transport": 0.0, "built": 0.0, "industrial": 0.0, "nature": 0.0}

    for cls, p in scenes:
        name = cls.lower()
        if any(k in name for k in INDUSTRIAL_KW):
            score["industrial"] += p
        if any(k in name for k in TRANSPORT_KW):
            score["transport"] += p
        if any(k in name for k in BUILT_KW):
            score["built"] += p
        if any(k in name for k in NATURE_KW):
            score["nature"] += p

    if score["industrial"] > 0.10:
        return "industrial"

    best = max(score.items(), key=lambda kv: kv[1])[0]
    return best if score[best] > 0.05 else "unknown"


def estimate_co2(out: Dict, grid_g_per_kwh: float, building_eui: float) -> Dict:
    stype = weighted_scene_type(out["scenes"])
    mult = LABEL_MULTIPLIER.get(out["label"], 1.0)

    if stype == "transport":
        return {
            "scene_type": stype,
            "unit": "gCO2/km",
            "value": EPA_PASSENGER_CAR_G_PER_KM * mult,
            "note": "EPA passenger car TTW CO2 per mile converted to per km, scaled by emission label."
        }

    base = (building_eui * grid_g_per_kwh) / 1000.0
    if stype == "industrial":
        base *= 1.4

    return {
        "scene_type": stype,
        "unit": "kgCO2e/m²/year",
        "value": base * mult,
        "note": "Operational proxy (EUI*grid intensity)/1000 scaled by emission label (industrial uplift fixed 1.4x)."
    }


def model_score(out: Dict, base_bias: float = 0.0) -> float:
    return (out["prob"] * 1.0) + (out["gap"] * 0.25) + base_bias


def choose_best(img_path: str, base_out: Dict, intel_out: Dict, prefer_base: bool = True) -> Tuple[str, Dict]:
    # If it's truly Intel dataset path, pick Intel model
    if is_intel_domain(img_path):
        return "INTEL_FINETUNED", intel_out

    base_type = weighted_scene_type(base_out["scenes"])
    intel_type = weighted_scene_type(intel_out["scenes"])
    agree = (base_type == intel_type) or (base_out["label"] == intel_out["label"])

    s_base = model_score(base_out, base_bias=(0.05 if prefer_base else 0.0))
    s_intel = model_score(intel_out)

    # switch to Intel only if clearly better AND safe agreement
    if agree and (s_intel > s_base + 0.12) and (intel_out["prob"] >= 0.80):
        return "INTEL_FINETUNED", intel_out

    return "BASE_PLACES", base_out


def print_model_block(name: str, out: Dict, topk: int):
    print(f"\n--- {name} ---")
    print(f"Emission: {out['label']} ({out['prob']*100:.2f}%) | gap={out['gap']*100:.2f}%")
    print(f"Scene-type: {weighted_scene_type(out['scenes'])}")
    print(f"Top-{topk} scenes:")
    for i, (cls, p) in enumerate(out["scenes"], start=1):
        print(f"  {i}. {cls} ({p*100:.2f}%)")


def print_final(img_path: str, chosen_name: str, chosen_out: Dict, co2: Dict, topk: int):
    print("\n==============================")
    print("✅ FINAL OUTPUT")
    print("==============================")
    print(f"Image: {img_path}")
    print(f"Chosen model: {chosen_name}")
    print(f"Emission: {chosen_out['label']} ({chosen_out['prob']*100:.2f}%) | gap={chosen_out['gap']*100:.2f}%")
    print(f"Scene-type: {co2['scene_type']}")
    print(f"CO2 estimate: {co2['value']:.2f} {co2['unit']}")
    print(f"Note: {co2['note']}")
    print(f"\nTop-{topk} scenes (chosen model):")
    for i, (cls, p) in enumerate(chosen_out["scenes"], start=1):
        print(f"  {i}. {cls} ({p*100:.2f}%)")


def main():
    ap = argparse.ArgumentParser("dual_inference_portable.py (no dataset root)")

    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", "-i", help="Single image path or direct image URL.")
    group.add_argument("--folder", help="Folder of images (recursive).")

    ap.add_argument("--topk", type=int, default=TOPK_DEFAULT)
    ap.add_argument("--num-images", type=int, default=30, help="Folder mode sample size.")
    ap.add_argument("--classes-file", default=ASSETS_CLASSES_FILE, help="Places365 categories file path.")
    ap.add_argument("--grid-gco2-kwh", type=float, default=DEFAULT_GRID_G_PER_KWH)
    ap.add_argument("--building-eui", type=float, default=DEFAULT_BUILDING_EUI_KWH_PER_M2_YR)
    ap.add_argument("--prefer-intel", action="store_true", help="Disable base preference in auto selection.")
    ap.add_argument("--final-only", action="store_true", help="Only print final output (no per-model blocks).")

    args = ap.parse_args()

    # Ensure correct 365-label class file (auto-download if missing/wrong)
    ensure_places365_classes_file(args.classes_file)
    classes = load_classes_from_file(args.classes_file)

    # IMPORTANT: checkpoints are trained with 365 scenes -> must be 365
    num_scenes = EXPECTED_NUM_SCENES

    print(f"\nLoading BASE:  {BASE_CKPT}")
    base_model = load_model(num_scenes, BASE_CKPT)

    print(f"Loading INTEL: {INTEL_CKPT}")
    intel_model = load_model(num_scenes, INTEL_CKPT)

    def run_one(path: str):
        x = preprocess_image(path)

        base_out = predict(base_model, x, classes, topk=args.topk)
        intel_out = predict(intel_model, x, classes, topk=args.topk)

        if not args.final_only:
            print_model_block("BASE_PLACES", base_out, args.topk)
            print_model_block("INTEL_FINETUNED", intel_out, args.topk)

        chosen_name, chosen_out = choose_best(
            path, base_out, intel_out, prefer_base=(not args.prefer_intel)
        )
        co2 = estimate_co2(chosen_out, args.grid_gco2_kwh, args.building_eui)

        print_final(path, chosen_name, chosen_out, co2, args.topk)

    if args.image:
        run_one(args.image)

    if args.folder:
        imgs = collect_images(args.folder)
        if not imgs:
            raise RuntimeError(f"No images found in: {args.folder}")
        random.shuffle(imgs)
        sample = imgs[:min(args.num_images, len(imgs))]
        print(f"\nFound {len(imgs)} images. Running {len(sample)} sample(s)...")

        for p in sample:
            try:
                run_one(p)
            except Exception as e:
                print(f"\n[SKIP] {p}\n  Reason: {e}")


if __name__ == "__main__":
    main()
