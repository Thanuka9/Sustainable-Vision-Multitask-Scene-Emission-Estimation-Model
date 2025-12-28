import os
import argparse
from io import BytesIO

import torch
from PIL import Image, UnidentifiedImageError
import requests

from dataset import Places365WithAttributes, get_default_transforms
from model import MultiTaskResNet50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = r"D:\datasets\torchvision_places365"
USE_SMALL_PLACES = True

BASE_CKPT = r"checkpoints\best_multitask_resnet50_emission.pt"
INTEL_CKPT = r"checkpoints\best_multitask_resnet50_emission_intel.pt"

EMISSION_LABELS = ["very_low", "low", "medium", "high", "very_high"]
TOPK_DEFAULT = 5


# -----------------------------
# STANDARD-BASED REFERENCES
# -----------------------------
# EPA Emission Factors Hub (Sep 2023):
# Passenger Car CO2 factor: 0.313 kg / vehicle-mile (combustion, tank-to-wheel)
# We'll use this as a transparent baseline for "transport intensity".
EPA_PASSENGER_CAR_KG_PER_MILE = 0.313
EPA_PASSENGER_CAR_G_PER_MILE = EPA_PASSENGER_CAR_KG_PER_MILE * 1000.0  # 313 g/mile
MILE_TO_KM = 1.609344
EPA_PASSENGER_CAR_G_PER_KM = EPA_PASSENGER_CAR_G_PER_MILE / MILE_TO_KM  # ~194.5 g/km

# Grid intensity is location-specific; keep it configurable.
DEFAULT_GRID_G_PER_KWH = 445.0

# Building EUI is context-specific; keep it configurable.
DEFAULT_BUILDING_EUI_KWH_PER_M2_YR = 120.0

LABEL_MULTIPLIER = {
    "very_low": 0.3,
    "low": 0.6,
    "medium": 1.0,
    "high": 1.5,
    "very_high": 2.1,
}

# Keywords for weighted scene typing
TRANSPORT_KW = [
    "street", "highway", "downtown", "crosswalk", "parking", "intersection", "road", "alley", "bridge"
]
BUILT_KW = [
    "building", "apartment", "office", "house", "hotel", "skyscraper", "mall", "library", "construction", "tower"
]
INDUSTRIAL_KW = [
    "industrial", "factory", "power_plant", "refinery", "warehouse", "plant"
]
NATURE_KW = [
    "forest", "rainforest", "mountain", "glacier", "ice", "beach", "coast", "ocean", "lake", "river", "valley", "field", "desert"
]


def load_classes():
    ds = Places365WithAttributes(
        root=DATA_ROOT,
        split="train-standard",
        small=USE_SMALL_PLACES,
        download=False,
        transform=get_default_transforms(train=False),
    )
    print(f"Loaded {len(ds.classes)} Places365 classes.")
    return ds.classes


def strip_module_prefix(state_dict):
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def load_model(num_scenes, ckpt_path):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = MultiTaskResNet50(num_scenes=num_scenes).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    # handle both {"model_state_dict": ...} and raw state_dict
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    state_dict = strip_module_prefix(state_dict)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _download_image_bytes(url: str) -> bytes:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    resp = requests.get(url, timeout=30, headers=headers, stream=True)
    resp.raise_for_status()

    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "image" not in ctype:
        raise RuntimeError(
            f"URL did not return an image (Content-Type={ctype}). "
            "Use a DIRECT image URL to the image file, not a page."
        )
    return resp.content


def preprocess_image(img_input: str):
    transform = get_default_transforms(train=False)

    if img_input.startswith("http://") or img_input.startswith("https://"):
        print(f"Downloading image from URL: {img_input}")
        try:
            raw = _download_image_bytes(img_input)
            img = Image.open(BytesIO(raw)).convert("RGB")
        except UnidentifiedImageError:
            raise RuntimeError("Downloaded content is not a valid image.")
    else:
        if not os.path.isfile(img_input):
            raise FileNotFoundError(f"Image not found: {img_input}")
        img = Image.open(img_input).convert("RGB")

    return transform(img).unsqueeze(0)


def is_intel_domain(img_input: str) -> bool:
    s = img_input.lower()
    if "intel image classification dataset" in s:
        return True
    if "\\seg_test\\" in s or "\\seg_train\\" in s:
        return True
    intel_folders = ["\\forest\\", "\\glacier\\", "\\mountain\\", "\\sea\\", "\\street\\", "\\buildings\\"]
    return any(tok in s for tok in intel_folders)


@torch.no_grad()
def predict(model, x, classes, topk=TOPK_DEFAULT):
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

    return {
        "label": label,
        "prob": best_prob,
        "gap": gap,
        "scenes": scenes,
        "full_emission": eprob.detach().cpu().tolist(),
    }


def weighted_scene_type(scenes):
    """
    Uses top-k weighted votes to reduce errors like 'mosque/outdoor' dominating.
    """
    score = {"transport": 0.0, "built": 0.0, "industrial": 0.0, "nature": 0.0}

    for cls, p in scenes:
        name = cls.lower()
        # weight by probability
        if any(k in name for k in INDUSTRIAL_KW):
            score["industrial"] += p
        if any(k in name for k in TRANSPORT_KW):
            score["transport"] += p
        if any(k in name for k in BUILT_KW):
            score["built"] += p
        if any(k in name for k in NATURE_KW):
            score["nature"] += p

    # If industrial is present, treat as built+industrial (but label industrial)
    best = max(score.items(), key=lambda kv: kv[1])[0]
    if score["industrial"] > 0.10:  # small threshold
        return "industrial"
    return best if score[best] > 0.05 else "unknown"


def estimate_co2_intensity(chosen_out, grid_g_per_kwh, building_eui_kwh_m2_yr):
    scene_type = weighted_scene_type(chosen_out["scenes"])
    label = chosen_out["label"]
    mult = LABEL_MULTIPLIER.get(label, 1.0)

    if scene_type == "transport":
        value = EPA_PASSENGER_CAR_G_PER_KM * mult
        return {
            "scene_type": "transport",
            "unit": "gCO2/km",
            "value": value,
            "note": "EPA passenger car tank-to-wheel CO2 per vehicle-mile converted to per-km, scaled by emission label."
        }

    # built / industrial / unknown -> building operational proxy
    base_kg_m2_yr = (building_eui_kwh_m2_yr * grid_g_per_kwh) / 1000.0
    value = base_kg_m2_yr * mult
    return {
        "scene_type": scene_type,
        "unit": "kgCO2e/m²/year",
        "value": value,
        "note": "Operational proxy = (EUI kWh/m²/yr * grid gCO2/kWh)/1000, scaled by emission label."
    }


def model_score(out, prefer_base_bias=0.0):
    """
    Conservative scoring:
    - Primary: emission probability
    - Secondary: confidence gap
    - Optional: bias (used to prefer BASE in AUTO mode)
    """
    return (out["prob"] * 1.00) + (out["gap"] * 0.25) + prefer_base_bias


def choose_best_output(img_input, base_out, intel_out, selector="auto", prefer_base=True):
    """
    FIXED selection logic:
    - selector=base or intel forces one model
    - selector=auto:
        * Intel path -> Intel
        * Else: default BASE (prevents Intel dominating out-of-domain)
        * Switch to Intel only if Intel is clearly stronger AND not contradictory
    """
    if selector == "base":
        return "BASE_PLACES", base_out
    if selector == "intel":
        return "INTEL_FINETUNED", intel_out

    # AUTO:
    if is_intel_domain(img_input):
        return "INTEL_FINETUNED", intel_out

    # For non-Intel inputs: prefer base unless Intel is *clearly* better
    base_type = weighted_scene_type(base_out["scenes"])
    intel_type = weighted_scene_type(intel_out["scenes"])

    # If both agree on type or emission label, we can compare more safely
    agree = (base_type == intel_type) or (base_out["label"] == intel_out["label"])

    base_bias = 0.05 if prefer_base else 0.0
    s_base = model_score(base_out, prefer_base_bias=base_bias)
    s_intel = model_score(intel_out, prefer_base_bias=0.0)

    # Require a margin to switch to Intel (prevents always switching)
    margin = 0.12  # tuneable
    if agree and (s_intel > s_base + margin) and (intel_out["prob"] >= 0.80):
        return "INTEL_FINETUNED", intel_out

    return "BASE_PLACES", base_out


def print_block(name, out, topk):
    print(f"\n--- {name} ---")
    print(f"Emission: {out['label']} ({out['prob']*100:.2f}%) | gap={out['gap']*100:.2f}%")
    print(f"Scene-type (weighted): {weighted_scene_type(out['scenes'])}")
    print(f"Top-{topk} scenes:")
    for i, (cls, p) in enumerate(out["scenes"], start=1):
        print(f"  {i}. {cls} ({p*100:.2f}%)")


def print_final(chosen_name, out, topk, co2_est):
    print("\n==============================")
    print("✅ FINAL OUTPUT (AUTO selection)")
    print("==============================")
    print("Chosen model:", chosen_name)
    print(f"Emission class: {out['label']} ({out['prob']*100:.2f}%)")
    print(f"Confidence gap: {out['gap']*100:.2f}%")
    print("\n--- Standard-based numeric estimate ---")
    print(f"Scene type: {co2_est['scene_type']}")
    print(f"Estimated intensity: {co2_est['value']:.2f} {co2_est['unit']}")
    print(f"Note: {co2_est['note']}")
    print(f"\nTop-{topk} scenes:")
    for i, (cls, p) in enumerate(out["scenes"], start=1):
        print(f"{i}. {cls} ({p*100:.2f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", "-i", required=True, help="Local path OR direct image URL (must return image/*)")
    ap.add_argument("--topk", type=int, default=TOPK_DEFAULT)

    ap.add_argument("--grid-gco2-kwh", type=float, default=DEFAULT_GRID_G_PER_KWH,
                    help="Grid carbon intensity in gCO2/kWh (override for your country/region).")
    ap.add_argument("--building-eui", type=float, default=DEFAULT_BUILDING_EUI_KWH_PER_M2_YR,
                    help="Assumed building EUI in kWh/m²/year for built/industrial scenes.")

    ap.add_argument("--selector", choices=["auto", "base", "intel"], default="auto",
                    help="Force model choice or let AUTO pick.")
    ap.add_argument("--no-prefer-base", action="store_true",
                    help="Disable conservative bias toward BASE in AUTO mode.")

    args = ap.parse_args()

    classes = load_classes()
    num_scenes = len(classes)

    print(f"\nLoading BASE model:  {BASE_CKPT}")
    base_model = load_model(num_scenes, BASE_CKPT)

    print(f"Loading INTEL model: {INTEL_CKPT}")
    intel_model = load_model(num_scenes, INTEL_CKPT)

    x = preprocess_image(args.image)

    base_out = predict(base_model, x, classes, topk=args.topk)
    intel_out = predict(intel_model, x, classes, topk=args.topk)

    # Always print both for debugging / evaluation
    print_block("BASE_PLACES", base_out, args.topk)
    print_block("INTEL_FINETUNED", intel_out, args.topk)

    chosen_name, chosen_out = choose_best_output(
        args.image, base_out, intel_out,
        selector=args.selector,
        prefer_base=(not args.no_prefer_base)
    )

    co2_est = estimate_co2_intensity(
        chosen_out=chosen_out,
        grid_g_per_kwh=args.grid_gco2_kwh,
        building_eui_kwh_m2_yr=args.building_eui,
    )

    print_final(chosen_name, chosen_out, args.topk, co2_est)


if __name__ == "__main__":
    main()
