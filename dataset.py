# dataset.py

from typing import Dict

import torch
from torchvision.datasets import Places365
from torchvision import transforms

from config import SCENE_ATTR_MAPPING, INDOOR_SCENES, ATTRIBUTE_ORDER


# -------------------------------------------------
# Heuristic: scene name -> carbon emission category
#   0: very_low   (mostly natural, untouched)
#   1: low        (rural / parks / light human impact)
#   2: medium     (generic urban / indoor)
#   3: high       (transport hubs, heavy traffic)
#   4: very_high  (industrial / energy production)
# -------------------------------------------------
def scene_name_to_emission(scene_name: str) -> int:
    name = scene_name.lower()

    # Very low - natural scenes
    very_low_keywords = [
        "forest", "mountain", "river", "lake", "desert", "field",
        "canyon", "glacier", "cliff", "valley", "ocean", "coast",
        "island", "snowfield", "iceberg", "pasture", "savanna",
    ]
    if any(k in name for k in very_low_keywords):
        return 0

    # Low - rural / small settlement / parks
    low_keywords = [
        "park", "garden", "village", "residential", "courtyard",
        "farm", "meadow", "orchard", "campsite", "playground",
        "suburb",
    ]
    if any(k in name for k in low_keywords):
        return 1

    # High - heavy transport / very busy
    high_keywords = [
        "highway", "traffic", "parking_garage", "parkinglot",
        "airport", "runway", "railroad", "train_station",
        "bus_station", "bus_interior", "bridge", "metro_station",
        "subway_station", "street",
    ]
    if any(k in name for k in high_keywords):
        return 3

    # Very high - industrial / energy / extraction
    very_high_keywords = [
        "industrial", "refinery", "power_plant", "powerplant",
        "factory", "smokestack", "oil", "gasworks", "mine",
        "mining", "warehouse", "construction_site",
    ]
    if any(k in name for k in very_high_keywords):
        return 4

    # Default: medium (typical city / indoor / mixed use)
    return 2


def attribute_from_scene(scene: str) -> torch.Tensor:
    """
    Given a scene name (e.g. 'cinema', 'airport_terminal'), return
    a 4-dim tensor [Indoor, Crowded, Bright_Lights, High_Tech_Equipment].
    Uses weak-supervision heuristics from config.py.
    """
    scene = scene.lower()
    attr: Dict[str, int] = {k: 0 for k in ATTRIBUTE_ORDER}

    # Indoor heuristic
    if scene in INDOOR_SCENES:
        attr["Indoor"] = 1

    # Scene-specific overrides (e.g. cinema -> Bright_Lights=1, High_Tech_Equipment=1)
    if scene in SCENE_ATTR_MAPPING:
        for k, v in SCENE_ATTR_MAPPING[scene].items():
            if k in attr:
                # OR-style merge – once 1, stays 1
                attr[k] = max(attr[k], int(v))

    return torch.tensor([attr[k] for k in ATTRIBUTE_ORDER], dtype=torch.float32)


class Places365WithAttributes(Places365):
    """
    Wraps torchvision Places365 to return a dict per sample:

        {
            "image": Tensor [3,H,W],
            "scene_label": LongTensor scalar (0..364),
            "attribute_label": FloatTensor [4] with 0/1 entries,
            "emission_label": LongTensor scalar (0..4)  # very_low..very_high
        }
    """

    def __init__(self, *args, **kwargs):
        """
        We just pass through to the original Places365 __init__,
        then build a mapping from scene class index -> emission level.
        """
        super().__init__(*args, **kwargs)

        # self.classes entries look like 'Places365-Standard/airport_terminal'
        # we take the last part as the scene key.
        self.class_to_emission = {}
        for idx, raw_name in enumerate(self.classes):
            scene_name = raw_name.split("/")[-1]
            self.class_to_emission[idx] = scene_name_to_emission(scene_name)

    def __getitem__(self, idx: int):
        img, scene_label = super().__getitem__(idx)

        # Recover scene name (last part of the string)
        scene_name_raw = self.classes[scene_label]
        scene_name = scene_name_raw.split("/")[-1]

        attr_vec = attribute_from_scene(scene_name)
        emission_label = self.class_to_emission[int(scene_label)]

        return {
            "image": img,
            "scene_label": torch.tensor(scene_label, dtype=torch.long),
            "attribute_label": attr_vec,
            "emission_label": torch.tensor(emission_label, dtype=torch.long),
        }


def get_default_transforms(train: bool = True):
    """
    Standard ImageNet / Places-style transforms.
    """
    if train:
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
