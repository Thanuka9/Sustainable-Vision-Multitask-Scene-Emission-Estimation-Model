# dataset.py

from typing import Dict

import torch
from torchvision.datasets import Places365
from torchvision import transforms

from config import SCENE_ATTR_MAPPING, INDOOR_SCENES, ATTRIBUTE_ORDER


# -------------------------------------------------
# Heuristic: scene name -> carbon emission category
# -------------------------------------------------
# This function maps a scene name to a carbon emission level (0-4).
# This is a heuristic approach because we don't have ground truth emission data for every image.
# We categorize scenes based on keywords found in their names.
#
# Levels:
#   0: very_low   (Nature, untouched environments)
#   1: low        (Rural areas, parks, light human impact)
#   2: medium     (Generic urban areas, indoor residential/commercial, mixed use)
#   3: high       (Transport hubs, heavy traffic, parking)
#   4: very_high  (Industrial zones, energy production, heavy construction)
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
    Generates a weak-supervision attribute vector for a given scene.

    Args:
        scene (str): The name of the scene (e.g., 'cinema', 'airport_terminal').

    Returns:
        torch.Tensor: A 4-element float tensor representing binary attributes:
                      [Indoor, Crowded, Bright_Lights, High_Tech_Equipment].
    """
    scene = scene.lower()
    attr: Dict[str, int] = {k: 0 for k in ATTRIBUTE_ORDER}

    # 1. Apply generic "Indoor" heuristic
    if scene in INDOOR_SCENES:
        attr["Indoor"] = 1

    # 2. Apply specific overrides from config.SCENE_ATTR_MAPPING
    # This allows fine-grained control, e.g., "cinema" implies specific attributes.
    if scene in SCENE_ATTR_MAPPING:
        for k, v in SCENE_ATTR_MAPPING[scene].items():
            if k in attr:
                # OR-style merge – once an attribute is set to 1, it stays 1.
                attr[k] = max(attr[k], int(v))

    return torch.tensor([attr[k] for k in ATTRIBUTE_ORDER], dtype=torch.float32)


class Places365WithAttributes(Places365):
    """
    Custom Dataset class that extends torchvision's Places365.
    It adds two auxiliary tasks: Attribute Prediction and Carbon Emission Estimation.

    Returns a dictionary for each sample:
        {
            "image": Tensor [3,H,W] - The transformed image.
            "scene_label": LongTensor scalar (0..364) - The original Places365 class index.
            "attribute_label": FloatTensor [4] - Binary vector for attributes (weakly supervised).
            "emission_label": LongTensor scalar (0..4) - Estimated carbon emission level (0=very_low..4=very_high).
        }
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the dataset.
        We pass all arguments to the parent Places365 class, then build
        a mapping from scene class index to emission level to speed up __getitem__.
        """
        super().__init__(*args, **kwargs)

        # self.classes contains strings like 'Places365-Standard/airport_terminal'
        # We pre-compute the emission label for each class index.
        self.class_to_emission = {}
        for idx, raw_name in enumerate(self.classes):
            scene_name = raw_name.split("/")[-1]
            self.class_to_emission[idx] = scene_name_to_emission(scene_name)

    def __getitem__(self, idx: int):
        # Get the image and scene label from the standard Places365 dataset
        img, scene_label = super().__getitem__(idx)

        # Recover scene name to generate attributes
        scene_name_raw = self.classes[scene_label]
        scene_name = scene_name_raw.split("/")[-1]

        # Generate weak labels on the fly
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
    Returns standard ImageNet / Places365 data augmentations/transformations.

    Args:
        train (bool): If True, applies random augmentations (Resize, RandomCrop, Flip, Jitter).
                      If False, applies deterministic transforms (Resize, CenterCrop) for validation/inference.
    """
    if train:
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                # Normalize using standard ImageNet mean and std
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
