import torch
import torch.nn as nn
import torchvision.models as models


class MultiTaskResNet50(nn.Module):
    def __init__(self, num_scenes: int, num_attrs: int = 4, num_emission: int = 5):
        super().__init__()

        # Base resnet50
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # [B, 2048, 1, 1]
        feature_dim = backbone.fc.in_features  # 2048

        # Scene head
        self.scene_head = nn.Linear(feature_dim, num_scenes)

        # Attribute head (4 binary attrs)
        self.attr_head = nn.Linear(feature_dim, num_attrs)

        # ✅ NEW: carbon-emission head (5 classes: very_low..very_high)
        self.emission_head = nn.Linear(feature_dim, num_emission)

    def forward(self, x):
        feats = self.features(x)          # [B, 2048, 1, 1]
        feats = feats.flatten(1)          # [B, 2048]

        scene_logits = self.scene_head(feats)
        attr_logits = self.attr_head(feats)
        emission_logits = self.emission_head(feats)  # ✅ NEW

        # 🔁 IMPORTANT: now we return THREE things
        return scene_logits, attr_logits, emission_logits
