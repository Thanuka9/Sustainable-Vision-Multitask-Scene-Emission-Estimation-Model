import torch
import torch.nn as nn
import torchvision.models as models


class MultiTaskResNet50(nn.Module):
    """
    A Multi-Task Learning model based on ResNet-50.
    It performs three tasks simultaneously:
    1. Scene Classification (365 classes)
    2. Attribute Prediction (4 binary attributes)
    3. Carbon Emission Level Estimation (5 ordinal classes)
    """
    def __init__(self, num_scenes: int, num_attrs: int = 4, num_emission: int = 5):
        """
        Args:
            num_scenes (int): Number of scene classes (typically 365 for Places365).
            num_attrs (int): Number of binary attributes (default 4).
            num_emission (int): Number of carbon emission levels (default 5: very_low..very_high).
        """
        super().__init__()

        # Load a pre-trained ResNet-50 backbone (trained on ImageNet)
        # We remove the final fully connected layer (fc) to use the features.
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # Output shape: [Batch, 2048, 1, 1]
        feature_dim = backbone.fc.in_features  # 2048

        # --- Task-Specific Heads ---

        # 1. Scene Classification Head
        # Predicts the specific scene category (e.g., 'airport_terminal', 'forest').
        self.scene_head = nn.Linear(feature_dim, num_scenes)

        # 2. Attribute Prediction Head
        # Predicts binary attributes (e.g., 'Indoor', 'Crowded').
        # Uses a linear layer; sigmoid activation is applied later (in loss or inference).
        self.attr_head = nn.Linear(feature_dim, num_attrs)

        # 3. Carbon Emission Estimation Head
        # Predicts the emission level category (0..4).
        # Treated as a classification problem here.
        self.emission_head = nn.Linear(feature_dim, num_emission)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (Tensor): Input images of shape [Batch, 3, 224, 224].

        Returns:
            tuple: (scene_logits, attr_logits, emission_logits)
        """
        # Extract features using the ResNet backbone
        feats = self.features(x)          # [Batch, 2048, 1, 1]
        feats = feats.flatten(1)          # [Batch, 2048] - Flatten to a vector

        # Pass features through each task-specific head
        scene_logits = self.scene_head(feats)       # [Batch, num_scenes]
        attr_logits = self.attr_head(feats)         # [Batch, num_attrs]
        emission_logits = self.emission_head(feats) # [Batch, num_emission]

        # Return raw logits for all three tasks
        return scene_logits, attr_logits, emission_logits
