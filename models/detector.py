import torch
import torch.nn as nn
from .backbone import SimpleCNNBackbone

class SimpleDetector(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        try:
            self.backbone = SimpleCNNBackbone()

            # Global pooling
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

            # Classification head
            self.classifier = nn.Linear(256, num_classes)

            # Bounding box regression head
            self.bbox_regressor = nn.Linear(256, 4)

        except Exception as e:
            print("Detector init error:", e)

    def forward(self, x):
        try:
            features = self.backbone(x)
            pooled = self.pool(features).view(features.size(0), -1)

            class_logits = self.classifier(pooled)
            bbox_preds = self.bbox_regressor(pooled)

            return class_logits, bbox_preds

        except Exception as e:
            print("Detector forward error:", e)
            return None, None
