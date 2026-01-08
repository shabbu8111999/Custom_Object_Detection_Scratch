import torch.nn as nn
from .backbone import SimpleCNNBackbone

class SimpleDetector(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        try:
            self.backbone = SimpleCNNBackbone()
            self.classifier = nn.Linear(256, num_classes)
        except Exception as e:
            print("Detector init error:", e)

    def forward(self, x):
        try:
            features = self.backbone(x)
            pooled = features.mean(dim=[2, 3])  # global avg pooling
            return self.classifier(pooled)
        except Exception as e:
            print("Detector forward error:", e)
            return None
