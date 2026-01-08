import torch.nn as nn

class SimpleCNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.layers = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),

                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
            )
        except Exception as e:
            print("Backbone init error:", e)

    def forward(self, x):
        try:
            return self.layers(x)
        except Exception as e:
            print("Backbone forward error:", e)
            return None
