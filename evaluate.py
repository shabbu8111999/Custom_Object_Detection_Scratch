import torch
from torchvision.ops import box_iou

def compute_iou(box1, box2):
    try:
        return box_iou(box1.unsqueeze(0), box2.unsqueeze(0)).item()
    except:
        return 0.0
