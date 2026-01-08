# Assignment 1 – Custom Object Detection (PASCAL VOC 2012)

## Overview
This project implements a custom object detection pipeline trained entirely
from scratch using the PASCAL VOC 2012 dataset. The goal of this assignment is
to demonstrate fundamental understanding of object detection concepts,
including dataset handling, model design, training, inference, and evaluation,
without relying on pre-trained weights.

The implementation focuses on clarity, modularity, and reproducibility rather
than achieving state-of-the-art performance.

---

## Dataset
- Dataset: PASCAL VOC 2012
- Object Classes Used:
  - person
  - car
  - bicycle
  - dog
  - chair

The dataset is **not included** in this repository due to size constraints.
Please download it manually from Kaggle:

https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012

After extraction, the dataset should be placed as:

VOCdevkit/
└── VOC2012/
    ├── JPEGImages/        # All input images (.jpg)
    ├── Annotations/       # XML annotation files (bounding boxes)
    ├── ImageSets/
    │   └── Main/          # train.txt, val.txt, trainval.txt



---

## Model Architecture
A lightweight CNN-based object detector was implemented and trained from scratch.

### Key Components:
- Custom CNN backbone for feature extraction
- Global average pooling
- Classification head for object class prediction
- Bounding box regression head for object localization

This design provides a simplified but complete object detection pipeline while
remaining easy to interpret and debug.

---

## Training Details
- Optimizer: Adam
- Learning Rate: 1e-4
- Input Image Size: 224 × 224
- Loss Functions:
  - CrossEntropyLoss (classification)
  - SmoothL1Loss (bounding box regression)
- Batch handling uses a custom collate function to support variable numbers
  of objects per image
- Only the first valid object per image is used for supervision, as part of
  a simplified detection strategy

No pre-trained models or external weights were used.

---

## Inference
During inference:
- The trained model predicts an object class and bounding box for a given image
- Bounding boxes are drawn on the input image
- Inference speed is measured by timing the forward pass and computing FPS

The output image is saved to:


---

## How to Run

### Setup
```bash
uv init
uv add (whatever requirements is)
```

### Train the Model
```bash
uv run train.py
```

### Run Inference
```bash
uv run inference.py
```

## Outputs
- model.pth – trained model weights
- outputs/result.jpg – image with predicted bounding box
- Console output showing predicted class and inference FPS

## Notes
***This project intentionally uses a simplified detector design to focus on ore object detection principles. The results demonstrate correct end-to-end functionality, including training from scratch, localization, visualization, and inference speed measurement.***