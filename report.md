# Assignment 1 – Custom Object Detection (PASCAL VOC 2012)

## Dataset
The PASCAL VOC 2012 dataset was used for this assignment. From the full dataset,
five commonly occurring object classes were selected: person, car, bicycle, dog,
and chair. Bounding box annotations provided in the VOC XML format were used for
training and evaluation.

Due to dataset size constraints, the dataset was downloaded manually and is not
included in the repository.

---

## Model Architecture
A custom CNN-based object detector was implemented and trained entirely from
scratch, without using any pre-trained weights.

The architecture consists of:
- A lightweight convolutional backbone for feature extraction
- Global average pooling to obtain fixed-size feature vectors
- A classification head for predicting object categories
- A bounding box regression head for predicting object locations

This simplified design was chosen to balance interpretability, computational
efficiency, and ease of implementation while still providing complete detection
functionality.

---

## Training Strategy
The model was trained using the Adam optimizer with a learning rate of 1e-4.
Input images were resized to 224 × 224. Two loss functions were used:
- Cross Entropy Loss for classification
- Smooth L1 Loss for bounding box regression

Since images in object detection datasets may contain varying numbers of objects,
a custom collate function was implemented. For training simplicity, only the
first valid object per image was used as supervision.

No pre-trained models or external weights were used at any stage of training.

---

## Evaluation
Evaluation was performed using a simplified object detection setup.
Bounding box predictions were verified using Intersection over Union (IoU)
between predicted and ground-truth boxes.

Inference speed was measured by timing the forward pass of the trained model
on a single image and computing Frames Per Second (FPS). Detection results were
validated visually by drawing predicted bounding boxes on images.

This evaluation approach focuses on correctness and efficiency rather than
optimizing for state-of-the-art accuracy.

---

## Results
The model successfully demonstrates end-to-end object detection capability,
including object classification, bounding box localization, and real-time
inference. While the detector is simplified, it provides a clear illustration
of fundamental object detection principles.

---

## Conclusion
This project presents a complete object detection pipeline built from scratch
using the PASCAL VOC 2012 dataset. It highlights key concepts such as dataset
handling, model design, training strategies, inference, and performance
measurement, while maintaining clarity and reproducibility without relying on
pre-trained models.