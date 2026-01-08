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
- Global average pooling to obtain fixed-size feature representations
- A classification head for predicting object categories
- A bounding box regression head for predicting object locations

This simplified design was chosen to maintain interpretability, computational
efficiency, and ease of implementation while still providing end-to-end object
detection functionality.

---

## Training Strategy
The model was trained using the Adam optimizer with a learning rate of 1e-4.
All input images were resized to 224 × 224. Two loss functions were used:
- Cross Entropy Loss for object classification
- Smooth L1 Loss for bounding box regression

Since object detection datasets contain a variable number of objects per image,
a custom collate function was implemented to handle batching. For training
simplicity, only a single object per image was used as supervision. Specifically,
one object instance was selected from each image to guide both classification
and bounding box regression.

No pre-trained models or external weights were used at any stage of training.

---

## Evaluation
Evaluation was performed using a simplified object detection setup.
Bounding box predictions were verified using Intersection over Union (IoU)
between predicted and ground-truth boxes.

Inference speed was measured by timing the forward pass of the trained model on a
single image and computing Frames Per Second (FPS). Detection results were also
validated visually by drawing predicted bounding boxes on input images.

This evaluation approach emphasizes correctness and efficiency rather than
state-of-the-art accuracy.

---

## Results and Observations
The model successfully demonstrates an end-to-end object detection pipeline,
including object classification, bounding box localization, visualization, and
inference speed measurement.

Due to class imbalance in the dataset and the simplified training strategy using
a single object per image, the classifier tends to favor the dominant class
(person). This behavior is expected and highlights a known limitation of global
image-level supervision in object detection tasks. The observation reinforces
the importance of region-level supervision and balanced sampling strategies in
more advanced detection architectures.

---

## Conclusion
This project presents a complete object detection pipeline built from scratch
using the PASCAL VOC 2012 dataset. It highlights key concepts such as dataset
handling, model design, training strategies, inference, visualization, and
performance measurement, while intentionally avoiding reliance on pre-trained
models.

The implementation prioritizes clarity and reproducibility, making it suitable
for demonstrating foundational object detection principles in an intern-level
setting.
