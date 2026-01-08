import cv2
import matplotlib.pyplot as plt

# Class names used in the project
CLASSES = ["person", "car", "bicycle", "dog", "chair"]

def draw_boxes(image, boxes, labels):
    """
    Draw bounding boxes on an image

    image  : numpy array (H, W, 3)
    boxes  : list of [xmin, ymin, xmax, ymax]
    labels : list of class indices
    """
    try:
        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = map(int, box)

            # Draw rectangle
            cv2.rectangle(
                image,
                (xmin, ymin),
                (xmax, ymax),
                color=(0, 255, 0),
                thickness=2
            )

            # Draw class name
            class_name = CLASSES[label]
            cv2.putText(
                image,
                class_name,
                (xmin, ymin - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )

        return image

    except Exception as e:
        print("Error drawing boxes:", e)
        return image


def show_image(image, title="Image"):
    """
    Display image using matplotlib
    """
    try:
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")
        plt.show()

    except Exception as e:
        print("Error showing image:", e)
