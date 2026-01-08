import os
import time
import torch
import cv2
from PIL import Image
from torchvision import transforms
from models.detector import SimpleDetector
from utils.visualization import draw_detection

CLASSES = ["person", "car", "bicycle", "dog", "chair"]

def run_inference(image_path):
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        image_pil = Image.open(image_path).convert("RGB")
        image_tensor = transform(image_pil).unsqueeze(0)

        model = SimpleDetector(num_classes=5)
        model.load_state_dict(torch.load("model.pth", map_location="cpu"))
        model.eval()

        start = time.time()
        cls_logits, bbox_preds = model(image_tensor)
        end = time.time()

        fps = 1 / (end - start)

        pred_label = cls_logits.argmax(dim=1).item()
        pred_box = bbox_preds[0].detach().numpy()

        image_cv = cv2.cvtColor(
            cv2.imread(image_path), cv2.COLOR_BGR2RGB
        )

        image_out = draw_detection(
            image_cv,
            pred_box,
            pred_label
        )

        os.makedirs("outputs", exist_ok=True)
        cv2.imwrite("outputs/result.jpg", cv2.cvtColor(image_out, cv2.COLOR_RGB2BGR))

        print("Prediction:", CLASSES[pred_label])
        print(f"Inference FPS: {fps:.2f}")

    except Exception as e:
        print("Inference error:", e)


if __name__ == "__main__":
    run_inference("VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg")
