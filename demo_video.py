import os
import cv2
import torch
from PIL import Image
from torchvision import transforms
from models.detector import SimpleDetector
from utils.visualization import draw_detection

CLASSES = ["person", "car", "bicycle", "dog", "chair"]

def run_demo():
    try:
        image_dir = "VOCdevkit/VOC2012/JPEGImages"
        image_files = os.listdir(image_dir)[:15]  # first 15 images

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        model = SimpleDetector(num_classes=5)
        model.load_state_dict(torch.load("model.pth", map_location="cpu"))
        model.eval()

        for img_name in image_files:
            img_path = os.path.join(image_dir, img_name)

            image_pil = Image.open(img_path).convert("RGB")
            image_tensor = transform(image_pil).unsqueeze(0)

            cls_logits, bbox_preds = model(image_tensor)

            label = cls_logits.argmax(dim=1).item()
            bbox = bbox_preds[0].detach().numpy()

            image_cv = cv2.cvtColor(
                cv2.imread(img_path), cv2.COLOR_BGR2RGB
            )

            image_out = draw_detection(image_cv, bbox, label)

            cv2.imshow("Detection Demo", cv2.cvtColor(image_out, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(500) & 0xFF == 27:  # 500 ms per image
                break

        cv2.destroyAllWindows()

    except Exception as e:
        print("Demo error:", e)


if __name__ == "__main__":
    run_demo()
