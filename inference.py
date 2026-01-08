import torch
from PIL import Image
from torchvision import transforms
from models.detector import SimpleDetector

CLASSES = ["person", "car", "bicycle", "dog", "chair"]

def run_inference(image_path):
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)

        model = SimpleDetector(num_classes=5)
        model.load_state_dict(torch.load("model.pth"))
        model.eval()

        outputs = model(image)
        pred = outputs.argmax(dim=1).item()

        print("Predicted class:", CLASSES[pred])

    except Exception as e:
        print("Inference error:", e)

if __name__ == "__main__":
    run_inference("test.jpg")
