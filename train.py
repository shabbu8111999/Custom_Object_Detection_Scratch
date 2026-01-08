import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data.dataset import VOCDataset
from models.detector import SimpleDetector

def train():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        dataset = VOCDataset(
            root_dir="VOCdevkit/VOC2012",
            split="train",
            transform=transform
        )

        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        model = SimpleDetector(num_classes=5).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()

        for epoch in range(5):
            print(f"Epoch {epoch+1}")
            for images, targets in loader:
                images = images.to(device)
                labels = targets["labels"][:, 0].to(device)  # first object

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            print("Loss:", loss.item())

        torch.save(model.state_dict(), "model.pth")
        print("Training completed ✅")

    except Exception as e:
        print("Training error:", e)

if __name__ == "__main__":
    train()
