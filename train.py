import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.dataset import VOCDataset
from models.detector import SimpleDetector
from utils.collate import collate_fn


def train():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

        # Transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Dataset
        dataset = VOCDataset(
            root_dir="VOCdevkit/VOC2012",
            split="train",
            transform=transform
        )

        # DataLoader
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn
        )

        # Model
        model = SimpleDetector(num_classes=5).to(device)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Loss functions (define ONCE)
        cls_loss_fn = torch.nn.CrossEntropyLoss()
        bbox_loss_fn = torch.nn.SmoothL1Loss()

        model.train()

        num_epochs = 5

        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

            epoch_loss = 0.0
            valid_batches = 0

            for images, targets in loader:

                valid_images = []
                valid_labels = []
                valid_boxes = []

                # Filter images without valid objects
                for img, tgt in zip(images, targets):
                    if len(tgt["labels"]) == 0:
                        continue

                    valid_images.append(img)
                    valid_labels.append(tgt["labels"][0])
                    valid_boxes.append(tgt["boxes"][0])

                if len(valid_images) == 0:
                    continue

                images_tensor = torch.stack(valid_images).to(device)
                labels_tensor = torch.tensor(
                    valid_labels, dtype=torch.long, device=device
                )
                boxes_tensor = torch.stack(valid_boxes).float().to(device)

                optimizer.zero_grad()

                cls_logits, bbox_preds = model(images_tensor)

                cls_loss = cls_loss_fn(cls_logits, labels_tensor)
                bbox_loss = bbox_loss_fn(bbox_preds, boxes_tensor)

                total_loss = cls_loss + bbox_loss
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()
                valid_batches += 1

            if valid_batches > 0:
                avg_loss = epoch_loss / valid_batches
                print(f"Average Loss: {avg_loss:.4f}")
            else:
                print("No valid batches in this epoch.")

        # Save model
        torch.save(model.state_dict(), "model.pth")
        print("\nTraining completed successfully ✅")

    except Exception as e:
        print("Training error:", e)


if __name__ == "__main__":
    train()
