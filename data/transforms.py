from torchvision import transforms

def get_train_transforms():
    """
    Transforms applied during training
    """
    try:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),   # resize image
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),                               # simple augmentation
            transforms.ToTensor()            # convert PIL image to tensor
        ])
        return train_transform

    except Exception as e:
        print("Error creating train transforms:", e)
        return None


def get_val_transforms():
    """
    Transforms applied during validation / inference
    """
    try:
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        return val_transform

    except Exception as e:
        print("Error creating validation transforms:", e)
        return None
