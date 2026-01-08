def collate_fn(batch):
    """
    Custom collate function for object detection.
    Keeps images and targets as lists.
    """
    try:
        images = []
        targets = []

        for item in batch:
            if item is None:
                continue
            image, target = item
            images.append(image)
            targets.append(target)

        return images, targets

    except Exception as e:
        print("Collate function error:", e)
        return None
