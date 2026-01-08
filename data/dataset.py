import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset


# I will only use the 5 classes that are in VOC2007
CLASSES = ["person", "car", "bicycle", "dog", "chair"]

class VOCDataset(Dataset):
    def __init__(self, root_dir, split="train", transform = None):
        """
        root_dir: VOCdevkit/VOC2012
        split: train / val
        """
        try:
            self.root_dir = root_dir
            self.transform = transform

            split_file = os.path.join(
                root_dir, "ImageSets", "Main", f"{split}.txt"
            )

            with open(split_file) as f:
                self.image_ids = f.read().strip().split()

            self.image_dir = os.path.join(root_dir, "JPEGImages")
            self.ann_dir = os.path.join(root_dir, "Annotations")

        except Exception as e:
            print(f"Dataset initialization error: {e}")


    def __len__(self):
        return len(self.image_ids)
    

    def __getitem__(self, idx):
        try:
            img_id = self.image_ids[idx]

            img_path = os.path.join(self.image_dir, img_id + ".jpg")
            ann_path = os.path.join(self.ann_dir, img_id + ".xml")

            image = Image.open(img_path).convert("RGB")

            tree = ET.parse(ann_path)
            root = tree.getroot()

            boxes = []
            labels = []

            for obj in root.findall("object"):
                name = obj.find("name").text
                if name in CLASSES:
                    label = CLASSES.index(name)

                    bbox = obj.find("bndbox")
                    xmin = int(bbox.find("xmin").text)
                    ymin = int(bbox.find("ymin").text)
                    xmax = int(bbox.find("xmax").text)
                    ymax = int(bbox.find("ymax").text)

                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(label)

            target = {
                "boxes" : torch.tensor(boxes, dtype=torch.float32),
                "labels" : torch.tensor(labels, dtype=torch.int64)
            }

            if self.transform:
                image = self.transform(image)

            return image, target
        
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            #return None
