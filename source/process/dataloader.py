import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

data_transform = transforms.Compose([
    transforms.Resize(size=(640, 640)),
    transforms.ToTensor()
])

class_to_idx = {
    "bus": 0, "car": 1, "motorbike": 2, "truck": 3,
}

device = "cuda" if torch.cuda.is_available() else "cpu"


class CustomDataset(Dataset):
    def __init__(self,
                 root_dir,
                 split="train",
                 class_to_idx=class_to_idx,
                 transform=data_transform):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.class_to_idx = class_to_idx if class_to_idx else {}
        self.device = device

        self.image_dir = os.path.join(root_dir, "images", split)
        self.label_dir = os.path.join(root_dir, "labels", split)

        self.image_files = [f for f in os.listdir(
            self.image_dir) if f.endswith(".jpg") or f.endswith(".png")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_filename)

        img_name, _ = os.path.splitext(img_filename)
        label_path = os.path.join(self.label_dir, img_name + ".txt")

        img = Image.open(img_path).convert("RGB")
        original_width, original_height = img.size

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])
                    x_center, y_center, w, h = map(float, parts[1:])

                    xmin = (x_center - w / 2) * original_width
                    ymin = (y_center - h / 2) * original_height
                    xmax = (x_center + w / 2) * original_width
                    ymax = (y_center + h / 2) * original_height

                    if xmin < xmax and ymin < ymax:
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(class_id + 1)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        if len(boxes) == 0:
            boxes = torch.zeros((1, 4), dtype=torch.float32)
            labels = torch.zeros((1,), dtype=torch.int64)

        if self.transform:
            img = self.transform(img)

        new_width, new_height = 640, 640
        scale_x = new_width / original_width
        scale_y = new_height / original_height

        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        target["boxes"] = boxes

        return img, target
