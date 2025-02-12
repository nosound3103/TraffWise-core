import os
import glob
import random
import numpy as np
import cv2


class DataLoader:
    def __init__(self, data_path, subset='train', img_size=640, batch_size=16, augment=False):
        self.data_path = os.path.join(data_path, 'images', subset)
        self.label_path = os.path.join(data_path, 'labels', subset)
        self.img_size = img_size
        self.batch_size = batch_size
        self.augment = augment
        self.img_files = glob.glob(os.path.join(self.data_path, '*.jpg'))
        self.label_files = [os.path.join(self.label_path, os.path.basename(
            x).replace('.jpg', '.txt')) for x in self.img_files]
        self.indices = list(range(len(self.img_files)))
        self.current_index = 0

    def __len__(self):
        return len(self.img_files) // self.batch_size

    def __iter__(self):
        self.current_index = 0
        if self.augment:
            random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current_index >= len(self.indices):
            raise StopIteration

        batch_indices = self.indices[self.current_index:
                                     self.current_index + self.batch_size]
        self.current_index += self.batch_size

        images, labels = [], []
        for idx in batch_indices:
            img_path = self.img_files[idx]
            label_path = self.label_files[idx]

            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.img_size, self.img_size))
            # BGR to RGB and HWC to CHW
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img, dtype=np.float32) / 255.0

            with open(label_path, 'r') as f:
                label = np.array([list(map(float, x.split()))
                                 for x in f.read().strip().splitlines()])

            images.append(img)
            labels.append(label)

        return np.stack(images, axis=0), labels

# Example usage:
# dataloader = DataLoader(data_path='/path/to/dataset', subset='train', img_size=640, batch_size=16, augment=True)
# for images, labels in dataloader:
#     # Training code here
