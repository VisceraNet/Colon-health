# datasets.py

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

IMG_SIZE = 288


def get_transforms(train=True):
    if train:
        return T.Compose([
            T.Resize(IMG_SIZE),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(0.1, 0.1, 0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std =[0.229,0.224,0.225])
        ])
    else:
        return T.Compose([
            T.Resize(IMG_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std =[0.229,0.224,0.225])
        ])


class LIMUCDataset(Dataset):
    def __init__(self, root_dir, train=True):
        self.samples = []
        self.transform = get_transforms(train)

        for folder in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, folder)
            if not os.path.isdir(class_dir):
                continue

            # Expect folder names like "Mayo 0", "Mayo 1", ...
            try:
                mayo = int(folder.split()[-1])
            except ValueError:
                continue

            for img in os.listdir(class_dir):
                if img.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                    self.samples.append(
                        (os.path.join(class_dir, img), mayo)
                    )

        self.samples.sort()


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, mayo = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        binary = 0 if mayo <= 1 else 1  # remission vs active

        return {
            "image": img,
            "binary": binary,
            "ordinal": mayo
        }