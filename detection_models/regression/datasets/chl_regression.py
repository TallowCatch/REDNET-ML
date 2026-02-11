import os, csv, cv2, numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

class ChlTiles(Dataset):
    def __init__(self, csv_path, split='train', size=224, augment=False):
        self.items = []
        self.size = size
        self.augment = augment
        with open(csv_path, newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                if row['split'] == split:
                    self.items.append((row['filepath'], float(row['chl'])))

        self.tf = A.Compose([
            A.Resize(self.size, self.size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.1,0.1,p=0.5),
            A.GaussianBlur(blur_limit=3,p=0.3),
            A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.3)
        ]) if augment else A.Compose([A.Resize(self.size, self.size)])

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug = self.tf(image=img)
        img = aug['image'].astype('float32')/255.0
        img = np.transpose(img, (2,0,1))  # CHW
        return torch.from_numpy(img), torch.tensor(y, dtype=torch.float32)
