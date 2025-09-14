import os, cv2, numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

class SegDataset(Dataset):
    def __init__(self, images_dir, masks_dir, size=256, split_ratio=0.8, train=True):
        self.images = sorted([os.path.join(images_dir,f) for f in os.listdir(images_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        self.masks  = [os.path.join(masks_dir, os.path.basename(p)) for p in self.images]
        n = int(len(self.images)*split_ratio)
        if train:
            self.images, self.masks = self.images[:n], self.masks[:n]
        else:
            self.images, self.masks = self.images[n:], self.masks[n:]
        self.tf = A.Compose([
            A.Resize(size,size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.1,0.1,p=0.5),
            A.GaussianBlur(blur_limit=3,p=0.3),
            A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.3)
        ]) if train else A.Compose([A.Resize(size,size)])

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.images[idx]), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        aug = self.tf(image=img, mask=msk)
        img, msk = aug['image'].astype('float32')/255.0, (aug['mask']>127).astype('float32')
        img = np.transpose(img, (2,0,1))
        msk = np.expand_dims(msk,0)
        return torch.from_numpy(img), torch.from_numpy(msk)
