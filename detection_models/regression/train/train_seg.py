import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from datasets.seg_dataset import SegDataset
from models.unet_lite import UNetLite
import numpy as np, os

def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs*targets).sum(dim=(2,3))
    union = probs.sum(dim=(2,3)) + targets.sum(dim=(2,3))
    dice = (2*inter+eps)/(union+eps)
    return 1 - dice.mean()

os.makedirs('outputs/seg', exist_ok=True)
tr = SegDataset('data/labels/segmentation/images','data/labels/segmentation/masks', train=True)
va = SegDataset('data/labels/segmentation/images','data/labels/segmentation/masks', train=False)
dl_tr, dl_va = DataLoader(tr,batch_size=8,shuffle=True,num_workers=2), DataLoader(va,batch_size=8,num_workers=2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNetLite().to(device)
bce = nn.BCEWithLogitsLoss()
opt = optim.AdamW(model.parameters(), lr=1e-3)

best = 1e9
for e in range(40):
    model.train()
    for x,y in dl_tr:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = bce(logits,y) + dice_loss(logits,y)
        loss.backward()
        opt.step()
    # val IoU
    model.eval(); ious=[]
    with torch.no_grad():
        for x,y in dl_va:
            x,y = x.to(device), y.to(device)
            p = (torch.sigmoid(model(x))>0.5).float()
            inter = (p*y).sum((2,3))
            union = p.sum((2,3)) + y.sum((2,3)) - inter + 1e-6
            iou = (inter+1e-6)/union
            ious += iou.mean(1).cpu().tolist()
    miou = float(np.mean(ious)) if ious else 0.0
    print(f'E{e}: mIoU={miou:.3f}')
    # save lowest (1-miou) as "loss"
    cur = 1-miou
    if cur<best:
        best=cur
        torch.save(model.state_dict(),'outputs/seg/best.pt')
print('Best mIoU:', 1-best)
