import yaml, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from datasets.chl_regression import ChlTiles
from models.reg_cnn import TinyRegressor
import os

cfg = yaml.safe_load(open('cfg/reg.yaml'))
os.makedirs('outputs/reg', exist_ok=True)

ds_tr = ChlTiles(csv_path='data/labels/regression.csv', split='train', size=224, augment=True)
ds_va = ChlTiles(csv_path='data/labels/regression.csv', split='val',   size=224, augment=False)
tr, va = DataLoader(ds_tr,batch_size=cfg['bs'],shuffle=True,num_workers=2), DataLoader(ds_va,batch_size=cfg['bs'],num_workers=2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TinyRegressor().to(device)
crit  = nn.SmoothL1Loss()
opt   = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)

best = 1e9
for epoch in range(cfg['epochs']):
    model.train()
    for x,y in tr:
        x,y = x.to(device), y.to(device).unsqueeze(1)
        opt.zero_grad()
        pred = model(x)
        loss = crit(pred, y)
        loss.backward()
        opt.step()
    # val
    model.eval(); mae=[]
    with torch.no_grad():
        for x,y in va:
            x,y = x.to(device), y.to(device).unsqueeze(1)
            p = model(x)
            mae.append((p-y).abs().mean().item())
    m = float(np.mean(mae))
    print(f'E{epoch}: val_MAE={m:.3f}')
    if m<best:
        best=m
        torch.save(model.state_dict(),'outputs/reg/best.pt')
print('Best MAE:', best)
