import torchvision.transforms as T, matplotlib.pyplot as plt
from pathlib import Path
import torch
from torchvision_det.ds_torchvision import HABDetDataset, collate_fn
from torchvision_det.eval_ap import load_model

def main(weights='runs/detect/frcnn/best.pt', split='val', k=6):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = load_model(weights, device=device).eval()
    tfm = T.Compose([T.Resize((640,640)), T.ToTensor()])
    ds = HABDetDataset(split, transforms=tfm)

    for i in range(min(k, len(ds))):
        img, target = ds[i]
        pred = model([img.to(device)])[0]
        im = img.permute(1,2,0).cpu().numpy()
        plt.figure(figsize=(4,4)); plt.imshow(im); plt.axis('off')
        for b,s in zip(pred['boxes'].cpu(), pred['scores'].cpu()):
            if s < 0.25: continue
            x1,y1,x2,y2 = b
            plt.gca().add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1, fill=False, lw=2, color='r'))
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
