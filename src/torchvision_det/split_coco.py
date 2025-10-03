from pathlib import Path
import json, random

COCO_IN  = Path("data/labels/coco/instances_all.json")
COCO_DIR = Path("data/labels/coco")

def main(seed=42, tvt=(0.8, 0.1, 0.1)):
    coco = json.loads(COCO_IN.read_text())
    ids = [int(im["id"]) for im in coco["images"]]
    random.Random(seed).shuffle(ids)
    n = len(ids); ntr = int(tvt[0]*n); nv = int(tvt[1]*n)
    splits = {
        "train": set(ids[:ntr]),
        "val":   set(ids[ntr:ntr+nv]),
        "test":  set(ids[ntr+nv:])
    }
    for name, keep in splits.items():
        ims = [im for im in coco["images"] if int(im["id"]) in keep]
        ann = [a for a in coco["annotations"] if int(a["image_id"]) in keep]
        out = {"images": ims, "annotations": ann, "categories": coco["categories"]}
        (COCO_DIR / f"instances_{name}.json").write_text(json.dumps(out, indent=2))
        print(f"{name}: {len(ims)} images, {len(ann)} anns")

if __name__ == "__main__":
    main()
