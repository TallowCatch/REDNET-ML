# REDNET-ML (Virtual Stack)

This repo contains a minimal, runnable machine-learning stack for the REDNET paper. It is **all-virtual** and uses open-source tools.

## Stack
- Python 3.11
- PyTorch (training + ONNX export)
- Ultralytics YOLOv8 (tiny detector)
- Albumentations, OpenCV, scikit-learn
- Optional: segmentation-models-pytorch

## Setup
```bash
conda create -n rednet-ml python=3.11 -y
conda activate rednet-ml
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or cpu wheels
pip install ultralytics==8.2.0 albumentations opencv-python shapely matplotlib scikit-learn pandas pyyaml tqdm onnx onnxruntime
# optional
pip install segmentation-models-pytorch
```

## Data layout
```
data/
  chl_tiles/                       # satellite-derived tiles or tri-band proxies
  uav_sim/                         # optional simulated frames/video
  labels/
    detection/
      images/{train,val,test}/
      labels/{train,val,test}/     # YOLO .txt per image
    segmentation/
      images/                      # PNG/JPG
      masks/                       # PNG masks aligned with images
    regression.csv                 # file,split,chl (µg/L)
```

## Training
### Regression (Chl-a)
```bash
python src/train/train_regression.py
```
Config: `cfg/reg.yaml`

### Segmentation (bloom mask)
```bash
python src/train/train_seg.py
```

### Detection (YOLOv8n)
```bash
yolo task=detect mode=train model=yolov8n.pt data=cfg/det.yaml imgsz=640 epochs=80 batch=16 project=outputs/det name=y8n
```

## Inference / Export
```bash
python src/infer/export_onnx.py          # export regression model to ONNX
python src/infer/infer_stream.py         # overlay demo on a video/frames
```

## Notes
- Fill `data/labels/regression.csv` with: `filepath,split,chl`
- Populate segmentation images+masks and detection YOLO labels accordingly.
- Outputs/models will appear in `outputs/`.
