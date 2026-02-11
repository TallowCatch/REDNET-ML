import cv2, torch, time, os
from models.reg_cnn import TinyRegressor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
m = TinyRegressor().to(device)
m.load_state_dict(torch.load('outputs/reg/best.pt', map_location=device))
m.eval()

cap = cv2.VideoCapture('data/uav_sim/stream.mp4')  # replace with your video path
if not cap.isOpened():
    print('Could not open video. Place a video at data/uav_sim/stream.mp4')
    raise SystemExit

while True:
    ok, frame = cap.read()
    if not ok: break
    img = cv2.resize(frame,(224,224)).astype('float32')/255.0
    img = img.transpose(2,0,1)
    x = torch.from_numpy(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = m(x).item()
    alert = (pred >= 2.0)
    out = frame.copy()
    cv2.putText(out, f'Chl-a: {pred:.2f} ug/L', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    if alert:
        cv2.putText(out, 'HAB ALERT', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    cv2.imshow('REDNET Scout (virtual)', out)
    if cv2.waitKey(1)==27: break
