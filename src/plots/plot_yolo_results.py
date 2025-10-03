# src/plots/plot_yolo_results.py (quick-and-dirty)
import pandas as pd, matplotlib.pyplot as plt, os, sys
csv = "runs/yolo/hab_yolov8n3/results.csv"
out = "runs/yolo/hab_yolov8n3/results_plot.png"
df = pd.read_csv(csv)
plt.figure(figsize=(9,4))
ax1 = plt.gca()
ax1.plot(df.index, df['train/box_loss'], label='box_loss')
ax1.plot(df.index, df['val/box_loss'], label='val_box_loss')
ax1.set_xlabel('epoch'); ax1.set_ylabel('loss'); ax1.legend(loc='upper left')
ax2 = ax1.twinx()
ax2.plot(df.index, df['metrics/mAP50(B)'], 'g-', label='mAP50')
ax2.set_ylabel('mAP50'); 
plt.title('YOLO training')
plt.tight_layout(); plt.savefig(out, dpi=200)
print('Saved', out)