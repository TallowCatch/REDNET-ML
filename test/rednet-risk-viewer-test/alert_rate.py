import pandas as pd
df = pd.read_csv("deployment/outputs/scene_alerts.csv")
print("alert rate:", df["scene_alert"].mean())
