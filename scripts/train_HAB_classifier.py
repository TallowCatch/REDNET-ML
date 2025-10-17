#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np, joblib, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score

FEATS = ["fai_mean","rednir_mean","ndwi_mean","chlor_a","kd490","flh"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)  # *_hablabels.csv
    ap.add_argument("--model_out", default="runs/reg_log_mobilenet/best.pt")  # keep naming vibe
    args=ap.parse_args()

    df = pd.read_csv(args.labels_csv)
    for c in FEATS: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATS + ["hab_label"])

    X = df[FEATS].values
    y = df["hab_label"].astype(int).values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced"))
    ])
    pipe.fit(Xtr, ytr)

    p = pipe.predict_proba(Xte)[:,1]
    print("AUPRC:", average_precision_score(yte, p))
    print("AUROC:", roc_auc_score(yte, p))
    print(classification_report(yte, p>0.5, digits=3))

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump({"pipe":pipe, "features":FEATS}, args.model_out)
    print("✓ saved", args.model_out)

if __name__=="__main__":
    main()
