import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import roc_curve, roc_auc_score
from typing import Optional, List, Dict, Tuple
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR=Path(__file__).resolve().parent.parent.parent
if BASE_DIR not in sys.path:
    sys.path.append(str(BASE_DIR))
DATAPATH=BASE_DIR/"outputs/"
DATASET_CSV=os.path.join(DATAPATH, "dataset/dataset.csv")
FEATURES_CSV=os.path.join(DATAPATH, "features/features.csv")
COLUMNS_TXT=os.path.join(DATAPATH, "dataset/dataset_columns.txt")
POINTCLOUDS_DIR=os.path.join(DATAPATH, "pointclouds/")
POINCLOUDS_VTP=os.path.join(DATAPATH, "pointclouds_vtp/")
TEST_IDS_NPY=os.path.join(DATAPATH, "splits/test_ids.npy")
CHECKPOINT_ROOT=os.path.join(DATAPATH, "checkpoint/seed2/")
OUTPUT_CSV=os.path.join(DATAPATH, "results/predictions.csv")

ENDOLEAK_COLS = [
    "endoleak_type1", "endoleak_type1a", "endoleak_type1b",
    "endoleak_type2", "endoleak_type3", "endoleak_type4",
]
OTHER_COLS = [
    "other_migration", "other_thrombosis", "other_reintervention", "other_rupture",
]
LABEL_COLS = ENDOLEAK_COLS + OTHER_COLS + ["any_endoleak"]

N_FOLDS=5
N_POINTS=8192
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_pointnet_fold(fold: int, feat_ch: int) -> Optional[torch.nn.Module]:
    from src.models.pointnet import PointNet
    ckpt=os.path.join(CHECKPOINT_ROOT, "pointnet", f"exp_fold{fold}", "best_auc.pth")
    if not os.path.exists(ckpt):
        print(f"[WARN] PointNet fold {fold} checkpoint not found: {ckpt}")
        return None
    model=PointNet(num_class=1, normal_channel=3, feat_channel=feat_ch).to(DEVICE)
    state=torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

def _load_pinn_fold(fold: int, feat_ch: int) -> Optional[torch.nn.Module]:
    from src.models.gnn_pinn import GNNPinn
    ckpt=os.path.join(CHECKPOINT_ROOT, "pinn", f"exp_fold{fold}", "best_auc.pth")
    if not os.path.exists(ckpt):
        print(f"[WARN] GNNPinn fold {fold} checkpoint not found: {ckpt}")
        return None
    model=GNNPinn(num_class=1, normal_channel=3, feat_channel=feat_ch).to(DEVICE)
    state=torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

def _load_xgboost_fold(fold: int) -> Optional[xgb.XGBClassifier]:
    ckpt=os.path.join(CHECKPOINT_ROOT, "xgboost", f"exp_fold{fold}", "best_auc.pth")
    if not os.path.exists(ckpt):
        print(f"[WARN] XGBoost fold {fold} checkpoint not found: {ckpt}")
        return None
    model=xgb.XGBClassifier()
    model.load_model(ckpt)
    return model

def load_feature_cols(columns_txt: str) -> List[str]:
    with open(columns_txt) as f:
        cols=[l.strip() for l in f if l.strip()]
    return cols

def run(force: bool = False):
    if os.path.exists(OUTPUT_CSV) and not force:
        print(f"[INFO] {OUTPUT_CSV} already eixsts. Use force to overwrite")
        return
    print("[1/6] Loading dataset")
    df=pd.read_csv(DATASET_CSV)
    df["patient_id"]=df["patient_id"].astype(str).str.strip()
    feat_df=pd.read_csv(FEATURES_CSV)
    feat_df["patient_id"]=feat_df["patient_id"].astype(str).str.strip()
    feat_cols=load_feature_cols(COLUMNS_TXT)
    test_ids=np.load(TEST_IDS_NPY, allow_pickle=True).astype(str)
    print(f" {len(test_ids)} test patients")

    n_feat=len(feat_cols)
    print(f"[2/6] Loading PointNet checkpoints")
    pn_models=[_load_pointnet_fold(f, n_feat) for f in range(N_FOLDS)]
    print("[3/6] Loading GNNPinn checkpoints")
    pinn_models=[_load_pinn_fold(f, n_feat) for f in range(N_FOLDS)]
    print("[4/6] Loading XGBoost checkpoints")
    xgb_models=[_load_xgboost_fold(f) for f in range(N_FOLDS)]
    pn_ok=sum(m is not None for m in pn_models)
    pinn_ok=sum(m is not None for m in pinn_models)
    xgb_ok=sum(m is not None for m in xgb_models)
    print(f"Loaded: PointNet {pn_ok}/{N_FOLDS}, GNNPinn {pinn_ok}/{N_FOLDS}, XGBoost {xgb_ok}/{N_FOLDS}")
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--datapath", default=DATAPATH, help="Path to the data directory")
    parser.add_argument("--force", action="store_true", help="Whether to force re-run inference even if predictions.csv exists")
    args=parser.parse_args()
    DATAPATH=args.datapath
    DATASET_CSV=os.path.join(DATAPATH, "dataset/dataset.csv")
    FEATURES_CSV=os.path.join(DATAPATH, "features/features.csv")
    COLUMNS_TXT=os.path.join(DATAPATH, "dataset/dataset_columns.txt")
    POINTCLOUDS_DIR=os.path.join(DATAPATH, "pointclouds/")
    POINCLOUDS_VTP=os.path.join(DATAPATH, "pointclouds_vtp/")
    TEST_IDS_NPY=os.path.join(DATAPATH, "splits/test_ids.npy")
    CHECKPOINT_ROOT=os.path.join(DATAPATH, "checkpoint/seed2/")
    OUTPUT_CSV=os.path.join(DATAPATH, "results/predictions.csv")
    run(force=args.force)