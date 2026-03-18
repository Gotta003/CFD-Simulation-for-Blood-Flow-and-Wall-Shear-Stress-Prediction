import argparse
import os
import numpy as np
from typing import List, Dict

try:
    import xgboost as xgb
    HAS_XGB=True
except ImportError:
    HAS_XGB=False
    
try:
    import torch
    import torch.nn as nn 
    HAS_TORCH=True
except ImportError:
    HAS_TORCH=False

COMPLICATIONS: Dict[str, dict]={
    "endoleak_type1": {
        "description": "Endoleak Type I",
        # ---- TO DEFINE ----
        "physical_driver": "pressure",
        "at_risk_feature": "pressure_pct_at_risk",
        # ---- TO DEFINE ----
        "class_weight": 1.0,
    },
    "endoleak_type2": {
        "description": "Endoleak Type II",
        # ---- TO DEFINE ----
        "physical_driver": "ecap",
        "at_risk_feature": "ecap_pct_at_risk",
        # ---- TO DEFINE ----
        "class_weight": 1.0,
    },
    "endoleak_type3": {
        "description": "Endoleak Type III",
        # ---- TO DEFINE ----
        "physical_driver": "wss",
        "at_risk_feature": "wss_pct_at_risk",
        # ---- TO DEFINE ----
        "class_weight": 1.0,
    },
}

COMPLICATION_KEYS: List[str]=list(COMPLICATIONS.keys())

def main():
    parser=argparse.ArgumentParser(description="Train EVAR risk predictor")
    parser.add_argument("--model", default="gnn_pinn", choices=["gnn_pinn", "xgboost"])
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--vtp-dir", default="data/vtp_files/", help="Spatial Features Directory")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--out", default="outputs/predictions/model.pkl")
    parser.add_argument("--device", default="auto")
    args=parser.parse_args()
    
    #GNN-PINN MODEL
    if args.model=="gnn_pinn":
        from src.models.gnn_pinn import (GNNPINNConfig, graph_dataset_composition, training_gnnpinn)
        cfg=GNNPINNConfig()
        print("Graph Dataset Composition...")
        dataset=graph_dataset_composition()
        if not dataset:
            print("No valid patients.")
            return 
        out=args.out.replace(".pkl", ".pt")
        training_gnnpinn()
        return
    #OTHER MODELS DOWN BELOW
    if args.model=="xgboost":
if __name__=="__main__":
    main()