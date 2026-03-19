import argparse
import os
import numpy as np
from typing import List, Dict
from src.models import COMPLICATION_KEYS
from src.models.xgboost import XGBoostConfig
from src.models.utils import load_tabular_data, tabular_dataset
from src.models.xgboost import training_xgboost

def main():
    parser=argparse.ArgumentParser(description="Train EVAR risk predictor")
    parser.add_argument("--model", default="gnn_pinn", choices=["gnn_pinn", "xgboost"])
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--vtp-dir", default="data/vtp_files/", help="Spatial Features Directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--complications", nargs="+", default=COMPLICATION_KEYS, choices=COMPLICATION_KEYS)
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
    elif args.model=="xgboost":
        cfg=XGBoostConfig(
            features_path=args.features,
            labels_path=args.labels,
            out_path=args.out or "outputs/models/xgboost/",
            cv_folds=args.cv
        )
        X, y, feat_cols=load_tabular_data(cfg)
        dataset=tabular_dataset(X, y, cfg)
        training_xgboost(dataset, cfg, feat_cols)
        return
    
if __name__=="__main__":
    main()