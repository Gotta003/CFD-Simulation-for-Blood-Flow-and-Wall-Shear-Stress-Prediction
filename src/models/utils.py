import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from src.models.gnn_pinn import GNNPINNConfig
from src.models.xgboost import XGBoostConfig

def update_class_weights(complications: Dict[str, dict], y: Dict[str, np.ndarray]) -> None:
    """
    Computation of the weight for each complication (N_neg/N_pos) and store in registry
    """
    for k in complications:
        if k not in y:
            continue
        n_pos=int(y[k].sum())
        n_neg=int((y[k]==0).sum())
        if n_pos==0:
            print(f"[WARN] {k}: 0 positive cases - class_weight stays 1.0")
        else:
            complications[k]['class_weight']=round(n_neg/n_pos, 3)
            print(f"{k:25s}: {n_pos} pos/{n_neg} neg -> class_weight={complications[k]['class_weight']:.2f}")
            
def load_tabular_data(cfg: XGBoostConfig | GNNPINNConfig, keys: List[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
    """
    Load and merge features CSV and labels CSV on patient_id. Converts features from raw file
    
    Return: X: float32 array[N,F]; y: dict complication_key -> binary label array[N]; feat_cols: list feature column  names
    """
    X_df=pd.read_csv(cfg.features_path)
    y_df=pd.read_csv(cfg.labels_path)
    df=X_df.merge(y_df, on="patient_id", how="inner")
    if df.empty:
        raise ValueError("Merge on patient_id returned 0 rows. Verify patient_id values match between features and labels CSV.")
    exclude={"patient_id"} | set(keys)
    feat_cols=[c for c in df.columns if c not in exclude]
    X=df[feat_cols].values.astype(np.float32)
    y: Dict[str, np.ndarray]={}
    for k in cfg.complications:
        if k in df.columns:
            y[k]=df[k].values.astype(int)
        else:
            print(f"[WARN] Label column '{k}' not found in CSV - skipping")
    print(f"\nTabular data loaded:")
    
    