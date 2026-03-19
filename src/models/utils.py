import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from src.models.gnn_pinn import GNNPINNConfig
from src.models.xgboost import XGBoostConfig
from src.models import COMPLICATIONS, COMPLICATION_KEYS

def update_class_weights(y: Dict[str, np.ndarray]) -> None:
    """
    Computation of the weight for each complication (N_neg/N_pos) and store in registry
    """
    for k in COMPLICATIONS:
        if k not in y:
            continue
        n_pos=int(y[k].sum())
        n_neg=int((y[k]==0).sum())
        if n_pos==0:
            print(f"[WARN] {k}: 0 positive cases - class_weight stays 1.0")
        else:
            COMPLICATIONS[k]['class_weight']=round(n_neg/n_pos, 3)
            print(f"{k:25s}: {n_pos} pos/{n_neg} neg -> class_weight={COMPLICATIONS[k]['class_weight']:.2f}")
            
def load_tabular_data(cfg: XGBoostConfig | GNNPINNConfig) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
    """
    Load and merge features CSV and labels CSV on patient_id. Converts features from raw file
    
    Return: X: float32 array[N,F]; y: dict complication_key -> binary label array[N]; feat_cols: list feature column  names
    """
    X_df=pd.read_csv(cfg.features_path)
    y_df=pd.read_csv(cfg.labels_path)
    df=X_df.merge(y_df, on="patient_id", how="inner")
    if df.empty:
        raise ValueError("Merge on patient_id returned 0 rows. Verify patient_id values match between features and labels CSV.")
    exclude={"patient_id"} | set(COMPLICATION_KEYS)
    feat_cols=[c for c in df.columns if c not in exclude]
    X=df[feat_cols].values.astype(np.float32)
    y: Dict[str, np.ndarray]={}
    for k in COMPLICATIONS:
        if k in df.columns:
            y[k]=df[k].to_numpy().astype(np.int32)
        else:
            print(f"[WARN] Label column '{k}' not found in CSV - skipping")
    print(f"\nTabular data loaded:")
    print(f" Patients: {len(df)}")
    print(f" Features: {len(feat_cols)}")
    for k, l in y.items():
        print(f" {k:25s}: {int(l.sum())} positives ({100*l.mean():.1f}%)")
    update_class_weights(y)
    return X, y, feat_cols
    
def tabular_dataset(X: np.ndarray, y: Dict[str, np.ndarray], cfg: XGBoostConfig | GNNPINNConfig) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compose per-complication (structure X,y) pairs ready for model. 

    Returns:
        dataset: dict with each complication_key -> (X[N,F], y_binary[N])
    """
    dataset: Dict[str, Tuple[np.ndarray, np.ndarray]]={}
    for k in COMPLICATIONS:
        if k not in y:
            continue
        labels=y[k]
        n_pos=int(labels.sum())
        if n_pos==0:
            print(f"    [SKIP] {k} - no positive cases, not trainable")
            continue
        if n_pos<3:
            print(f" [WARN] {k} - only {n_pos} positive cases")
        dataset[k]=(X,labels)
    print(f"\nDataset composed: {len(dataset)}/{len(COMPLICATIONS)} trainable\n")
    return dataset

def safe_cv_folds(labels: np.ndarray, requested: int) -> int:
    n_pos=int(labels.sum())
    n_neg=int((labels==0).sum())
    safe=min(requested, n_pos, n_neg)
    if safe<requested:
        print(f"    [WARN] Reducing CV folds {requested} -> {safe} (only {n_pos} positive cases)")
    return max(2, safe)

def print_fold_summary(mean_va: float, std_va: float, mean_tr: float) -> None:
    print("-"*50)
    print("Mean val AUC: {mean_va:.3f} +- {std_va:.3f}")
    print(f"Mean train AUC: {mean_tr:.3f} (overfit gap={mean_tr-mean_va:.3f})")
    
def print_summary(model_name: str, results: Dict[str, dict]) -> None:
    print("\n"+"="*50)
    print(f"{model_name} - Summary")
    print("="*50)
    for k, r in results.items():
        bar="█"*int(r["val_auc"]*20)
        print(f"    {k:25s} {bar:<20}   AUC={r['val_auc']:.3f} +- {r['val_auc_std']:.3f}")
    print("="*50)