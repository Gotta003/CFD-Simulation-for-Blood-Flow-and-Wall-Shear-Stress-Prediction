from src.models.registry import COMPLICATIONS, COMPLICATION_KEYS, HAS_XGB, HAS_TORCH
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import os
import joblib
import numpy as np
from src.models.utils import safe_cv_folds, print_fold_summary, print_summary

from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

@dataclass 
class XGBoostConfig:
    features_path: str="outputs/features.csv"
    labels_path: str="data/labels/outcomes.csv"
    out_path: str="outputs/predictions/xgboost"
    nun_estimators: int=300
    max_depth: int=4
    learning_rate: float=0.05
    subsample: float=0.8
    colsample: float=0.8
    cv_folds: int=5
    random_state: int=42
    
def build_xgb_pipeline(cfg: XGBoostConfig, pos_weight: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", xgb.XGBClassifier(
            n_estimators=cfg.nun_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample,
            scale_pos_weight=pos_weight,
            eval_metric="auc",
            random_state=cfg.random_state,
            verbosity=0
        ))
    ])

def training_xgboost(dataset: Dict[str, Tuple[np.ndarray, np.ndarray]], cfg: XGBoostConfig, feat_cols: List[str]) -> Dict[str, dict]:
    """
    Train a Classifier per complification, then refit on full data and save
   
    Returns:
        results: dict key of complication -> [val_auc, val_auc_std, train_auc, model_path]
    """
    if not HAS_XGB:
        raise ImportError("xgboost not installed correctly")
    os.makedirs(cfg.out_path, exist_ok=True)
    results: Dict[str, dict]={}
    for k, (X, labels) in dataset.items():
        pos_w=COMPLICATIONS[k]["class_weight"]
        n_splits=safe_cv_folds(labels, cfg.cv_folds)
        print(f"{k} [XGBoost | {int(labels.sum())} pos | pos_weight={pos_w:.2f} | {n_splits}-foldsCV]")
        
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)
        aucs_va, aucs_tr=[],[]
        for fold, (tr_idx, va_idx) in enumerate(cv.split(X, labels)):
            pipeline=build_xgb_pipeline(cfg, pos_w)
            pipeline.fit(X[tr_idx, labels[tr_idx]])
            p_va=pipeline.predict_proba(X[va_idx])[:,1]
            p_tr=pipeline.predict_proba(X[tr_idx])[:,1]
            auc_va=roc_auc_score(labels[va_idx], p_va)
            auc_tr=roc_auc_score(labels[tr_idx], p_tr)
            aucs_va.append(auc_va)
            aucs_tr.append(auc_tr)
            print(f"Fold {fold+1}/{n_splits} val_AUC={auc_va:.3f} train_AUC={auc_tr:.3f}")
        
        mean_va, std_va=float(np.mean(aucs_va)), float(np.std(aucs_va))
        mean_tr=float(np.mean(aucs_tr))
        print_fold_summary(mean_va, std_va, mean_tr)
        
        #Refiting on full dataset
        
        final=build_xgb_pipeline(cfg, pos_w)
        final.fit(X,labels)
        model_path=os.path.join(cfg.out_path, f"{k}.pkl")
        joblib.dump({"model": final, "features": feat_cols}, model_path)
        print(f"Saved -> {model_path}\n")
        
        results[k]={
            "val_auc": mean_va,
            "val_auc_std": std_va,
            "train_auc": mean_tr,
            "model_path": model_path
        }
    print_summary("XGBoost", results)
    return results