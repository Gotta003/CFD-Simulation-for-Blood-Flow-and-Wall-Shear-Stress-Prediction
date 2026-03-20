from src.models.registry import COMPLICATIONS, COMPLICATION_KEYS, HAS_XGB, HAS_TORCH
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

@dataclass 
class XGBoostConfig:
    features_path: str="outputs/features.csv"
    labels_path: str="data/labels/outcomes.csv"
    out_path: str="outputs/predictions/xgboost"

def training_xgboost():
    return
