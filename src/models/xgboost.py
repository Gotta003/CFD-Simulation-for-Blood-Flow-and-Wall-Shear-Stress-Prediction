from dataclasses import dataclass, field
from typing import List

@dataclass 
class XGBoostConfig:
    features_path: str="outputs/features.csv"
    labels_path: str="data/labels/outcomes.csv"
    out_path: str="outputs/predictions/xgboost"
    complications: List[str]=field()


    