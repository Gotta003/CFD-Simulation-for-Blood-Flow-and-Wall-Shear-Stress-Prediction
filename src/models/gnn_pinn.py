from dataclasses import dataclass, field
from typing import List

@dataclass
class GNNPINNConfig:
    features_path: str="outputs/features.csv"
    labels_path: str="data/labels/outcomes.csv"
    out_path: str="outputs/predictions/xgboost"
    
def vtp_to_graph():
    return

def graph_dataset_composition():
    return

def training_gnnpinn():
    return