from dataclasses import dataclass, field
from typing import List

COMPLICATIONS: Dict[str, dict]={
    "endoleak_type1": {
        "description": "Endoleak Type I",
        # ---- TO DEFINE ----
        "physical_driver": "pressure",
        # ---- TO DEFINE ----
        "class_weight": 1.0,
    },
    "endoleak_type2": {
        "description": "Endoleak Type II",
        # ---- TO DEFINE ----
        "physical_driver": "pressure",
        # ---- TO DEFINE ----
        "class_weight": 1.0,
    },
    "endoleak_type3": {
        "description": "Endoleak Type III",
        # ---- TO DEFINE ----
        "physical_driver": "pressure",
        # ---- TO DEFINE ----
        "class_weight": 1.0,
    },
    
}

@dataclass
class GNNPINNConfig:
    node_features: List[str]=field
    
def vtp_to_graph():
    return

def graph_dataset_composition():
    return

def training_gnnpinn():
    return