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
    
from typing import Dict, List

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