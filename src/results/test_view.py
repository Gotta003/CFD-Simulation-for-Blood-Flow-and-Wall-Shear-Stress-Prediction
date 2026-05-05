import os
import sys
import json
import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

from patient_list_panel import PatientListPanel
from results_view import (
    PatientResult, open_results_window, C_NAV, C_NAV_LIGHT, C_BG, C_WHITE, C_BORDER, C_GREEN, C_RED, C_ORANGE, C_MUTED, C_TEXT, FONT_H1, FONT_H2, FONT_H3, FONT_BODY, FONT_SM, ENDOLEAK_DISPLAY, OTHER_DISPLAY, SummaryHeaderStrip, ColorLegendStrip, EndoleakProbCard, ROCCurvePanel, FeatureBreakdownPanel
)

PREDS_CSV="outputs/predictions.csv"
DATASET_CSV="outputs/dataset/dataset.csv"
COLUMNS_TXT="outputs/dataset/dataset_columns.txt"

CFD_COLS=["tawss", "osi", "ecap", "rrt", "pressure", "wss", "velocity", "vorticity", "divergence", "traction"]
MORPHO_COLS=["D1","D2","D3","D4","D5","Dmax","Proximal_Neck_Length","L_Curvilinear","L_Straight","Tortuosity","Deformation_Ratio","Saccularization_Index","AAA_Volume_mm3","AAA_SurfaceArea_mm2", "CL_Trunk_Length_mm","CL_AAA_Length_mm","CL_AAA_Straight_mm","CL_AAA_Tortuosity", "CL_Prox_Neck_Length_mm","CL_AAA_MeanCurvature","CL_AAA_MaxCurvature", "CL_AAA_MeanTorsion","CL_AAA_MeanRadius_mm","CL_AAA_MaxRadius_mm"]