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

def _build_patient_result(pid: str, rec: Dict[str, Any]) -> PatientResult:
    complication_prob=float(rec.get("complication_prob", rec.get("confidence", 0.5)))
    risk_label=bool(int(rec.get("risk_label", complication_prob>=0.5)))
    model_name=str(rec.get("model_name", "PINN"))
    endoleak_probs: Dict[str, float]={}
    for c in ENDOLEAK_DISPLAY:
        endoleak_probs[c]=float(rec.get(f"prob_{c}", 0.0))
    other_probs: Dict[str, float]={}
    for c in OTHER_DISPLAY:
        other_probs[c]=float(rec.get(f"prob_{c}", 0.0))
    #Ground Truth
    ground_truth: Dict[str, int]={}
    for c in list(ENDOLEAK_DISPLAY.keys())+list(OTHER_DISPLAY.keys())+["any_endoleak"]:
        key=f"gt_{c}"
        if key in rec:
            ground_truth[c]=int(rec[key])
    #ROC
    try:
        fpr=np.array(json.loads(rec.get("roc_fpr", "[]")), dtype=np.float32)
        tpr=np.array(json.loads(rec.get("roc_tpr", "[]")), dtype=np.float32)
    except Exception:
        fpr=np.array([])
        tpr=np.array([])
    roc_auc=float(rec.get("roc_auc", 0.5))
    if len(fpr)==0:
        fpr=None
        tpr=None
    #Feature Groups (Dataset.csv)
    feats_groups: Dict[str, Dict[str, float]]={}
    if os.path.exists(DATASET_CSV):
        try:
            df=pd.read_csv(DATASET_CSV)
            df["patient_id"]=df["patient_id"].astype(str).str.strip()
            row_df=df[df["patient_id"]==pid]
            if not row_df.empty:
                row=row_df.iloc[0]
                cfd_group: Dict[str, float]={}
                for c in CFD_COLS:
                    cols=[c for c in df.columns if c.lower().startswith(c) and pd.api.types.is_numeric_dtype(df[c])]
                    if cols:
                        vals=[row[c] for c in cols if not pd.isna(row.get(c, float("nan")))]
                        if vals:
                            cfd_group[c.upper()]=round(float(np.mean(vals)), 4)
                morpho_group: Dict[str, float]={}
                for c in MORPHO_COLS:
                    if c in row.index and not pd.isna(row[c]):
                        morpho_group[c.replace("_", " ")]=round(float(row[c]), 4)
                if morpho_group:
                    feats_groups["Morphometric"]=morpho_group
                if cfd_group:
                    feats_groups["CFD"]=cfd_group
        except Exception:
            pass
    
    attention: Dict[str, float]={}
    for k, l in [
        ("pred_pointnet", "PointNet"),
        ("pred_pinn", "GNNPinn"),
        ("pred_xgboost", "xGBoost")
    ]:
        val=rec.get(k, float("nan"))
        try:
            v=float(val)
            if not np.isnan(v):
                attention[l]=v
        except (TypeError, ValueError):
            pass
    
    return PatientResult(
        patient_id=pid,
        risk_label=risk_label,
        model_name=model_name,
        confidence=complication_prob,
        endoleak_probs=endoleak_probs,
        ground_truth=ground_truth,
        feature_groups=feats_groups,
        attention_weights=attention,
        roc_fpr=fpr,
        roc_tpr=tpr,
        roc_auc=roc_auc,
    )
    
class WelcomePanel(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=C_BG, **kw)
        inner=tk.Frame(self, bg=C_BG)
        inner.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(inner, text="🫀", font=("Helvetica", 48), bg=C_BG).pack(pady=(0, 12))
        tk.Label(inner, text="Select a patient", font=FONT_H1, bg=C_BG, fg=C_TEXT).pack()
        tk.Label(inner, text="Click any patient in the list on the left into their ML prediction results", font=FONT_BODY, bg=C_BG, fg=C_MUTED, justify="center").pack(pady=(8, 0))
    
class EmbeddedResultsPane(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=C_BG, **kw)
        self._current: Optional[PatientResult]=None
        self._show_welcome()
        
    def _show_welcome(self):
        for child in self.winfo_children():
            child.destroy()
        WelcomePanel(self).pack(fill="both", expand=True)
        
    def load_patient(self, result: PatientResult):
        self._current=result
        for child in self.winfo_children():
            child.destroy()
        SummaryHeaderStrip(self, result).pack(fill="x")
        ColorLegendStrip(self).pack(fill="x")
        outer=tk.Frame(self, bg=C_BG)
        outer.pack(fill="both", expand=True)
        canvas=tk.Canvas(outer, bg=C_BG, highlightthickness=0)
        vscroll=ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner=tk.Frame(canvas, bg=C_BG)
        win_id=canvas.create_window((0,0), window=inner, anchor="nw")
        
        def _cfg(_e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        def _resize(e):
            canvas.itemconfig(win_id, width=e.width)
            
        inner.bind("<Configure>", _cfg)
        canvas.bind("<Configure>", _resize)
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", lambda ev: canvas.yview_scroll(int(-1*ev.delta/120), "units")))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        PAD=dict(padx=10, pady=8, sticky="nsew")
        inner.columnconfigure(0, weight=1)
        EndoleakProbCard(inner, result).grid(row=0, column=0, **PAD)
        ROCCurvePanel(inner, result).grid(row=1, column=0, padx=10, pady=(0,4), sticky="ew")
        FeatureBreakdownPanel(inner, result).grid(row=2, column=0, padx=10, pady=(0, 4), sticky="ew")
        
        btn_bar=tk.Frame(inner, bg=C_NAV_LIGHT, padx=16, pady=8)
        btn_bar.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 16))
        tk.Button(btn_bar, text="Open Window", command=lambda r=result: open_results_window(self, r), bg=C_NAV, fg=C_WHITE, font=FONT_H3, relief="flat", padx=20, pady=6).pack()
        
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AAA EVAR - ML Results Viewer")
        self.geometry("1400x900")
        self.minsize(900, 600)
        self.configure(bg=C_BG)
        self._build_ui()
    
    def _build_ui(self):
        top_bar=tk.Frame(self, bg=C_NAV, padx=20, pady=8)
        top_bar.pack(fill="x")
        tk.Label(top_bar, text="EVAR ML Results Viewer", font=FONT_H1, bg=C_NAV, fg=C_WHITE).pack(side="left")
        tk.Label(top_bar, text="Point Net - GNNPinn - XGBoost", font=FONT_SM, bg=C_NAV, fg="#AAB8D0").pack(side="right")
        paned=tk.PanedWindow(self, orient="horizontal", bg=C_BORDER, sashwidth=5, sashrelief="flat", bd=0)
        paned.pack(fill="both", expand=True)
        #Left Panel
        self._list_panel=PatientListPanel(paned, on_select=self._on_patient_select, predicitons_csv=PREDS_CSV)
        #Right Panel
        self._results_pane=EmbeddedResultsPane(paned)
        paned.add(self._results_pane, minsize=600)
        
    def _on_patient_Selected(self, pid:str, rec: Dict[str, Any]):
        try:
            result=_build_patient_result(pid, rec)
            self._results_pane.load_patient(result)
        except Exception as e:
            import traceback 
            traceback.print_exc()
            tk.messagebox.showerror("Error loading patient", f"Could not build PatientResult for {pid}:\n{e}")
            
if __name__=="__main__":
    app=App()
    app.mainloop()