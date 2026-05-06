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
    PatientResult, open_results_window, C_NAV, C_NAV_LIGHT, C_BG, C_WHITE, C_BORDER, C_GREEN, C_RED, C_ORANGE, C_MUTED, C_TEXT, FONT_H1, FONT_H2, FONT_H3, FONT_BODY, FONT_SM, ENDOLEAK_DISPLAY, OTHER_DISPLAY, SummaryHeaderStrip, ColorLegendStrip, EndoleakProbCard, ROCCurvePanel, FeatureBreakdownPanel, BaseCard, prob_colors
)

PREDS_NPY = "predizioni_multilabel_svm.npy"
LABELS_NPY = "labels_multilabel_svm.npy"
TEST_IDS_NPY = "outputs/splits/test_ids.npy" 
DATASET_CSV = "outputs/dataset/dataset.csv" 
COLUMNS_TXT="outputs/dataset/dataset_columns.txt"
METRICS_JSON="metrics_multilabel_svm.json"

CFD_COLS=["tawss", "osi", "ecap", "rrt", "pressure", "wss", "velocity", "vorticity", "divergence", "traction"]
MORPHO_COLS=["D1","D2","D3","D4","D5","Dmax","Proximal_Neck_Length","L_Curvilinear","L_Straight","Tortuosity","Deformation_Ratio","Saccularization_Index","AAA_Volume_mm3","AAA_SurfaceArea_mm2", "CL_Trunk_Length_mm","CL_AAA_Length_mm","CL_AAA_Straight_mm","CL_AAA_Tortuosity", "CL_Prox_Neck_Length_mm","CL_AAA_MeanCurvature","CL_AAA_MaxCurvature", "CL_AAA_MeanTorsion","CL_AAA_MeanRadius_mm","CL_AAA_MaxRadius_mm"]

#Labels
IDX_ENDOLEAK_START=0
IDX_ENDOLEAK_END=IDX_ENDOLEAK_START+len(ENDOLEAK_DISPLAY)
IDX_OTHER_START=IDX_ENDOLEAK_END
IDX_OTHER_END=IDX_OTHER_START+len(OTHER_DISPLAY)
IDX_ANY_ENDOLEAK=IDX_OTHER_END

def _auc_calc(m: dict, idx: int)-> float:
    v=m.get("auc", {}).get(str(idx), 0.0)
    try:
        v=float(v)
        return 0.0 if (v is None or np.isnan(v)) else v
    except Exception:
        return 0.0

def _roc_calc(m: dict, idx: int):
    fpr=np.nan_to_num(np.array(m.get("fpr", {}).get(str(idx), [0.0, 1.0]), dtype=np.float64))
    tpr=np.nan_to_num(np.array(m.get("tpr", {}).get(str(idx), [0.0, 1.0]), dtype=np.float64))
    valid=np.isfinite(fpr) & np.isfinite(tpr)
    if valid.sum()<2:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    return fpr[valid], tpr[valid]

def _build_patient_result(pid: str, rec: Dict[str, Any]) -> PatientResult:
    try:
        with open(METRICS_JSON, "r") as f:
            m=json.load(f)
    except Exception:
        m={"fpr": {}, "tpr": {}, "thresholds": {}, "auc": {}, "macro_auc": 0.0}

    full_probs=np.array(rec.get("svm_probs", np.zeros(11)), dtype=np.float64)
    full_gt=np.array(rec.get("gt_labels", np.zeros(10)), dtype=np.float64)
    endoleak_keys=list(ENDOLEAK_DISPLAY.keys())
    other_keys=list(OTHER_DISPLAY.keys())
    endoleak_probs: Dict[str, float]={}
    other_probs: Dict[str, float]={}
    ground_truth: Dict[str, int]={}
    for i, key in enumerate(endoleak_keys):
        idx=IDX_ENDOLEAK_START+i
        if idx>=len(full_probs):
            break
        endoleak_probs[key]=float(full_probs[idx])
        ground_truth[key]=int(full_gt[idx])
    for j, key in enumerate(other_keys):
        idx=IDX_OTHER_START+j
        if idx>=len(full_probs):
            break
        other_probs[key]=float(full_probs[idx])
        ground_truth[key]=int(full_gt[idx])
    if IDX_ANY_ENDOLEAK<len(full_gt):
        ground_truth["any_endoleak"]=int(full_gt[IDX_ANY_ENDOLEAK])
    complication_prob=float(full_probs[IDX_ANY_ENDOLEAK]) if len(full_probs)>0 else 0.0
    risk_label=(complication_prob>=0.5)
    best_idx=None
    best_auc=0.0
    for idx in range(IDX_ENDOLEAK_START, min(IDX_OTHER_END, len(full_probs))):
        a=_auc_calc(m, idx)
        if a>best_auc:
            best_auc=a
            best_idx=idx
    if best_idx is None:
        fpr_out=np.array([0.0, 1.0])
        tpr_out=np.array([0.0, 1.0])
        auc_val=0.0
    else:
        fpr_out, tpr_out=_roc_calc(m, best_idx)
        auc_val=best_auc
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
                for col in CFD_COLS:
                    cols=[c for c in df.columns if c.lower().startswith(col.lower()) and pd.api.types.is_numeric_dtype(df[c])]
                    if cols:
                        vals=[row[c] for c in cols if not pd.isna(row.get(c, float("nan")))]
                        if vals:
                            cfd_group[col.upper()]=round(float(np.mean(vals)), 4)
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
    
    return PatientResult(
        patient_id=pid,
        risk_label=risk_label,
        model_name="SVM Multilabel",
        confidence=complication_prob,
        endoleak_probs=endoleak_probs,
        other_probs=other_probs,
        ground_truth=ground_truth,
        feature_groups=feats_groups,
        roc_fpr=fpr_out,
        roc_tpr=tpr_out,
        roc_auc=auc_val,
    )

class AdverseEventsProbCard(BaseCard):
    def __init__(self, parent, result: PatientResult, **kw):
        super().__init__(parent, title="Adverse Events Risk", **kw)
        b=self.body
        grid=tk.Frame(b, bg=C_WHITE)
        grid.pack(fill="x")
        grid.columnconfigure((0, 1), weight=1)
        items=list(result.other_probs.items())
        for i, (c, p) in enumerate(items):
            row_i, col_i=divmod(i, 2)
            fg, bg=prob_colors(p)
            gt=result.ground_truth.get(c, None)
            display_name=OTHER_DISPLAY.get(c, c)
            self._make_pill(grid, display_name, p, fg, bg, gt, row_i, col_i)
    
    def _make_pill(self, parent, label, prob, fg, bg, gt, row_i, col_i):
        pill=tk.Frame(parent, bg=bg, padx=12, pady=10, highlightbackground=fg, highlightthickness=1)
        pill.grid(row=row_i, column=col_i, padx=6, pady=6, sticky="nsew")
        tk.Label(pill, text=label, font=FONT_H3, bg=bg, fg=fg).pack()
        tk.Label(pill, text=f"{prob*100:.1f}%", font=FONT_H1, bg=bg, fg=fg).pack(pady=(2,0))
        bar_c=tk.Canvas(pill, height=6, bg="#DDDDDD", highlightthickness=0)
        bar_c.pack(fill="x", pady=(4,0))
        bar_c.bind("<Configure>", lambda e, c=bar_c, v=prob, col=fg: self._draw_bar(c, v, col))
        if gt is not None:
            gt_txt="Confirmed" if gt else "Not Confirmed"
            gt_color=C_GREEN if gt else C_MUTED
            tk.Label(pill, text=gt_txt, font=FONT_SM, bg=bg, fg=gt_color).pack(pady=(4,0))

    def _draw_bar(self, canvas, value, color):
        canvas.delete("all")
        w=canvas.winfo_width()
        h=canvas.winfo_height()
        fill_w=max(2, int(w*value))
        canvas.create_rectangle(0, 0, fill_w, h, fill=color, outline="")

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
        AdverseEventsProbCard(inner, result).grid(row=1, column=0, **PAD)
        ROCCurvePanel(inner, result).grid(row=2, column=0, padx=10, pady=(0,4), sticky="ew")
        FeatureBreakdownPanel(inner, result).grid(row=3, column=0, padx=10, pady=(0, 4), sticky="ew")
        
        btn_bar=tk.Frame(inner, bg=C_NAV_LIGHT, padx=16, pady=8)
        btn_bar.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 16))
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
        paned=tk.PanedWindow(self, orient="horizontal", bg=C_BORDER, sashwidth=5, sashrelief="flat", bd=0)
        paned.pack(fill="both", expand=True)
        #Left Panel
        self._list_panel=PatientListPanel(paned, on_select=self._on_patient_select)
        paned.add(self._list_panel, minsize=260, width=310)
        #Right Panel
        self._results_pane=EmbeddedResultsPane(paned)
        paned.add(self._results_pane, minsize=600)
        
    def _on_patient_select(self, pid:str, rec: Dict[str, Any]):
        try:
            result=_build_patient_result(pid, rec)
            self._results_pane.load_patient(result)
        except Exception as e:
            tk.messagebox.showerror("Error loading patient", f"Could not build PatientResult for {pid}:\n{e}")
            
if __name__=="__main__":
    app=App()
    app.mainloop()