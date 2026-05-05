import os
import tkinter as tk
import tkinter.ttk as ttk
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

C_NAV="#1A2A4A"
C_NAV_LIGHT="#2C3E6B"
C_BG="#F4F4F4"
C_WHITE="#FFFFFF"
C_BORDER="#DDE0E3"
C_GREEN="#27AE60"
C_GREEN_BG="#E8F8F0"
C_RED="#C0392B"
C_RED_BG="#FDE8E8"
C_ORANGE="#E67E22"
C_ORANGE_BG="#FEF5E7"
C_BLUE="#2980B9"
C_TEXT="#2C3E50"
C_MUTED="#7F8C8D"

FONT_H1=("Helvetica", 14, "bold")
FONT_H2=("Helvetica", 11, "bold")
FONT_H3=("Helvetica", 10, "bold")
FONT_BODY=("Helvetica", 10)
FONT_SM=("Helvetica", 8)

CFD_COLS=["tawss", "osi", "ecap", "rrt", "pressure", "wss", "velocity", "vorticity", "divergence", "traction"]

MORPHO_COLS=["D1", "D2", "D3", "D4", "D5", "Dmax", "Proximal_Neck_Length", "L_Curvilinear", "L_Straight", "Tortuosity", "Deformation_Ratio", "Saccularization_Index", "AAA_Volume_mm3", "AAA_SurfaceArea_mm2", "CL_Trunk_Length_mm", "CL_AAA_Length_mm", "CL_AAA_Straight_mm","CL_AAA_Tortuosity", "CL_Prox_Neck_Length_mm", "CL_AAA_MeanCurvature", "CL_AAA_MaxCurvature", "CL_AAA_MeanTorsion", "CL_AAA_MeanRadius_mm", "CL_AAA_MaxRadius_mm"]

ENDOLEAK_DISPLAY = {
    "endoleak_type1":  "Type I",
    "endoleak_type1a": "Type IA",
    "endoleak_type1b": "Type IB",
    "endoleak_type2":  "Type II",
    "endoleak_type3":  "Type III",
    "endoleak_type4":  "Type IV",
}

OTHER_DISPLAY = {
    "other_migration":      "Graft Migration",
    "other_thrombosis":     "Thrombosis",
    "other_reintervention": "Reintervention",
    "other_rupture":        "Rupture",
}

def prob_colors(prob: float) -> tuple[str, str]:
    if prob>=0.7:
        return C_RED, C_RED_BG
    if prob>=0.5:
        return C_ORANGE, C_ORANGE_BG
    return C_GREEN, C_GREEN_BG

@dataclass
class PatientResult:
    patient_id: str
    risk_label: bool
    model_name: str
    confidence: float
    endoleak_probs: Dict[str, float]
    other_probs: Dict[str, float]
    ground_truth: Dict[str, int]=field(default_factory=dict)
    feature_groups: Dict[str, Dict[str, float]]=field(default_factory=dict)
    attention_weights: Dict[str, float]=field(default_factory=dict)
    survival_times: List[float]=field(default_factory=list)
    survival_probs: List[float]=field(default_factory=list)
    roc_fpr: Optional[np.ndarray]=None
    roc_tpr: Optional[np.ndarray]=None
    roc_auc: float=0.0

    @classmethod
    def from_dataset(
        cls,
        dataset_csv: str,
        patient_id: str,
        model_name: str,
        model_predictions: Dict[str, float],
        attention_weights: Dict[str, float]=None,
        survival_times: List[float]=None,
        survival_probs: List[float]=None,
        roc_fpr: np.ndarray=None,
        roc_tpr: np.ndarray=None,
        roc_auc: float=0.0
    ) -> "PatientResult":
        if not os.path.exists(dataset_csv):
            raise FileNotFoundError(f"Dataset not found: {dataset_csv}")
        df=pd.read_csv(dataset_csv)
        df["patient_id"]=df["patient_id"].astype(str).str.strip()
        pid_norm=patient_id.strip()
        row_df=df[df["patient_id"]==pid_norm]
        if row_df.empty:
            raise ValueError(
                f"Patient {pid_norm} not found in dataset.\nAvailable IDs: {df['patient_id'].tolist()[:10]}"
            )
        row=row_df.iloc[0]
        all_label_cols=list(ENDOLEAK_DISPLAY.keys())+list(OTHER_DISPLAY.keys())+["any_endoleak"]
        ground_truth={
            col: int(row[col]) for col in all_label_cols if col in row.index
        }
        endoleak_probs={
            col: float(model_predictions.get(col, 0.0)) for col in ENDOLEAK_DISPLAY
        }
        other_probs={
            col: float(model_predictions.get(col, 0.0)) for col in OTHER_DISPLAY
        }
        confidence=float(model_predictions.get("any_endoleak", max(endoleak_probs.values()) if endoleak_probs else 0.0))
        risk_label=(confidence>=0.5)
        
        cfd_group: Dict[str, float]={}
        for col in CFD_COLS:
            cols=[c for c in df.columns if c.lower().startswith(col) and pd.api.types.is_numeric_dtype(df[c])]
            if cols:
                vals=[row[c] for c in cols if not pd.isna(row.get(c, np.nan))]
                if vals:
                    cfd_group[col.upper()]=round(float(np.mean(vals)), 4)
                    
        morpho_group: Dict[str, float]={}
        for col in MORPHO_COLS:
            if col in row.index and not pd.isna(row[col]):
                morpho_group[col.replace("_", " ")]=round(float(row[col]), 4)
        feature_groups: Dict[str, Dict[str, float]]={}
        if morpho_group:
            feature_groups["Morphometric"]=morpho_group
        if cfd_group:
            feature_groups["CFD"]=cfd_group
        return cls(
            patient_id=pid_norm,
            risk_label=risk_label,
            confidence=confidence,
            endoleak_probs=endoleak_probs,
            other_probs=other_probs,
            ground_truth=ground_truth,
            attention_weights=attention_weights or {},
            survival_times=survival_times or [],
            survival_probs=survival_probs or [],
            feature_groups=feature_groups,
            roc_fpr=roc_fpr,
            roc_tpr=roc_tpr,
            roc_auc=roc_auc,
            model_name=model_name,
        )

class BaseCard(tk.Frame):
    def __init__(self, parent, title:str, **kw):
        super().__init__(parent, bg=C_WHITE, bd=1, relief="solid", highlightbackground=C_BORDER, highlightthickness=1, **kw)
        hdr=tk.Frame(self, bg=C_NAV, padx=10, pady=6)
        hdr.pack(fill="x")
        tk.Label(hdr, text=title, font=FONT_H2, bg=C_NAV, fg=C_WHITE).pack(anchor="w")
        self.body=tk.Frame(self, bg=C_WHITE, padx=12, pady=10)
        self.body.pack(fill="both", expand=True)
            
class CollapsibleCard(BaseCard):
    def __init__(self, parent, title: str, collapsed: bool=False, **kw):
        tk.Frame.__init__(self, parent, bg=C_WHITE, bd=0, highlightbackground=C_BORDER, highlightthickness=1, **kw)
        self._collapsed=tk.BooleanVar(value=collapsed)
        hdr=tk.Frame(self, bg=C_NAV, padx=12, pady=7)
        hdr.pack(fill="x")
        tk.Label(hdr, text=title, font=FONT_H2, bg=C_NAV, fg=C_WHITE).pack(side="left")
        self._toggle_lbl=tk.Label(hdr, text="▶ Show" if collapsed else "▼ Hide", font=FONT_SM, bg=C_NAV_LIGHT, fg=C_WHITE, padx=8, pady=2, cursor="hand2")
        self._toggle_lbl.pack(side="right", padx=4)
        self._toggle_lbl.bind("<Button-1>", self._toggle)
        self.body=tk.Frame(self, bg=C_WHITE, padx=14, pady=10)
        if not collapsed:
            self.body.pack(fill="both", expand=True)
    
    def _toggle(self, _event=None):
        if self._collapsed.get():
            self.body.pack(fill="both", expand=True)
            self._collapsed.set(False)
            self._toggle_lbl.config(text="▼ Hide")
        else:
            self.body.pack_forget()
            self._collapsed.set(True)
            self._toggle_lbl.config(text="▶ Show")
                 
class ColorLegendStrip(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg="#EEF1F5", padx=16, pady=6, **kw)
        tk.Label(self, text="Probability Colour code:", font=FONT_SM, bg="#EEF1F5", fg=C_MUTED).pack(side="left", padx=(0, 12))
        tiers=[
            (C_RED, C_RED_BG, "HIGH"),
            (C_ORANGE, C_ORANGE_BG, "MEDIUM"),
            (C_GREEN, C_GREEN_BG, "LOW"),
        ]                 
        for fg, bg, label in tiers:
            pill=tk.Frame(self, bg=bg, padx=10, pady=3, highlightbackground=fg, highlightthickness=1)
            pill.pack(side="left", padx=6)
            tk.Label(pill, text=label, font=FONT_SM, bg=bg, fg=fg).pack()
            
class SummaryHeaderStrip(tk.Frame):
    """
    header with patientID, model name, color coded risk
    """
    def __init__(self, parent, result:PatientResult, **kw):
        super().__init__(parent, bg=C_NAV, padx=20, pady=12, **kw)
        tk.Label(self, text=f"ML Results - Patient {result.patient_id}", font=FONT_H1, bg=C_NAV, fg=C_WHITE).pack(side="left")
        pill_color=C_RED if result.risk_label else C_GREEN
        pill_text="HIGH RISK" if result.risk_label else "LOW RISK"
        tk.Label(self, text=pill_text, font=FONT_H2, bg=pill_color, fg=C_WHITE, padx=12, pady=4).pack(side="right", padx=8)
        tk.Label(self, text=result.model_name, font=FONT_SM, bg=C_NAV, fg="#AAB8D0").pack(side="right", padx=16)

class EndoleakProbCard(BaseCard):
    def __init__(self, parent, result: PatientResult, **kw):
        super().__init__(parent, title="Endoleak Risk by Type", **kw)
        b=self.body
        ov_fg, ov_bg=prob_colors(result.confidence)
        banner=tk.Frame(b, bg=ov_bg, padx=14, pady=8, highlightbackground=ov_fg, highlightthickness=1)
        banner.pack(fill="x", pady=(0, 12))
        label_txt="HIGH RISK" if result.risk_label else "LOW RISK"
        tk.Label(banner, text=f"Overall: {label_txt}", font=FONT_H1, bg=ov_bg, fg=ov_fg).pack(side="left")
        tk.Label(banner, text=f"P(any endoleak)={result.confidence:.3f}", font=FONT_BODY, bg=ov_bg, fg=ov_fg).pack(side="right")
        grid=tk.Frame(b, bg=C_WHITE)
        grid.pack(fill="x")
        grid.columnconfigure((0, 1), weight=1)
        items=list(result.endoleak_probs.items())
        for i, (c, p) in enumerate(items):
            row_i, col_i=divmod(i, 2)
            fg, bg=prob_colors(p)
            gt=result.ground_truth.get(c, None)
            self._make_pill(grid, ENDOLEAK_DISPLAY[c], p, fg, bg, gt, row_i, col_i)
    
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
   
class RiskBadgeCard(BaseCard):
    def __init__(self, parent, result: PatientResult, **kw):
        super().__init__(parent, title="Risk Prediction", **kw)
        b=self.body
        #Badge + Confidence
        top=tk.Frame(b, bg=C_WHITE)
        top.pack(fill="x", pady=(0, 10))
        badge_color=C_RED if result.risk_label else C_GREEN
        badge_text="HIGH RISK" if result.risk_label else "LOW RISK"
        badge_bg="#FDE8E8" if result.risk_label else "#E8F8F0"
        badge_frame=tk.Frame(top, bg=badge_color, padx=18, pady=10)
        badge_frame.pack(side="left")
        tk.Label(badge_frame, text=badge_text, font=FONT_H1, bg=badge_color, fg=C_WHITE).pack()
        tk.Label(badge_frame, text=f"Confidence: {result.confidence*100:.1f}%", font=FONT_SM, bg=badge_color, fg=C_WHITE).pack()
        #Bar
        bar_frame=tk.Frame(top, bg=C_WHITE, padx=15)
        bar_frame.pack(side="left", fill="both", expand=True)
        tk.Label(bar_frame, text="Model Confidence", font=FONT_H3, bg=C_WHITE, fg=C_TEXT).pack(anchor="w")
        canvas=tk.Canvas(bar_frame, height=22, bg="#ECF0F1", highlightthickness=0)
        canvas.pack(fill="x", pady=4)
        canvas.bind("<Configure>", lambda e, c=canvas, v=result.confidence, col=badge_color: self._draw_bar(c, v, col))
        tk.Label(bar_frame, text=f"P(endoleak)={result.confidence:.3f} Threshold=?", font=FONT_SM, bg=C_WHITE, fg=C_MUTED).pack(anchor="w")
        tk.Frame(b, bg=C_BORDER, height=1).pack(fill="x", pady=6)
        tk.Label(b, text="Per-Type Endoleak Probability", font=FONT_H3, bg=C_WHITE, fg=C_TEXT).pack(anchor="w", pady=(0,6))
        for l, p in result.endoleak_probs.items():
            row=tk.Frame(b, bg=C_WHITE)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=f"{l}:", font=FONT_BODY, bg=C_WHITE, fg=C_TEXT, width=10, anchor="w").pack(side="left")
            bar_c=tk.Canvas(row, height=16, bg="#ECF0F1", highlightthickness=0)
            bar_c.pack(side="left", fill="x", expand=True, padx=(4, 8))
            pct_lbl=tk.Label(row, text=f"{p*100:.1f}%", font=FONT_SM, bg=C_WHITE, fg=C_MUTED, width=5)
            pct_lbl.pack(side="right")
            color=C_RED if p>=0.5 else (C_ORANGE if p>=0.3 else C_GREEN)
            bar_c.bind("<Configure>", lambda e, c=bar_c, v=p, col=color: self._draw_bar(c, v, col))
        
    def _draw_bar(self, c, v, col):
        c.delete("all")
        w=c.winfo_width()
        h=c.winfo_height()
        fill_w=max(4, int(w*v))
        c.create_rectangle(0, 0, fill_w, h, fill=col, outline="")
        c.create_text(fill_w+4, h//2, text="", anchor="w")
 
class FeatureBreakdownPanel(CollapsibleCard):
    def __init__(self, parent, result: PatientResult, **kw):
        super().__init__(parent, title="Input Feature Details", collapsed=True, **kw)
        columns=("Feature", "Value")
        tree=ttk.Treeview(self.body, columns=columns, show="headings", height=12)
        tree.heading("Feature", text="Feature")
        tree.heading("Value", text="Value")
        tree.column("Feature", width=230, anchor="w")
        tree.column("Value", width=120, anchor="e")
        style=ttk.Style()
        style.configure("Treeview", font=FONT_BODY, rowheight=22, background=C_WHITE, fieldbackground=C_WHITE)
        style.configure("Treeview.Heading", font=FONT_H3, background=C_BG, foreground=C_TEXT)
        scroll=ttk.Scrollbar(self.body, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        tree.pack(side="left", fill="both", expand=True)
        tree.tag_configure("group", font=FONT_H3, background="#EAF0FB", foreground=C_NAV)
        tree.tag_configure("normal", background=C_WHITE)
        tree.tag_configure("alt", background="#F7F9FC")
        row_idx=0
        for g_name, feats in result.feature_groups.items():
            tree.insert("", "end", values=(f"   {g_name}", ""), tags=("group",))
            for f_name, v in feats.items():
                tag="normal" if row_idx%2==0 else "alt"
                formatted=f"{v:,.4g}" if isinstance(v, (int, float)) else str(v)
                tree.insert("", "end", values=(f"   {f_name}", formatted), tags=(tag,))
                row_idx+=1

class ROCCurvePanel(CollapsibleCard):
    def __init__(self, parent, result: PatientResult, **kw):
        super().__init__(parent, title=f"ROC Curve (AUC={result.roc_auc:.3f})", collapsed=True, **kw)
        fpr=result.roc_fpr
        tpr=result.roc_tpr
        if fpr is None or tpr is None:
            fpr=np.linspace(0,1,100)
            tpr=np.clip(fpr**(1/max(result.roc_auc*2, 0.5)), 0, 1)
        fig, ax=plt.subplots(figsize=(3.8, 3.0))
        fig.patch.set_facecolor(C_WHITE)
        ax.set_facecolor(C_WHITE)
        ax.plot([0, 1], [0, 1], "--", color=C_MUTED, linewidth=0.8)
        ax.plot(fpr, tpr, color=C_NAV, linewidth=2, label=f"Model AUC={result.roc_auc:.3f}")
        ax.fill_between(fpr, tpr, alpha=0.08, color=C_NAV)
        ax.set_xlabel("False Positive Rate", fontsize=8, color=C_MUTED)
        ax.set_ylabel("True Positive Rate", fontsize=8, color=C_MUTED)
        ax.tick_params(labelsize=7, colors=C_TEXT)
        ax.legend(fontsize=7, loc="lower right", framealpha=0.9, edgecolor=C_BORDER)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(pad=0.5)
        
        canvas=FigureCanvasTkAgg(fig, master=self.body)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)
        tk.Label(self.body, text=f"Model: {result.model_name}", font=FONT_SM, bg=C_WHITE, fg=C_MUTED).pack(anchor="w", pady=(4,0))
        
class ResultsWindow(tk.Toplevel):
    """TopLevel window with two-column

    Column Left -> RiskBadgeCard | FeatureBreakdownPanel
    Column Right -> AttentionBarChart | ROCCurvePAnel | SurivalCurvePanel
    """
    def __init__(self, parent, result: PatientResult):
        super().__init__(parent)
        self.title(f"ML Results - Patient {result.patient_id}")
        self.geometry("1200x860")
        self.configure(bg=C_BG)
        self.resizable(True, True)
        self.minsize(860, 600)
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
        
        def _on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        def _on_canvas_resize(event):
            canvas.itemconfig(win_id, width=event.width)
            
        inner.bind("<Configure>", _on_configure)
        canvas.bind("<Configure>", _on_canvas_resize)
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", lambda ev: canvas.yview_scroll(int(-1*(ev.delta/120)), "units")))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("MouseWheel>"))
        #Columns Section
        inner.columnconfigure(0, weight=3)
        inner.columnconfigure(1, weight=2)
        PAD=dict(padx=10, pady=8, sticky="nsew")
        #Column Left
        EndoleakProbCard(inner, result).grid(row=0, column=0, columnspan=2, **PAD)
        
        ROCCurvePanel(inner, result).grid(row=2, column=0, columnspan=2, padx=10, pady=(0,4), sticky="ew")
        FeatureBreakdownPanel(inner, result).grid(row=3, column=0, columnspan=2, padx=10, pady=(0,4), sticky="ew")
        #Column Right
        
        footer=tk.Frame(inner, bg=C_NAV_LIGHT, padx=16, pady=8)
        footer.grid(row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=(0,16))
        tk.Button(footer, text="Close", command=self.destroy, bg=C_NAV, fg=C_WHITE, font=FONT_H3, relief="flat", padx=20, pady=6).pack()
        
def open_results_window(parent: tk.Widget, result: PatientResult) -> ResultsWindow:
    win=ResultsWindow(parent, result)
    win.grab_set()
    win.focus_force()
    return win
 
if __name__=="__main__":
    roc_fpr=np.linspace(0, 1, 80)
    roc_tpr=np.clip(1-(1-roc_fpr)**3.5, 0, 1)
    results=PatientResult(
        patient_id="pz001",
        risk_label=True,
        model_name="CFD + PCNN + Morphometric",
        confidence=0.83,
        endoleak_probs = {
            "endoleak_type1":  0.83,
            "endoleak_type1a": 0.76,
            "endoleak_type1b": 0.62,
            "endoleak_type2":  0.21,
            "endoleak_type3":  0.05,
            "endoleak_type4":  0.03,
        },
        other_probs = {
            "other_migration":      0.08,
            "other_thrombosis":     0.12,
            "other_reintervention": 0.55,
            "other_rupture":        0.04,
        },
        ground_truth = {
            "endoleak_type1":  1,
            "endoleak_type1a": 1,
            "endoleak_type1b": 0,
            "endoleak_type2":  0,
            "endoleak_type3":  0,
            "endoleak_type4":  0,
            "other_migration": 0,
            "other_thrombosis": 0,
            "other_reintervention": 0,
            "other_rupture":   0,
            "any_endoleak":    1,
        },
        feature_groups={
            "Morphometric": {
                "Dmax (mm)":            118.13,
                "Tortuosity":           1.116,
                "AAA Volume (mm³)":     125864,
                "Proximal Neck (mm)":   14.3,
                "Deformation Ratio":    0.117,
                "Mean Curvature":       0.0221,
                "Max Curvature":        0.0404,
                "Surface Area (mm²)":   26675,
            },
            "CFD (mean per patient)": {
                "TAWSS":     0.84,
                "OSI":       0.21,
                "ECAP":      0.18,
                "RRT":       1.42,
                "Pressure":  98.3,
                "WSS":       0.76,
                "Velocity":  0.31,
            },
        },
        attention_weights={
            "Point cloud": 0.54,
            "Morphometric data": 0.15,
            "CFD Information": 0.31
        },
        roc_fpr=roc_fpr,
        roc_tpr=roc_tpr,
        roc_auc=0.82,
        survival_times = [0, 6, 12, 18, 24, 30, 36, 42, 48],
        survival_probs = [1.0, 0.91, 0.78, 0.64, 0.51, 0.42, 0.38, 0.35, 0.33],
    )
    root=tk.Tk()
    root.withdraw()
    win=open_results_window(root, results)
    win.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()