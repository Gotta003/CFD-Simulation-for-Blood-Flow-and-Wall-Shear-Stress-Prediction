import os
import json
import threading
import subprocess
import sys
import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
import pandas as pd
from typing import Callable, Optional, Dict, Any

C_NAV        = "#1A2A4A"
C_NAV_LIGHT  = "#2C3E6B"
C_BG         = "#F4F4F4"
C_WHITE      = "#FFFFFF"
C_BORDER     = "#DDE0E3"
C_GREEN      = "#27AE60"
C_GREEN_BG   = "#E8F8F0"
C_RED        = "#C0392B"
C_RED_BG     = "#FDE8E8"
C_ORANGE     = "#E67E22"
C_ORANGE_BG  = "#FEF5E7"
C_TEXT       = "#2C3E50"
C_MUTED      = "#7F8C8D"
C_HOVER      = "#EAF0FB"

FONT_H1   = ("Helvetica", 14, "bold")
FONT_H2   = ("Helvetica", 11, "bold")
FONT_H3   = ("Helvetica", 10, "bold")
FONT_BODY = ("Helvetica", 10)
FONT_SM   = ("Helvetica", 8)

PREDS_CSV="outputs/results/predictions.csv"
TEST_IDS="outputs/splits/test_ids.npy"

def _risk_color(conf: float) -> tuple:
    if conf>=0.70:
        return C_RED, C_RED_BG
    elif conf>=0.50:
        return C_ORANGE, C_ORANGE_BG
    return C_GREEN, C_GREEN_BG


class PatientRow(tk.Frame):
    def __init__(self, parent, pid: str, conf: float, on_click:Callable[[str], None], **kw):
        super().__init__(parent, bg=C_WHITE, cursor="hand2", **kw)
        self.pid=pid
        self.conf=conf
        self.on_click=on_click
        fg, bg, label= _risk_color(conf)
        #PAtient
        id_lbl=tk.Label(self, text=f"   {pid}", font=FONT_BODY, bg=bg, fg=C_TEXT, anchor="w")
        id_lbl.pack(side="left", fill="x", expand=True, ipady=6)
        #Risk
        pill=tk.Frame(self, bg=bg, padx=8, pady=3, highlightbackground=fg, highlightthickness=1)

class PatientListPanel(tk.Frame):
    def __init__(self, parent, on_select: Callable[[str, Dict[str, Any]], None], preds_csv: str=PREDS_CSV, **kw):
        super().__init__(parent, bg=C_BG, **kw)
        self.on_select=on_select
        self.preds_csv=preds_csv
        self._records: Dict[str, Dict]={}
        self._rows: Dict[str, PatientRow]={}
        self._selected: Optional[str]=None
        #Interface
        self._build_header()
        self._build_search()
        self._build_list()
        self._build_footer()
        self._load()

    def _show_status(self, msg: str):
        for child in self._list_frame.winfo_children():
            child.destroy()
        tk.Label(self._list_frame, text=msg, font=FONT_BODY, bg=C_WHITE, fg=C_MUTED, pady=30, wraplength=220, justify="center").pack()

    def _load(self):
        if not os.path.exists(self.preds_csv):
            self._show_status("No predictions found\nRun inference to generate")
            return
        try:
            df=pd.read_csv(self.preds_csv)
            df["patient_id"]=df["patient_id"].astype(str).str.strip()
            self._records={
                row["patient_id"]: row.to_dict() for _, row in df.iterrows()
            }
            self._render_rows(list(self._records.keys()))
        except Exception as e:
            self._show_status(f"Error loading preds:\n{e}")

    def _build_list(self):
        c=tk.Frame(self, bg=C_BG)
        c.pack(fill="both", expand=True)
        canvas=tk.Canvas(c, bg=C_WHITE, highlightthickness=0)
        scroll=ttk.Scrollbar(c, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        self._list_frame=tk.Frame(canvas, bg=C_WHITE)
        win_id=canvas.create_window((0,0), window=self._list_frame, anchor="nw")

        def _on_frame_configure(_e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        def _on_canvas_resize(e):
            canvas.itemconfig(win_id, width=e.width)
        self._list_frame.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_resize)
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", lambda ev: canvas.yview_scroll(int(-1*(ev.delta/120)), "units")))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        self._status_lbl=tk.Label(self._list_frame, text="Loading...", font=FONT_BODY, bg=C_WHITE, fg=C_MUTED, pady=20)
        self._status_lbl.pack()

    def _run_inference(self):
        self._run_btn.configure(state="disabled", text="Running...")
        self._run_status.configure(text="Starting inference script...")
        def _worker():
            try:
                location="src/results/run_inference.py"
                process=subprocess.Popen(
                    [sys.executable, "-u", location, "--force"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text="true",
                    bufsize=1
                )
                while True:
                    line=process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        clean_line=line.strip()
                        self.after(0, lambda msg=clean_line: self._run_status.configure(text=msg))
                if process.returncode==0:
                    self.after(0, self._inference_done, True, "DONE")
                #else:
                    #self.after(0, self._inference_done, False, f"Error:\n{process.returncode}")
            except Exception as e:
                self.after(0, self._inference_done, False, f"Exception:\n{str(e)}")
        threading.Thread(target=_worker, daemon=True).start()

    def _inference_done(self, success: bool, msg: str):
        self._run_btn.configure(state="normal", text="▶  Run Inference", bg=C_GREEN if success else C_RED)
        self._run_status.configure(text=msg)
        if success:
            self._load()

    def _build_footer(self):
        footer=tk.Frame(self, bg=C_NAV_LIGHT, padx=14, pady=8)
        footer.pack(fill="x", side="bottom")
        self._run_btn=tk.Button(footer, text="▶  Run Inference", font=FONT_H3, bg=C_GREEN, fg=C_WHITE, relief="flat", padx=14, pady=6, cursor="hand2", command=self._run_inference)
        self._run_btn.pack(side="left")
        self._run_status=tk.Label(footer, text="", font=FONT_SM, bg=C_NAV_LIGHT, fg=C_WHITE, wraplength=160, anchor="w")
        self._run_status.pack(side="left", padx=10)

    def _render_rows(self, pids):
        for child in self._list_frame.winfo_children():
            child.destroy()
        self._rows.clear()
        if not pids:
            tk.Label(self._list_frame, text="No patients match filter.", font=FONT_BODY, bg=C_WHITE, fg=C_MUTED, pady=20).pack()
            return
        
        def sort_key(pid):
            conf=self._records[pid].get("conf", 0.5)
            return (-conf, pid)
        
        for pid in sorted(pids, key=sort_key):
            rec=self._records[pid]
            conf=float(rec.get("conf", 0.5))
            row=PatientRow(self._list_frame, pid, conf, on_click=lambda p=pid: self._patient_clicked(p))
            row.pack(fill="x")
            self._rows[pid]=row
        n=len(pids)
        self._count_lbl.configure(text=f"{n} patient {'s' if n!=1 else ''}")

    def _filter(self):
        q=self._search_var.get().strip().lower()
        filt=self._filter_var.get()
        visible=[]
        for pid, rec in self._records.items():
            conf=float(rec.get("conf", 0.5))
            _,_,risk=_risk_color(conf)
            if filt!="ALL" and risk!=filt:
                continue
            if q and q not in pid.lower():
                continue
            visible.append(pid)
        self._render_rows(visible)

    def refresh(self):
        self._records.clear()
        self._load()

    def _build_search(self):
        bar=tk.Frame(self, bg=C_BG, padx=10, pady=6)
        bar.pack(fill="x")
        tk.Label(bar, text="Search:", font=FONT_BODY, bg=C_BG).pack(side="left")
        self._search_var=tk.StringVar()
        self._search_var.trace_add("write", lambda *_: self._filter())
        entry=tk.Entry(bar, textvariable=self._search_var, font=FONT_BODY, bd=1, relief="solid", highlightthickness=1, highlightcolor=C_NAV)
        entry.pack(side="left", fill="x", expand=True, padx=(4, 8))
        self._filter_var=tk.StringVar(value="ALL")
        for val, fg, bg in [
            ("ALL", C_TEXT, C_BG),
            ("HIGH", C_RED, C_RED_BG),
            ("MEDIUM", C_ORANGE, C_ORANGE_BG),
            ("LOW", C_GREEN, C_GREEN_BG),
        ]:
            btn=tk.Label(bar, text=val, font=FONT_SM, bg=bg, fg=fg, padx=7, pady=2, cursor="hand2", highlightbackground=fg, highlightthickness=1)
            btn.pack(side="left", padx=2)
            btn.bind("<Button-1>", lambda e, v=val: self._set_filter(v))

    def _build_header(self):
        hdr=tk.Frame(self, bg=C_NAV, padx=14, pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Test Patients", font=FONT_H1, bg=C_NAV, fg=C_WHITE).pack(side="left")
        self._count_lbl=tk.Label(hdr, text="", font=FONT_SM, bg=C_NAV_LIGHT, fg=C_WHITE, padx=8, pady=2)
        self._count_lbl.pack(side="right")


if __name__=="__main__":
    root=tk.Tk()
    root.title("Patient List - Test")
    root.geometry("320x700")
    root.configure(bg=C_BG)
    
    def _on_select(pid, rec):
        print(f"Selected: {pid} conf={rec['conf']:.2f}")

    panel=PatientListPanel(root, on_select=_on_select)
    panel.pack(fill="both", expand=True)
    root.mainloop()