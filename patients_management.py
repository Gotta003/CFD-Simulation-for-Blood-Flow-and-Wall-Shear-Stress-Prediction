import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import pandas as pd
import os
import subprocess
import platform

FILE_NAME="./data/labels/outcomes.csv"
REPORTS_BASE="../report/"
IMAGES_BASE="./outputs/mesh_heatmaps/"
FEATURES_FILE="./outputs/features/features.csv"

try:
    from PIL import Image, ImageTk
except ImportError:
    print("Warning: Pillow not installed. Image gallery will not work. Run: pip install Pillow")

# Create report folder if it doesn't exist
if not os.path.exists(REPORTS_BASE):
    os.makedirs(REPORTS_BASE)

COMP_STRUCTURE={
    "Endoleak": {"Type I": ["Type IA", "Type IB"], "Type II": [], "Type III": []},
    "Other": ["Graft_Migration", "Thrombosis", "Reintervention", "Rupture"]
}

class PatientApp:
    def __init__(self, root):
        self.root=root
        self.root.title("Patient Management")
        self.root.geometry("1500x850")
        
        self.original_id=None 
        self.image_previews=[] 
        self.load_databases()
        self.setup_ui()
        self.set_form_state("disabled")
        self.refresh_list()

    def load_databases(self):
        if os.path.exists(FILE_NAME):
            self.df=pd.read_csv(FILE_NAME).fillna("")
        else:
            self.df=pd.DataFrame(columns=["ID", "Requires_Op", "Complications", "Notes", "Examined_Files"])
        
        if os.path.exists(FEATURES_FILE):
            self.feat_df=pd.read_csv(FEATURES_FILE)
        else:
            self.feat_df=pd.DataFrame()

    def get_pz_path(self, base, patient_id):
        """Standardizes path to pz001 format."""
        try:
            folder_name=f"pz{int(patient_id):03d}"
            return os.path.join(base, folder_name)
        except: return None

    def setup_ui(self):
        # LEFT SIDEBAR (IDs + Filters)
        sidebar=tk.Frame(self.root, width=280, bg="#f4f4f4", padx=10, pady=10)
        sidebar.pack(side="left", fill="y")
        tk.Label(sidebar, text="Search ID:", bg="#f4f4f4", font=("Arial", 10, "bold")).pack(anchor="w")
        self.search_var=tk.StringVar()
        self.search_var.trace("w", lambda *args: self.refresh_list())
        tk.Entry(sidebar, textvariable=self.search_var).pack(fill="x", pady=5)
        self.filter_var=tk.StringVar(value="All")
        filter_menu=ttk.Combobox(sidebar, textvariable=self.filter_var, values=["All", "Red (Not Started)", "Yellow (In Progress)", "Green (Complete)"], state="readonly")
        filter_menu.pack(fill="x", pady=5)
        filter_menu.bind("<<ComboboxSelected>>", lambda e: self.refresh_list())
        self.listbox=tk.Listbox(sidebar, font=("Arial", 10, "bold"), selectmode="single")
        self.listbox.pack(fill="both", expand=True, pady=5)
        self.listbox.bind("<<ListboxSelect>>", self.on_patient_select)
        self.stats_label=tk.Label(sidebar, text="", bg="#f4f4f4", font=("Arial", 9))
        self.stats_label.pack(fill="x", pady=5)
        tk.Button(sidebar, text="+ Add Patient", command=self.prepare_new_patient, bg="#3498db", fg="white", font=("Arial", 10, "bold")).pack(fill="x")
        # CENTER PANEL (Data)
        self.center_panel=tk.Frame(self.root, padx=20, pady=20, borderwidth=1, relief="sunken")
        self.center_panel.pack(side="left", fill="both", expand=True)
        self.header_var=tk.StringVar(value="Select a Patient")
        tk.Label(self.center_panel, textvariable=self.header_var, font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 10))
        tk.Label(self.center_panel, text="Patient ID:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.id_entry=tk.Entry(self.center_panel, font=("Arial", 11))
        self.id_entry.pack(fill="x", pady=5)
        wf_frame=tk.LabelFrame(self.center_panel, text="Workflow Status", padx=10, pady=10)
        wf_frame.pack(fill="x", pady=10)
        self.report_var=tk.BooleanVar()
        self.cfd_var=tk.BooleanVar()
        self.img_var=tk.BooleanVar()
        self.report_cb=tk.Checkbutton(wf_frame, text="Report Analysis", variable=self.report_var)
        self.cfd_cb=tk.Checkbutton(wf_frame, text="CFD Simulations", variable=self.cfd_var)
        self.img_cb=tk.Checkbutton(wf_frame, text="Images Processing", variable=self.img_var)
        self.report_cb.pack(side="left", padx=5)
        self.cfd_cb.pack(side="left", padx=5)
        self.img_cb.pack(side="left", padx=5)
        self.op_var=tk.BooleanVar()
        self.op_check=tk.Checkbutton(self.center_panel, text="Requires Operation", variable=self.op_var, command=self.handle_op_toggle)
        self.op_check.pack(anchor="w")

        self.comp_frame=tk.LabelFrame(self.center_panel, text="Complications", padx=10, pady=10)
        self.comp_frame.pack(fill="both", expand=True, pady=10)
        self.check_vars={}
        self.check_widgets=[]
        self.create_comp_widgets()
        self.notes_text=tk.Text(self.center_panel, height=4)
        self.notes_text.pack(fill="x", pady=10)
        btn_f=tk.Frame(self.center_panel)
        btn_f.pack(fill="x")
        self.modify_btn=tk.Button(btn_f, text="Modify", command=self.enter_modify_mode, bg="#f39c12", fg="white", width=10)
        self.modify_btn.pack(side="left", padx=5)
        self.save_btn=tk.Button(btn_f, text="Save Changes", command=self.save_data, bg="#2ecc71", fg="white", width=15)
        self.save_btn.pack(side="left", padx=5)
        # RIGHT PANEL (Tabs)
        self.notebook=ttk.Notebook(self.root, width=450)
        self.notebook.pack(side="right", fill="both", padx=10, pady=10)

        # Tab 1: Documents
        self.doc_tab=tk.Frame(self.notebook)
        self.notebook.add(self.doc_tab, text="Reports Checklist")
        self.setup_doc_tab()

        # Tab 2: Features
        self.feat_tab=tk.Frame(self.notebook)
        self.notebook.add(self.feat_tab, text="Extracted Features")
        self.setup_feat_tab()

        # Tab 3: Gallery
        self.img_tab=tk.Frame(self.notebook)
        self.notebook.add(self.img_tab, text="Image Gallery")
        self.setup_img_tab()

    def setup_doc_tab(self):
        self.doc_canvas=tk.Canvas(self.doc_tab)
        self.doc_scroll=ttk.Scrollbar(self.doc_tab, orient="vertical", command=self.doc_canvas.yview)
        self.doc_list_frame=tk.Frame(self.doc_canvas)
        self.doc_canvas.create_window((0,0), window=self.doc_list_frame, anchor="nw")
        self.doc_canvas.configure(yscrollcommand=self.doc_scroll.set)
        self.doc_canvas.pack(side="left", fill="both", expand=True)
        self.doc_scroll.pack(side="right", fill="y")
        self.examined_files_vars={}

    def setup_feat_tab(self):
        columns=("Metric", "Value")
        self.feat_tree=ttk.Treeview(self.feat_tab, columns=columns, show="headings")
        self.feat_tree.heading("Metric", text="CFD Metric")
        self.feat_tree.heading("Value", text="Value")
        self.feat_tree.column("Metric", width=300)
        self.feat_tree.column("Value", width=150)
        feat_scroll=ttk.Scrollbar(self.feat_tab, orient="vertical", command=self.feat_tree.yview)
        self.feat_tree.configure(yscrollcommand=feat_scroll.set)
        self.feat_tree.pack(side="left", fill="both", expand=True)
        feat_scroll.pack(side="right", fill="y")

    def setup_img_tab(self):
        self.img_canvas=tk.Canvas(self.img_tab, bg="#e0e0e0")
        self.img_scroll=ttk.Scrollbar(self.img_tab, orient="vertical", command=self.img_canvas.yview)
        self.img_container=tk.Frame(self.img_canvas, bg="#e0e0e0")
        self.img_canvas.create_window((0,0), window=self.img_container, anchor="nw")
        self.img_canvas.configure(yscrollcommand=self.img_scroll.set)
        self.img_canvas.pack(side="left", fill="both", expand=True)
        self.img_scroll.pack(side="right", fill="y")

    def refresh_features(self, patient_id):
        self.feat_tree.delete(*self.feat_tree.get_children())
        if self.feat_df.empty:
            return
        id_str=f"pz{int(patient_id):03d}"
        patient_row=self.feat_df[self.feat_df.iloc[:,0].str.startswith(id_str, na=False)]
        if not patient_row.empty:
            row_data=patient_row.iloc[0]
            current_group=""
            for col in self.feat_df.columns[1:]:
                category=col.split('_'[0].upper())
                if category!=current_group:
                    current_group=category
                    self.feat_tree.insert("", "end", values=(f"--- {current_group} ---", ""), tags=('group',))
                val=row_data[col]
                formatted_val=f"{float(val):.4f}" if isinstance(val, (int, float)) else val
                self.feat_tree.insert("", "end", value=(col, formatted_val))
        self.feat_tree.tag_configure('group', font=('Arial', 10, "bold"), background="#f0f0f0")

    def refresh_media(self, patient_id):
        for w in self.doc_list_frame.winfo_children(): w.destroy()
        self.examined_files_vars={}
        for w in self.img_container.winfo_children(): w.destroy()
        self.image_previews=[]
        report_dir=self.get_pz_path(REPORTS_BASE, patient_id)
        img_dir=self.get_pz_path(IMAGES_BASE, patient_id)

        if report_dir and os.path.exists(report_dir):
            examined_str=str(self.df[self.df['ID']==int(patient_id)].iloc[0]['Examined_Files']) if int(patient_id) in self.df['ID'].values else ""
            examined_list=examined_str.split("|")
            for f in sorted(os.listdir(report_dir)):
                if os.path.isfile(os.path.join(report_dir, f)):
                    f_f=tk.Frame(self.doc_list_frame)
                    f_f.pack(fill="x")
                    v=tk.BooleanVar(value=f in examined_list)
                    self.examined_files_vars[f]=v
                    tk.Checkbutton(f_f, variable=v).pack(side="left")
                    btn=tk.Button(f_f, text=f, relief="flat", fg="blue", cursor="hand2", anchor="w")
                    btn.pack(side="left", fill="x")
                    btn.bind("<Button-1>", lambda e, p=os.path.join(report_dir, f): self.open_file(p))
        
        if img_dir and os.path.exists(img_dir):
            valid_ext=('.jpg', '.jpeg', '.png', '.bmp')
            for f in sorted(os.listdir(img_dir)):
                if f.lower().endswith(valid_ext):
                    p=os.path.join(img_dir, f)
                    try:
                        img=Image.open(p)
                        img.thumbnail((180, 180))
                        photo=ImageTk.PhotoImage(img)
                        self.image_previews.append(photo)
                        lbl=tk.Label(self.img_container, image=photo, relief="ridge", cursor="hand2")
                        lbl.pack(pady=5)
                        lbl.bind("<Button-1>", lambda e, path=p: self.open_file(path))
                        tk.Label(self.img_container, text=f, bg="#e0e0e0", font=("Arial", 8)).pack()
                    except: pass

        self.doc_list_frame.update_idletasks()
        self.doc_canvas.config(scrollregion=self.doc_canvas.bbox("all"))
        self.img_container.update_idletasks()
        self.img_canvas.config(scrollregion=self.img_canvas.bbox("all"))

    def create_comp_widgets(self):
        r=0
        for cat, sub in COMP_STRUCTURE.items():
            if isinstance(sub, dict):
                tk.Label(self.comp_frame, text=cat, font=("Arial", 9, "bold")).grid(row=r, column=0, sticky="w")
                r += 1
                for s_key, s_list in sub.items():
                    var=tk.BooleanVar()
                    self.check_vars[s_key]=var
                    cb=tk.Checkbutton(self.comp_frame, text=s_key, variable=var)
                    cb.grid(row=r, column=0, sticky="w", padx=15)
                    self.check_widgets.append(cb)
                    r += 1
                    for item in s_list:
                        i_var=tk.BooleanVar()
                        self.check_vars[item]=i_var
                        cb_sub=tk.Checkbutton(self.comp_frame, text=item, variable=i_var)
                        cb_sub.grid(row=r, column=0, sticky="w", padx=35)
                        self.check_widgets.append(cb_sub)
                        r += 1
            else:
                for item in sub:
                    var=tk.BooleanVar()
                    self.check_vars[item]=var
                    cb_other=tk.Checkbutton(self.comp_frame, text=item, variable=var) 
                    cb_other.grid(row=r, column=0, sticky="w")
                    self.check_widgets.append(cb_other)
                    r += 1

    def open_file(self, path):
        if platform.system()=='Darwin': subprocess.call(('open', path))
        elif platform.system()=='Windows': os.startfile(path)
        else: subprocess.call(('xdg-open', path))

    def on_patient_select(self, event):
        idx=self.listbox.curselection()
        if not idx: return
        pid=int(self.listbox.get(idx[0]))
        row=self.df[self.df['ID']==pid].iloc[0]

        self.original_id=pid
        self.header_var.set(f"Viewing Patient: {pid}")
        self.id_entry.config(state="normal")
        self.notes_text.config(state="normal")
        self.clear_fields()
        
        self.id_entry.insert(0, str(row['ID']))
        self.op_var.set(row['Requires_Op']=="Yes")
        self.report_var.set(bool(row['Report_Analysis']))
        self.cfd_var.set(bool(row['CFD_Simulations']))
        self.img_var.set(bool(row['Image_Processing']))
        
        comps=str(row['Complications']).split(", ")
        for c in comps:
            if c in self.check_vars: self.check_vars[c].set(True)
        self.notes_text.insert("1.0", str(row['Notes']))
        self.refresh_features(pid)
        self.set_form_state("disabled")
        self.refresh_media(pid)

    def save_data(self):
        new_id_raw=self.id_entry.get().strip()
        if not new_id_raw: return
        new_id=int(new_id_raw)

        examined=[f for f, v in self.examined_files_vars.items() if v.get()]
        examined_str="|".join(examined)
        comp_list=[n for n, v in self.check_vars.items() if v.get()] if self.op_var.get() else []
        
        data={
            "ID": new_id, "Requires_Op": "Yes" if self.op_var.get() else "No", "Complications": ", ".join(comp_list),
            "Notes": self.notes_text.get("1.0", tk.END).strip(), "Report_Analysis": self.report_var.get(),
            "CFD_Simulations": self.cfd_var.get(), "Image_Processing": self.img_var.get(),
            "Examined_Files": examined_str
        }

        if self.original_id is not None:
            if self.original_id != new_id: self.df=self.df[self.df['ID'] != self.original_id]
            self.df=self.df[self.df['ID'] != new_id]
        
        self.df=pd.concat([self.df, pd.DataFrame([data])], ignore_index=True)
        self.df=self.df.sort_values(by="ID").reset_index(drop=True)
        self.df.to_csv(FILE_NAME, index=False)
        
        messagebox.showinfo("Saved", f"Patient {new_id} updated.")
        self.original_id=new_id
        self.set_form_state("disabled")
        self.refresh_list()

    def handle_op_toggle(self):
        st="normal" if (self.op_var.get() and self.save_btn['state']=='normal') else "disabled"
        for cb in self.check_widgets: cb.config(state=st)

    def set_form_state(self, state):
        widgets_to_toggle=[self.id_entry, self.op_check, self.notes_text, self.report_cb, self.cfd_cb, self.img_cb]
        for w in widgets_to_toggle:
            w.config(state="normal" if state=="disabled" and w==self.modify_btn else state)
        comp_state=state if (state=="normal" and self.op_var.get()) else "disabled"
        for child in self.comp_frame.winfo_children():
            if isinstance(child, tk.Checkbutton):
                child.config(state=comp_state)
        self.save_btn.config(state=state)
        self.modify_btn.config(state="normal" if state=="disabled" and self.listbox.curselection() else "disabled")
        self.update_colors()

    def update_colors(self):
        is_readonly=(self.save_btn['state']=='disabled')
        is_op_req=self.op_var.get()
        for child in self.comp_frame.winfo_children():
            if isinstance(child, tk.Checkbutton):
                if is_readonly:
                    child.config(disabledforeground="black")
                else:
                    child.config(fg="black" if is_op_req else "gray") 

    def refresh_list(self):
        self.listbox.delete(0, tk.END)
        term=self.search_var.get()
        s_filt=self.filter_var.get()
        f_df=self.df[self.df['ID'].astype(str).str.contains(term)].copy()
        s_ids=sorted(f_df['ID'].unique(), key=int)
        
        r,y,g=0,0,0
        for pid in s_ids:
            row=self.df[self.df['ID']==pid].iloc[0]
            count=sum([1 for c in [row['Report_Analysis'], row['CFD_Simulations'], row['Image_Processing']] if str(c).lower()=='true' or c==True])
            color="red" if count==0 else "orange" if count < 3 else "green"
            if color=="red": r += 1
            elif color=="orange": y += 1
            else: g += 1
            
            if s_filt=="All" or (s_filt.startswith("Red") and color=="red") or (s_filt.startswith("Yellow") and color=="orange") or (s_filt.startswith("Green") and color=="green"):
                self.listbox.insert(tk.END, pid)
                self.listbox.itemconfig(tk.END, {'fg': color})
        self.stats_label.config(text=f"To Do {r}  |  Doing {y}  |  Complete {g}")

    def enter_modify_mode(self):
        self.set_form_state("normal")
        self.header_var.set(f"Modifying Patient: {self.original_id}")

    def prepare_new_patient(self):
        self.original_id=None
        self.set_form_state("normal")
        self.clear_fields()
        self.id_entry.delete(0, tk.END)
        self.id_entry.focus()
        self.header_var.set("Adding New Patient")

    def clear_fields(self):
        self.id_entry.delete(0, tk.END)
        self.op_var.set(False)
        self.notes_text.delete("1.0", tk.END)
        for var in list(self.check_vars.values()) + [self.report_var, self.cfd_var, self.img_var]: var.set(False)

if __name__=="__main__":
    root=tk.Tk()
    app=PatientApp(root)
    root.mainloop()