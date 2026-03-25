import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import pandas as pd
import os
import subprocess
import platform
import shutil

FILE_NAME="./data/labels/outcomes.csv"
SEG_BASE="../simulation_db/"
REPORTS_BASE="../report/"
IMAGES_BASE="./outputs/mesh_heatmaps/"
FEATURES_FILE="./outputs/features/features.csv"
VTP_BASE="./data/vtp_files/"

try:
    from PIL import Image, ImageTk
    HAS_PIL=True
except ImportError:
    print("Warning: Pillow not installed. Image gallery will not work. Run: pip install Pillow")
    HAS_PIL=False

# Create report folder if it doesn't exist
if not os.path.exists(REPORTS_BASE):
    os.makedirs(REPORTS_BASE)

COMP_STRUCTURE={
    "Endoleak": {"Type I": ["Type IA", "Type IB"], "Type II": [], "Type III": []},
    "Other": ["Graft_Migration", "Thrombosis", "Reintervention", "Rupture"]
}

STATUS_OK="OK"
STATUS_FAIL="x"
COLOR_OK="#2ecc71"
COLOR_FAIL="#c0392b"

SIM_COLOR_MAP={
    "To Run": "#c0392b",
    "Post processed": "#27ae60",
    "Completed: post processing required": "#e67e22",
    "Running": "#3498db",
    "not available": "#dde0e3"
}

REPORT_COLOR_MAP={
    "OK": "#27ae60",
    "Doing": "#e67e22",
    "Missing": "#dde0e3"
}

def run_post_processing(patient_id, vtp_dir):
    id_str=f"pz{int(patient_id):03d}"
    print(f"--- Extract features for {id_str} ---")
    try:
        subprocess.run(["python", "src/extraction/extract_features.py", "--input", vtp_dir, "--out", FEATURES_FILE], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during feature extraction for {id_str}: {e}")

    print(f"--- Generating tmaps for {id_str} ---")
    dest_dir=os.path.join(IMAGES_BASE, id_str)
    os.makedirs(dest_dir, exist_ok=True)
    for vtp in os.listdir(vtp_dir):
        if vtp.lower().endswith(".vtp"):
            src_path=os.path.join(vtp_dir, vtp)
            try:
                subprocess.run(["python", "src/visualization/visualize.py", "--vtp", src_path, "--out", dest_dir, "--all_fields"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error generating tmap for {vtp} of {id_str}: {e}")

def detect_segmentation_status(patient_id: int) -> bool:
    try:
        id_str=f"pz{int(patient_id):03d}"
    except (ValueError, TypeError):
        return False
    if not os.path.exists(SEG_BASE):
        return False
    
    patient_folder= os.path.join(SEG_BASE, id_str)
    return os.path.isdir(patient_folder)

def detect_report_status(patient_id: int, examined_files_str: str) -> bool:
    base=REPORTS_BASE
    try:
        folder=os.path.join(base, f"pz{int(patient_id):03d}")
    except (ValueError, TypeError):
        return "Missing"
    
    if not os.path.exists(folder):
        return "Missing"

    all_files=[f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    #if not all_files:
    #    return "Missing"

    examined=set(examined_files_str.split("|")) if examined_files_str else set()
    if all(f in examined for f in all_files):
        return "OK"
    if any(f in examined for f in all_files):
        return "Doing"
    return "Missing"

def detect_cfd_status(patient_id: int) -> bool:
    try:
        id_str=f"pz{int(patient_id):03d}"
    except (ValueError, TypeError):
        return False
    
    path_to_check= os.path.join(SEG_BASE, id_str, "Simulations", id_str)
    if not os.path.exists(path_to_check):
        return False 
    
    try: 
        for item in os.listdir(path_to_check):
            item_path = os.path.join(path_to_check, item)
            if os.path.isdir(item_path) and item.endswith("-procs"):
                if item[:2].isdigit():
                    return True
    except OSError:
        return False

    return False

def detect_simulation_status(patient_id: int) -> str:
    try:
        id_str = f"pz{int(patient_id):03d}"
    except (ValueError, TypeError):
        return "not available"

    patient_main_folder = os.path.join(SEG_BASE, id_str)
    sim_path = os.path.join(patient_main_folder, "Simulations", id_str)

    if not os.path.exists(patient_main_folder):
        return "not available"

    try:
        main_contents = os.listdir(patient_main_folder)
        proc_folders = []
        if os.path.exists(sim_path):
            proc_folders = [d for d in os.listdir(sim_path) 
                           if d.endswith("-procs") and os.path.isdir(os.path.join(sim_path, d))]

        if not proc_folders:
            if len(main_contents) > 0:
                return "To Run"
            else:
                return "not available"

        target_dir = os.path.join(sim_path, sorted(proc_folders)[-1])
        files = os.listdir(target_dir)
        vtp_files=[f for f in files if f.lower().endswith(".vtp")]
        if vtp_files:
            patient_vtp_dir=os.path.join(VTP_BASE, id_str)
            os.makedirs(patient_vtp_dir, exist_ok=True)
            new_file_copy=False
            for vtp in vtp_files:
                src_path=os.path.join(target_dir, vtp)
                if vtp.startswith(id_str):
                    new_filename=vtp
                else:
                    new_filename=f"{id_str}_{vtp}"
                    
                dst_path=os.path.join(patient_vtp_dir, new_filename)
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                    new_file_copy=True
            if new_file_copy:
                run_post_processing(patient_id, patient_vtp_dir)
            return "Post processed"

        if "result_960.vtu" in files:
            return "Completed: post processing required"
            
        return "Running"

    except OSError:
        return "not available"
    

def detect_image_status(patient_id: int) -> bool:
    try:
        folder=os.path.join(IMAGES_BASE, f"pz{int(patient_id):03d}")
    except (ValueError, TypeError):
        return False
    if not os.path.exists(folder):
        return False
    valid_ext=('.jpg', '.jpeg', '.png')
    return any(f.lower().endswith(valid_ext) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)))

class PatientApp:
    def __init__(self, root):
        self.root=root
        self.root.title("Patient Management")
        self.root.geometry("1500x850")
        
        self.original_id=None 
        self.image_previews=[] 
        self._status_cache={}
        self.load_databases()
        self.setup_ui()
        self.set_form_state("disabled")
        self._build_all_rows()
        self.refresh_list()

    def _update_row_colors(self, pid):
        if pid not in self._row_widgets:
            return
        row_f, id_lbl, sep, sq_labels=self._row_widgets[pid]
        st=self._status_cache.get(pid, {})
        for key, sq in sq_labels:
            if key=="overall":
                color=st.get("overall", "#e74c3c")
            else:
                is_ok=st.get(key, False)
                color=self._rect_color(is_ok)
            sq.config(bg=color)

    def load_databases(self):
        if os.path.exists(FILE_NAME):
            self.df=pd.read_csv(FILE_NAME).fillna("")
            if "Report_Analysis" in self.df.columns:
                self.df["Report_Analysis"]=self.df["Report_Analysis"].astype(object)
        else:
            self.df=pd.DataFrame(columns=["ID", "Requires_Op", "Complications", "Notes", "Examined_Files"])
        
        if os.path.exists(FEATURES_FILE):
            self.feat_df=pd.read_csv(FEATURES_FILE)
        else:
            self.feat_df=pd.DataFrame()
        self._rebuild_cache()

    def get_pz_path(self, base, patient_id):
        """Standardizes path to pz001 format."""
        try:
            folder_name=f"pz{int(patient_id):03d}"
            return os.path.join(base, folder_name)
        except: return None

    def _set_status_label(self, lbl, status):
        is_ok = (status is True or status == "OK")
        is_doing = (status == "Doing")
        
        if is_ok:
            lbl.config(text=STATUS_OK, fg=COLOR_OK, bg="#e8f8f0")
        elif is_doing:
            lbl.config(text="...", fg="#e67e22", bg="#fef5e7")
        else:
            lbl.config(text=STATUS_FAIL, fg=COLOR_FAIL, bg="#fde8e8")

    def refresh_auto_status(self, patient_id, examined_files_str=""):
        seg_ok=detect_segmentation_status(patient_id)
        report_status=detect_report_status(patient_id, examined_files_str)
        cfd_ok=detect_cfd_status(patient_id)
        img_ok=detect_image_status(patient_id)
        sim_status=detect_simulation_status(patient_id)

        self.seg_var.set(seg_ok)
        self.report_var.set(report_status=="OK")
        self.cfd_var.set(cfd_ok)
        self.img_var.set(img_ok)
        self._set_status_label(self.seg_status_lbl, seg_ok)
        self._set_status_label(self.report_status_lbl, report_status)
        self._set_status_label(self.cfd_status_lbl, cfd_ok)
        self._set_status_label(self.img_status_lbl, img_ok)
     
        text_color=SIM_COLOR_MAP.get(sim_status, "#7f8c8d")
        self.cfd_text_lbl.config(text=f"({sim_status})", fg=text_color)

    def setup_ui(self):
        # LEFT SIDEBAR (IDs + Filters)
        sidebar=tk.Frame(self.root, width=280, bg="#f4f4f4", padx=10, pady=10)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        
        #Search
        tk.Label(sidebar, text="Search ID:", bg="#f4f4f4", font=("Arial", 10, "bold")).pack(anchor="w")
        search_frame=tk.Frame(sidebar, bg="#f4f4f4")
        search_frame.pack(fill="x", pady=(2,8))
        self.search_var=tk.StringVar()
        self.search_entry=tk.Entry(search_frame, textvariable=self.search_var, font=("Arial", 10))
        self.search_entry.pack(side="left", fill="x", expand=True)
        self.search_entry.bind("<Return>", lambda e: self.refresh_list())
        self.search_btn=tk.Button(search_frame, text="Find", command=self.refresh_list, bg="#ecf0f1", relief="flat", padx=5)
        self.search_btn.pack(side="right", padx=(2,0))

        #Colour Filter
        tk.Label(sidebar, text="Filter by Status", bg="#f4f4f4", font=("Arial", 9, "bold")).pack(anchor="w")
        filt_frame=tk.Frame(sidebar, bg="#f4f4f4")
        filt_frame.pack(fill="x", pady=(2,8))
        self.filter_var=tk.StringVar(value="All")
        filter_defs=[
            ("All", "#555555"),
            ("Red", "#e74c3c"),
            ("Yellow", "#e67e22"),
            ("Green", "#27ae60")
        ]
        self._filter_btns={}
        for label, color in filter_defs:
            b=tk.Button(filt_frame, text=label, bg=color, fg="white", font=("Arial", 8, "bold"), relief="raised", bd=2, command=lambda l=label: self._set_filter(l))
            b.pack(side="left", expand=True, fill="x", padx=2)
            self._filter_btns[label]=b
        
        #Substatus Filter   ,, , 
        tk.Label(sidebar, text="Filter by Step:", bg="#f4f4f4", font=("Arial", 9, "bold")).pack(anchor="w", pady=(5,0))
        self.sub_filter_var=tk.StringVar(value="All")
        sub_options=[
            "All",
            "Segmentation: Missing", "Segmentation: OK",
            "Report Analysis: Missing", "Report Analysis: Doing", "Report Analysis: OK",
            "CFD Simulations: Missing", "CFD Simulations: To Run", "CFD Simulations: Running", "CFD Simulations: Post Processing Req", "CFD Simulations: OK" ,
            "Image Processing: Missing", "Image Processing: OK",
            "Labeling: Missing", "Labeling: OK"
        ]
        self.sub_filter_combo=ttk.Combobox(sidebar, textvariable=self.sub_filter_var, values=sub_options, state="readonly")
        self.sub_filter_combo.pack(fill="x", pady=(2,8))
        self.sub_filter_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh_list())
        #Patient List
        list_frame=tk.Frame(sidebar, bg="white", relief="sunken", bd=1)
        list_frame.pack(fill="both", expand=True, pady=(0,8))
        self._list_canvas=tk.Canvas(list_frame, bg="white", highlightthickness=0)
        list_scroll=ttk.Scrollbar(list_frame, orient="vertical", command=self._list_canvas.yview)
        self._list_canvas.configure(yscrollcommand=list_scroll.set)
        list_scroll.pack(side="right", fill="y")
        self._list_canvas.pack(side="left", fill="both", expand=True)
        self._list_inner=tk.Frame(self._list_canvas, bg="white")
        self._list_canvas_window=self._list_canvas.create_window(
            (0,0), window=self._list_inner, anchor="nw"
        )
        self._list_inner.bind("<Configure>", lambda e: self._list_canvas.configure(scrollregion=self._list_canvas.bbox("all")))
        self._list_canvas.bind("<Configure>", lambda e: self._list_canvas.itemconfig(self._list_canvas_window, width=e.width))
        
        self._list_canvas.bind("<Enter>", lambda e: self._list_canvas.bind_all("<MouseWheel>", self._on_mousewheel))
        self._list_canvas.bind("<Leave>", lambda e: self._list_canvas.unbind_all("<MouseWheel>"))
        self._patient_rows=[]
        
        # Stats + Add Button
        self.stats_label=tk.Label(sidebar, text="", bg="#f4f4f4", font=("Arial", 8), anchor="w")
        self.stats_label.pack(fill="x", pady=(0,4))
        tk.Button(sidebar, text="+ Add Patient", command=self.prepare_new_patient, bg="#3498db", fg="white", font=("Arial", 10, "bold"), relief="flat", pady=6).pack(fill="x")
        self._set_filter("All")
        
        # CENTER PANEL (Data)
        self.center_panel=tk.Frame(self.root, padx=20, pady=20, borderwidth=1, relief="sunken")
        self.center_panel.pack(side="left", fill="both", expand=True)
        self.header_var=tk.StringVar(value="Select a Patient")
        tk.Label(self.center_panel, textvariable=self.header_var, font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 10))
        tk.Label(self.center_panel, text="Patient ID:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.id_entry=tk.Entry(self.center_panel, font=("Arial", 11))
        self.id_entry.pack(fill="x", pady=5)
        #Workflow Status
        wf_frame=tk.LabelFrame(self.center_panel, text="Workflow Status", padx=10, pady=8)
        wf_frame.pack(fill="x", pady=10)
        self.seg_var=tk.BooleanVar()
        self.report_var=tk.BooleanVar()
        self.cfd_var=tk.BooleanVar()
        self.img_var=tk.BooleanVar()
        
        def _status_row(parent, text, row):
            tk.Label(parent, text=text, font=("Arial", 10)).grid(row=row, column=0, sticky="w", padx=(0, 6))
            lbl=tk.Label(parent, text=STATUS_FAIL, fg=COLOR_FAIL, font=("Arial", 11, "bold"), width=3)
            lbl.grid(row=row, column=1, sticky="w")
            #per aggiungere stato della simulazione 
            txt_lbl = tk.Label(parent, text="", font=("Arial", 9, "italic"), fg="#7f8c8d", anchor="w")
            txt_lbl.grid(row=row, column=2, sticky="w", padx=10)
            return lbl, txt_lbl
        self.seg_status_lbl, _=_status_row(wf_frame, "Segmentation", 0)
        self.report_status_lbl, _=_status_row(wf_frame, "Report Analysis", 1)
        self.cfd_status_lbl, self.cfd_text_lbl=_status_row(wf_frame, "CFD Simulations", 2)
        self.img_status_lbl, _=_status_row(wf_frame, "Image Processing", 3)
        
        #Labeling
        label_frame=tk.LabelFrame(self.center_panel, text="Labeling", padx=10, pady=8)
        label_frame.pack(fill="x", pady=(0,6))
        self.labeling_var=tk.BooleanVar()
        self.labeling_cb=tk.Checkbutton(label_frame, text="Outcome labels verified and confirmed", variable=self.labeling_var, font=("Arial", 10))
        self.labeling_cb.pack(anchor="w")
        tk.Label(label_frame, text="Manually click once complication labels have been reviewed.", font=("Arial", 7), fg="#aaaaaa", wraplength=380, justify="left").pack(anchor="w")
        
        #Requires Operation Button
        self.op_var=tk.BooleanVar()
        self.op_check=tk.Checkbutton(self.center_panel, text="Requires Operation", variable=self.op_var, command=self.handle_op_toggle)
        self.op_check.pack(anchor="w")

        #Complications
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
        self.remove_btn=tk.Button(btn_f, text="Remove Patient", command=self.remove_patient, bg="#e74c3c", fg="white", width=15)
        self.remove_btn.pack(side="left", padx=5)
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
        if os.path.exists(FEATURES_FILE):
            try:
                self.feat_df=pd.read_csv(FEATURES_FILE)
            except Exception as e:
                print(f"Error loading features file: {e}")
                return
        if self.feat_df.empty:
            return
        id_str=f"pz{int(patient_id):03d}"
        patient_row = self.feat_df[self.feat_df.iloc[:,0].astype(str).str.contains(id_str, na=False)]
        if not patient_row.empty:
            row_data=patient_row.iloc[0]
            current_group=""
            for col in self.feat_df.columns[1:]:
                parts=col.split('_')
                new_group=parts[0].upper()
                if new_group!=current_group:
                    current_group=new_group
                    self.feat_tree.insert("", "end", values=(f"--- {current_group} ---", ""), tags=('group',))

                val=row_data[col]
                formatted_val=f"{float(val):.4f}" if isinstance(val, (int, float)) else val
                display_name=" ".join(parts[1:]).upper() if len(parts)>1 else col
                self.feat_tree.insert("", "end", values=(display_name, formatted_val))
        self.feat_tree.tag_configure('group', font=('Arial', 10, "bold"), background="#f0f0f0")

    def refresh_media(self, patient_id):
        for w in self.doc_list_frame.winfo_children(): w.destroy()
        self.examined_files_vars={}
        for w in self.img_container.winfo_children(): w.destroy()
        self.image_previews=[]
        report_dir=self.get_pz_path(REPORTS_BASE, patient_id)
        img_dir=self.get_pz_path(IMAGES_BASE, patient_id)

        if report_dir and os.path.exists(report_dir):
            rows=self.df[self.df["ID"]==int(patient_id)]
            exam_str=str(rows.iloc[0]["Examined_Files"]) if not rows.empty else ""
            exam_list=exam_str.split("|") if exam_str else []
            for f in sorted(os.listdir(report_dir)):
                if not os.path.isfile(os.path.join(report_dir, f)):
                    continue
                f_f=tk.Frame(self.doc_list_frame)
                f_f.pack(fill="x")
                v=tk.BooleanVar(value=f in exam_list)
                v.trace("w", lambda *args, pid=patient_id: self._on_examined_change(pid))
                self.examined_files_vars[f]=v
                tk.Checkbutton(f_f, variable=v).pack(side="left")
                btn=tk.Button(f_f, text=f, relief="flat", fg="blue", cursor="hand2", anchor="w")
                btn.pack(side="left", fill="x")
                btn.bind("<Button-1>", lambda e, p=os.path.join(report_dir, f): self.open_file(p))
        
        if img_dir and os.path.exists(img_dir) and HAS_PIL:
            valid_ext=('.jpg', '.jpeg', '.png')
            images=[f for f in os.listdir(img_dir) if f.lower().endswith(valid_ext)]
            groups={}
            for f in sorted(images):
                parts=f.replace(".png","").replace(".jpg","").split("_")
                category=parts[1].upper() if len(parts)>1 else "OTHER"
                if category not in groups:
                    groups[category]=[]
                groups[category].append(f)

            for cat, names in groups.items():
                cat_frame=tk.Frame(self.img_container, bg="#f0f0f0", pady=5)
                cat_frame.pack(fill="x", pady=(10, 2))
                tk.Label(cat_frame, text=f" {cat} ", bg="#f0f0f0", font=("Arial", 9, "bold"), fg="#2c3e50").pack(anchor="w")

                grid_frame=tk.Frame(self.img_container, bg="#e0e0e0")
                grid_frame.pack(fill="x", padx=5)
                cols=2
                for i, f in enumerate(names):
                    p=os.path.join(img_dir, f)
                    try:
                        img=Image.open(p)
                        img.thumbnail((200,200))
                        photo=ImageTk.PhotoImage(img)
                        self.image_previews.append(photo)
                        img_box=tk.Frame(grid_frame, bg="#e0e0e0", padx=5, pady=5)
                        img_box.grid(row=i//cols, column=i%cols, sticky="nw")
                        lbl=tk.Label(img_box, image=photo, relief="ridge", bd=2, cursor="hand2")
                        lbl.pack()
                        lbl.bind("<Button-1>", lambda e, path=p: self.open_file(path))
                        tk.Label(img_box, text=f[:20], bg="#e0e0e0", font=("Arial", 7)).pack()
                    except Exception as e:
                        print(f"Error loading image {p}: {e}")

        self.doc_list_frame.update_idletasks()
        self.doc_canvas.config(scrollregion=self.doc_canvas.bbox("all"))
        self.img_container.update_idletasks()
        self.img_canvas.config(scrollregion=self.img_canvas.bbox("all"))
        self._on_examined_change(patient_id)
        
    def _on_examined_change(self, patient_id):
        examined_str="|".join(
            f for f, v in self.examined_files_vars.items() if v.get()
        )
        idx=self.df.index[self.df["ID"]==int(patient_id)]
        if not idx.empty:
            report_status=detect_report_status(patient_id, examined_str)
            self.df.at[idx[0], "Examined_Files"]=examined_str
            self.df.at[idx[0], "Report_Analysis"]=report_status
            self.df.to_csv(FILE_NAME, index=False)
            self._invalidate_patient(int(patient_id))
            self._update_row_colors(int(patient_id))
            self.refresh_auto_status(patient_id, examined_str)
            self.refresh_list()

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
        
    def save_data(self):
        new_id_raw=self.id_entry.get().strip()
        if not new_id_raw: 
            return
        new_id=int(new_id_raw)

        examined=[f for f, v in self.examined_files_vars.items() if v.get()]
        examined_str="|".join(examined)
        comp_list=[n for n, v in self.check_vars.items() if v.get()] if self.op_var.get() else []
        
        seg_ok=detect_segmentation_status(new_id)
        report_ok=detect_report_status(new_id, examined_str)
        cfd_ok=detect_cfd_status(new_id)
        img_ok=detect_image_status(new_id)
        
        data={
            "ID": new_id, "Requires_Op": "Yes" if self.op_var.get() else "No", "Complications": ", ".join(comp_list),
            "Notes": self.notes_text.get("1.0", tk.END).strip(), 
            "Segmentation": seg_ok, 
            "Report_Analysis": report_ok,
            "CFD_Simulations": cfd_ok, "Image_Processing": img_ok,
            "Labeling": self.labeling_var.get(),
            "Examined_Files": examined_str
        }

        if self.original_id is not None:
            if self.original_id != new_id: 
                self.df=self.df[self.df['ID'] != self.original_id]
            self.df=self.df[self.df['ID'] != new_id]
        
        self.df=pd.concat([self.df, pd.DataFrame([data])], ignore_index=True)
        self.df=self.df.sort_values(by="ID").reset_index(drop=True)
        self.df.to_csv(FILE_NAME, index=False)
        
        messagebox.showinfo("Saved", f"Patient {new_id} updated.")
        self.original_id=new_id
        self.set_form_state("disabled")
        self._invalidate_patient(new_id)
        if new_id not in self._row_widgets:
            self._build_all_rows()
        else:
            self._update_row_colors(new_id)
        self.refresh_list()

    def handle_op_toggle(self):
        st="normal" if (self.op_var.get() and self.save_btn['state']=='normal') else "disabled"
        for cb in self.check_widgets: 
            cb.config(state=st)

    def set_form_state(self, state):
        widgets_to_toggle=[self.id_entry, self.op_check, self.notes_text,self.labeling_cb]
        for w in widgets_to_toggle:
            w.config(state=state)
        comp_state=state if (state=="normal" and self.op_var.get()) else "disabled"
        for child in self.comp_frame.winfo_children():
            if isinstance(child, tk.Checkbutton):
                child.config(state=comp_state)
        self.save_btn.config(state=state)
        can_action=(state=="disabled" and self.original_id is not None)
        self.modify_btn.config(state="normal" if can_action else "disabled")
        self.remove_btn.config(state="normal" if can_action else "disabled")
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

    def _on_mousewheel(self, event):
        self._list_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
    def _set_filter(self, label):
        self.filter_var.set(label)
        for lbl, btn in self._filter_btns.items():
            btn.config(relief="sunken" if lbl==label else "raised")
        self.refresh_list()
        
    def _bool_val(self, v) -> bool:
        return str(v).lower()=="true" or v is True
    
    def _rect_color(self, ok) -> str:
        if isinstance(ok, str) and ok.startswith("#"):
            return ok
        if ok in REPORT_COLOR_MAP:
            return REPORT_COLOR_MAP[ok]
        return "#2ecc71" if ok else "#dde0e3"
    
    def _overall_color(self, count: int) -> str:
        if count==0: return "#e74c3c"
        if count<5: return "#e67e22"
        return "#27ae60"

    def _debounce_refresh(self):
        if self._refresh_job:
            self.root.after_cancel(self._refresh_job)
        self._refresh_job=self.root.after(200, self.refresh_list)

    def _rebuild_cache(self):
        self._status_cache={}
        if self.df.empty:
            return
        for index, row in self.df.iterrows():
            try:
                pid=int(row["ID"])
            except (ValueError, TypeError):
                continue
            sim_status=detect_simulation_status(pid)
            cfd_rect_color=SIM_COLOR_MAP.get(sim_status, "#dde0e3")
            seg=detect_segmentation_status(pid)
            rep_status=detect_report_status(pid, str(row.get("Examined_Files", "")))
            img=detect_image_status(pid)
            lbl=self._bool_val(row.get("Labeling", False))

            self.df.at[index, "Segmentation"]=seg 
            self.df.at[index, "Report_Analysis"]=rep_status
            self.df.at[index, "Image_Processing"]=img

            count=sum([seg, rep_status=="OK", (sim_status=="Post processed"), img, lbl])
            self._status_cache[pid]={
                "seg": seg,
                "rep": rep_status,
                "cfd": cfd_rect_color,
                "img": img,
                "lbl": lbl,
                "sim_text": sim_status,
                "count": count,
                "overall": self._overall_color(count)
            }
        
    def _invalidate_patient(self, pid: int):
        rows=self.df[self.df["ID"]==pid]
        if rows.empty:
            self._status_cache.pop(pid, None)
            return
        row=rows.iloc[0]
        sim_status=detect_simulation_status(pid)
        cfd_rect_color=SIM_COLOR_MAP.get(sim_status, "#dde0e3")
        seg_ok=self._bool_val(row["Segmentation"])
        rep_status=detect_report_status(pid, str(row.get("Examined_Files", "")))
        img_ok=self._bool_val(row["Image_Processing"])
        lbl_ok=self._bool_val(row.get("Labeling", False))
        count=sum([seg_ok, rep_status=="OK", (sim_status=="Post processed"), img_ok, lbl_ok])
        self._status_cache[pid]={
            "seg": seg_ok,
            "rep": rep_status,
            "cfd": cfd_rect_color,
            "img": img_ok,
            "lbl": lbl_ok,
            "sim_text": sim_status,
            "count": count,
            "overall": self._overall_color(count)
        }

    def _build_all_rows(self):
        for w in self._list_inner.winfo_children():
            w.destroy()
        self._patient_rows=[]
        self._row_widgets={}
        for pid in sorted(self._status_cache.keys()):
            st=self._status_cache[pid]
            sep=tk.Frame(self._list_inner, bg="#eeeeee", height=1)
            row_f=tk.Frame(self._list_inner, bg="white", cursor="hand2", pady=3, padx=4)
            id_lbl=tk.Label(row_f, text=str(pid), font=("Arial", 10, "bold"), anchor="w")
            id_lbl.pack(side="left", fill="x", expand=True)
            rect_specs=["seg", "rep", "cfd", "img", "lbl", "overall"]
            sq_labels=[]
            for key in reversed(rect_specs):
                color=st["overall"] if key=="overall" else self._rect_color(st[key])
                sq=tk.Label(row_f, bg=color, width=3, relief="flat")
                sq.pack(side="right", padx=2, ipady=6)
                sq_labels.append((key, sq))
                
            def _click(e, p=pid, rf=row_f):
                self._select_patient_row(p, rf)
            for w in [row_f, id_lbl] + list(row_f.winfo_children()):
                w.bind("<Button-1>", _click)
            self._row_widgets[pid]=(row_f, id_lbl, sep, sq_labels)
            self._patient_rows.append((pid, row_f))
            
    def refresh_list(self):
        if not hasattr(self, "_row_widgets"):
            return
        term=self.search_var.get().lower()
        s_filt=self.filter_var.get()
        sub_filt=self.sub_filter_var.get()
        
        for pid in self._status_cache.keys():
            row_f, _, sep, _=self._row_widgets[pid]
            row_f.pack_forget()
            sep.pack_forget()

        r,y,g=0,0,0
        sorted_pids=sorted(self._status_cache.keys())
        for pid in sorted_pids:
            st=self._status_cache[pid]
            if st is None:
                continue

            overall=st["overall"]
            if overall=="#e74c3c": r += 1
            elif overall=="#e67e22": y += 1
            else: g += 1
            
            match_search=(term=="" or term in str(pid).lower())
            match_filter=(s_filt=="All" or (s_filt=="Red" and overall=="#e74c3c") or (s_filt=="Yellow" and overall=="#e67e22") or (s_filt=="Green" and overall=="#27ae60"))
            match_sub=True
            if sub_filt!="All":
                try:
                    parts=sub_filt.split(":")
                    step_name=parts[0].strip()
                    status_goal=parts[1].strip()
                    if step_name=="CFD Simulations":
                        cfd_map={
                            "Missing": "not available",
                            "To Run": "To Run",
                            "Running": "Running",
                            "Post Processing Req": "Completed: post processing required",
                            "OK": "Post processed"
                        }
                        match_sub=(st.get("sim_text")==cfd_map.get(status_goal))
                    elif step_name=="Report Analysis":
                        current_status=st.get("rep", False)
                        match_sub=(current_status==status_goal)
                    else:
                        mapping={
                            "Segmentation": "seg",
                            "Image Processing": "img",
                            "Labeling": "lbl"
                        }
                        cache_key=mapping.get(step_name)
                        if cache_key:
                            actual_val=st.get(cache_key, False)
                            match_sub=(actual_val is True) if status_goal=="OK" else (actual_val is False)
                except (IndexError, ValueError):
                    match_sub=True

            if match_search and match_filter and match_sub:
                row_f, _, sep, _=self._row_widgets[pid]
                sep.pack(fill="x", side="top")
                row_f.pack(fill="x", side="top")
        
        self.stats_label.config(text=f"To Do {r}  |  Doing {y}  |  CompleteS {g}")
    
    def _select_patient_row(self, pid, row_frame):
        SEL_BG="#d6eaf8"
        for p_id, rf in self._patient_rows:
            widgets=self._row_widgets.get(p_id)
            if not widgets:
                continue
            row_f, id_lbl, sep, sq_labels=widgets
            row_f.config(bg="white")
            id_lbl.config(bg="white")
            st=self._status_cache.get(p_id, {})
            for key, sq in sq_labels:
                color=st.get("overall") if key=="overall" else self._rect_color(st.get(key, False))
                sq.config(bg=color)
    
        row_frame.config(bg=SEL_BG)
        self._row_widgets[pid][1].config(bg=SEL_BG)  
        #for child in row_frame.winfo_children():
        #    if isinstance(child, tk.Label) and child.cget("width")==0:
        #        child.config(bg=SEL_BG)
        self._load_patient(pid)
        
    def _load_patient(self, pid):
        if pid not in self.df["ID"].values:
            return
        row=self.df[self.df["ID"]==pid].iloc[0]
        self.original_id=pid
        self.header_var.set(f"Viewing Patient: {pid}")
        self.id_entry.config(state="normal")
        self.notes_text.config(state="normal")
        self.clear_fields()
        self.id_entry.insert(0, str(row["ID"]))
        self.op_var.set(row["Requires_Op"]=="Yes")
        labeling_raw=row.get("Labeling", "")
        self.labeling_var.set(str(labeling_raw).lower()=="true" or labeling_raw is True)
        comps=str(row["Complications"]).split(", ")
        for c in comps:
            if c in self.check_vars:
                self.check_vars[c].set(True)
        self.notes_text.insert("1.0", str(row["Notes"]))
        self.refresh_features(pid)
        self.refresh_media(pid)
        self.set_form_state("disabled")

    def remove_patient(self):
        if self.original_id is None:
            return
        confirmed=messagebox.askyesno(
            "Remove Patient",
            f"Permanently remove Patient {self.original_id} from the database?\n\nThis cannot be undone.",
            icon="warning"
        )
        if not confirmed:
            return
        removed_id=self.original_id
        self.df=self.df[self.df["ID"]!=self.original_id].reset_index(drop=True)
        self.df.to_csv(FILE_NAME, index=False)
        self._status_cache.pop(removed_id, None)
        self._build_all_rows()
        self.original_id=None
        self.clear_fields()
        self.header_var.set("Select a Patient")
        self.set_form_state("disabled")
        self.refresh_list()
        
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
        self.labeling_var.set(False)
        self.notes_text.delete("1.0", tk.END)
        for var in list(self.check_vars.values()):
            var.set(False)
        for lbl in [self.seg_status_lbl, self.report_status_lbl, self.cfd_status_lbl, self.img_status_lbl]:
            self._set_status_label(lbl, False)
        if hasattr(self, "cfd_text_lbl"):
            self.cfd_text_lbl.config(text="")
if __name__=="__main__":
    root=tk.Tk()
    app=PatientApp(root)
    root.mainloop()