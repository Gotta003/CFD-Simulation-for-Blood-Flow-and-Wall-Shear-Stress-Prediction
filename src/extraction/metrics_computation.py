import os
import argparse 
import pandas as pd
import numpy as np

def extract_3d_stats(volume_path):
    """Parses Slicer SegmentStatistics for Volume and Surface Area."""
    stats = {"Volume": 0, "SurfaceArea": 0}
    if not os.path.exists(volume_path):
        return stats
    
    v_df = pd.read_csv(volume_path)
    # Mapping specific Slicer headers 
    col_vol = "LabelmapSegmentStatisticsPlugin.volume_mm3"
    col_surf = "ClosedSurfaceSegmentStatisticsPlugin.surface_mm2"
    
    if col_vol in v_df.columns:
        stats["Volume"] = v_df[col_vol].iloc[0]
    if col_surf in v_df.columns:
        stats["SurfaceArea"] = v_df[col_surf].iloc[0]
    return stats

def build_main_trunk(cl: pd.DataFrame, tol_mm: float=2.0) -> pd.DataFrame:
    n=len(cl)
    starts=cl[['StartPoinPosition_R', 'StartPointPosition_A', 'StartPointPosition_S']].values
    ends=cl[['EndPointPosition_R', 'EndPointPosition_A', 'EndPointPosition_S']].values
    children={i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i==j:
                continue
            if np.linalg.norm(ends[i]-starts[j])<tol_mm:
                children[i].append(j)
    all_children={c for lst in children.values() for c in lst}
    roots=[i for i in range(n) if i not in all_children]
    if not roots:
        roots=[0]
    best_length=-1
    best_path=[]
    
    def dfs(node, path, length):
        nonlocal best_length, best_path
        path=path+[node]
        length+=cl.iloc[node]['Length']
        if length>best_length:
            best_length=length
            best_path=path
        for child in children[node]:
            if child not in path:
                dfs(child, path, length)
    
    for r in roots:
        dfs(r, [], 0.0)
    return cl.iloc[best_path].reset_index(drop=True)

def locate_aaa_on_trunk(trunk: pd.DataFrame, aaa_radius_thresh_mm: float=15.0, min_aaa_segment: int=2) -> tuple:
    radii=trunk['Radius'].values
    in_aaa=(radii>=aaa_radius_thresh_mm)
    if in_aaa.sum()>=min_aaa_segment:
        best_start, best_end, cur_start=0, 0, None
        best_len=0
        for i, flag in enumerate(in_aaa):
            if flag and cur_start is None:
                cur_start=i
            elif not flag and cur_start is not None:
                if (i-cur_start)>best_len:
                    best_len=i-cur_start
                    best_start, best_end=cur_start, i-1
                cur_start=None
        if cur_start is not None and (len(in_aaa)-cur_start)>best_len:
            best_start, best_end=cur_start, len(in_aaa)-1
        prox_idx, dist_idx=best_start, best_end
    else:
        seed=int(np.argmax(radii))
        fallback_thresh=12.0
        prox_idx=seed
        while prox_idx>0 and radii[prox_idx-1]>=fallback_thresh:
            prox_idx-=1
        dist_idx=seed
        while dist_idx<len(trunk)-1 and radii[dist_idx+1]>=fallback_thresh:
            dist_idx+=1
    aaa_df=trunk.iloc[prox_idx: dist_idx+1].copy()
    return aaa_df, prox_idx, dist_idx

def compute_centerline_metrics(cl_path: str, aaa_radius_thresh_mm: float=15.0, torsion_outlier_sd: float=3.0, tol_mm: float=2.0) -> dict:
    empty={k: np.nan for k in [
        "CL_Trunk_Length_mm",
        "CL_AAA_Length_mm",
        "CL_AAA_Straight_mm",
        "CL_AAA_Tortuosity",
        "CL_Prox_Neck_Length_mm",
        "CL_AAA_MeanCurvature",
        "CL_AAA_MaxCurvature",
        "CL_AAA_MeanTorsion",
        "CL_AAA_MeanRadius_mm",
        "CL_AAA_MaxRadius_mm",
        "CL_Segments_AAA",
        "CL_AAA_found"
    ]}
    if not os.path.exists(cl_path):
        print(f"[WARN] Centerline not found: {cl_path}")
        return empty
    cl=pd.read_csv(cl_path)
    required={"Radius", "Length", "Curvature", "Torsion", "StartPointPosition_R", "StartPointPosition_A", "StartPointPosition_S", "EndPointPosition_R", "EndPointPosition_A", "EndPointPosition_S"}
    if not required.issubset(cl.columns):
        missing=required-cl.columns
        print(f"[WARN] Centerline CSV missing columns: {missing}")
        return empty
    #Extract Main Trunk
    trunk=build_main_trunk(cl, tol_mm=tol_mm)
    trunk_length=trunk['Length'].sum()
    #Localise AAA
    aaa_found_primary=(trunk['Radius']>=aaa_radius_thresh_mm).sum()>=2
    aaa_df, prox_idx, dist_idx=locate_aaa_on_trunk(trunk, aaa_radius_thresh_mm=aaa_radius_thresh_mm)
    #AAA arc-length
    aaa_length=aaa_df['Length'].sum()
    #AAA straight-line
    prox_pt=trunk.iloc[prox_idx][['StartPointPosition_R', 'StartPointPosition_A', 'StartPointPosition_S']].values.astype(float)
    dist_pt=trunk.iloc[dist_idx][['EndPointPosition_R', 'EndPointPosition_A', 'EndPointPosition_S']].values.astype(float)
    aaa_straight=np.linalg.norm(dist_pt-prox_pt)
    #Tortuosity
    aaa_tortuosity=aaa_length/aaa_straight if aaa_straight>0 else 1.0
    #Proximal neck length
    prox_neck_df=trunk.iloc[:prox_idx]
    prox_neck_length=prox_neck_df['Length'].sum()
    #Curvature
    w=aaa_df['Length']
    total_w=w.sum() if w.sum()>0 else 1.0
    mean_curv=(aaa_df['Curvature']*w).sum()/total_w
    max_curv=aaa_df['Curvature'].max()
    #Torsion
    tors=aaa_df['Torsion'].abs()
    mean_t, std_t=tors.mean(), tors.std()
    if std_t>0:
        mask=(tors-mean_t).abs()<torsion_outlier_sd*std_t
    else:
        mask=pd.Series([True]*len(tors), index=tors.index)
    w_clean=aaa_df.loc[mask, 'Length']
    mean_torsion=(tors[mask]*w_clean).sum()/w_clean.sum() if w_clean.sum()>0 else np.nan
    #Radius
    mean_radius=(aaa_df['Radius']*w).sum()/total_w
    max_radius=aaa_df['Radius'].max()
    return {
        'CL_Trunk_Length_mm': round(trunk_length, 2),
        'CL_AAA_Length_mm': round(aaa_length, 2),
        'CL_AAA_Straight_mm': round(aaa_straight, 2),
        'CL_AAA_Tortuosity': round(aaa_tortuosity, 4),
        'CL_Prox_Neck_Length_mm': round(prox_neck_length, 2),
        'CL_AAA_MeanCurvature': round(mean_curv, 5),
        'CL_AAA_MaxCurvature': round(max_curv, 5),
        'CL_AAA_MeanTorsion': round(mean_torsion, 5),
        'CL_AAA_MeanRadius_mm': round(mean_radius, 2),
        'CL_AAA_MaxRadius_mm': round(max_radius, 2),
        'CL_Segments_AAA': int(len(aaa_df)),
        'CL_AAA_found': int(aaa_found_primary),
    }

def main():
    parser = argparse.ArgumentParser(description="Full Morphological Feature Extraction")
    parser.add_argument("--patientid", required=True)
    parser.add_argument("--in_folder", required=True, help=r"Path to pz{ID} folders")
    parser.add_argument("--out_folder", required=True)
    parser.add_argument("--voxel_spacing", type=float, default=1.0, help="Slice thickness in mm to convert Slice Index to mm")
    parser.add_argument("--aaa_radius_thresh", type=float, default=15.0, help="Radius threshold (mm) to flag a centerline segment as aneurysmal")
    args = parser.parse_args()

    base_path = os.path.join(args.in_folder, f"pz{args.patientid}")
    try:
        df_axial = pd.read_csv(os.path.join(base_path, "parameters_S.csv"))
    except FileNotFoundError:
        print(f"[ERROR] Missing CSVs for pz{args.patientid}")
        return
    
    valid_df = df_axial[df_axial["Max Diameter (mm)"] > 10].copy()
    if valid_df.empty: return
    p1 = valid_df.iloc[0]
    idx_max = valid_df["Max Diameter (mm)"].idxmax()
    p_max = valid_df.loc[idx_max]
    baseline_diam = valid_df.iloc[0:5]["Max Diameter (mm)"].mean()
    p2_candidates = valid_df[valid_df['Max Diameter (mm)'] > (baseline_diam * 1.1)]
    p2 = p2_candidates.iloc[0] if not p2_candidates.empty else p1
    p3 = valid_df.iloc[-1]
    d4 = p3["Max Diameter (mm)"] * 0.55  
    d5 = p3["Max Diameter (mm)"] * 0.50
    l_curvilinear = abs(p3['Length (mm)'] - p1['Length (mm)'])
    # Straight length L' 
    p1_pos = np.array([p1['Cx'], p1['Cy'], p1['Slice Index']])
    p3_pos = np.array([p3['Cx'], p3['Cy'], p3['Slice Index']])
    l_straight = np.linalg.norm(p3_pos - p1_pos)
    deformation_ratio = p_max['Rmajor (mm)'] / p_max['Rminor (mm)'] if p_max['Rminor (mm)'] > 0 else 1.0
    saccular_index = p_max['Max Diameter (mm)'] / p_max['Perimeter (mm)'] * np.pi if p_max['Perimeter (mm)'] > 0 else 0
    stats_3d = extract_3d_stats(os.path.join(base_path, "volume.csv"))
    cl_path=os.path.join(base_path, "centerline_data.csv")  
    cl_metrics=compute_centerline_metrics(cl_path, aaa_radius_thresh_mm=args.aaa_radius_thresh)
    cl_neck=cl_metrics.get('CL_Prox_Neck_Length_mm', np.nan)
    axial_neck=round(abs(p2['Length (mm)'] - p1['Length (mm)']), 2)
    proximal_neck_length=cl_neck if not np.isnan(cl_neck) else axial_neck

    unified_row = {
        "patient_id": args.patientid,
        "D1": round(p1['Max Diameter (mm)'], 2),
        "D2": round(p2['Max Diameter (mm)'], 2),
        "D3": round(p3['Max Diameter (mm)'], 2),
        "D4": round(d4, 2), 
        "D5": round(d5, 2),
        "Dmax": round(p_max['Max Diameter (mm)'], 2),
        "Proximal_Neck_Length": round(proximal_neck_length, 2),
        "L_Curvilinear": cl_metrics.get('CL_AAA_Length_mm', np.nan),
        "L_Straight": cl_metrics.get('CL_AAA_Straight_mm', np.nan),
        "Tortuosity": cl_metrics.get('CL_AAA_Tortuosity', np.nan),
        "Deformation_Ratio": round(deformation_ratio, 3),
        "Saccularization_Index": round(saccular_index, 3),
        "AAA_Volume_mm3": round(stats_3d["Volume"], 1),
        "AAA_SurfaceArea_mm2": round(stats_3d["SurfaceArea"], 1),
        "CL_Trunk_Length_mm": cl_metrics.get('CL_Trunk_Length_mm', np.nan),
        "CL_AAA_MeanCurvature": cl_metrics.get('CL_AAA_MeanCurvature', np.nan),
        "CL_AAA_MaxCurvature": cl_metrics.get('CL_AAA_MaxCurvature', np.nan),
        "CL_AAA_MeanTorsion": cl_metrics.get('CL_AAA_MeanTorsion', np.nan),
        "CL_AAA_MeanRadius_mm": cl_metrics.get('CL_AAA_MeanRadius_mm', np.nan),
        "CL_AAA_MaxRadius_mm": cl_metrics.get('CL_AAA_MaxRadius_mm', np.nan),
        "CL_Segments_AAA": cl_metrics.get('CL_Segments_AAA',np.nan),
        "CL_AAA_found": cl_metrics.get('CL_AAA_found', np.nan),
    }

    # Save to unified CSV
    output_path = os.path.join(args.out_folder, "morpho_unified_metrics.csv")
    final_df = pd.DataFrame([unified_row])
    final_df.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
    print(f"[SUCCESS] pz{args.patientid} unified.")

if __name__ == "__main__":
    main()