# TO DO
# Code to compute specifics metrics like P1, P2, P3, etc. based on extracted features, and composing a unified table with all morphological features for each patient

#P1 -> Renal Margin, last superior branch point = highest Slice Index
#P2 -> End Neck, where diameter diverge exceeding baseline by >10%, below P1
#P3 -> Bifurcation point, where the centerline bifurcates, lowest Slice Index (most inferior point of aneurysm sac)
#P4/P5 -> Start iliac branches
#Pmax -> Max diameter point

import os
import argparse 
import pandas as pd
import numpy as np

def main():
    parser=argparse.ArgumentParser(description="Compute morphological metrics from extracted features and aneurism location")
    parser.add_argument("--patientid", required=True, help="Path to CSV files containing patient CSV files")
    parser.add_argument("--in_folder", required=True, help="Parent path containing the pz\{PATIENT_ID\} format subfolders of the extracted features CSV files")
    parser.add_argument("--out_folder", required=True, help="Path to folder where the unified metrics CSV files will be saved")
    args=parser.parse_args()
    paths={
        'centerline': os.path.join(args.in_folder, f"pz{args.patientid}", f"centerline_data.csv"),
        's_view': os.path.join(args.in_folder, f"pz{args.patientid}", f"parameters_S.csv"),
        'a_view': os.path.join(args.in_folder, f"pz{args.patientid}", f"parameters_A.csv"),
        'r_view': os.path.join(args.in_folder, f"pz{args.patientid}", f"parameters_R.csv"),
        'volume': os.path.join(args.in_folder, f"pz{args.patientid}", f"volume.csv")
    }
    for f in paths:
        if not os.path.isfile(paths[f]):
            print(f"{f} file not found for patient {args.patientid} at {paths[f]}")
            return
    df_center=pd.read_csv(paths['centerline'])
    df_volume=pd.read_csv(paths['volume'])
    
    def rename_view(df, prefix):
        return df.add_prefix(f"{prefix}_").rename(columns={f"{prefix}_Slice Index": "Slice Index"})
    
    df_s=rename_view(pd.read_csv(paths['s_view']), "S")
    df_a=rename_view(pd.read_csv(paths['a_view']), "A")
    df_r=rename_view(pd.read_csv(paths['r_view']), "R")
    
    master_df=(df_s.merge(df_a, on="Slice Index", how="outer").merge(df_r, on="Slice Index", how="outer").sort_values(by="Slice Index").reset_index(drop=True))
    
    #Computation landmarks P1-P5
    
    length_col="S_Length (mm)"
    if length_col not in master_df.columns:
        length_col=next((c for c in master_df.columns if "ength" in c and "S" in c), None)
    diam_S="S_Max Diameter (mm)"
    diam_A="A_Max Diameter (mm)"
    diam_R="R_Max Diameter (mm)"
    for col in [diam_S, diam_A, diam_R]:
        if col not in master_df.columns:
            print(f"[ERROR] Expected column '{col}' not found. Available: {list(master_df.columns)}")
            return
    master_df["D_Global_Max"]=master_df[[diam_S, diam_A, diam_R]].max(axis=1)
    idx_max=master_df["D_Global_Max"].idmax()
    p_max=master_df.loc[idx_max]
    #P1
    p1=master_df.iloc[-1]
    #P3
    p3=master_df.iloc[0]
    #P2
    top_slices=master_df.nlargest(10, 'Slice Index')
    d_baseline=top_slices[diam_S].mean()
    p1_slice_idx=p1['Slice Index']
    p3_slice_idx=p3['Slice Index']
    p2_candidates=master_df[(master_df[diam_S]>d_baseline*1.1)&(master_df['Slice Index']<p1_slice_idx)&(master_df['Slice Index']>p3_slice_idx)].sort_values('Slice Index', ascending=False)
    p2=p2_candidates.iloc[0] if not p2_candidates.empty else p1
    
    p1_coords=np.array([p1['S_Cx'], p1['S_Cy'], p1['Slice Index']])
    p3_coords=np.array([p3['S_Cx'], p3['A_Cy'], p3['Slice Index']])
    dist_straight=np.linalg.norm(p3_coords-p1_coords)
    if length_col and length_col in master_df.columns:
        total_length=master_df[length_col].max()
    else:
        print(f"[WARN] Length column not found - tortuosity will be estimated from  slice spacing")
        total_length=dist_straight
    tortuosity=total_length/dist_straight if dist_straight>0 else 1.0
    A_at_max=p_max[diam_A]
    S_at_max=p_max[diam_S]
    if pd.isna(A_at_max) or S_at_max==0:
        print(f"[WARN] S diameter at Pmax is {S_at_max} - saccularity set to NaN")
        saccularity=float('nan')
    else:
        saccularity=S_at_max/A_at_max
    df_volume=pd.read_csv(paths['volume'])
    volume_col_priority=["Volume mm3 (CS)", "Volume mm3 (LM)", "Volume mm3 (SV)"]
    total_volume=float('nan')
    for vcol in volume_col_priority:
        if vcol in df_volume.columns and not df_volume[vcol].isna().all():
            total_volume=round(float(df_volume[vcol].iloc[0]), 1)
            print(f"[INFO] Using volume column: '{vcol}'={total_volume} mm3")
            break
    if pd.isna(total_volume):
        print(f"[WARN] No volume column found. Available: {list(df_volume.columns)}")
        
    def safe_angle(a, b, c):
        v1=np.array(a)-np.array(b)
        v2=np.array(c)-np.array(b)
        n1, n2=np.linalg.norm(v1), np.linalg.norm(v2)
        if n1==0 or n2==0:
            return float('nan')
        cos_angle=np.clip(np.dot(v1, v2)/(n1*n2), -1.0, 1.0)
        return round(float(np.degrees(np.arccos(cos_angle))), 2)
    
    p2_coords=np.array([p2['S_Cx'], p2['S_Cy'], p2['Slice Index']])
    pmax_coords=np.array([p_max['S_Cx'], p_max['S_Cy'], p_max['Slice Index']])
    neck_angle=safe_angle(p1_coords, p2_coords, pmax_coords)
            
    unified_row = {
        "Patient_ID":           args.patientid,
 
        # Landmark diameters
        "P1_Diam_Axial_mm":     round(p1[diam_S],   2),
        "P2_Diam_Axial_mm":     round(p2[diam_S],   2),
        "Pmax_Diam_Axial_mm":   round(p_max[diam_S], 2),
        "Pmax_Diam_Coronal_mm": round(p_max[diam_A], 2) if not pd.isna(p_max[diam_A]) else float('nan'),
        "P3_Diam_Axial_mm":     round(p3[diam_S],   2),
 
        # Lengths
        "L_Neck_mm":            round(abs(p2['Slice Index'] - p1['Slice Index']), 2),
        "L_Aneurysm_mm":        round(abs(p3['Slice Index'] - p2['Slice Index']), 2),
        "L_Total_mm":           round(total_length, 2) if not pd.isna(total_length) else float('nan'),
 
        # Shape metrics
        "Tortuosity":           round(tortuosity, 3),
        "Saccularity_Ratio":    round(saccularity, 3) if not pd.isna(saccularity) else float('nan'),
        "Neck_Angle_deg":       neck_angle,
 
        # Volume
        "Total_Volume_mm3":     total_volume,
 
        # Baseline (diagnostic context)
        "D_Baseline_mm":        round(d_baseline, 2),
    }
 
    os.makedirs(args.out_folder, exist_ok=True)
    output_path = os.path.join(args.out_folder, "morpho_unified_metrics.csv")
    final_df    = pd.DataFrame([unified_row])
    file_exists = os.path.isfile(output_path)
    final_df.to_csv(output_path, mode='a', header=not file_exists, index=False)
    print(f"[INFO] Patient {args.patientid} metrics saved to {output_path}")
    print(final_df.to_string(index=False))
    
if __name__=="__main__":
    main()