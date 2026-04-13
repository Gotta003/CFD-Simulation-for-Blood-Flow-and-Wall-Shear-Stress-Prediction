# TO DO
# Code to compute specifics metrics like P1, P2, P3, etc. based on extracted features, and composing a unified table with all morphological features for each patient

#P1 -> Renal Margin, last superior branch point
#P2 -> End Neck, where diameter diverge
#P3 -> Biforcation point, where the centerline bifurcates
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
        'centerline': os.path.join(args.in_folder, f"pz{args.patientid}", f"centerlines.csv"),
        's_view': os.path.join(args.in_folder, f"pz{args.patientid}", f"parameters_s.csv"),
        'a_view': os.path.join(args.in_folder, f"pz{args.patientid}", f"parameters_a.csv"),
        'r_view': os.path.join(args.in_folder, f"pz{args.patientid}", f"parameters_r.csv"),
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
    
    df_s=rename_view(pd.read_csv(paths['s_view']), "Axial")
    df_a=rename_view(pd.read_csv(paths['a_view']), "Coronal")
    df_r=rename_view(pd.read_csv(paths['r_view']), "Sagittal")
    
    master_df=df_s.merge(df_a, on='Slice Index', how='outer')
    master_df=master_df.merge(df_r, on='Slice Index', how='outer')
    master_df=master_df.sort_values(by='Slice Index').reset_index(drop=True)
    
    #Computation landmarks P1-P5
    p1=master_df.iloc[0]
    d_baseline=master_df.iloc[0:10] ["Axial_Max Diameter (mm)"].mean()
    master_df['D_Global_Max']=master_df[["Axial_Max Diameter (mm)", "Coronal_Max Diameter (mm)", "Sagittal_Max Diameter (mm)"]].max(axis=1)
    idx_max=master_df['D_Global_Max'].idxmax()
    p_max=master_df.iloc[idx_max]
    
    p2_candidates=master_df[(master_df['Axial_Max Diameter (mm)'] > d_baseline * 1.1) & (master_df["Slice Index"] < p1["Slice Index"])]
    p2=p2_candidates.iloc[0] if not p2_candidates.empty else p1
    p3=master_df.iloc[-1]
    
    #Derived Metrics
    p1_coords=np.array([p1['Axial_Cx'], p1['Axial_Cy'], p1['Slice Index']])
    p3_coords=np.array([p3['Axial_Cx'], p3['Axial_Cy'], p3['Slice Index']])
    dist_straight=np.linalg.norm(p3_coords - p1_coords)
    total_length=master_df['Axial_length (mm)'].max() if 'Axial_length (mm)' in master_df else dist_straight*1.2
    tortuosity=total_length/dist_straight if dist_straight>0 else 1
    saccularity=p_max['Axial_Max Diameter (mm)']/p_max['Coronal_Max Diameter (mm)']

    unified_row = {
        "Patient_ID": args.patientid,
        "P1_Diam_Axial": round(p1['Axial_Max Diameter (mm)'], 2),
        "P2_Diam_Axial": round(p2['Axial_Max Diameter (mm)'], 2),
        "Pmax_Diam_Axial": round(p_max['Axial_Max Diameter (mm)'], 2),
        "Pmax_Diam_Coronal": round(p_max['Coronal_Max Diameter (mm)'], 2),
        "P3_Diam_Axial": round(p3['Axial_Max Diameter (mm)'], 2),
        "L_Neck_mm": abs(round(p2['Slice Index'] - p1['Slice Index'], 2)),
        "L_Aneurysm_mm": abs(round(p3['Slice Index'] - p2['Slice Index'], 2)),
        "Tortuosity": round(tortuosity, 3),
        "Saccularity_Ratio": round(saccularity, 3),
        "Total_Volume_mm3": round(df_volume['Volume'].iloc[0], 1) if 'Volume' in df_volume.columns else 0
    }
    
    final_df=pd.DataFrame([unified_row])
    output_path=os.path.join(args.out_folder, f"morpho_unified_metrics.csv")
    file_exists=os.path.isfile(output_path)
    final_df.to_csv(output_path, mode='a', header=not file_exists, index=False)
    print(f"[INFO] Patient {args.patientid} metrics computed and saved to {output_path}")

if __name__=="__main__":
    main()