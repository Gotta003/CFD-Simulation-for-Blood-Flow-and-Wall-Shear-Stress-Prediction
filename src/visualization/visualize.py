import os
import argparse
import numpy as np
import pandas as pd

try:
    import pyvista as pv
    pv.set_plot_theme("document")
    HAS_PYVISTA=True
except ImportError:
    HAS_PYVISTA=False

AT_RISK_THRESHOLDS={
    "TAWSS": ("lt", 0.4),
    "OSI": ("gt", 0.3),
    "ECAP": ("gt", 1.4),
    "RRT": ("gt", 5.0),
    "Pressure": ("gt", None),
}

COLORMAPS={
    "TAWSS": "Blues_r",
    "OSI": "hot",
    "ECAP": "RdYlGn_r",
    "RRT": "plasma",
    "Pressure": "viridis",
    "Velocity": "jet",
    "WSS": "coolwarm"
}

def visualize_field(vtp_path, field, out_dir, metrics_csv=None):
    if not HAS_PYVISTA:
        raise ImportError("Pyvista is required for mesh visualization (pip install pyvista)")
    mesh=pv.read(vtp_path)
    patient_id=os.path.splitext(os.path.basename(vtp_path))[0]
    landmarks_coords=[]
    landmarks_labels=[]
    if metrics_csv and os.path.exists(metrics_csv):
        df_metrics=pd.read_csv(metrics_csv)
        row=df_metrics[df_metrics["Patient_ID"].astype(str)==patient_id.replace("pz", "")]
        if not row.empty:
            for p_name in ["P1", "P2", "Pmax", "P3"]:
                try:
                    x=row[f'{p_name}_Cx'].values[0]
                    y=row[f'{p_name}_Cy'].values[0]
                    z=row[f'{p_name}_Slice Index'].values[0]
                    landmarks_coords.append([x,y,z])
                    landmarks_labels.append(p_name)
                except KeyError:
                    continue
    
    plotter=pv.Plotter(off_screen=True)
    cmap=COLORMAPS.get(field, "viridis")
    title=f"{patient_id} - {field}"
    if field not in mesh.point_data:
        plotter.add_mesh(mesh, color="lightgrey", opacity=0.5)
    else:
        arr=mesh.point_data[field]
        if arr.ndim==2:
            arr=np.linalg.norm(arr, axis=1)
            mesh.point_data[f"{field}_mag"]=arr
            plot_field=f"{field}_mag"
        else:
            plot_field=field
        plotter.add_mesh(mesh, scalars=plot_field, cmap=cmap, show_scalar_bar=True, scalar_bar_args={"title": field}, opacity=0.8)

    plotter.add_title(title, font_size=12)
    plotter.camera_position="xz"
    
    if landmarks_coords:
        points=np.array(landmarks_coords)
        plotter.add_points(points, color="red", point_size=15, render_points_as_spheres=True)
        plotter.add_point_labels(points, labels=landmarks_labels, font_size=18, text_color="black", point_color="yellow", shape_opacity=0.7, always_visible=True)
        screenshot_path=os.path.join(out_dir, f"{patient_id}_landmarks.png")
        plotter.screenshot(screenshot_path)
        plotter.close()
        print(f"Saved visualization for {field} to {screenshot_path}")
        return
        
    screenshot_path=os.path.join(out_dir, f"{patient_id}_{field}.png")
    plotter.screenshot(screenshot_path)
    plotter.close()
    print(f"Saved visualization for {field} to {screenshot_path}")

    if field in AT_RISK_THRESHOLDS:
        direction, threshold=AT_RISK_THRESHOLDS[field]
        if threshold is None:
            threshold=float(np.mean(arr)+2*np.std(arr)) 
        risk_mask=(arr<threshold) if direction=="lt" else (arr>threshold)
        mesh.point_data["at_risk"]=risk_mask.astype(np.float32)

        plotter2=pv.Plotter(off_screen=True)
        plotter2.add_mesh(mesh, scalars="at_risk", cmap="RdYlGn_r", clim=[0,1], show_scalar_bar=True, scalar_bar_args={"title": f"{field} - At Risk"})
        plotter2.add_title(f"{title} - At Risk Regions", font_size=12)
        plotter2.camera_position="xz"
        risk_path=os.path.join(out_dir, f"{patient_id}_{field}_at_risk.png")
        plotter2.screenshot(risk_path)
        plotter2.close()
        print(f"Saved at-risk visualization for {field} to {risk_path}")

def main():
    parser=argparse.ArgumentParser(description="Visualie CFD heatmaps")
    parser.add_argument("--vtp", required=True, help="Path to .vtp file")
    parser.add_argument("--field", default="TAWSS", choices=list(COLORMAPS.keys()), help="Field to visualize")
    parser.add_argument("--out", default="outputs/mesh_heatmaps/")
    parser.add_argument("--all_fields", action="store_true", help="render all fields")
    parser.add_argument("--landmarks_en", help="Argument to enable landmark visualization, provide path to metrics CSV file for coordinates")
    args=parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.all_fields:
        for f in COLORMAPS:
            visualize_field(args.vtp, f, args.out)
    else:
        if args.landmarks_en:
            visualize_field(args.vtp, args.field, args.out, metrics_csv=args.landmarks_en)
        else:
            visualize_field(args.vtp, args.field, args.out)
        
if __name__=="__main__":
    main()