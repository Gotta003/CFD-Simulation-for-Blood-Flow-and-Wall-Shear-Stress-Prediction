import argparse
import os
import glob
import numpy as np
import pandas as pd
from typing import Optional

try:
    import pyvista as pv
    USE_PYVISTA = True
except ImportError:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    USE_PYVISTA = False

# Physical Limits - To check if they are okay
PHYSICAL_BONDS: dict[str, tuple[float,float]]={
    "TAWSS": (0.0, 200.0), #Pa
    "OSI": (0.0, 0.499), #pure number (0.5 should be maximum)
    "ECAP": (0.0, 100.0), #Pa^(-1) - >1000 artifact 
    "RRT": (0.0, 100.0), #Pa^(-1) - >100k artifact
    "Pressure": (0.0, 300000.0), #Pa 
    "Vorticity": (0.0, 5000.0), #1/s
    "Divergence": (-5000.0, 5000.0), 
    "WSS": (0.0, 200.0), #Pa
    "Velocity": (0.0, 5.0), #m/s
    "Traction": (0.0, 300000.0), #Pa
}
TAWSS_FLOOR=1e-3
# Clinical Thresholds - To setup properly 
AT_RISK_THRESHOLD: dict[str, tuple[str, Optional[float]]]={
    "TAWSS":("lt", 0.4), #<0.4Pa
    "OSI":("gt", 0.3), #>0.3
    "ECAP":("gt", 1.4), #>1.4Pa^(-1)
    "RRT":("gt", 5.0), #>5.0Pa^(-1)
    "Pressure":("gt", None), #Dynamic
}

SCALAR_FIELDS=["TAWSS", "OSI", "ECAP", "RRT", "WSS", "Pressure", "Vorticity", "Divergence"]
VECTOR_FIELDS=["Velocity", "Traction"]

SANITY_BOUNDS: dict[str, tuple[float,float]]={
    "tawss_mean": (0.0, 200.0),
    "osi_mean": (0.0, 0.499),
    "ecap_mean": (0.0, 100.0),
    "rrt_mean": (0.0, 100.0),
    "pressure_mean": (0.0,300000.0),
    "wss_mean": (0.0, 200.0),
    "velocity_mean": (0.0, 5.0),
}

def load_vtp_pyvista(path:str):
    mesh=pv.read(path)
    return mesh

def get_array_pyvista(mesh, name:str):
    if name in mesh.point_data:
        arr=np.array(mesh.point_data[name], dtype=np.float64)
        if arr.ndim==2:
            arr=np.linalg.norm(arr, axis=1)
        return arr
    return None

def get_point_areas_pyvista(mesh):
    try:
        mesh_with_areas=mesh.compute_cell_sizes(length=False, area=True, volume=False)
        point_mesh=mesh_with_areas.cell_data_to_point_data()
        if "Area" in point_mesh.point_data:
            areas=np.array(point_mesh.point_data["Area"], dtype=np.float64)
            areas=np.clip(areas, 0.0, None)
            return areas
    except Exception as e:
        print(f"  [WARN]: Failed Area: {e}")
    return None

def load_vtp_vtk(path:str):
    reader=vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()

def get_array_vtk(mesh, name:str):
    pd=mesh.GetPointData()
    arr_vtk=pd.GetArray(name)
    if arr_vtk is None:
        return None
    arr=vtk_to_numpy(arr_vtk).astype(np.float64)
    if arr.ndim==2:
        arr=np.linalg.norm(arr, axis=1)
    return arr

def clean_array(arr, field_name) -> np.ndarray:
    arr=arr[np.isfinite(arr)]
    if field_name in PHYSICAL_BONDS:
        l, h=PHYSICAL_BONDS[field_name]
        if l==h==0.0:
            return np.array([], dtype=np.float64)
        arr=np.clip(arr, l, h)
    return arr

def is_zero_variance(arr, tol=1e-10):
    return len(arr)==0 or np.std(arr)<tol

def compute_stats(arr:np.ndarray, prefix:str, areas=Optional[np.ndarray]) -> dict:
    arr=arr[np.isfinite(arr)]
    if len(arr)==0:
        return {}
    stats={
        f"{prefix}_mean": np.mean(arr),
        f"{prefix}_std": np.std(arr),
        f"{prefix}_median": np.median(arr),
        f"{prefix}_min": np.min(arr),
        f"{prefix}_p99": np.percentile(arr, 99),
        f"{prefix}_p05": np.percentile(arr, 5)
    }
    if areas is not None and len(areas)==len(arr):
        total=areas.sum()
        if total>0:
            w=areas/areas.sum()
            stats[f"{prefix}_area_mean"]=float(np.dot(arr, w))

    return stats

def compute_at_risk(arr:np.ndarray, field_name:str, prefix:str) -> dict:
    if field_name not in AT_RISK_THRESHOLD or len(arr)==0:
        return {}
    direction, threshold=AT_RISK_THRESHOLD[field_name]
    if threshold is None:
        threshold=float(np.mean(arr)+2.0*np.std(arr))
    if direction=="lt":
        mask=arr<threshold
    else:
        mask=arr>threshold
   
    return {f"{prefix}_pct_at_risk": float(np.mean(mask))}

def validate_row(features: dict, patient_id: str) -> list[str]:
    issues = []
    for col, (lo, hi) in PHYSICAL_BONDS.items():
        if col in features:
            val = features[col]
            if not np.isfinite(val) or not (lo <= val <= hi):
                issues.append(f"{col}={val:.3e} outside [{lo}, {hi}]")

    for prefix in ("tawss", "osi", "ecap", "rrt", "wss", "pressure", "velocity"):
        mean_col = f"{prefix}_mean"
        p99_col  = f"{prefix}_p99"
        if mean_col in features and p99_col in features:
            mean_val = features[mean_col]
            p99_val  = features[p99_col]
            if np.isfinite(mean_val) and np.isfinite(p99_val):
                if mean_val > p99_val * 1.05:  # 5% tolerance 
                    issues.append(
                        f"{prefix}: mean ({mean_val:.3f}) > p99 ({p99_val:.3f}) "
                        f"— clipping applied inconsistently, check clean_array call order"
                    )

    return issues

def extract_features(vtp_path: str) -> dict:
    id=os.path.splitext(os.path.basename(vtp_path))[0]
    features={"patient_id": id}
    if USE_PYVISTA:
        mesh=load_vtp_pyvista(vtp_path)
        get_array=lambda name: get_array_pyvista(mesh, name)
        point_areas=get_point_areas_pyvista(mesh)
    else:
        mesh=load_vtp_vtk(vtp_path)
        get_array=lambda name: get_array_vtk(mesh, name)
        point_areas=None

    all_fields=SCALAR_FIELDS+VECTOR_FIELDS
    for field in all_fields:
        arr_raw=get_array(field)
        if arr_raw is None:
            print(f"  [WARN]: Field '{field}' not found in patient {id}")
            continue
        arr=clean_array(arr_raw, field)
        if(len(arr)==0):
            if field not in ("Divergence",):
                print(f"  [WARN]: Field '{field}' has no valid values for patient {id}")
            continue
        if is_zero_variance(arr):
            print(f"  [WARN]: Field '{field}' has zero variance for patient {id}")

        prefix=field.lower()
        aligned_areas=None
        if point_areas is not None and len(point_areas)==len(arr_raw):
            finite_mask=np.isfinite(arr_raw)
            if finite_mask.all():
                aligned_areas=point_areas

        features.update(compute_stats(arr, prefix, aligned_areas))
        features.update(compute_at_risk(arr, field, prefix))
    return features

def main():
    parser=argparse.ArgumentParser(description="Extract hermodynamic features from CFD .vtp files")
    parser.add_argument("--input", required=True, help="Dir containing .vtp files")
    parser.add_argument("--out", default="outputs/features/features.csv", help="Path to output CSV file")
    args=parser.parse_args()
    if os.path.isdir(args.input):
        vtp_files=sorted(glob.glob(os.path.join(args.input, "*.vtp")))
    else:
        vtp_files=[args.input]
    if not vtp_files:
        print(f"No .vtp files found in {args.input}")
        return
    
    print(f"Processing {len(vtp_files)} file(s)...\n")
    all_features=[]
    validation_failures=[]
    for path in vtp_files:
        print(f"  - {os.path.basename(path)}")
        try:
            features=extract_features(path)
            issues=validate_row(features, features["patient_id"])
            if issues:
                print(f"  [VALIDATION] {features['patient_id']}: {'; '.join(issues)}")
                validation_failures.append((features["patient_id"]))
            all_features.append(features)
        except Exception as e:
            print(f"     [ERROR]: {e}")

    if not all_features:
        print("No features extracted.")
        return

    df=pd.DataFrame(all_features)
    before=df.shape[1]
    df=df.dropna(axis=1, how="all")
    dropped=before-df.shape[1]
    if dropped:
        print(f"\nDropped {dropped} all-NaN columns")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nFeatures saved in {args.out}")
    print(f"Shape: {df.shape[0]} patients x {df.shape[1]-1} features")

    if validation_failures:
        print(f"\n{len(validation_failures)} patients failed with out-of.range values:")
        for pid in validation_failures:
            print(f"  - {pid}")
        print("Check CFD for these cases")

if __name__ == "__main__":
    main()