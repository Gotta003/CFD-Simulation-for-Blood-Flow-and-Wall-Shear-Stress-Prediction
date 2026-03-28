import argparse
import os
import glob
from pathlib import Path
import numpy as np

try:
    import pyvista as pv
    HAS_PYVISTA=True
except ImportError:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    HAS_PYVISTA=False

ALL_EMBED=["TAWSS", "OSI", "RRT"]

PHYSICAL_BOUNDS: dict[str, tuple[float, float]]={
    "TAWSS": (0.0, 200.0),
    "OSI": (0.0, 0.499),
    "RRT": (0.0, 100.0),
}

def _load_mesh(path: str):
    if HAS_PYVISTA:
        return pv.read(path), "pyvista"
    reader=vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput(), "vtk"

def _get_points(mesh, backend: str) -> np.ndarray:
    if backend=="pyvista":
        return np.array(mesh.points, dtype=np.float32)
    from vtk.util.numpy_support import vtk_to_numpy
    return vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float32)

def _get_array(mesh, field: str, backend: str) -> np.ndarray | None:
    if backend=="pyvista":
        if field not in mesh.point_data:
            return None
        arr=np.array(mesh.point_data[field], dtype=np.float64)
    else:
        from vtk.util.numpy_support import vtk_to_numpy
        pd=mesh.GetPointData()
        raw=pd.GetArray(field)
        if raw is None:
            return None
        arr=vtk_to_numpy(raw).astype(np.float64)
        
    if arr.ndim==2:
        arr=np.linalg.norm(arr, axis=1)
    return arr

def _clamp(arr: np.ndarray, field: str) -> np.ndarray:
    if field in PHYSICAL_BOUNDS:
        l, h=PHYSICAL_BOUNDS[field]
        arr=np.clip(arr, l, h)
    return arr

# SAMPLING STRATEGIES
def _random_sample(n_points: int, n_sample: int, rng: np.random.Generator) -> np.ndarray:
    if(n_sample>=n_points):
        return np.arange(n_points)
    return rng.choice(n_points, size=n_sample, replace=False)
    
def _fps(points: np.ndarray, n_sample: int, rng: np.random.Generator) -> np.ndarray:
    n_points=len(points)
    if n_sample>=n_points:
        return np.arange(n_points)
    selected=np.zeros(n_sample, dtype=np.int64)
    selected[0]=rng.integers(0, n_points)
    dists=np.full(n_points, np.inf)
    for i in range(1, n_sample):
        last=points[selected[i-1]]
        d=np.sum((points-last)**2, axis=1)
        dists=np.minimum(dists, d)
        selected[i]=int(np.argmax(dists))
    return selected
    
def process_vtp(vtp_path: str, n_points: int, strategy: str, rng: np.random.Generator) -> dict | None:
    mesh, backend=_load_mesh(vtp_path)
    points=_get_points(mesh, backend)
    n_mesh=len(points)
    if n_mesh==0:
        print(f"[WARN] No points in {vtp_path}")
        return None
    cfd:dict[str, np.ndarray]={}
    # CFD Arrays
    for field in ALL_EMBED:
        arr=_get_array(mesh, field, backend)
        if arr is None:
            print(f"[WARN] Field {field} missing")
            arr=np.zeros(n_mesh, dtype=np.float64)
        arr=_clamp(arr, field)
        cfd[field.lower()]=arr.astype(np.float32)
    # Valid-point mask
    valid_mask=np.ones(n_mesh, dtype=bool)
    for arr in cfd.values():
        valid_mask&=np.isfinite(arr)
    valid_mask&=np.all(np.isfinite(points), axis=1)
    valid_idx=np.where(valid_mask)[0]
    if len(valid_idx)==0:
        print(f"[ERROR] No valid points in {vtp_path}")
        return None
    #Sampling
    points_valid=points[valid_idx]
    if strategy=="fps":
        local_idx=_fps(points_valid, n_points, rng)
    else:
        local_idx=_random_sample(len(valid_idx), n_points, rng)
    sel=valid_idx[local_idx]
    
    if len(sel)<n_points:
        pad=rng.choice(sel, size=n_points-len(sel), replace=True)
        sel=np.concatenate([sel,pad])
    result: dict[str, np.ndarray]={
        "xyz": points[sel]
    }
    for key, arr in cfd.items():
        result[key]=arr[sel]
    result["mask"]=valid_mask[sel]
    return result
    
def main():
    parser=argparse.ArgumentParser(description="Sample .vtp meshes into .npz point clouds")
    parser.add_argument("--vtp_dir", required=True, help="Dir containing .vtp files")
    parser.add_argument("--out_dir", default="data/pointclouds/")
    parser.add_argument("--n_points", type=int, default=4096, help="Points per cloud")
    parser.add_argument("--strategy", choices=["fps", "random"], default="fps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args=parser.parse_args()
    
    vtp_root=Path(args.vtp_dir)
    vtp_files=sorted(list(vtp_root.rglob("*.vtp")))
    #vtp_files=sorted(glob.glob(os.path.join(args.vtp_dir, "*.vtp")))
    if not vtp_files:
        print(f"No .vtp files found in {args.vtp_dir} or subdirs")
        return
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    rng=np.random.default_rng(args.seed)
    ok, skipped, failed=0,0,0
    for vtp_path in vtp_files:
        pid=vtp_path.parent.name
        out_path=os.path.join(args.out_dir, f"pz{pid:03d}.npz")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[SKIP] {pid} - already exists")
            skipped+=1
            continue
        print(f"Processing {pid} ({args.strategy}, N={args.n_points})")
        try:
            data=process_vtp(str(vtp_path), args.n_points, args.strategy, rng)
            if data is None:
                failed+=1
                continue
            np.savez_compressed(out_path, **data)
            print(f"{out_path} xyz{data['xyz'].shape}")
            ok+=1
        except Exception as e:
            print(f"[ERROR] {pid}: {e}")
            failed+=1
    print(f"\nDone - ok: {ok} skipped: {skipped} failed: {failed}")
    print(f"Point clouds saved to {args.out_dir}")
    print(f"Channels per point: xyz + {[f.lower() for f in ALL_EMBED]}")
    
if __name__=="__main__":
    main()