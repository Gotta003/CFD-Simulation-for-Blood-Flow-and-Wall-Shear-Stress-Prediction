import argparse
import os
import glob
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
  
try:
    import pyvista as pv
    HAS_PYVISTA=True
except ImportError:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    HAS_PYVISTA=False


ALL_EMBED= ["p", "vx", "vy", "vz", "wss"]

FIELD_ALIASES: dict[str, list[str]]={
    "velocity": ["Velocity"],
    "pressure": ["Pressure"],
    "wss": ["WSS"],
}
    
PHYSICAL_BOUND: dict[str, tuple[float, float]]={
    "velocity": (0.0, 10.0),
    "pressure": (-500.0, 500.0),
    "wss": (0.0, 200.0)
}


def _array(mesh, name: str, backend: str) -> np.ndarray | None:
    if backend=="pv":
        for store in (mesh.point_data, mesh.cell_data):
            if name in store:
                arr=np.asarray(store[name], dtype=np.float64)
                return arr
        return None
    from vtk.util.numpy_support import vtk_to_numpy
    pd=mesh.GetPointData()
    raw=pd.GetArray(name)
    if raw is None:
        return None
    return vtk_to_numpy(raw).astype(np.float64)

def _get_field(mesh, field: str, backend: str) -> np.ndarray | None:
    for alias in FIELD_ALIASES[field]:
        arr=_array(mesh, alias, backend)
        if arr is not None:
            if arr.ndim==2:
                arr=np.linalg.norm(arr, axis=1)
            return arr.astype(np.float32)
    return None    

def _get_velocity_components(mesh, backend: str) -> tuple[np.ndarray | None, ...]:
    for alias in FIELD_ALIASES["velocity"]:
        arr=_array(mesh, alias, backend)
        if arr is not None and arr.ndim==2 and arr.shape[1]>=3:
            return(
                arr[:,0].astype(np.float32),
                arr[:,1].astype(np.float32),
                arr[:,2].astype(np.float32)
            )
    return None, None, None

def _get_wss_components(mesh, backend: str) -> tuple[np.ndarray | None, ...]:
    for alias in FIELD_ALIASES["wss"]:
        arr=_array(mesh, alias, backend)
        if arr is not None and arr.ndim==2 and arr.shape[1]>=3:
            return(
                arr[:,0].astype(np.float32),
                arr[:,1].astype(np.float32),
                arr[:,2].astype(np.float32)
            )
    return None, None, None

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

 
def _clamp(arr: np.ndarray, field: str) -> np.ndarray:
    if field in PHYSICAL_BOUND:
        l, h=PHYSICAL_BOUND[field]
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
    print(n_mesh, "points loaded")
    cfd:dict[str, np.ndarray]={}
    # CFD Arrays
    p = _get_field(mesh, "pressure", backend)
    p=_clamp(p, "pressure")
    cfd["p"]=p.astype(np.float32)
    vx, vy, vz=_get_velocity_components(mesh, backend)
    vx  =_clamp(vx, "velocity")
    vy  =_clamp(vy, "velocity")
    vz  =_clamp(vz, "velocity")
    cfd["vx"]=vx.astype(np.float32)
    cfd["vy"]=vy.astype(np.float32)
    cfd["vz"]=vz.astype(np.float32)
    wx, wy, wz=_get_wss_components(mesh, backend)
    wx =_clamp(wx, "wss")
    wy =_clamp(wy, "wss")
    wz =_clamp(wz, "wss")
    wss = np.linalg.norm([wx, wy, wz], axis=0).astype(np.float32)
    cfd["wss"]=wss.astype(np.float32)
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
    parser.add_argument("--out_dir", default="/home/group4/Challenge3/vtp_analysis/outputs/pointclouds_vtp/")
    parser.add_argument("--n_points", type=int, default=10000, help="Points per cloud")
    parser.add_argument("--strategy", choices=["fps", "random"], default="fps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args=parser.parse_args()
    
    vtp_root=Path(args.vtp_dir)
    print(vtp_root)
    vtp_files=sorted(list(vtp_root.rglob("*.vtp")))
    rng=np.random.default_rng(args.seed)
    ok, skipped, failed=0,0,0
    for vtp_path in vtp_files:
        name=vtp_path.parent.name
        out_path=os.path.join(args.out_dir, f"{name}.npz")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[SKIP] {name} - already exists")
            skipped+=1
            continue
        print(f"Processing {name} ({args.strategy}, N={args.n_points})")
        try:
            data=process_vtp(vtp_path, args.n_points, args.strategy, rng)
            if data is None:
                failed+=1
                continue
            np.savez_compressed(out_path, **data)
            print(f"{out_path} xyz{data['xyz'].shape}")
            ok+=1
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            failed+=1
    print(f"\nDone - ok: {ok} skipped: {skipped} failed: {failed}")
    print(f"Point clouds saved to {args.out_dir}")
    print(f"Channels per point: xyz + {[f.lower() for f in ALL_EMBED]}")
    
if __name__=="__main__":
    main()