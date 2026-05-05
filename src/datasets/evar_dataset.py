import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Sequence
import torch
from torch.utils.data import Dataset, DataLoader

POINT_CHANNELS=["pressure", "wss", "velocity"]

SUBSAMPLED_INDEXES = range(0, 480, 20) # change the last value to include more timesteps

BASE_DIR=Path(__file__).resolve().parent.parent.parent

 
class NormStats:
    def __init__(self):
        self.feat_max=None
        self.feat_min=None
        self.cfd_max=None
        self.cfd_min=None
        self.point_max=None
        self.point_min=None
        
    def fit(self, points: np.ndarray, feats: np.ndarray, cfd_samples: list[np.ndarray]) -> None:
        self.feat_max=np.nanmax(feats, axis=0).astype(np.float32)
        self.feat_min=np.nanmin(feats, axis=0).astype(np.float32)
        all_points=np.concatenate(points, axis=0)
        self.point_max=np.nanmax(all_points, axis=0).astype(np.float32)
        self.point_min=np.nanmin(all_points, axis=0).astype(np.float32)
        all_cfd=np.concatenate(cfd_samples, axis=0)
        self.cfd_max=np.nanmax(all_cfd, axis=0).astype(np.float32)
        self.cfd_min=np.nanmin(all_cfd, axis=0).astype(np.float32)
            
def _augment_point_cloud(xyz: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    angle=rng.uniform(0,2*np.pi)
    c,s=np.cos(angle), np.sin(angle)
    Rz=np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
    xyz=xyz@Rz.T
    xyz*=rng.uniform(0.95, 1.05)
    xyz+=rng.normal(0,0.002, size=xyz.shape).astype(np.float32)
    return xyz

def _random_resample(xyz, cfd, n_points, rng, w_time = False):
    n=xyz.shape[1] if w_time else len(xyz)
    idx=rng.choice(n, size=n_points, replace=(n<n_points))
    if w_time: 
        return xyz[:, idx, :], cfd[:, :, idx] 
    else:
        return xyz[idx], cfd[idx]

class EVARDataset(Dataset):
    def __init__(self, datapath: str = str(BASE_DIR) + "/outputs/dataset", 
                 features_cols_txt: str = str(BASE_DIR) + "/outputs/dataset/dataset_columns.txt", 
                 pointcloud_dir: str = str(BASE_DIR) + "/outputs/pointclouds", 
                 split_ids: np.ndarray | Sequence[str] = np.load(str(BASE_DIR) + "/outputs/splits/fold_0_train_ids.npy"), 
                 n_points: int=8192, augment: bool=False, norm_stats: NormStats | None=None, 
                 fallback_zeros: bool=True, seed: int=0, normalize = True):
        self.pointcloud_dir=Path(pointcloud_dir)
        self.n_points=n_points
        self.augment=augment
        self.norm_stats=norm_stats  
        self.fallback_zeros=fallback_zeros
        self._rng=np.random.default_rng(seed)
        df=pd.read_csv(os.path.join(datapath, "dataset.csv"))
        df["patient_id"]=df["patient_id"].astype(str).str.strip()
        if type(split_ids) is np.ndarray:
            split_ids=np.array([str(s).strip() for s in split_ids])
        self.df=df[df["patient_id"].isin(split_ids)].reset_index(drop=True)
        with open(features_cols_txt) as f:
            self.feature_cols=[l.strip() for l in f if l.strip()]
        self.n_cfd_channels=len(POINT_CHANNELS)
        self.n_feat=len(self.feature_cols)
        if self.norm_stats is None and normalize:
            self.norm_stats=self.compute_norm_stats()
        labels_path = datapath + "/dataset.csv"
        self.labels = pd.read_csv(labels_path)[pd.read_csv(labels_path)["patient_id"].isin(split_ids)]["complication_raw"].to_numpy().reshape([-1, 1])
        self.labels = (self.labels != 'none').astype(int)
        
    def compute_norm_stats(self) -> NormStats:
        feats=self.df[self.feature_cols]
        cfd_samples=[]
        points=[]
        for pid in self.df["patient_id"]:
            npz_path=self.pointcloud_dir/f"{pid}.npz"
            if npz_path.exists():
                data=np.load(npz_path)
                xyz=data["xyz"][data["xyz"].shape[0]-1, :, :].astype(np.float32)
                points.append(xyz)
                pressure_last_step = data["p"][data["p"].shape[0] -1, :].astype(np.float32)
                v_x = data["vx"][data["vx"].shape[0]-1, :].astype(np.float32)
                v_y = data["vy"][data["vy"].shape[0]-1, :].astype(np.float32)
                v_z = data["vz"][data["vz"].shape[0]-1, :].astype(np.float32)
                v_last_step = np.sqrt(v_x**2 + v_y**2 + v_z**2)
                wss_x = data["wx"][data["wx"].shape[0]-1, :].astype(np.float32)
                wss_y = data["wy"][data["wy"].shape[0]-1, :].astype(np.float32)
                wss_z = data["wz"][data["wz"].shape[0]-1, :].astype(np.float32)
                wss_last_step = np.sqrt(wss_x**2 + wss_y**2 + wss_z**2)
                cfd=np.stack([pressure_last_step, wss_last_step, v_last_step], axis=1)
                cfd_samples.append(cfd.astype(np.float32))
        stats=NormStats()
        stats.fit(points, feats, cfd_samples)
        return stats
    
    def _load_pointcloud(self, patient_id: str):
        npz_path=self.pointcloud_dir/f"{patient_id}.npz"
        if not npz_path.exists():
            return np.zeros((self.n_points, 3), dtype=np.float32), np.zeros((self.n_points, self.n_cfd_channels), dtype=np.float32)
        data=np.load(npz_path)
        xyz=data["xyz"][data["xyz"].shape[0]-1, :, :].astype(np.float32)
        pressure_last_step = data["p"][data["p"].shape[0] -1, :].astype(np.float32)
        v_x = data["vx"][data["vx"].shape[0]-1, :].astype(np.float32)
        v_y = data["vy"][data["vy"].shape[0]-1, :].astype(np.float32)
        v_z = data["vz"][data["vz"].shape[0]-1, :].astype(np.float32)
        v_last_step = np.sqrt(v_x**2 + v_y**2 + v_z**2)
        wss_x = data["wx"][data["wx"].shape[0]-1, :].astype(np.float32)
        wss_y = data["wy"][data["wy"].shape[0]-1, :].astype(np.float32)
        wss_z = data["wz"][data["wz"].shape[0]-1, :].astype(np.float32)
        wss_last_step = np.sqrt(wss_x**2 + wss_y**2 + wss_z**2)
        cfd=np.stack([pressure_last_step, wss_last_step, v_last_step], axis=1)
        return xyz, cfd
    
    def _normalise(self, feat, xyz, cfd):
        feat=2*((feat-self.norm_stats.feat_min)/(self.norm_stats.feat_max-self.norm_stats.feat_min+1e-8))-1
        xyz=2*((xyz-self.norm_stats.point_min)/(self.norm_stats.point_max-self.norm_stats.point_min+1e-8))-1
        cfd=2*((cfd-self.norm_stats.cfd_min)/(self.norm_stats.cfd_max-self.norm_stats.cfd_min+1e-8))-1
        return feat, xyz, cfd
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int):
        row=self.df.iloc[idx]
        patient_id=row["patient_id"]
        label=torch.tensor(self.labels[idx], dtype=torch.float32)
        feat=self.df.iloc[idx][self.feature_cols].values.astype(np.float32)
        feat=np.nan_to_num(feat, nan=0.0)
        xyz, cfd=self._load_pointcloud(patient_id)
        xyz, cfd=_random_resample(xyz, cfd, self.n_points, self._rng)
        if self.augment:
            xyz=_augment_point_cloud(xyz, self._rng)
        if self.norm_stats is not None:
            feat, xyz, cfd=self._normalise(feat, xyz, cfd)
        point=np.concatenate([xyz, cfd], axis=1).T.astype(np.float32)
        feat_t=torch.tensor(feat, dtype=torch.float32)
        return point, label, feat_t
    

class TimeEVARDataset(Dataset):
    def __init__(self, datapath: str = str(BASE_DIR) + "/outputs/dataset", 
                 features_cols_txt: str = str(BASE_DIR) + "/outputs/dataset/dataset_columns.txt", 
                 pointcloud_dir: str = str(BASE_DIR) + "/outputs/pointclouds", 
                 pointcloud_wall_dir: str = str(BASE_DIR) + "/outputs/pointclouds_vtp", 
                 split_ids: np.ndarray | Sequence[str] = np.load(str(BASE_DIR) + "/outputs/splits/fold_0_train_ids.npy"), 
                 n_points: int=8192, augment: bool=False, norm_stats: NormStats | None=None, 
                 fallback_zeros: bool=True, seed: int=0, normalize = True):
        self.pointcloud_dir=Path(pointcloud_dir)
        self.pointcloud_wall_dir=Path(pointcloud_wall_dir)
        self.n_points=n_points
        self.augment=augment
        self.norm_stats=norm_stats  
        self.fallback_zeros=fallback_zeros
        self._rng=np.random.default_rng(seed)
        df=pd.read_csv(os.path.join(datapath, "dataset.csv"))
        df["patient_id"]=df["patient_id"].astype(str).str.strip()
        if type(split_ids) is np.ndarray:
            split_ids=np.array([str(s).strip() for s in split_ids])
        self.df=df[df["patient_id"].isin(split_ids)].reset_index(drop=True)
        with open(features_cols_txt) as f:
            self.feature_cols=[l.strip() for l in f if l.strip()]
        self.n_cfd_channels=len(POINT_CHANNELS)
        self.n_feat=len(self.feature_cols)
        if self.norm_stats is None and normalize:
            self.norm_stats=self.compute_norm_stats()
        labels_path = datapath + "/dataset.csv"
        self.labels = pd.read_csv(labels_path)[pd.read_csv(labels_path)["patient_id"].isin(split_ids)]["complication_raw"].to_numpy().reshape([-1, 1])
        self.labels = (self.labels != 'none').astype(int)
        self.other_patologies_labels = pd.read_csv(labels_path).iloc[:, 3:14].to_numpy()
        
    def compute_norm_stats(self) -> NormStats:
        feats=self.df[self.feature_cols]
        cfd_samples=[]
        points=[]
        for pid in self.df["patient_id"]:
            npz_path=self.pointcloud_dir/f"{pid}.npz"
            if npz_path.exists():
                data=np.load(npz_path)
                xyz=data["xyz"].astype(np.float32)
            if (xyz.shape[0]==480):     
                xyz_subsampled=xyz[SUBSAMPLED_INDEXES, :, :].astype(np.float32)
                t, n, _ = xyz_subsampled.shape
                time_steps = np.arange(t).reshape(t, 1, 1)
                time_column = np.broadcast_to(time_steps, (t, n, 1))
                xyz_subsampled = np.concatenate([time_column, xyz_subsampled], axis=2) 
                p_subsampled=data["p"][SUBSAMPLED_INDEXES, :].astype(np.float32)
                vx_subsampled=data["vx"][ SUBSAMPLED_INDEXES, :].astype(np.float32)
                vy_subsampled=data["vy"][SUBSAMPLED_INDEXES, :].astype(np.float32)
                vz_subsampled=data["vz"][SUBSAMPLED_INDEXES, :].astype(np.float32)
                wx_subsampled=data["wx"][SUBSAMPLED_INDEXES, :].astype(np.float32)
                wy_subsampled=data["wy"][SUBSAMPLED_INDEXES, :].astype(np.float32)
                wz_subsampled=data["wz"][SUBSAMPLED_INDEXES, :].astype(np.float32)
                wss_subsampled = np.sqrt(wx_subsampled**2 + wy_subsampled**2 + wz_subsampled**2)
                cfd_subsampled=np.stack([p_subsampled, vx_subsampled, vy_subsampled, vz_subsampled, wss_subsampled], axis=1)
                for i in range(xyz_subsampled.shape[0]):
                    xyz=xyz_subsampled[i].squeeze().astype(np.float32) 
                    cfd = cfd_subsampled[i].squeeze().astype(np.float32)   
                    print(xyz.shape, cfd.shape)       
                    points.append(xyz)
                    cfd_samples.append(np.transpose(cfd, (1, 0)).astype(np.float32))
                npz_path=self.pointcloud_wall_dir/f"{pid}.npz"
                if npz_path.exists():
                    data_wall=np.load(npz_path)
                    xyz_wall=data_wall["xyz"].astype(np.float32)
                    xyz_wall= np.column_stack((xyz_wall, xyz_subsampled[xyz_subsampled.shape[0]-1, :xyz_wall.shape[0], 3])).astype(np.float32)            
                    p_wall=data_wall["p"].astype(np.float32)
                    vx_wall=data_wall["vx"].astype(np.float32)
                    vy_wall=data_wall["vy"].astype(np.float32)
                    vz_wall=data_wall["vz"].astype(np.float32)
                    wss_wall=data_wall["wss"].astype(np.float32)
                    cfd_wall=np.stack([p_wall, vx_wall, vy_wall, vz_wall, wss_wall], axis=1)
                    xyz_wall = np.tile(xyz_wall, (5,1))
                    points.append(xyz_wall)
                    cfd_wall = np.tile(cfd_wall, (5,1))
                    print(xyz_wall.shape, cfd_wall.shape)
                    cfd_samples.append(cfd_wall.astype(np.float32))
        stats=NormStats()
        stats.fit(points, feats, cfd_samples)
        return stats
    
    def _load_pointcloud(self, patient_id: str):
        npz_path=self.pointcloud_dir/f"{patient_id}.npz"
        if not npz_path.exists():
            return np.zeros((self.n_points, 3), dtype=np.float32), np.zeros((self.n_points, self.n_cfd_channels), dtype=np.float32)
        data=np.load(npz_path)
        xyz=data["xyz"].astype(np.float32)
        xyz_subsampled=xyz[SUBSAMPLED_INDEXES, :, :].astype(np.float32)
        t, n, _ = xyz_subsampled.shape
        time_steps = np.arange(t).reshape(t, 1, 1)
        time_column = np.broadcast_to(time_steps, (t, n, 1))
        xyz_subsampled = np.concatenate([time_column, xyz_subsampled], axis=2) 
        p_subsampled=data["p"][SUBSAMPLED_INDEXES, :].astype(np.float32)
        vx_subsampled=data["vx"][ SUBSAMPLED_INDEXES, :].astype(np.float32)
        vy_subsampled=data["vy"][SUBSAMPLED_INDEXES, :].astype(np.float32)
        vz_subsampled=data["vz"][SUBSAMPLED_INDEXES, :].astype(np.float32)
        wx_subsampled=data["wx"][SUBSAMPLED_INDEXES, :].astype(np.float32)
        wy_subsampled=data["wy"][SUBSAMPLED_INDEXES, :].astype(np.float32)
        wz_subsampled=data["wz"][SUBSAMPLED_INDEXES, :].astype(np.float32)
        wss_subsampled = np.sqrt(wx_subsampled**2 + wy_subsampled**2 + wz_subsampled**2)
        cfd_subsampled=np.stack([p_subsampled, vx_subsampled, vy_subsampled, vz_subsampled, wss_subsampled], axis=1)
        npz_path_wall=self.pointcloud_wall_dir/f"{patient_id}.npz"
        data_wall=np.load(npz_path_wall)
        xyz_wall=data_wall["xyz"].astype(np.float32)
        p_wall=data_wall["p"].astype(np.float32)
        vx_wall=data_wall["vx"].astype(np.float32)
        vy_wall=data_wall["vy"].astype(np.float32)
        vz_wall=data_wall["vz"].astype(np.float32)
        wss_wall=data_wall["wss"].astype(np.float32)
        cfd_wall=np.stack([p_wall, vx_wall, vy_wall, vz_wall, wss_wall], axis=1)
        return xyz_subsampled, cfd_subsampled, xyz_wall, cfd_wall
    
    def _normalise(self, feat, xyz, cfd, xyz_wall, cfd_wall):
        feat=2*((feat-self.norm_stats.feat_min)/(self.norm_stats.feat_max-self.norm_stats.feat_min+1e-8))-1
        xyz=2*((xyz-self.norm_stats.point_min)/(self.norm_stats.point_max-self.norm_stats.point_min+1e-8))-1
        cfd=2*((cfd-self.norm_stats.cfd_min)/(self.norm_stats.cfd_max-self.norm_stats.cfd_min+1e-8))-1
        xyz_wall=2*((xyz_wall-self.norm_stats.point_min[:3])/(self.norm_stats.point_max[:3]-self.norm_stats.point_min[:3]+1e-8))-1
        cfd_wall=2*((cfd_wall-self.norm_stats.cfd_min)/(self.norm_stats.cfd_max-self.norm_stats.cfd_min+1e-8))-1
        return feat, xyz, cfd, xyz_wall, cfd_wall  
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int):
        row=self.df.iloc[idx]
        patient_id=row["patient_id"]
        label_complication=torch.tensor(self.labels[idx], dtype=torch.float32)
        other_patologies_label = torch.tensor(self.other_patologies_labels[idx], dtype=torch.float32)
        feat=self.df.iloc[idx][self.feature_cols].values.astype(np.float32)
        feat=np.nan_to_num(feat, nan=0.0)
        xyz, cfd, xyz_wall, cfd_wall = self._load_pointcloud(patient_id)
        xyz, cfd =_random_resample(xyz, cfd, self.n_points, self._rng, w_time=True)
        xyz_wall, cfd_wall = _random_resample(xyz_wall, cfd_wall, self.n_points, self._rng)
        if self.augment:
            xyz=_augment_point_cloud(xyz, self._rng)
        if self.norm_stats is not None:
            
            feat, xyz, cfd, xyz_wall, cfd_wall = self._normalise(feat, xyz, np.transpose(cfd, (0, 2, 1)), xyz_wall, cfd_wall)
            cfd_wall = np.transpose(cfd_wall, (1, 0))
        else:
            cfd = np.transpose(cfd, (0, 2, 1))
            cfd_wall = np.transpose(cfd_wall, (1, 0))
        points = {
                  "point": xyz.T.astype(np.float32),
                  "point_wall": xyz_wall.T.astype(np.float32)
                  }
        cfd = cfd.T.astype(np.float32)
        labels = {"complication": label_complication, "other_patologies": other_patologies_label, "cfd": cfd, "cfd_wall": cfd_wall}
        feat_t=torch.tensor(feat, dtype=torch.float32)
        return points, labels, feat_t  # (xyzt, point_index, time) (cfd, point_index, time)