import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Sequence
import torch
from torch.utils.data import Dataset, DataLoader

POINT_CHANNELS=["tawss", "osi", "ecap", "rrt", "pressure", "wss", "velocity"]
TARGET_COLS=["endoleak_type1", "endoleak_type1a", "endoleak_type1b", "endoleak_type2", "endoleak_type3", "endoleak_type4", "other_migration", "other_thrombosis", "other_reintervention", "other_rupture"]
ANY_ENDOLEAK="any_endoleak"

class NormStats:
    def __init__(self):
        self.feat_mean=None
        self.feat_std=None
        self.cfd_mean=None
        self.cfd_std=None
        
    def fit(self, feats: np.ndarray, cfd_samples: list[np.ndarray]) -> None:
        self.feat_mean=np.nanmean(feats, axis=0).astype(np.float32)
        self.feat_std=np.nanstd(feats, axis=0).astype(np.float32)
        self.feat_std[self.feat_std<1e-8]=1.0
        if cfd_samples:
            all_cfd=np.concatenate(cfd_samples, axis=0)
            self.cfd_mean=np.nanmean(all_cfd, axis=0).astype(np.float32)
            self.cfd_std=np.nanstd(all_cfd, axis=0).astype(np.float32)
            self.cfd_std[self.cfd_std<1e-8]=1.0
            
def _augment_point_cloud(xyz: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    angle=rng.uniform(0,2*np.pi)
    c,s=np.cos(angle), np.sin(angle)
    Rz=np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
    xyz=xyz@Rz.T
    xyz*=rng.uniform(0.95, 1.05)
    xyz+=rng.normal(0,0.002, size=xyz.shape).astype(np.float32)
    return xyz

def _random_resample(xyz, cfd, n_points, rng):
    n=len(xyz)
    idx=rng.choice(n, size=n_points, replace=(n<n_points))
    return xyz[idx], cfd[idx]

class EVARDataset(Dataset):
    def __init__(self, dataset_csv: str, features_cols_txt: str, pointcloud_dir: str, split_ids: np.ndarray | Sequence[str], n_points: int=8192, augment: bool=False, norm_stats: NormStats | None=None, fallback_zeros: bool=True, seed: int=0):
        self.pointcloud_dir=Path(pointcloud_dir)
        self.n_points=n_points
        self.augment=augment
        self.norm_stats=norm_stats
        self.fallback_zeros=fallback_zeros
        self._rng=np.random.default_rng(seed)
        df=pd.read_csv(dataset_csv)
        df["patient_id"]=df["patient_id"].astype(str).str.strip()
        split_ids=np.array([str(s).strip() for s in split_ids])
        self.df=df[df["patient_id"].isin(split_ids)].reset_index(drop=True)
        with open(features_cols_txt) as f:
            self.feature_cols=[l.strip() for l in f if l.strip()]
        self.n_cfd_channels=len(POINT_CHANNELS)
        self.n_feat=len(self.feature_cols)
        
    def compute_norm_stats(self) -> NormStats:
        feats=self.df[self.feature_cols]
        cfd_samples=[]
        for pid in self.df["patient_id"]:
            npz_path=self.pointcloud_dir/f"{pid}.npz"
            if npz_path.exists():
                data=np.load(npz_path)
                cfd=np.stack([data[ch] for ch in POINT_CHANNELS], axis=1)
                cfd_samples.append(cfd.astype(np.float32))
        stats=NormStats()
        stats.fit(feats, cfd_samples)
        return stats
    
    def _load_pointcloud(self, patient_id: str):
        npz_path=self.pointcloud_dir/f"{patient_id}.npz"
        if not npz_path.exists():
            return np.zeros((self.n_points, 3), dtype=np.float32), np.zeros((self.n_points, self.n_cfd_channels), dtype=np.float32)
        data=np.load(npz_path)
        xyz=data["xyz"].astype(np.float32)
        cfd=np.stack([data[ch].astype(np.float32) for ch in POINT_CHANNELS], axis=1)
        return xyz, cfd
    
    def _normalise(self, feat, xyz, cfd):
        if self.norm_stats:
            if self.norm_stats.feat_mean is not None:
                feat=(feat-self.norm_stats.feat_mean)/self.norm_stats.feat_std
            xyz=xyz-xyz.mean(axis=0)
            scale=np.abs(xyz).max()
            if scale>1e-8: xyz/=scale
            if self.norm_stats.cfd_mean is not None:
                cfd=(cfd-self.norm_stats.cfd_mean)/self.norm_stats.cfd_std
        return feat, xyz, cfd
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int):
        row=self.df.iloc[idx]
        patient_id=row["patient_id"]
        labels=torch.tensor([float(row[c]) for c in TARGET_COLS], dtype=torch.float32)
        feat=self.df.iloc[idx][self.feature_cols].values.astype(np.float32)
        feat=np.nan_to_num(feat, nan=0.0)
        xyz, cfd=self._load_pointcloud(patient_id)
        xyz, cfd=_random_resample(xyz, cfd, self.n_points, self._rng)
        if self.augment:
            xyz=_augment_point_cloud(xyz, self._rng)
        feat, xyz, cfd=self._normalise(feat, xyz, cfd)
        point=np.concatenate([xyz, cfd], axis=1).T.astype(np.float32)
        feat_t=torch.tensor(feat, dtype=torch.float32)
        return point, feat_t, labels