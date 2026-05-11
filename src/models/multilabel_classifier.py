from matplotlib.pyplot import clf
import os
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import json
from pathlib import Path
import sys

BASE_DIR=Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.models import pointnet
from src.models import gnn_pinn
from src.datasets import evar_dataset  

CH_P=0
CH_VX=1
CH_VY=2
CH_VZ=3
CH_WSS=4

def _mse(pred: torch.Tensor, gt: torch.Tensor) -> float:
    diff=(pred-gt)**2
    return float(diff[torch.isfinite(diff)].mean().item())

parser=argparse.ArgumentParser(description="Train and classify pathological conditions")
#parser.add_argument("--modelpath", default="/home/group4/Challenge3/vtp_analysis/outputs/checkpoint/seed2/pointnet/exp_fold1/best_auc.pth", help="Path to the trained model checkpoint")
parser.add_argument("--modelpath", default="/home/group4/Challenge3/vtp_analysis/outputs/checkpoint/seed12/pinn/exp_fold3/best_auc.pth", help="Path to the trained model checkpoint")
opt = parser.parse_args()

def get_classification(model_path = None, datapath = '/home/group4/Challenge3/vtp_analysis/outputs/', 
                       setting = "pointnet", num_class=1, normal_channel=3, feat_channel=97):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # extract model info 
    if setting == "pointnet":
        model = pointnet.PointNet().to(device)
    elif setting == "pinn":
        model = gnn_pinn.GNNPinn().to(device)
    else:
        raise ValueError("Invalid setting. Choose 'pointnet' or 'pinn'.")
    
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # extract path info 
    features_cols_txt_path: str = datapath + "/dataset/dataset_columns.txt"
    pointcloud_dir_path: str = datapath + "/pointclouds"
    all_subjects = set(pd.read_csv(datapath + "/dataset/dataset.csv")["patient_id"].unique())
    testset_splits = np.load(datapath + "/splits/test_ids.npy")
    for elem in testset_splits.tolist():
        all_subjects.discard(elem)
    all_subjects = np.array(list(all_subjects))


    # train
    trainset = None 
    if setting != "pinn" :    
        trainset = evar_dataset.EVARDataset(datapath = datapath + "/dataset", features_cols_txt = features_cols_txt_path, 
                                           pointcloud_dir = pointcloud_dir_path, split_ids = all_subjects) 
    else:
        trainset = evar_dataset.TimeEVARDataset(datapath = datapath + "/dataset", features_cols_txt = features_cols_txt_path,
                                                pointcloud_dir = pointcloud_dir_path, split_ids = all_subjects)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=2, shuffle=False,
        num_workers=1, drop_last=False
    )
    reprs = []
    labels = []
    for point, label, feat in tqdm(train_loader):
        if setting == "pinn":
            point_wall = point["point_wall"].to(device)
            point = point["point"].to(device)
        else:
            point = point.to(device)
        feat = feat.to(device)
        label_other_patologies = label["other_patologies"]
        repr = None
        with torch.no_grad():
            if setting == "pinn":
                pred, h, pinn_out_wall, repr = model(point, feat, point_wall)
            else:
                pred, repr = model(point, feat)
        reprs.append(repr.detach().cpu())
        labels.append(label_other_patologies)
    training_reprs = torch.cat(reprs, dim=0).numpy()
    training_labels = torch.cat(labels, dim=0).numpy()

    clf = OneVsRestClassifier(SVC(kernel='linear', probability=True))
    clf.fit(training_reprs, training_labels)

    # test 
    testset = None
    if setting != "pinn" :    
        testset = evar_dataset.EVARDataset(datapath = datapath + "/dataset", features_cols_txt = features_cols_txt_path, 
                                           pointcloud_dir = pointcloud_dir_path, split_ids = testset_splits) 
    else:
        testset = evar_dataset.TimeEVARDataset(datapath = datapath + "/dataset", features_cols_txt = features_cols_txt_path,
                                                pointcloud_dir = pointcloud_dir_path, split_ids = testset_splits)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=2, shuffle=False,
        num_workers=1, drop_last=False
    )
    reprs = []
    labels = []
    patient_mse: dict={}
    pid_list=testset_splits.tolist()
    for batch_idx, (point, label, feat) in enumerate(tqdm(test_loader)):
        pid=str(pid_list[batch_idx]).strip()
        if setting == "pinn":
            point_wall = point["point_wall"].to(device)
            point = point["point"].to(device)
            cfd=label["cfd"].to(device)
            cfd_wall=label.get("cfd_wall", None)
            if cfd_wall is not None:
                cfd_wall=cfd_wall.to(device)
        else:
            point = point.to(device)
        feat = feat.to(device)
        label_other_patologies = label["other_patologies"]
        repr = None
        with torch.no_grad():
            if setting == "pinn":
                pred, h, pinn_out_wall, repr = model(point, feat, point_wall)
            else:
                pred, repr = model(point, feat)
        reprs.append(repr.detach().cpu())
        labels.append(label_other_patologies)
        if setting=="pinn":
            h_last=h[:,:,:,-1]
            cfd_last=cfd[:,:,:,-1]
            mse_p=_mse(h_last[:,CH_P,:], cfd_last[:,CH_P,:])
            mse_vx=_mse(h_last[:,CH_VX,:], cfd_last[:,CH_VX,:])
            mse_vy=_mse(h_last[:,CH_VY,:], cfd_last[:,CH_VY,:])
            mse_vz=_mse(h_last[:,CH_VZ,:], cfd_last[:,CH_VZ,:])
            mse_wss=_mse(h_last[:,CH_WSS,:], cfd_last[:,CH_WSS,:])
            v_pred=h_last[:,CH_VX:CH_VZ+1,:]
            v_gt=cfd_last[:,CH_VX:CH_VZ+1,:]
            mse_vmag=_mse((v_pred**2).sum(dim=1), (v_gt**2).sum(dim=1))
            patient_mse[pid]={
                "pressure_mse": round(mse_p, 6),
                "wss_mse": round(mse_wss, 6),
                "velocity_x_mse": round(mse_vx, 6),
                "velocity_y_mse": round(mse_vy, 6),
                "velocity_z_mse": round(mse_vz, 6),
                "velocity_magnitude_mse": round(mse_vmag, 6)
            }

    testing_reprs = torch.cat(reprs, dim=0).numpy()
    testing_labels = torch.cat(labels, dim=0).numpy()
     
    #test_predictions =  clf.predict(testing_reprs)
    test_prob_predictions=clf.predict_proba(testing_reprs)
    fpr=dict()
    tpr=dict()
    roc_auc=dict()
    n_classes=testing_labels.shape[1]
    for i in range(n_classes):
        try:
            fpr[i], tpr[i], _ = roc_curve(testing_labels[:, i], test_prob_predictions[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        except Exception:
            fpr[i]=np.array([0.0, 1.0])
            tpr[i]=np.array([0.0, 1.0])
            roc_auc[i]=float("nan")
    all_fpr=np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr=np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr+=np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr/=n_classes
    macro_auc=auc(all_fpr, mean_tpr)
    metrics_to_save={
        "fpr": {str(i): fpr[i].tolist   () for i in range(n_classes)},
        "tpr": {str(i): tpr[i].tolist   () for i in range(n_classes)},
        "auc": {str(i): roc_auc[i] for i in range(n_classes)},
        "macro_auc": macro_auc
    }
    with open("metrics_multilabel_svm.json", "w") as f:
        json.dump(metrics_to_save, f)
    np.save('predizioni_multilabel_svm.npy', test_prob_predictions)
    np.save('labels_multilabel_svm.npy', testing_labels)
    print(f"Macro AUC: {macro_auc:.4f}")

    if setting=="pinn" and patient_mse:
        fields=["pressure_mse", "wss_mse", "velocity_x_mse", "velocity_y_mse", "velocity_z_mse", "velocity_magnitude_mse"]
        aggregate={}
        for field in fields:
            vals=[v[field] for v in patient_mse.values() if field in v and np.isfinite(v[field])]
            if vals:
                aggregate[f"{field}_mean"]=round(float(np.mean(vals)), 6)
                aggregate[f"{field}_std"]=round(float(np.std(vals)), 6)
                aggregate[f"{field}_min"]=round(float(np.min(vals)), 6)
                aggregate[f"{field}_max"]=round(float(np.max(vals)), 6)
        pinn_metrics={
            "per_patient": patient_mse,
            "aggregate": aggregate
        }
        with open("pinn_metrics.json", "w") as f:
            json.dump(pinn_metrics, f, indent=2)
        print("PINN MSE metrics saved to pinn_metrics.json")
        _print_mse_summary(aggregate)
    return test_prob_predictions

def _print_mse_summary(agg: dict):
    print("\n── PINN Physics MSE (test set) ──────────────────────────")
    labels = {
        "pressure_mse":     "Pressure      ",
        "wss_mse":          "WSS           ",
        "velocity_x_mse":   "Velocity X    ",
        "velocity_y_mse":   "Velocity Y    ",
        "velocity_z_mse":   "Velocity Z    ",
        "velocity_mag_mse": "Velocity |mag|",
    }
    for key, display in labels.items():
        mean = agg.get(f"{key}_mean", float("nan"))
        std  = agg.get(f"{key}_std",  float("nan"))
        print(f"  {display}  MSE = {mean:.6f}  ±{std:.6f}")
    print("─────────────────────────────────────────────────────────\n")

if __name__ == '__main__':
    get_classification(model_path=opt.modelpath, setting="pinn") 