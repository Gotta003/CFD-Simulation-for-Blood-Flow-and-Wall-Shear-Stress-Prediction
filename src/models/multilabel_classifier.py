from matplotlib.pyplot import clf
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

def get_classification(model_path = '/home/group4/Challenge3/vtp_analysis/outputs/checkpoint/seed2/pointnet/exp_fold1/best_auc.pth', datapath = '/home/group4/Challenge3/vtp_analysis/outputs/', 
                       setting = "pointnet", num_class=1, normal_channel=3, feat_channel=97):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # extract model info 
    if setting == "pointnet":
        model = pointnet.PointNet().to(device)
    elif setting == "pinn":
        model = gnn_pinn.GNN_PINN().to(device)
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
            cfd = label["cfd"].to(device)
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
    for point, label, feat in tqdm(test_loader):
        if setting == "pinn":
            point_wall = point["point_wall"].to(device)
            point = point["point"].to(device)
            cfd = label["cfd"].to(device)
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
    testing_reprs = torch.cat(reprs, dim=0).numpy()
    testing_labels = torch.cat(labels, dim=0).numpy()
     
    #test_predictions =  clf.predict(testing_reprs)
    test_prob_predictions=clf.predict_proba(testing_reprs)
    fpr=dict()
    tpr=dict()
    thresholds=dict()
    roc_auc=dict()
    n_classes=testing_labels.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(testing_labels[:, i], test_prob_predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
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
    return test_prob_predictions

if __name__ == '__main__':
    get_classification() 