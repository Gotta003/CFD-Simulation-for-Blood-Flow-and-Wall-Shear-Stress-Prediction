# ======================== LIBRARIES ========================
# external libraries
import os
import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm import tqdm
import time
import torch
from torch.nn import BCELoss
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
# custom libraries
from src.datasets import evar_dataset 
from src.models.pointnet import PointNet, Fakemodel
from src.models.gnn_pinn import GNNPinn, PinnLoss
from data import *

# ======================== HANDLING MODEL CODE ========================
def _create_lr_scheduler(optimizer):
    lr = ExponentialLR(optimizer, gamma=0.99)
    return lr

def get_model(setting = "pointnet"):
    if setting == "pointnet":
       model = PointNet()
    elif setting == "pinn":
        model = GNNPinn()
    return model

def get_loss(setting = "pointnet", model = None):
    if setting == "pointnet":
        return BCELoss()
    else:
        return PinnLoss(model)

# ======================== TRAINING CODE ========================
# training
def train(model, opt, train_loader, val_loader, loss, fold, board, lr_scheduler, trainF, path, 
          num_epoch, predthreshold, device, setting = "pointnet"):
    print("training fold {}".format(fold))
    best_acc = 0.0
    best_auc = 0.0
    acc_bestauc = 0.0
    auc_bestauc = 0.0
    recall_bestauc = 0.0
    precision_bestauc = 0.0
    f1_score_bestauc = 0.0
    # ROC
    fpr_bestauc, tpr_bestauc, thresholds_bestauc = None, None, None
    for epoch in range(num_epoch):
        train_loss, train_acc, train_auc = train_one_epoch(model, train_loader, opt, loss, 
                                                           epoch, predthreshold, device, setting = setting)
        board.add_scalar('train_loss', train_loss, epoch)
        board.add_scalar('train_acc', train_acc, epoch)
        board.add_scalar('train_auc', train_auc, epoch)
        val_loss, val_acc, val_auc, val_labels, val_preds, val_preds_post = val_one_epoch(
            model, val_loader, loss,  predthreshold, device, setting = setting
        )
        lr_scheduler.step(val_loss)
        board.add_scalar('val_loss', val_loss, epoch)
        board.add_scalar('val_acc', val_acc, epoch)
        board.add_scalar('val_auc', val_auc, epoch)
        print('fold {} epoch {}:'.format(fold, epoch))
        print('  val_loss    : {}'.format(val_loss))
        print('  val_acc     : {}'.format(val_acc))
        print('  val_auc     : {}'.format(val_auc))
        trainF.write('fold{},epoch,{}\n'.format(fold, epoch))
        trainF.write('fold{},val_acc,{}\n'.format(fold, val_acc))
        trainF.write('fold{},val_auc,{}\n'.format(fold, val_auc))
        trainF.write('####################################################\n')
        if val_auc > best_auc:
            fold_dir = os.path.join(path, 'exp_fold{}'.format(fold))
            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir)
            torch.save(model.state_dict(), os.path.join(fold_dir, 'best_auc.pth'))
            best_auc = val_auc
            acc_bestauc = val_acc
            auc_bestauc = val_auc
            fpr_bestauc, tpr_bestauc, thresholds_bestauc = roc_curve(
                val_labels, val_preds, drop_intermediate=False
            )
            recall_bestauc = recall_score(val_labels, val_preds_post, average='binary')
            precision_bestauc = precision_score(val_labels, val_preds_post, average='binary')
            f1_score_bestauc = f1_score(val_labels, val_preds_post, average='binary')
        if val_acc > best_acc:
            best_acc = val_acc
        if setting == "pinn":
            loss.update_epoch() # update epoch for adaptive loss balancing in PINN
    metric = {
        'acc': best_acc,
        'auc': best_auc,
        'acc_bestauc': acc_bestauc,
        'auc_bestauc': auc_bestauc,
        'recall_bestauc': recall_bestauc,
        'precision_bestauc': precision_bestauc,
        'f1_score_bestauc': f1_score_bestauc,
        'fpr_bestauc': fpr_bestauc,
        'tpr_bestauc': tpr_bestauc,
        'thresholds_bestauc': thresholds_bestauc
    }
    return metric 


def train_one_epoch(model, train_loader, opt, selected_loss, epoch, predthreshold, device, setting = "pointnet"):
    since = time.time()
    model.train()
    total_loss = 0.0
    total_batch = 0
    preds = []
    labels = []
    pred_posts = []
    for point, label, feat in tqdm(train_loader):
        #batch_size = len(label)
        batch_size=feat.size(0)
        total_batch += batch_size
        opt.zero_grad()
        point_wall = None
 
        cfd = None 
        cfd_wall = None
        if setting != "pinn":
            point = point.to(device)
        else:
            point_wall = point["point_wall"].to(device)
            point = point["point"].to(device)
        feat = feat.to(device)
        if setting == "pinn":
            comp_label = label["complication"].to(device).reshape(batch_size, -1)
            cfd = label["cfd"].to(device)
            cfd_wall = label["cfd_wall"].to(device)
            label=comp_label
        else: 
            label = label.to(device).reshape(batch_size, -1)
        if setting == "pinn":
                pred, h_t, pinn_out_wall, _ = model(point.requires_grad_(True), feat,  point_wall)
                loss = selected_loss(point, pred, h_t, label, pinn_out_wall, cfd, cfd_wall)
        else:
            pred, _ = model(point, feat)
            loss = selected_loss(pred, label)
        loss.backward()
        opt.step()
        pred_post = (pred > predthreshold).float()
        total_loss += loss.item() * batch_size
        labels.append(label.detach().cpu().squeeze(0).numpy())
        preds.append(pred.detach().cpu().squeeze(0).numpy())
        pred_posts.append(pred_post.detach().cpu().squeeze(0).numpy())
    a = np.concatenate(labels, axis=0)
    b = np.concatenate(preds, axis=0)
    c = np.concatenate(pred_posts, axis=0)
    time_one_epoch = time.time() - since
    print('Training epoch {:0d} time: {:0f}m {:0f}s'.format(
        epoch, time_one_epoch // 60, time_one_epoch % 60))
    return (total_loss / total_batch,
            accuracy_score(a, c),
            roc_auc_score(a, b))

# testing & validation

def val_one_epoch(model, val_loader, selected_loss, predthreshold, device, setting = "pointnet"):
    model.eval()
    total_loss = 0.0
    total_batch = 0
    preds = []
    labels = []
    pred_posts = []
    for point, label, feat in tqdm(val_loader):
        #batch_size_val = len(label)
        batch_size_val=feat.size(0)
        total_batch += batch_size_val
        point_wall = None
        if setting != "pinn":
            point = point.to(device)
        else:
            point_wall = point["point_wall"].to(device)
            point = point["point"].to(device)
        feat = feat.to(device)
        if setting == "pinn":
            comp_label = label["complication"].to(device).reshape(batch_size_val, -1)
            cfd = label["cfd"].to(device)
            cfd_wall = label["cfd_wall"].to(device)
            label=comp_label
        else: 
            label = label.to(device).reshape(batch_size_val, -1)
        with torch.no_grad():
            if setting == "pinn":
                pred, h_t, pinn_out_wall, _ = model(point.requires_grad_(True), feat, point_wall)
                loss = selected_loss(point, pred, h_t, label, pinn_out_wall, cfd, cfd_wall) 
            else:
                pred, _ = model(point, feat)
                loss = selected_loss(pred, label)
        pred_post = (pred > predthreshold).float()
        total_loss += loss.item() * batch_size_val
        labels.append(label.detach().cpu().squeeze(0).numpy())
        preds.append(pred.detach().cpu().squeeze(0).numpy())
        pred_posts.append(pred_post.detach().cpu().squeeze(0).numpy())
        a = np.concatenate(labels, axis=0)
        b = np.concatenate(preds, axis=0)
        c = np.concatenate(pred_posts, axis=0)
    return (total_loss / total_batch,
            accuracy_score(a, c),
            roc_auc_score(a, b),
            a, b, c)


def test(model, test_loader, predthreshold, device, setting="pointnet"):
    model.eval()
    total_batch = 0
    preds = []
    labels = []
    pred_posts = []
    pressures = []
    label_pressures = []
    stresses = []
    label_stresses = []
    velocities_x = []
    label_velocities_x = []
    velocities_y = []
    label_velocities_y = []
    velocities_z = []
    label_velocities_z = []
    start = time.time()
    for point, label, feat in tqdm(test_loader):
        #batch_size = len(label)
        batch_size=feat.size(0)
        total_batch += batch_size
        point_wall = None
        h = None
        if setting != "pinn":
            point = point.to(device)
        else:
            point_wall = point["point_wall"].to(device)
            point = point["point"].to(device)
        feat = feat.to(device)
        if setting == "pinn":
            label = label["complication"].to(device).reshape(batch_size, -1)
            cfd = label["cfd"].to(device)
        else: 
            label = label.to(device).reshape(batch_size, -1)
        with torch.no_grad():
            if setting == "pinn":
                pred, h, pinn_out_wall, _ = model(point, feat, point_wall)
            else:
                pred, _ = model(point, feat)
        pred_post = (pred > predthreshold).float()
        labels.append(label.detach().cpu().squeeze(0).numpy())
        preds.append(pred.detach().cpu().squeeze(0).numpy())
        pred_posts.append(pred_post.detach().cpu().squeeze(0).numpy())
        pressures.append(h[:, 0, :, -1].detach().cpu().squeeze(0).numpy())
        stresses.append(h[:, 4, :, -1].detach().cpu().squeeze(0).numpy())
        velocities_x.append(h[:, 1, :, -1].detach().cpu().squeeze(0).numpy())
        velocities_y.append(h[:, 2, :, -1].detach().cpu().squeeze(0).numpy())
        velocities_z.append(h[:, 3, :, -1].detach().cpu().squeeze(0).numpy())
        label_pressures.append(cfd[:, 0, :, -1].detach().cpu().squeeze(0).numpy())
        label_stresses.append(cfd[:, 4, :, -1].detach().cpu().squeeze(0).numpy())
        label_velocities_x.append(cfd[:, 1, :, -1].detach().cpu().squeeze(0).numpy())
        label_velocities_y.append(cfd[:, 2, :, -1].detach().cpu().squeeze(0).numpy())
        label_velocities_z.append(cfd[:, 3, :, -1].detach().cpu().squeeze(0).numpy())
    a = np.concatenate(labels, axis=0)
    b = np.concatenate(preds, axis=0)
    c = np.concatenate(pred_posts, axis=0)
    velocities_x = np.concatenate(velocities_x, axis=0)
    label_velocities_x = np.concatenate(label_velocities_x, axis=0)
    velocities_y = np.concatenate(velocities_y, axis=0)
    label_velocities_y = np.concatenate(label_velocities_y, axis=0)
    velocities_z = np.concatenate(velocities_z, axis=0)
    label_velocities_z = np.concatenate(label_velocities_z, axis=0)
    stresses = np.concatenate(stresses, axis=0)
    label_stresses = np.concatenate(label_stresses, axis=0)
    pressures = np.concatenate(pressures, axis=0)
    label_pressures = np.concatenate(label_pressures, axis=0)
    stop = time.time()
    metric = {
        'acc_test': accuracy_score(a, c),
        'auc_test': roc_auc_score(a, b),
        'recall_test': recall_score(a, c, average='binary'),
        'precision_test': precision_score(a, c, average='binary'),
        'f1_score_test': f1_score(a, c, average='binary'),
        'fpr_test': None,
        'tpr_test': None,
        'thresholds_test': None,
        'test_time': stop - start,
        "pressure_error": mean_squared_error(label_pressures, pressures),
        "wss_error": mean_squared_error(label_stresses, stresses),
        "velocity_error_x": mean_squared_error(label_velocities_x, velocities_x),
        "velocity_error_y": mean_squared_error(label_velocities_y, velocities_y),
        "velocity_error_z": mean_squared_error(label_velocities_z, velocities_z)
    }
    metric["fpr_test"], metric["tpr_test"], metric["thresholds_test"] = roc_curve(a, b, drop_intermediate=False)
    return metric 

def train_nn_model(setting = "pointnet", boards = None, trainFs = None,
                    batch_size = 2, batch_size_val = 1, num_worker = 2, learning_rate = 0.0001, 
                    device = None, datapath = '/home/group4/Challenge3/vtp_analysis/outputs/', 
                    output_path = None, num_epoch = 50, predthreshold = 0.5):
    features_cols_txt_path: str = datapath + "/dataset/dataset_columns.txt"
    pointcloud_dir_path: str = datapath + "/pointclouds"
    split_ids: dict[str, str] = {
        "train": [datapath + "/splits/fold_0_train_ids.npy", 
                  datapath + "/splits/fold_1_train_ids.npy",
                  datapath + "/splits/fold_2_train_ids.npy",
                  datapath + "/splits/fold_3_train_ids.npy",
                  datapath + "/splits/fold_4_train_ids.npy"],
        "test": datapath + "/splits/test_ids.npy",
        "val": [datapath + "/splits/fold_0_val_ids.npy",
                datapath + "/splits/fold_1_val_ids.npy",
                datapath + "/splits/fold_2_val_ids.npy",
                datapath + "/splits/fold_3_val_ids.npy",
                datapath + "/splits/fold_4_val_ids.npy"]
    }
    # ====== training ======
    metrics = []
    for i in range(5):
        # trainset & valset
        trainset = evar_dataset.EVARDataset(datapath = datapath + "/dataset", features_cols_txt = features_cols_txt_path, 
                                            pointcloud_dir = pointcloud_dir_path, 
                                            split_ids = np.load(split_ids["train"][i])) if setting != "pinn" else evar_dataset.TimeEVARDataset(datapath = datapath + "/dataset", features_cols_txt = features_cols_txt_path, 
                                            pointcloud_dir = pointcloud_dir_path, 
                                            split_ids = np.load(split_ids["train"][i]))
        valset = evar_dataset.EVARDataset(datapath = datapath + "/dataset", features_cols_txt = features_cols_txt_path, 
                                          pointcloud_dir = pointcloud_dir_path, 
                                          split_ids = np.load(split_ids["val"][i])) if setting != "pinn" else evar_dataset.TimeEVARDataset(datapath = datapath + "/dataset", features_cols_txt = features_cols_txt_path,
                                          pointcloud_dir = pointcloud_dir_path, 
                                          split_ids = np.load(split_ids["val"][i]))
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=False,
            num_workers=num_worker, drop_last=True
            )
        val_loader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size_val, shuffle=False,
            num_workers=num_worker, drop_last=False
            )
        # model handling
        model = get_model(setting).to(device)
        # optimizer
        loss = get_loss(setting, model)
        opt = optim.Adam(model.parameters(), lr=learning_rate)
        lr_scheduler = _create_lr_scheduler(opt)
        metrics.append(train(model, opt, train_loader, val_loader, loss, i, 
                             boards[i], lr_scheduler, trainFs[i], output_path, num_epoch, predthreshold, device, setting = setting))
        print('#########################################################################################')
    # ====== testing ======
    max_auc = max([metric['auc_bestauc'] for metric in metrics])
    best_fold = [metric['auc_bestauc'] for metric in metrics ].index(max_auc)
    print(f"Best model is from fold {best_fold} with AUC: {max_auc}")
    model_best = get_model(setting = setting).to(device)
    checkpoint_best = os.path.join(output_path, 'exp_fold{}'.format(best_fold), 'best_auc.pth')
    checkpoint_best = torch.load(checkpoint_best)
    model_best.load_state_dict(checkpoint_best)
    testset = evar_dataset.EVARDataset(datapath = datapath + "/dataset", features_cols_txt = features_cols_txt_path, pointcloud_dir = pointcloud_dir_path, split_ids = np.load(split_ids["test"])) if setting != "pinn" else evar_dataset.TimeEVARDataset(datapath = datapath + "/dataset", features_cols_txt = features_cols_txt_path, pointcloud_dir = pointcloud_dir_path, split_ids = np.load(split_ids["test"]))
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size_val, shuffle=False,
        num_workers=num_worker, drop_last=False
    )
    metric_test = test(model_best, test_loader, predthreshold, device, setting = setting)
    print('#########################################################################################')
    return metrics, metric_test

def train_xgboost_model(boards = None, trainFs = None,  
                        datapath = '/home/group4/Challenge3/vtp_analysis/outputs/', output_path = None, 
                        device = None):
    features_path: str=datapath + "features/features.csv"
    labels_path: str=datapath + "dataset/dataset.csv"
    split_ids: dict[str, str] = {
        "train": [datapath + "splits/fold_0_train_ids.npy", 
                  datapath + "splits/fold_1_train_ids.npy",
                  datapath + "splits/fold_2_train_ids.npy",
                  datapath + "splits/fold_3_train_ids.npy",
                  datapath + "splits/fold_4_train_ids.npy"],
        "test": datapath + "splits/test_ids.npy",
        "val": [datapath + "splits/fold_0_val_ids.npy",
                datapath + "splits/fold_1_val_ids.npy",
                datapath + "splits/fold_2_val_ids.npy",
                datapath + "splits/fold_3_val_ids.npy",
                datapath + "splits/fold_4_val_ids.npy"]
    }
    # ====== training ======
    metrics = []
    for i in range(5):
        model = xgb.XGBClassifier()
        trainset =  pd.read_csv(features_path)[pd.read_csv(features_path)["patient_id"].str.contains('|'.join(np.load(split_ids["train"][i]).astype(str)))].to_numpy()[:, 1:]
        trainlabel = pd.read_csv(labels_path)[pd.read_csv(labels_path)["patient_id"].isin(np.load(split_ids["train"][i]))]["complication_raw"].to_numpy().reshape([-1, 1])
        trainlabel = (trainlabel != 'none').astype(int)
        valset =  pd.read_csv(features_path)[pd.read_csv(features_path)["patient_id"].str.contains('|'.join(np.load(split_ids["val"][i]).astype(str)))].to_numpy()[:, 1:]
        vallabel = pd.read_csv(labels_path)[pd.read_csv(labels_path)["patient_id"].isin(np.load(split_ids["val"][i]).tolist())]["complication_raw"].to_numpy().reshape([-1, 1])
        vallabel = (vallabel != 'none').astype(int)
        model.fit(trainset, trainlabel, eval_set=[(valset, vallabel)])
        predictions = model.predict(valset)
        metric = {
        'acc': accuracy_score(vallabel, predictions),
        'auc': roc_auc_score(vallabel, predictions),
        'acc_bestauc': accuracy_score(vallabel, predictions),
        'auc_bestauc': roc_auc_score(vallabel, predictions),
        'recall_bestauc': recall_score(vallabel, predictions, average='binary'),
        'precision_bestauc': precision_score(vallabel, predictions, average='binary'),
        'f1_score_bestauc': f1_score(vallabel, predictions, average='binary'),
        'fpr_bestauc': None,
        'tpr_bestauc': None,
        'thresholds_bestauc': None
        } 

        fpr_bestauc, tpr_bestauc, thresholds_bestauc = roc_curve(vallabel, predictions, drop_intermediate=False )
        metric["fpr_bestauc"] = fpr_bestauc
        metric["tpr_bestauc"] = tpr_bestauc
        metric["thresholds_bestauc"] = thresholds_bestauc
     
        metrics.append(metric)
    

        boards[i].add_scalar('val_acc', metric["acc"], 0)
        boards[i].add_scalar('val_auc', metric["auc"], 0)
        print('fold {} epoch {}:'.format(i, 0))
        print('  val_acc     : {}'.format(metric["acc"]))
        print('  val_auc     : {}'.format(metric["auc"]))
        trainFs[i].write('fold{},epoch,{}\n'.format(i, 0))
        trainFs[i].write('fold{},val_acc,{}\n'.format(i, metric["acc"]))
        trainFs[i].write('fold{},val_auc,{}\n'.format(i, metric["auc"]))
        trainFs[i].write('####################################################\n')
    # ====== testing ======
    max_auc = max([ metric['auc_bestauc'] for metric in metrics ])
    best_fold = [metric['auc_bestauc'] for metric in metrics ].index(max_auc)
    print(f"Best model is from fold {best_fold} with AUC: {max_auc}")
    checkpoint_best = os.path.join(output_path, 'exp_fold{}'.format(best_fold), 'best_auc.pth')
    print(checkpoint_best)
    model_best = xgb.XGBClassifier()
    model_best.load_model(checkpoint_best)
    testset =  pd.read_csv(features_path)[pd.read_csv(features_path)["patient_id"].str.contains('|'.join(np.load(split_ids["test"]).astype(str)))].to_numpy()[:, 1:]
    testlabel = pd.read_csv(labels_path)[pd.read_csv(labels_path)["patient_id"].isin(np.load(split_ids["test"]).tolist())]["complication_raw"].to_numpy().reshape([-1, 1])
    testlabel = (testlabel != 'none').astype(int)
    predictions = model_best.predict(testset)
    metric_test = {
        'acc_test': accuracy_score(testlabel, predictions),
        'auc_test': roc_auc_score(testlabel, predictions),
        'recall_test': recall_score(testlabel, predictions, average='binary'),
        'precision_test': precision_score(testlabel, predictions, average='binary'),
        'f1_score_test': f1_score(testlabel, predictions, average='binary'),
        'fpr_test': None,
        'tpr_test': None,
        'thresholds_test': None
    } 
    metric_test["fpr_test"], metric_test["tpr_test"], metric_test["thresholds_test"] = roc_curve(testlabel, predictions, drop_intermediate=False)
    print('#########################################################################################')
    return metrics, metric_test

def train_model(setting = "pointnet", boards = None, trainFs = None, batch_size = 2, 
                 batch_size_val = 1, num_worker = 2, learning_rate = 0.0001, 
                 device = None, datapath = '/home/group4/Challenge3/vtp_analysis/outputs/', output_path = None,
                 num_epoch = 50, predthreshold = 0.5):
    if setting == "xgboost":
        return train_xgboost_model(boards = boards, trainFs = trainFs, 
                                   datapath = datapath, 
                                   output_path = output_path)
    else:
        return train_nn_model(setting = setting, boards = boards, trainFs = trainFs, 
                              batch_size = batch_size, batch_size_val = batch_size_val,
                              num_worker = num_worker, learning_rate = learning_rate, 
                              device = device, datapath = datapath, output_path = output_path, 
                              num_epoch = num_epoch, predthreshold = predthreshold)
     